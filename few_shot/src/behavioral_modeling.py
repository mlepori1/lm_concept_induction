import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.stats
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from helpers import model_helpers


def compute_effective_directions(prototypes, U, sigma):
    """Compute the per-row anisotropically stretched prototype direction w2(S), such that
    s_m(y, S) = e_y . w2(S) for a raw (non-unit-normalized) vocab embedding e_y.

    w2 = normalize(U @ diag(sigma) @ U.T @ p) for each prototype p: decompose p along A's
    own input (left-singular-vector) directions, scale each coefficient by that
    direction's singular value, then reconstruct in the SAME U basis.

    Args:
        prototypes: (n, d) support prototypes p_S
        U: (d, d) left singular vectors of the best-layer ridge map A (A's own input/
            prototype-space directions -- NOT A's right singular vectors V/Vh)
        sigma: (d,) singular values, matching U's columns

    Returns:
        w2: (n, d)
    """
    P = prototypes
    norm = lambda x: np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    P_in_U_basis   = P @ U                              # (n, d) prototype coefficients along A's input directions
    P_stretched    = P_in_U_basis * sigma[None, :]       # (n, d) each coefficient scaled by its direction's gain
    P_stretched_unit = P_stretched / norm(P_stretched)   # (n, d) unit-normalized before reconstructing (U is
                                                          #        orthonormal, so this is equivalent to normalizing after)
    w2 = P_stretched_unit @ U.T   # (n, d) reconstructed in the SAME U basis -- a pure stretch, no rotation

    return w2


def compute_vocab_scores(W, vocab_table, row_chunk_size, device):
    """s_m(y, S) for every row S (rows of W, already the effective direction w_m(S)) and
    EVERY vocab item y (not just the true target -- this full matrix is what the log
    partition function sums over). Returns an (n, vocab_size) torch tensor on `device`,
    computed in row chunks to bound peak memory.

    This allows us to amortize the cost of computing the (n, vocab_size) score matrix.
    """
    n = W.shape[0]
    all_vocab_scores = torch.empty((n, vocab_table.shape[0]), dtype=torch.float32, device=device)
    W_t = torch.as_tensor(W, dtype=torch.float32, device=device)
    for start in range(0, n, row_chunk_size):
        end = min(start + row_chunk_size, n)
        all_vocab_scores[start:end] = W_t[start:end] @ vocab_table.T
    return all_vocab_scores


def fit_beta(all_vocab_scores_train, target_scores_train, base_log_probs, observed_boost_train,
             fit_intercept, max_iter, lr):
    """LBFGS fit of beta (and optionally an intercept) minimizing MSE between the
    predicted and observed contextual log-probability boost, on train rows only.

        predicted_boost_i = [intercept +] beta * target_scores_train[i]
                             - logsumexp_y'[log_p0(y') + beta * all_vocab_scores_train[i, y']]

    beta is softplus-constrained to stay positive. all_vocab_scores_train/target_scores_train
    are already restricted to train ROWS by the caller; the logsumexp below still sums over
    the full vocab (dim=1) for each of those rows -- only which support sets are used to
    fit beta is restricted, never the vocabulary the partition function sums over.
    """
    device = all_vocab_scores_train.device
    raw_beta = torch.nn.Parameter(torch.tensor(-2.0, device=device))
    intercept = torch.nn.Parameter(torch.tensor(0.0, device=device))
    params = [raw_beta] + ([intercept] if fit_intercept else [])

    optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter, history_size=10,
                                   line_search_fn="strong_wolfe")

    def predicted_boost(beta):
        # dim=1 is the vocab axis -- full vocab, every time.
        log_partition = torch.logsumexp(base_log_probs[None, :] + beta * all_vocab_scores_train, dim=1)
        pred = beta * target_scores_train - log_partition
        if fit_intercept:
            pred = pred + intercept
        return pred

    def closure():
        optimizer.zero_grad()
        beta = torch.nn.functional.softplus(raw_beta)
        loss = torch.nn.functional.mse_loss(predicted_boost(beta), observed_boost_train)
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        beta = torch.nn.functional.softplus(raw_beta)
    return float(beta.item()), float(intercept.item()) if fit_intercept else 0.0


def predicted_boost_for(beta, intercept, all_vocab_scores, target_scores, base_log_probs):
    with torch.no_grad():
        beta_t = torch.tensor(beta, device=all_vocab_scores.device)
        log_partition = torch.logsumexp(base_log_probs[None, :] + beta_t * all_vocab_scores, dim=1)
        pred = beta_t * target_scores - log_partition + intercept
    return pred.cpu().numpy()


def fit_direction_model(W, target_embeddings, vocab_table, all_base_log_probs,
                         observed_boost_t, observed_boost, train_mask, heldout_mask,
                         noise_ceiling_r, args, device,
                         all_vocab_scores=None, target_scores=None):
    """Fits beta/intercept for effective-direction matrix W (n, d) on train rows, and
    evaluates held-out pearson r / rmse against the observed contextual log-prob boost.

    all_vocab_scores/target_scores: pass these in (already computed for this exact W) to
    skip recomputing compute_vocab_scores -- useful when the same direction is refit
    against many different train/heldout splits without the direction itself ever
    changing (e.g. full_rank_prototype's w1 across every --leave_one_concept_out concept).
    When omitted, they're computed fresh here and freed before returning, as before.

    Returns:
        beta, intercept: fitted temperature/intercept
        pred_heldout: (n_heldout,) predicted boost
        summary: dict with beta/intercept/pearson_r/pearson_p/rmse/n/noise_ceiling_r/pearson_r_normalized
    """
    # Check if the scores and target scores are already computed.
    owns_scores = all_vocab_scores is None
    if owns_scores:
        all_vocab_scores = compute_vocab_scores(W, vocab_table, args.row_chunk_size, device)  # (n, vocab)
    if target_scores is None:
        target_scores = torch.as_tensor(
            (W * target_embeddings).sum(axis=1), dtype=torch.float32, device=device
        )

    # [train_mask]/[heldout_mask] select which ROWS (support-set instances) to use for
    # fitting vs. evaluation -- the partition function inside fit_beta always sums over the complete vocabulary.
    beta, intercept = fit_beta(
        all_vocab_scores[train_mask], target_scores[train_mask], all_base_log_probs,
        observed_boost_t[train_mask], args.fit_intercept, args.lbfgs_max_iter, args.lbfgs_lr,
    )
    pred_heldout = predicted_boost_for(
        beta, intercept, all_vocab_scores[heldout_mask], target_scores[heldout_mask], all_base_log_probs
    )
    if owns_scores:
        del all_vocab_scores  # free the (n, vocab_size) matrix before the next layer/model

    obs_heldout = observed_boost[heldout_mask]
    pearson = scipy.stats.pearsonr(obs_heldout, pred_heldout)
    rmse = float(np.sqrt(np.mean((obs_heldout - pred_heldout) ** 2)))

    summary = {
        "beta": beta,
        "intercept": intercept,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "rmse": rmse,
        "n": int(heldout_mask.sum()),
        "noise_ceiling_r": noise_ceiling_r,
        "pearson_r_normalized": float(pearson.statistic) / noise_ceiling_r,
    }
    return beta, intercept, pred_heldout, summary


if __name__ == "__main__":
    """Fits the episode-score behavioral models, using the ridge map A fit by latent_regressions.py and the
    log-probabilities/embeddings collected by harvest_model_data.py.

    Each model tilts the base (BOS-only) vocab distribution by an episode-specific score
    s_m(y, S) and a fitted temperature beta_m, and is evaluated by how well the resulting
    predicted contextual log-probability boost correlates with the actually observed one
    on held-out support sets -- mirroring latent_regressions.py's own train/held-out split
    (all categories visible in both).

    full_rank_prototype doesn't depend on A at all, so it's fit once. anisotropically_stretched_direction
    does depend on A, so it's fit separately at EVERY layer (requires latent_regressions.py to have
    been run with --save_all_A) -- layer_comparison.csv has one row per (layer, model), with
    full_rank_prototype's single fit repeated at every layer for direct comparison/plotting.
    predictions.csv/summary.json still hold only the canonical single fit per model: full_rank_prototype's
    fit, and anisotropically_stretched_direction's fit at the best layer (per best_layer.json).

    With --leave_one_concept_out (requires latent_regressions.py's own --leave_one_concept_out
    to have been run first), additionally evaluates full_rank_prototype and
    anisotropically_stretched_direction (AT THE BEST LAYER ONLY) as a clean unseen-concept
    test: for each concept, BOTH the direction and the beta/intercept temperature are refit
    using only every other concept's train rows (this concept's train AND held-out rows are
    untouched by either fit), then evaluated on this concept's held-out rows.
    anisotropically_stretched_direction's direction additionally swaps in that concept's own
    loco ridge map (from latent_regressions.py's loco_A.npy). Predictions are pooled across
    all LOCO concepts into one aggregate pearson r/rmse per model, in summary.json --
    loco_per_concept.csv additionally holds each individual concept's OWN pearson r/rmse
    (n = that concept's held-out row count, typically ~5, so noisy; use the pooled,
    aggregate number in summary.json for the statistically meaningful comparison).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="HF model id")
    parser.add_argument("--save_root", type=str, default="../results")
    parser.add_argument("--fit_intercept", action="store_true", default=False,
                        help="Fit a free intercept per model (default: no intercept")
    parser.add_argument("--lbfgs_max_iter", type=int, default=100)
    parser.add_argument("--lbfgs_lr", type=float, default=1.0)
    parser.add_argument("--row_chunk_size", type=int, default=256,
                        help="Rows per chunk when computing the (n_rows, vocab_size) "
                             "score matrices, to bound peak memory")
    parser.add_argument("--leave_one_concept_out", action="store_true", default=False,
                        help="Clean unseen-concept test: for each concept, refit both the "
                             "direction and the beta/intercept temperature on every other "
                             "concept's train rows only, then evaluate on this concept's "
                             "held-out rows. Substantially more expensive than the main fit "
                             "for anisotropically_stretched_direction (see run_loco).")
    parser.add_argument("--baseline_log_probs_path", type=str, default=None,
                        help="Path to an alternative (vocab_size,) P_0(y) tensor, used in place of the "
                             "BOS-only baseline (all_base_log_probs.pt) EVERYWHERE P_0(y) "
                             "appears -- both re-deriving observed_boost/shuffled_boost per "
                             "row (records['log_p_target_bos'] was baked in at harvest time "
                             "against BOS specifically, so it can't be reused here) and "
                             "inside fit_beta's own partition function, so the model's "
                             "predicted boost and the observed boost it's compared against "
                             "stay defined relative to the same P_0. Output goes to "
                             "'behavioral_modeling_<tag>' instead of 'behavioral_modeling', "
                             "<tag> being this path's basename without extension.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available -- this script requires a GPU.")
    device = torch.device("cuda")

    model_name_safe = args.model_name.replace("/", "_")
    data_dir = os.path.join(args.save_root, model_name_safe, "concept_list_data")
    regression_dir = os.path.join(args.save_root, model_name_safe, "displacement_regression")

    # ── Load harvest_model_data.py output ───────────────────────────────────────
    records = pd.read_csv(
        os.path.join(data_dir, "records.csv"),
        keep_default_na=False, # Make sure that "nan" is not coerced into a float 
        na_values={"log_p_target_given_shuffled_context": [""]},
    )
    prototypes = np.load(os.path.join(data_dir, "prototypes.npy")).astype(np.float32)
    target_embeddings = np.load(os.path.join(data_dir, "target_embeddings.npy")).astype(np.float32)
    baseline_path = args.baseline_log_probs_path or os.path.join(data_dir, "all_base_log_probs.pt")
    all_base_log_probs = torch.load(baseline_path, map_location=device).float()
    if args.baseline_log_probs_path is not None:
        print(f"Using alternative P_0(y) baseline: {baseline_path}")

    with open(os.path.join(data_dir, "hyperparams.json")) as f:
        embedding_scale = json.load(f)["embedding_scale"]

    # ── Load latent_regressions.py output ───────────────────────────────────────
    all_A_path = os.path.join(regression_dir, "all_A.npy")
    if not os.path.exists(all_A_path):
        raise FileNotFoundError(
            f"{all_A_path} not found -- run latent_regressions.py with --save_all_A first "
            f"(behavioral_modeling.py fits anisotropically_stretched_direction at every layer)."
        )
    all_A = np.load(all_A_path).astype(np.float32)   # (n_layers, d, d)
    with open(os.path.join(regression_dir, "best_layer.json")) as f:
        best_layer_info = json.load(f)
    best_layer = best_layer_info["best_layer"]
    n_layers = all_A.shape[0]
    print(f"Loaded {n_layers} layers' A matrices from {all_A_path}; best layer = {best_layer}")

    # ── Vocab table, in the same embedding space as prototypes/target_embeddings ──
    print(f"Loading model for vocab embeddings: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", trust_remote_code=True)
    vocab_table = (
        model.get_input_embeddings().weight.detach().float().cpu() * embedding_scale
    ).to(device)
    del model


    train_mask   = (records["split"] == "train").to_numpy()
    heldout_mask = (records["split"] == "heldout").to_numpy()

    if args.baseline_log_probs_path is not None:
        # records["log_p_target_bos"] was baked in at harvest time specifically against the
        # BOS-only baseline, so it can't be reused for a different P_0(y) -- recompute per
        # row by looking up each row's own target token in the new baseline instead.
        target_ids = np.array([model_helpers.get_token_id(t, tokenizer) for t in records["target"]])
        log_p_target_baseline = all_base_log_probs[target_ids].cpu().numpy()
    else:
        log_p_target_baseline = records["log_p_target_bos"].to_numpy()

    observed_boost = records["log_p_target_given_context"].to_numpy() - log_p_target_baseline
    shuffled_boost = records["log_p_target_given_shuffled_context"].to_numpy() - log_p_target_baseline

    noise_ceiling = scipy.stats.pearsonr(
        observed_boost[heldout_mask], shuffled_boost[heldout_mask]
    )
    noise_ceiling_r = float(noise_ceiling.statistic)
    print(f"Noise ceiling (held-out): r = {noise_ceiling_r:.4f} (p = {noise_ceiling.pvalue:.3e})")

    observed_boost_t = torch.as_tensor(observed_boost, dtype=torch.float32, device=device)

    all_predictions = []
    all_summaries = {"noise_ceiling": {"pearson_r": noise_ceiling_r}}
    layer_rows = []     # one row per (layer, model): the cross-layer stretched-vs-prototype comparison

    # Arguments for fit_direction_model, defined once here for convenience.
    fit_kwargs = dict(
        target_embeddings=target_embeddings, vocab_table=vocab_table, all_base_log_probs=all_base_log_probs,
        observed_boost_t=observed_boost_t, observed_boost=observed_boost, train_mask=train_mask,
        heldout_mask=heldout_mask, noise_ceiling_r=noise_ceiling_r, args=args, device=device,
    )

    # ── full_rank_prototype: doesn't depend on A, fit once ──────────────────────
    print("Fitting model: full_rank_prototype")
    w1 = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True).clip(min=1e-12)
    if args.leave_one_concept_out:
        # Kept around for reuse in the LOCO loop below: w1 never varies by concept, 
        # so there's no need to recompute this (n, vocab) matrix once per LOCO concept.
        w1_all_vocab_scores = compute_vocab_scores(w1, vocab_table, args.row_chunk_size, device)
        w1_target_scores = torch.as_tensor(
            (w1 * target_embeddings).sum(axis=1), dtype=torch.float32, device=device
        )
        beta, intercept, pred_heldout, summary = fit_direction_model(
            w1, all_vocab_scores=w1_all_vocab_scores, target_scores=w1_target_scores, **fit_kwargs
        )
    else:
        beta, intercept, pred_heldout, summary = fit_direction_model(w1, **fit_kwargs)

    results = records[heldout_mask].copy()
    results["model"] = "full_rank_prototype"
    results["observed_boost"] = observed_boost[heldout_mask]
    results["predicted_boost"] = pred_heldout
    results["residual"] = observed_boost[heldout_mask] - pred_heldout
    all_predictions.append(results)

    all_summaries["full_rank_prototype"] = summary
    for layer_idx in range(n_layers):
        layer_rows.append({"layer": layer_idx, "model": "full_rank_prototype", **summary})
    print(f"  [full_rank_prototype] r = {summary['pearson_r']:.4f} "
          f"(r/ceiling = {summary['pearson_r_normalized']:.4f})  beta = {beta:.4f}")

    # ── anisotropically_stretched_direction: depends on A, fit at EVERY layer ──
    for layer_idx in tqdm(range(n_layers), desc="Fitting stretched-direction model per layer"):
        # Vh (A's right singular vectors) is intentionally discarded here: compute_effective_directions
        # only needs A's input-side directions (U) and singular values to build a pure anisotropic
        # stretch in embedding space.
        U_layer, sigma_layer, _ = np.linalg.svd(all_A[layer_idx], full_matrices=False)
        w2_layer = compute_effective_directions(prototypes, U_layer, sigma_layer)

        beta, intercept, pred_heldout, summary = fit_direction_model(w2_layer, **fit_kwargs)
        layer_rows.append({"layer": layer_idx, "model": "anisotropically_stretched_direction", **summary})

        if layer_idx == best_layer:
            # Canonical fit for predictions.csv/summary.json/--leave_one_concept_out
            results = records[heldout_mask].copy()
            results["model"] = "anisotropically_stretched_direction"
            results["observed_boost"] = observed_boost[heldout_mask]
            results["predicted_boost"] = pred_heldout
            results["residual"] = observed_boost[heldout_mask] - pred_heldout
            all_predictions.append(results)

            all_summaries["anisotropically_stretched_direction"] = summary
            print(f"  [anisotropically_stretched_direction] best layer {best_layer}: "
                  f"r = {summary['pearson_r']:.4f} (r/ceiling = {summary['pearson_r_normalized']:.4f})  "
                  f"beta = {beta:.4f}")

    # ── Leave-one-concept-out: a clean unseen-concept test ──────────────────────
    # For EACH concept, both the DIRECTION and the beta/intercept TEMPERATURE are refit
    # using only every OTHER concept's train rows -- this concept's data (train or
    # held-out) is never touched by either fit -- then evaluated on this concept's
    # held-out rows. Predictions are pooled across all LOCO concepts (each scored with
    # its own refit beta) into one aggregate pearson r/rmse.

    if args.leave_one_concept_out:
        loco_A = np.load(os.path.join(regression_dir, "loco_A.npy")).astype(np.float32)  # (n_concepts, d, d)
        with open(os.path.join(regression_dir, "loco_categories.json")) as f:
            loco_categories = json.load(f)

        categories_arr = records["category"].to_numpy()
        eval_mask = heldout_mask & records["category"].isin(loco_categories).to_numpy()

        # This is a well-formedness check.
        if eval_mask.sum() < heldout_mask.sum():
            print(f"leave_one_concept_out: {heldout_mask.sum() - eval_mask.sum()} held-out rows have no "
                  f"matching concept in loco_categories.json and will be skipped")

        def run_loco(direction_fn, precomputed=None):
            """For each LOCO concept: build a direction matrix via direction_fn(A_cat,
            combined_mask), refit beta/intercept using every OTHER concept's train rows
            only (fit_mask_cat), then predict this concept's held-out rows (test_mask_cat).

            precomputed, if given, is an (all_vocab_scores_full, target_scores_full) pair
            -- only valid when direction_fn's output doesn't actually vary by concept.
            This is the case for full_rank_prototype, but not for anisotropically_stretched_direction.

            Returns (row_positions, pred, betas, intercepts, per_concept_rows):
            row_positions are integer positions into `records`, pooled across concepts
            (grouped by concept rather than original row order); 
            betas/intercepts are broadcast to one value per row (that row's concept-specific fit), for easy
            per-row inspection in predictions.csv; 
            per_concept_rows: has one dict per concept with that concept's OWN (small-n, so noisy -- see docstring above)
            pearson r/rmse, for loco_per_concept.csv.
            """
            row_positions, preds, betas, intercepts, per_concept_rows = [], [], [], [], []
            for cat, A_cat in zip(loco_categories, loco_A):
                test_mask_cat = eval_mask & (categories_arr == cat)
                if not test_mask_cat.any():
                    continue
                fit_mask_cat = train_mask & (categories_arr != cat)
                combined_mask = fit_mask_cat | test_mask_cat

                W_sub = direction_fn(A_cat, combined_mask)
                # fit_mask_cat/test_mask_cat are boolean arrays over ALL n_total records, but
                # W_sub only has combined_mask.sum() rows -- so re-index them by combined_mask
                # (boolean-array-by-boolean-array) to get a same-length boolean mask marking,
                # among just the rows that survived into W_sub, which are fit vs. test rows.
                model_kwargs = dict(
                    target_embeddings=target_embeddings[combined_mask], vocab_table=vocab_table,
                    all_base_log_probs=all_base_log_probs, observed_boost_t=observed_boost_t[combined_mask],
                    observed_boost=observed_boost[combined_mask], train_mask=fit_mask_cat[combined_mask],
                    heldout_mask=test_mask_cat[combined_mask], noise_ceiling_r=noise_ceiling_r,
                    args=args, device=device,
                )
                if precomputed is not None:
                    all_scores_full, target_scores_full = precomputed
                    model_kwargs["all_vocab_scores"] = all_scores_full[combined_mask]
                    model_kwargs["target_scores"] = target_scores_full[combined_mask]

                beta_cat, intercept_cat, pred_cat, _ = fit_direction_model(W_sub, **model_kwargs)

                n_cat = int(test_mask_cat.sum())
                obs_cat = observed_boost[test_mask_cat]
                rmse_cat = float(np.sqrt(np.mean((obs_cat - pred_cat) ** 2)))
                if n_cat >= 2:
                    pearson_cat = scipy.stats.pearsonr(obs_cat, pred_cat)
                    pearson_r_cat, pearson_p_cat = float(pearson_cat.statistic), float(pearson_cat.pvalue)
                else:
                    # A single held-out row has no variance to correlate against.
                    pearson_r_cat, pearson_p_cat = float("nan"), float("nan")

                row_positions.append(np.where(test_mask_cat)[0])
                preds.append(pred_cat)
                betas.append(np.full(n_cat, beta_cat))
                intercepts.append(np.full(n_cat, intercept_cat))
                per_concept_rows.append({
                    "category": cat, "n": n_cat, "beta": beta_cat, "intercept": intercept_cat,
                    "pearson_r": pearson_r_cat, "pearson_p": pearson_p_cat, "rmse": rmse_cat,
                })
            return (np.concatenate(row_positions), np.concatenate(preds),
                    np.concatenate(betas), np.concatenate(intercepts), per_concept_rows)

        def summarize_loco(model_name, row_positions, pred_loco, betas, intercepts, per_concept_rows):
            obs_loco = observed_boost[row_positions]
            pearson_loco = scipy.stats.pearsonr(obs_loco, pred_loco)
            rmse_loco = float(np.sqrt(np.mean((obs_loco - pred_loco) ** 2)))

            loco_results = records.iloc[row_positions].copy()
            loco_results["model"] = model_name
            loco_results["beta"] = betas
            loco_results["intercept"] = intercepts
            loco_results["observed_boost"] = obs_loco
            loco_results["predicted_boost"] = pred_loco
            loco_results["residual"] = obs_loco - pred_loco
            all_predictions.append(loco_results)

            for row in per_concept_rows:
                loco_per_concept_rows.append({"model": model_name, **row})

            n_concepts = loco_results["category"].nunique()
            all_summaries[model_name] = {
                "mean_beta": float(np.mean(betas)),
                "mean_intercept": float(np.mean(intercepts)),
                "pearson_r": float(pearson_loco.statistic),
                "pearson_p": float(pearson_loco.pvalue),
                "rmse": rmse_loco,
                "n": int(len(row_positions)),
                "n_concepts": int(n_concepts),
                "noise_ceiling_r": noise_ceiling_r,
                "pearson_r_normalized": float(pearson_loco.statistic) / noise_ceiling_r,
            }
            print(f"  [{model_name}] r = {pearson_loco.statistic:.4f} "
                  f"(r/ceiling = {float(pearson_loco.statistic) / noise_ceiling_r:.4f}) "
                  f"over {n_concepts} concepts, each with its own refit beta/intercept "
                  f"(mean beta = {np.mean(betas):.4f})")

        # Collects one row per (model, concept) with that concept's OWN pearson r/rmse
        # (small n per concept, so noisy).
        loco_per_concept_rows = []

        # full_rank_prototype_loco: direction never depends on A, so it's identical across
        # concepts -- only beta/intercept differ per concept, and w1's scores are reused
        # from the main fit above rather than recomputed.
        row_pos, pred_loco, betas, intercepts, per_concept = run_loco(
            lambda A_cat, mask: w1[mask], precomputed=(w1_all_vocab_scores, w1_target_scores)
        )
        summarize_loco("full_rank_prototype_loco", row_pos, pred_loco, betas, intercepts, per_concept)

        # anisotropically_stretched_direction_loco: both the direction (via that concept's
        # own loco ridge map) and the temperature are refit excluding this concept entirely.
        def stretched_direction_for(A_cat, mask):
            U_cat, sigma_cat, _ = np.linalg.svd(A_cat, full_matrices=False)  # Vh discarded, see compute_effective_directions
            return compute_effective_directions(prototypes[mask], U_cat, sigma_cat)

        row_pos, pred_loco, betas, intercepts, per_concept = run_loco(stretched_direction_for)
        summarize_loco("anisotropically_stretched_direction_loco", row_pos, pred_loco, betas, intercepts, per_concept)

        del w1_all_vocab_scores, w1_target_scores  # free the (n, vocab_size) matrix now that LOCO is done

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.baseline_log_probs_path is not None:
        baseline_tag = os.path.splitext(os.path.basename(args.baseline_log_probs_path))[0]
        results_dirname = f"behavioral_modeling_{baseline_tag}"
    else:
        results_dirname = "behavioral_modeling"
    save_dir = os.path.join(args.save_root, model_name_safe, results_dirname)
    os.makedirs(save_dir, exist_ok=True)

    pd.concat(all_predictions, ignore_index=True).to_csv(
        os.path.join(save_dir, "predictions.csv"), index=False
    )

    pd.DataFrame(layer_rows).to_csv(os.path.join(save_dir, "layer_comparison.csv"), index=False)

    if args.leave_one_concept_out:
        pd.DataFrame(loco_per_concept_rows).to_csv(
            os.path.join(save_dir, "loco_per_concept.csv"), index=False
        )

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    hyperparams = vars(args)
    with open(os.path.join(save_dir, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    print(f"Saved predictions, summary, and layer_comparison.csv to {save_dir}")
