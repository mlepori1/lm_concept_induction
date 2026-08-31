import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from helpers import data as data_module
from helpers import model_helpers


def draw_distinct_support_sequences(pool, support_len, n_draws, label):
    """Draw up to n_draws support sequences from pool, each of length support_len, with no
    two draws sharing the exact same ORDERED sequence.

    Whenever pool can't support n_draws distinct orderings, 
    this caps at however many exist and warns, rather than silently repeating a draw.

    Args:
        pool: list of tokens to draw from
        support_len: length of the support sequences to draw
        n_draws: number of distinct support sequences to draw
        label: label for the pool, used for error messages
    Returns a list of up to n_draws lists, each of length support_len.
    """
    if len(pool) < support_len:
        raise ValueError(
            f"'{label}' pool has only {len(pool)} members, fewer than "
            f"support_len={support_len}; cannot draw a support set."
        )

    max_distinct = math.perm(len(pool), support_len)
    n_draws_actual = min(n_draws, max_distinct)
    if n_draws_actual < n_draws:
        print(
            f"Warning: '{label}' pool (size {len(pool)}) only supports {max_distinct} "
            f"distinct ordered support draws of length {support_len}; requested {n_draws}, "
            f"using {n_draws_actual}."
        )

    seen = set()
    draws = []
    while len(draws) < n_draws_actual:
        candidate = rng.permutation(pool).tolist()[:support_len]
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        draws.append(candidate)
    return draws


def compute_mean_context_displacement_vector(
    args, train_members, tokenizer, model, context_id, mean_disp, label
):
    """Compute the mean residualised context-displacement vector for a concept.

    For each of n_supports DISTINCT ordered support sets drawn from train_members:
        1. Run the full concept list (e.g. "[BOS] robin sparrow finch wren")
        2. Run the last token alone (e.g. "[BOS] wren")
        3. Diff them
        4. Residualise: subtract mean_disp to remove position/length effects

    Return the unit-normalised mean over all support sets: shape (n_layers, d).
    """
    displacements = []
    for support_tokens in draw_distinct_support_sequences(
        train_members, args.support_len, args.n_supports, label
    ):
        target_token = support_tokens[-1]
        hs_full, _ = model_helpers.get_hidden_states_and_logprobs_last_token(
            " " + " ".join(support_tokens), tokenizer, model, context_id
        )
        hs_alone, _ = model_helpers.get_hidden_states_and_logprobs_last_token(
            " " + target_token, tokenizer, model, context_id
        )
        residualised = hs_full - hs_alone - mean_disp
        displacements.append(residualised)
    return F.normalize(torch.stack(displacements).mean(dim=0), dim=-1)  # (n_layers, d)


if __name__ == "__main__":
    """Causal steering experiment.

    Tests whether the average context displacement direction (derived from context_displacement_analysis.py)
    causally controls the model's concept predictions.

    We take a support sequence like [BOS, robin, sparrow, finch, wren] and, at a chosen
    transformer layer, add a steering vector at the last residual stream.

    We compute these average context displacement vectors from a training set of items,
    and evaludate whether the model becomes more or less likely to predict held out
    in-category words.

    We test several conditions:
        baseline logprobs
        positive steering
        negative steering
        control category steering (steer toward a different concept)
        random steering (unit normed Gaussian vector)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b")
    parser.add_argument("--n_categories", type=int, default=113)
    parser.add_argument("--n_supports", type=int, default=5)
    parser.add_argument("--support_len", type=int, default=4)
    parser.add_argument(
        "--n_holdout_in_cat",
        type=int,
        default=4,
        help="In-category tokens held out per concept for evaluation; "
        "the mean displacement vector is recomputed using only the remaining members",
    )
    parser.add_argument(
        "--n_holdout_out_cat",
        type=int,
        default=4,
        help="Held-out out-of-category tokens per concept",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[5, 10, 20, 50],
        help="Steering magnitude (scalar multiplier on the unit-normed steering vector)",
    )
    parser.add_argument(
        "--steer_layers",
        type=str,
        default="all",
        help="Comma-separated layer indices to inject the steering vector at, or 'all'",
    )
    parser.add_argument(
        "--context_displacement_dir",
        type=str,
        default=None,
        help="Path to context_displacement_analysis results dir (defaults to "
             "results/MODEL_NAME/context_displacement_analysis)",
    )
    parser.add_argument("--save_root", type=str, default="../results")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../data/categories_data.csv"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Model loading ─────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype="auto", trust_remote_code=True
    )
    model.eval()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available -- this script requires a GPU.")
    device = torch.device("cuda")
    model = model.to(device)
    print(f"Model loaded on {device}")

    context_id = tokenizer.bos_token_id
    if context_id is None:
        raise ValueError("Tokenizer does not have a BOS token ID")

    # Locate the list of transformer layers (works for LLaMA/Gemma/GPT-2 families)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model_layers = model.model.layers  # LLaMA / Gemma / OLMo
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        model_layers = model.transformer.h  # GPT-2
    else:
        raise ValueError(f"Cannot locate transformer layers for {type(model)}")

    n_transformer_layers = len(model_layers)

    if args.steer_layers == "all":
        layers_to_steer = list(range(n_transformer_layers))
    else:
        layers_to_steer = [int(x.strip()) for x in args.steer_layers.split(",")]

    # ── Load context_displacement_analysis results ──────────────────────────────
    model_name_safe = args.model_name.replace("/", "_")
    if args.context_displacement_dir is None:
        args.context_displacement_dir = os.path.join(
            args.save_root, model_name_safe, "context_displacement_analysis"
        )

    # mean_disp[k, :] is the average context displacement at hidden-state index k.
    #
    # HuggingFace outputs.hidden_states indexing convention (used throughout this codebase):
    #   index 0   → input embeddings (before any transformer block)
    #   index k≥1 → output of transformer block k-1  (i.e. model_layers[k-1])
    #
    # So mean_disp has shape (n_transformer_layers + 1, d):
    #   mean_disp[0]   = embedding-layer displacement
    #   mean_disp[k]   = displacement at the OUTPUT of transformer block k-1
    mean_disp = torch.tensor(
        np.load(os.path.join(args.context_displacement_dir, "mean_disp.npy")),
        dtype=torch.float32,
    )  # shape (n_transformer_layers + 1, d)
    assert mean_disp.shape[0] == n_transformer_layers + 1, (
        f"mean_disp has {mean_disp.shape[0]} entries but expected "
        f"n_transformer_layers + 1 = {n_transformer_layers + 1}. "
        "Make sure mean_disp.npy was produced by the same model."
    )

    with open(os.path.join(args.context_displacement_dir, "categories.json")) as f:
        cdv_categories = json.load(f)

    # For each concept, find the semantically most dissimilar other concept.
    # Used as the "control_cat" steering condition.
    embed_hyp = np.load(
        os.path.join(args.context_displacement_dir, "embed_hyp.npy")
    )  # (n_cat, d), unit-normed per context_displacement_analysis.py
    concept_similarity = (
        embed_hyp @ embed_hyp.T
    )  # (n_cat, n_cat) cosine-similarity matrix
    least_similar_idx = np.argmin(np.abs(concept_similarity), axis=1)
    control_cats = {
        cat: cdv_categories[least_similar_idx[i]]
        for i, cat in enumerate(cdv_categories)
    }

    # ── Data loading ──────────────────────────────────────────────────────────
    print("Loading and filtering data...")

    raw_data = pd.read_csv(args.data_path, keep_default_na=False) # Don't turn word 'nan' into a float NaN
    filtered_data = data_module.filter_multitoken_data(raw_data, tokenizer)

    # Keep only concepts that appeared in the context_displacement_analysis run. This keeps
    # analyses consistent.
    cat_members = {
        cat: filtered_data[filtered_data["category"] == cat]["category.member"].tolist()
        for cat in cdv_categories
        if cat in filtered_data["category"].values
    }
    categories = [c for c in cdv_categories if c in cat_members][: args.n_categories]

    # ── Fixed random steering vectors (one per concept) ─────────-------------
    # Used as the "random" control condition: same magnitude, random direction.
    hidden_dim = mean_disp.shape[-1]
    random_steering_vecs = {
        cat: F.normalize(
            torch.tensor(rng.standard_normal(hidden_dim), dtype=torch.float32), dim=0
        )
        for cat in categories
    }

    # ─────────-------------Steering Section ─────────-------------#
    # For each concept, compute a stable averaged trajectory vector (using only
    # training members) and steer at each layer × alpha combination.
    rows = []

    for cat in tqdm(categories, desc="Steering"):
        members = cat_members[cat]
        if len(members) < args.n_holdout_in_cat + args.support_len:
            raise ValueError(
                f"{cat} has too few items to support {args.n_holdout_in_cat} held out examples"
            )

        # Hold out some in-category tokens for evaluation (never seen during mean context displacement computation)
        all_shuffled = rng.permutation(members).tolist()
        holdout_members = all_shuffled[: args.n_holdout_in_cat]
        train_members = all_shuffled[args.n_holdout_in_cat :]

        # mean context displacement: averaged, residualised displacement direction for this concept
        mean_context_displacement = compute_mean_context_displacement_vector(
            args, train_members, tokenizer, model, context_id, mean_disp, label=cat
        )  # (n_layers, d)

        # Control concept: most semantically dissimilar concept, for the control_cat condition
        control_cat = control_cats[cat]
        control_displacement = compute_mean_context_displacement_vector(
            args, cat_members[control_cat], tokenizer, model, context_id, mean_disp,
            label=f"{control_cat} (control for {cat})",
        )  # (n_layers, d)

        # Token IDs to evaluate: held-out in-category and randomly sampled out-of-category
        in_cat_eval_ids = [
            model_helpers.get_token_id(m, tokenizer) for m in holdout_members
        ]
        out_of_cat_pool = [
            m
            for other_cat, mlist in cat_members.items()
            if other_cat != cat
            for m in mlist
        ]

        # Ensure that the out of cat pool doesn't contain in-cat tokens
        out_of_cat_pool = [item for item in out_of_cat_pool if item not in members]
        out_cat_eval_ids = [
            model_helpers.get_token_id(m, tokenizer)
            for m in rng.choice(
                out_of_cat_pool,
                size=min(args.n_holdout_out_cat, len(out_of_cat_pool)),
                replace=False,
            )
        ]

        # Distinct ordered support draws for evaluation
        eval_support_draws = draw_distinct_support_sequences(
            train_members, args.support_len, args.n_supports, label=f"{cat} (evaluation)"
        )

        for support_id, support_tokens in enumerate(eval_support_draws):
            input_text = " " + " ".join(support_tokens)
            input_ids = [context_id] + tokenizer.encode(
                input_text, add_special_tokens=False
            )

            # Baseline: no steering
            baseline_lp = model_helpers.get_next_token_log_probs(input_ids, model, model_layers)
            rows.append(
                {
                    "category": cat,
                    "control_category": control_cat,
                    "support_id": support_id,
                    "layer": -1,
                    "condition": "baseline",
                    "alpha": 0.0,
                    "mean_log_prob_in_cat": baseline_lp[in_cat_eval_ids].mean().item(),
                    "mean_log_prob_out_cat": baseline_lp[out_cat_eval_ids]
                    .mean()
                    .item(),
                }
            )

            for layer_idx in layers_to_steer:
                # Indexing note:
                # Hooking model_layers[layer_idx] fires *after* transformer block layer_idx runs,
                # so the injection lands at hidden-state index layer_idx + 1 (per the convention
                # above: hidden_states[k] = output of transformer block k-1).
                # We therefore use the displacement vector measured at that same index so that
                # the steering direction matches the layer being modified.
                layer_cdv = mean_context_displacement[layer_idx + 1]  # (d,) direction at hidden_states[layer_idx+1]
                layer_ctrl = control_displacement[layer_idx + 1]    # (d,) control direction at same index

                for condition, steering_vec in [
                    ("match", layer_cdv),  # steer toward concept direction
                    ("anti", -layer_cdv),  # steer away from concept direction
                    ("control_cat", layer_ctrl),  # steer toward dissimilar concept
                    ("random", random_steering_vecs[cat]),  # random direction control
                ]:
                    for magnitude in args.alpha:
                        lp = model_helpers.get_next_token_log_probs(
                            input_ids, model, model_layers, steering_vec, magnitude, layer_idx
                        )
                        rows.append(
                            {
                                "category": cat,
                                "control_category": control_cat,
                                "support_id": support_id,
                                "layer": layer_idx,
                                "condition": condition,
                                "alpha": magnitude,
                                "mean_log_prob_in_cat": lp[in_cat_eval_ids]
                                .mean()
                                .item(),
                                "mean_log_prob_out_cat": lp[out_cat_eval_ids]
                                .mean()
                                .item(),
                            }
                        )

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dir = os.path.join(args.save_root, model_name_safe, "steering")
    os.makedirs(save_dir, exist_ok=True)

    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "steering.csv"), index=False)

    with open(os.path.join(save_dir, "hyperparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"{len(rows)} rows → steering.csv")
    print(f"Saved to {save_dir}")
