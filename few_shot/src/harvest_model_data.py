import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from helpers import data as data_module
from helpers import model_helpers
from helpers import compute_quantities


def sample_distinct_support_sets(rng, pool, support_len, n_draws, cat, split_name):
    """Draw up to n_draws support sets from pool, each of size support_len, with no two
    draws containing the same set of support tokens (order-independent).

    A pool of size support_len+1 only admits C(support_len+1, support_len) = support_len+1
    distinct sets, so requesting more draws than that would otherwise force duplicate
    (tokens, target) records under different support_ids. When the pool can't support
    n_draws distinct sets, this caps at however many distinct sets exist and warns.
    """
    max_distinct = math.comb(len(pool), support_len)
    n_draws_actual = min(n_draws, max_distinct)
    if n_draws_actual < n_draws:
        print(
            f"Warning: category '{cat}' {split_name} pool (size {len(pool)}) only "
            f"supports {max_distinct} distinct support sets of size {support_len}; "
            f"requested {n_draws}, using {n_draws_actual}."
        )

    seen = set()
    draws = []
    while len(draws) < n_draws_actual:
        tokens = rng.choice(pool, size=support_len, replace=False).tolist()
        key = frozenset(tokens)
        if key in seen:
            continue
        seen.add(key)
        draws.append(tokens)
    return draws


if __name__ == "__main__":
    """Collects the data (activations and log-probs) needed by latent_regressions.py and by behavioral_modeling.py.

    Output files are stored under
    {save_root}/{model_name_safe}/concept_list_data/:
    records.csv                  per-row category/support_id/split/tokens + log-probs
    prototypes.npy               (n_records, d) mean of context tokens' isolated layer-0 states
    displacements.npy            (n_records, n_layers, d) residualized context displacement vectors
    shuffled_displacements.npy   (n_heldout, n_layers, d), rows in the same order as the
                                  split=="heldout" rows of records.csv, permuted context tokens' displacement vectors
    target_embeddings.npy        (n_records, d) isolated layer-0 state of the target token
    all_base_log_probs.pt        (vocab_size,) log p_0(y), BOS-only
    global_mean_displacement.npy (n_layers, d) global mean of the displacement vectors for subtracting out position information
    hyperparams.json             (includes embedding_scale)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="HF model id")
    parser.add_argument("--n_categories", type=int, default=113)
    parser.add_argument("--n_train", type=int, default=8,
                        help="Training support sets per concept")
    parser.add_argument("--n_heldout", type=int, default=5,
                        help="Held-out support sets per concept (tokens disjoint from train).")
    parser.add_argument("--n_supports", type=int, default=4,
                        help="Number of context tokens in each support set (displacement is "
                             "read off the last one, same role as support_len in "
                             "displacement_regression.py)")
    parser.add_argument("--save_root", type=str, default="../results")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--data_path", type=str,
        default=os.path.join(os.path.dirname(__file__), "../data/categories_data.csv"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    aux_rng = np.random.default_rng(args.seed + 1)

    # Model loading
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

    if tokenizer.bos_token_id is None:
        raise ValueError(
            "Tokenizer has no bos_token_id; hidden states and log-probs collected "
            "here must share the same context convention as eval_model.py's "
            "compute_base_probability."
        )
    context_id = tokenizer.bos_token_id
 
    # Dataset creation
    print("Loading and filtering data...")
    # keep_default_na=False: a category member that happens to spell a pandas NA sentinel
    # ("nan", "null", "NA", "None", ...) must survive as literal text, not silently become
    # a float NaN that then blows up downstream tokenization.
    raw_data = pd.read_csv(args.data_path, keep_default_na=False)
    # A category needs at least n_supports+1 members per split (context tokens + the
    # disjoint target), so 2*(n_supports+1) members overall -- filter accordingly.
    min_tokens = 2 * (args.n_supports + 1)
    filtered_data = data_module.filter_multitoken_data(raw_data, tokenizer, min_tokens=min_tokens)
    print(
        f"Filtered to {len(filtered_data)} single-token members across "
        f"{filtered_data['category'].nunique()} categories"
    )

    all_categories = filtered_data["category"].unique()
    selected_categories = rng.choice(
        all_categories, size=min(args.n_categories, len(all_categories)), replace=False
    )

    # Each record describes one support set:
    #   cat, support_id, split ("train"/"heldout"), tokens
    # tokens[-1] is where the displacement vector is computed from. 
    # 'pool' is the token pool this record was drawn from,
    # kept around only to draw its disjoint target token.
    records = []

    for cat in selected_categories:
        members = filtered_data[filtered_data["category"] == cat]["category.member"].tolist()
        assert len(members) >= min_tokens, f"{cat} selected but does not have enough members"

        shuffled     = rng.permutation(members).tolist()
        mid          = len(shuffled) // 2
        train_pool   = shuffled[:mid]
        heldout_pool = shuffled[mid:]

        train_draws = sample_distinct_support_sets(
            rng, train_pool, args.n_supports, args.n_train, cat, "train"
        )
        for support_id, tokens in enumerate(train_draws):
            records.append({
                "cat": cat, "support_id": support_id, "split": "train",
                "tokens": tokens, "pool": train_pool,
            })

        heldout_draws = sample_distinct_support_sets(
            rng, heldout_pool, args.n_supports, args.n_heldout, cat, "heldout"
        )
        for support_id, tokens in enumerate(heldout_draws):
            records.append({
                "cat": cat, "support_id": support_id, "split": "heldout",
                "tokens": tokens, "pool": heldout_pool,
            })

    categories = sorted({r["cat"] for r in records})
    n_train   = sum(1 for r in records if r["split"] == "train")
    n_heldout = sum(1 for r in records if r["split"] == "heldout")
    print(f"Categories: {len(categories)} | Total train sets: {n_train} | Total held-out sets: {n_heldout}")

    # Select disjoint target tokens for each record
    for rec in records:
        remaining = [m for m in rec["pool"] if m not in rec["tokens"]]
        rec["target"] = str(aux_rng.choice(remaining))

    ### Collect hidden states + log-probs for each record
    # isolated_states[token] = (n_layers, d) hidden states at position 1 (after BOS)
    isolated_states = {}

    # full_list_states[record_idx] = (n_layers, d) hidden states at the last position of
    # the context (i.e. of context_tokens[-1], contextualized)
    full_list_states = {}
    log_p_target_given_context = {}   # record_idx -> float

    # For held-out records: also collect hidden states + log-probs for a context whose
    # token order is permuted (the target token is untouched). This gives a noise
    # ceiling for both the displacement vector and the conditional log-prob.
    shuffled_list_states = {}                 # idx -> (n_layers, d), only for held-out
    log_p_target_given_shuffled_context = {}  # idx -> float, only for held-out

    for idx, rec in enumerate(tqdm(records, desc="Collecting hidden states")):
        context_str = " " + " ".join(rec["tokens"])
        hidden_states, log_probs = model_helpers.get_hidden_states_and_logprobs_last_token(
            context_str, tokenizer, model, context_id
        )
        full_list_states[idx] = hidden_states
        target_id = model_helpers.get_token_id(rec["target"], tokenizer)
        log_p_target_given_context[idx] = log_probs[target_id].item()

        for token in rec["tokens"] + [rec["target"]]:
            if token not in isolated_states:
                isolated_states[token], _ = model_helpers.get_hidden_states_and_logprobs_last_token(
                    " " + token, tokenizer, model, context_id
                )

        if rec["split"] == "heldout":
            shuffled_tokens = rng.permutation(rec["tokens"]).tolist()
            shuffled_str = " " + " ".join(shuffled_tokens)
            shuffled_hidden, shuffled_log_probs = model_helpers.get_hidden_states_and_logprobs_last_token(
                shuffled_str, tokenizer, model, context_id
            )
            shuffled_list_states[idx] = (shuffled_hidden, shuffled_tokens[-1])
            log_p_target_given_shuffled_context[idx] = shuffled_log_probs[target_id].item()

    n_layers = next(iter(full_list_states.values())).shape[0]

    # Compute support prototypes and context displacements
    support_prototype     = {}   # record_idx -> (d,) mean over context tokens embeddings
    context_displacement  = {}   # record_idx -> (n_layers, d)

    for idx, rec in enumerate(records):
        layer0_embeddings = np.stack([isolated_states[tok][0].numpy() for tok in rec["tokens"]])
        support_prototype[idx] = layer0_embeddings.mean(axis=0)   # (d,)

        last_context_token = rec["tokens"][-1]
        # Displacement: being in context vs. being alone, at every layer, read off the
        # last context token
        context_displacement[idx] = (full_list_states[idx] - isolated_states[last_context_token]).numpy()

    train_record_indices   = [i for i, r in enumerate(records) if r["split"] == "train"]
    heldout_record_indices = [i for i, r in enumerate(records) if r["split"] == "heldout"]

    # Residualise out position information: subtract the global mean displacement vector,
    # (computed only on train sets), in order to remove position information from the displacement vectors.
    global_mean_displacement = np.stack(
        [context_displacement[i] for i in train_record_indices]
    ).mean(axis=0)  # (n_layers, d)

    for idx in range(len(records)):
        context_displacement[idx] = context_displacement[idx] - global_mean_displacement

    shuffled_displacement = {}   # idx -> (n_layers, d), held-out only
    for idx in heldout_record_indices:
        hs, shuffled_last_token = shuffled_list_states[idx]
        shuffled_displacement[idx] = (
            hs - isolated_states[shuffled_last_token]
        ).numpy() - global_mean_displacement

    # Compute base (BOS-only) log-probs for the whole vocab
    print("Computing base (BOS-only) vocab log-probs...")
    all_base_log_probs = compute_quantities.compute_base_probability(model, tokenizer)  # (vocab_size,)

    # Important Note: Embedding-scale correction!
    # prototype / target-token embeddings come from hidden_states[0] (layer-0, post-embedding),
    # which some model families scale (e.g. Gemma multiplies by sqrt(hidden_size)).
    # behavioral_modeling.py needs the full vocab table in the same space, so record the
    # scalar correction here rather than saving a redundant multi-GB scaled copy of it.
    probe_token = records[0]["tokens"][0]
    probe_id = model_helpers.get_token_id(probe_token, tokenizer)
    probe_isolated_norm = isolated_states[probe_token][0].norm().item()
    probe_raw_norm = (
        model.get_input_embeddings().weight[probe_id].detach().float().cpu().norm().item()
    )
    embedding_scale = probe_isolated_norm / probe_raw_norm

    # Construct output dataframes
    d = support_prototype[0].shape[0]
    prototypes         = np.zeros((len(records), d), dtype=np.float32)
    displacements       = np.zeros((len(records), n_layers, d), dtype=np.float32)
    target_embeddings   = np.zeros((len(records), d), dtype=np.float32)
    row_records = []

    for idx, rec in enumerate(records):
        prototypes[idx]        = support_prototype[idx]
        displacements[idx]     = context_displacement[idx]
        target_embeddings[idx] = isolated_states[rec["target"]][0].numpy()

        target_id = model_helpers.get_token_id(rec["target"], tokenizer)
        row_records.append({
            "category":                           rec["cat"],
            "support_id":                         rec["support_id"],
            "split":                               rec["split"],
            "context_tokens":                     " ".join(rec["tokens"]),
            "target":                             rec["target"],
            "log_p_target_given_context":         log_p_target_given_context[idx],
            "log_p_target_bos":                   all_base_log_probs[target_id].item(),
            "log_p_target_given_shuffled_context": log_p_target_given_shuffled_context.get(idx, float("nan")),
        })

    shuffled_displacements = np.stack([shuffled_displacement[i] for i in heldout_record_indices])

    # Save output dataframes
    model_name_safe = args.model_name.replace("/", "_")
    save_dir = os.path.join(args.save_root, model_name_safe, "concept_list_data")
    os.makedirs(save_dir, exist_ok=True)

    pd.DataFrame(row_records).to_csv(os.path.join(save_dir, "records.csv"), index=False)
    np.save(os.path.join(save_dir, "prototypes.npy"), prototypes)
    np.save(os.path.join(save_dir, "displacements.npy"), displacements)
    np.save(os.path.join(save_dir, "shuffled_displacements.npy"), shuffled_displacements)
    np.save(os.path.join(save_dir, "target_embeddings.npy"), target_embeddings)
    np.save(os.path.join(save_dir, "global_mean_displacement.npy"), global_mean_displacement)
    torch.save(all_base_log_probs, os.path.join(save_dir, "all_base_log_probs.pt"))

    hyperparams = vars(args)
    hyperparams["embedding_scale"] = embedding_scale
    with open(os.path.join(save_dir, "hyperparams.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    print(f"Saved {len(records)} rows to {save_dir}")
