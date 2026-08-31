"""Inspection of the ridge map's input directions.

The ridge map A (fit in latent_regressions.py) maps a support-set prototype vector
to a predicted context-displacement vector.  Its left singular vectors (A's own input,
prototype-space directions) are the directions in embedding space that A amplifies
most strongly.

What kinds of words point along each of those directions?
We answer by finding the vocabulary tokens whose static (layer-0) embedding vectors
have the highest cosine similarity with each singular vector.
"""

import argparse
import json
import os

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def nearest_and_farthest_tokens(direction, embedding_matrix_normed, tokenizer, n_top_tokens):
    """Cosine similarity between a (d,) direction and every row of embedding_matrix_normed
    (already unit-normalised), returning the n_top_tokens most- and least-similar tokens.
    """
    direction_unit = direction / np.maximum(np.linalg.norm(direction), 1e-12)
    cosine_sims = embedding_matrix_normed @ direction_unit   # (vocab_size,)

    top_ids = np.argsort(cosine_sims)[::-1][:n_top_tokens]
    bot_ids = np.argsort(cosine_sims)[:n_top_tokens]

    return {
        "top_tokens":    tokenizer.convert_ids_to_tokens(top_ids.tolist()),
        "top_sims":      cosine_sims[top_ids].tolist(),
        "bottom_tokens": tokenizer.convert_ids_to_tokens(bot_ids.tolist()),
        "bottom_sims":   cosine_sims[bot_ids].tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b",
                        help="HuggingFace model id")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Directory containing top_input_svecs.npy and singular_values.npy")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of singular vectors to inspect (per layer)")
    parser.add_argument("--n_top_tokens", type=int, default=20,
                        help="Number of most- and least-similar tokens to report per singular vector")
    parser.add_argument("--target_layers", type=int, nargs="+",
                        default=[6, 10, 14, 16, 18, 20, 21, 22, 23, 24],
                        help="Layers to analyze")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the random-direction control")
    args = parser.parse_args()

    # automatically set results_dir based on model_name if not provided
    if args.results_dir is None:
        model_id = args.model_name.replace("/", "_")
        args.results_dir = os.path.join("../results", model_id, "displacement_regression")

    # ── Load model just to get the static token embedding matrix ─────────────
    # We only need embed_tokens.weight (the lookup table that maps token IDs to
    # d-dimensional vectors before any transformer computation).
    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="float32",
        trust_remote_code=True,
    )
    model.eval()

    # Shape: (vocab_size, d_model) — one row per token in the vocabulary
    embedding_matrix = model.get_input_embeddings().weight.data.float().numpy()
    del model   # free GPU/CPU memory; we only needed the embeddings
    print(f"Embedding matrix shape: {embedding_matrix.shape}  "
          f"({embedding_matrix.shape[0]:,} tokens × {embedding_matrix.shape[1]} dims)")

    # Unit-normalise every row so dot products become cosine similarities.
    row_norms = np.linalg.norm(embedding_matrix, axis=-1, keepdims=True)   # (vocab, 1)
    embedding_matrix_normed = embedding_matrix / np.maximum(row_norms, 1e-12)

    # ── Load pre-computed singular vectors of the ridge map A ─────────────────
    # top_input_svecs[layer, i, :] is the i-th left singular vector of A at that layer.
    # A left singular vector points in prototype space; a high absolute cosine similarity between
    # a token's embedding and this vector means that token "activates" this gain direction.
    sv_matrix = np.load(os.path.join(args.results_dir, "top_input_svecs.npy"))  # (n_layers, k_saved, d)
    sigma_matrix = np.load(os.path.join(args.results_dir, "singular_values.npy"))  # (n_layers, r)

    n_layers = sv_matrix.shape[0]
    k_saved  = sv_matrix.shape[1]
    k_use    = min(args.k, k_saved)

    print(f"Saved singular vectors: {sv_matrix.shape}  |  inspecting top k={k_use} per layer")

    # ── Random-direction control ────────────────────────────────────────────────
    # A single random unit direction in the same (static, layer-independent) embedding
    # space, analyzed the same way as the singular vectors -- a null baseline for how
    # tight/interpretable a generic, non-amplified direction's nearest tokens look by
    # chance, to compare against the real singular vectors' neighborhoods.
    d = sv_matrix.shape[-1]
    random_direction = np.random.default_rng(args.seed).standard_normal(d)
    random_control = nearest_and_farthest_tokens(
        random_direction, embedding_matrix_normed, tokenizer, args.n_top_tokens
    )
    print(f"\nRandom-direction control (seed={args.seed})  |  "
          f"{' '.join(random_control['top_tokens'][:10])}")

    # For each layer × singular vector, find nearest vocab tokens
    results = {}

    for layer_idx in args.target_layers:
        if layer_idx >= n_layers:
            print(f"Warning: layer {layer_idx} >= n_layers ({n_layers}), skipping")
            continue

        print(f"\n{'─'*60}\nLayer {layer_idx}\n{'─'*60}")

        layer_results = []

        for sv_i in range(k_use):
            direction = sv_matrix[layer_idx, sv_i]            # (d,) un-normalised direction
            sigma     = float(sigma_matrix[layer_idx, sv_i])  # how much A amplifies this direction

            if float(np.linalg.norm(direction)) < 1e-12:
                layer_results.append(None)
                continue

            nearest = nearest_and_farthest_tokens(
                direction, embedding_matrix_normed, tokenizer, args.n_top_tokens
            )
            layer_results.append({
                "sv_index":       sv_i,
                "singular_value": sigma,
                **nearest,
            })

            print(f"  SV {sv_i:2d}  σ={sigma:7.2f}  |  {' '.join(nearest['top_tokens'][:10])}")

        results[str(layer_idx)] = layer_results

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {"args": vars(args), "results": results, "random_control": random_control}
    out_path = os.path.join(args.results_dir, "concept_stretching.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {out_path}")
