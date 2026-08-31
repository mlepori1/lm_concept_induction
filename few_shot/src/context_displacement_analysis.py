import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from helpers import compute_quantities
from helpers import data as data_module
from helpers import model_helpers


if __name__ == "__main__":
    """Concept Displacement Vector Analysis.

    This script analyzes the context displacement vectors of a model.

    We ask whether the context displacement vector is consistent across 
    different support sets from the same concept.

    We next ask whether the average context displacement vector per concept
    captures similar information as embedding-space hypothesis vectors contrastively generated 
    from all positive examples in the concept's vocabulary and all negative examples in the dataset. 
    To do this, we create an RSM of context displacement vectors (averaged over concept) and
    and RSM of embedding-space hypothesis vectors, and compare them using spearman r. If this is high, then the
    context-displacement vectors are similar to embedding-space hypothesis vectors, up to rotation.

    We also track how RSA metrics change as you average more context displacement vectors.

    Also performs RSA permutation test to assess the significance of the RSM similarities


    OUTPUTS
    -------
    Saved under SAVE_ROOT/MODEL_NAME/context_displacement_analysis/:
        within_consistency.csv      — per (category, layer): within_cat_cosine_sim
        between_consistency.csv     — per (category, layer): between category cosine sim of supports
        rsa_by_n_supports.csv       — per (n_support_sets, layer): spearman_r, pearson_r, perm_p
        mean_cd.npy                — (n_categories, n_layers, d)
        mean_cd_single.npy         — (n_categories, n_layers, d): Sample (rather than avg) a support set context displacement vector
        embed_hyp.npy               — (n_categories, d)
        mean_disp.npy               - (n_layers, d): average of displacement vectors over categories, for removing positional info
        categories.json             — ordered list of category names aligned with arrays
        hyperparams.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="google/gemma-2-2b", help="HF model id"
    )
    parser.add_argument("--n_categories", type=int, default=113)
    parser.add_argument("--n_supports", type=int, default=10)
    parser.add_argument("--support_len", type=int, default=4)
    parser.add_argument("--save_root", type=str, default="../results")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../data/categories_data.csv"),
        help="Path to categories_data.csv",
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=500,
        help="Number of permutations for RSA null distribution",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

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
        raise ValueError("Tokenizer has no bos_token_id; this script requires one.")
    context_id = tokenizer.bos_token_id

    ### Data ###
    print("Loading and filtering data...")
    raw_data = pd.read_csv(args.data_path, keep_default_na=False) # Make sure that "nan" is not coerced into a float 
    filtered_data = data_module.filter_multitoken_data(raw_data, tokenizer)
    print(
        f"Filtered to {len(filtered_data)} single-token members across "
        f"{filtered_data['category'].nunique()} categories"
    )

    dataset = data_module.create_cdv_dataset(
        filtered_data,
        n_categories=args.n_categories,
        n_supports=args.n_supports,
        support_len=args.support_len,
        random_seed=args.seed,
    )
    categories = sorted(dataset["category"].unique().tolist())
    print(f"Dataset: {len(dataset)} rows | {len(categories)} categories")

    ### Embedding hypothesis vectors ###
    print("Computing embedding-space hypothesis vectors...")
    embed_hyps = {}  # cat -> (d,) unit-normed tensor
    for cat in tqdm(categories, desc="Embedding hypotheses"):
        raw_hyp = compute_quantities.compute_hypothesis_vector(
            model, tokenizer, filtered_data, cat
        ).cpu()
        # Unit norm the embedding hypothesis vectors
        embed_hyps[cat] = F.normalize(raw_hyp, dim=-1)

    ### Collect hidden states for each concept list for context displacement vector analysis ###
    # This collects hidden states from disjoint supports to enable clean consistency analysis later on

    hs_store_0_first = {}
    hs_store_1_first = {}
    hs_store_0_last = {}
    hs_store_1_last = {}

    for _, row in tqdm(
        dataset.iterrows(), desc="Collecting hidden states", total=len(dataset)
    ):
        cat = row["category"]
        support_id = int(row["support_id"])

        cl_0 = row["concept_list_0"]
        at_0 = row["analysis_target_0"]

        cl_1 = row["concept_list_1"]
        at_1 = row["analysis_target_1"]

        assert cl_0.strip().split()[-1] == at_0.strip(), (cl_0, at_0)
        assert cl_1.strip().split()[-1] == at_1.strip(), (cl_1, at_1)

        hs_store_0_last[(cat, support_id)], _ = model_helpers.get_hidden_states_and_logprobs_last_token(
            cl_0, tokenizer, model, context_id
        )
        hs_store_1_last[(cat, support_id)], _ = model_helpers.get_hidden_states_and_logprobs_last_token(
            cl_1, tokenizer, model, context_id
        )
        hs_store_0_first[(cat, support_id)], _ = (
            model_helpers.get_hidden_states_and_logprobs_last_token(
                " " + at_0.strip(), tokenizer, model, context_id
            )
        )
        hs_store_1_first[(cat, support_id)], _ = (
            model_helpers.get_hidden_states_and_logprobs_last_token(
                " " + at_1.strip(), tokenizer, model, context_id
            )
        )

    n_layers = next(iter(hs_store_0_last.values())).shape[0]

    ### Compute context displacement vectors ###
    context_displacement_0 = {
        (cat, support_id): hs_store_0_last[(cat, support_id)]
        - hs_store_0_first[(cat, support_id)]
        for (cat, support_id) in hs_store_0_last
    }

    context_displacement_1 = {
        (cat, support_id): hs_store_1_last[(cat, support_id)]
        - hs_store_1_first[(cat, support_id)]
        for (cat, support_id) in hs_store_1_last
    }

    ### Compute position-based difference (to residualize) ###
    # Average displacement vector over ALL (cat, support_id) across both sets.
    # Note: index 0 is embedding-layer, index 1 is after the first transformer block, etc.

    all_disps = list(context_displacement_0.values()) + list(
        context_displacement_1.values()
    )
    mean_disp = torch.stack(all_disps).mean(dim=0)  # (n_transformer_layers + 1, d)

    ### Residualize: remove positional baseline ###
    context_displacement_0 = {
        (cat, support_id): cd - mean_disp
        for (cat, support_id), cd in context_displacement_0.items()
    }
    context_displacement_1 = {
        (cat, support_id): cd - mean_disp
        for (cat, support_id), cd in context_displacement_1.items()
    }

    ### Compute within-category consistency over disjoint supports ###
    within_rows = []
    for cat in categories:
        support_ids = [s for (c, s) in context_displacement_0 if c == cat]
        if not support_ids:
            continue
        cds_0 = F.normalize(
            torch.stack([context_displacement_0[(cat, s)] for s in support_ids]), dim=-1
        )  # (n_s, n_layers, d)
        cds_1 = F.normalize(
            torch.stack([context_displacement_1[(cat, s)] for s in support_ids]), dim=-1
        )  # (n_s, n_layers, d)
        for layer_idx in range(n_layers):
            within_cos = (
                (cds_0[:, layer_idx, :] * cds_1[:, layer_idx, :])
                .sum(dim=-1)
                .mean()
                .item()
            )
            within_rows.append(
                {
                    "category": cat,
                    "layer": layer_idx,
                    "within_cat_cosine_sim": within_cos,
                }
            )

    within_df = pd.DataFrame(within_rows)

    ### Compute between-category consistency ###
    # For each (cat, layer): mean cosine sim between that cat's cds_0 and all other cats' cds_1
    between_rows = []
    for cat in categories:
        support_ids = [s for (c, s) in context_displacement_0 if c == cat]
        if not support_ids:
            continue
        cds_0_cat = F.normalize(
            torch.stack([context_displacement_0[(cat, s)] for s in support_ids]), dim=-1
        )  # (n_support, n_layers, d)
        other_cds_1 = [
            context_displacement_1[(c, s)]
            for (c, s) in context_displacement_1
            if c != cat
        ]  # List of context_displacement vectors from every support set in every other category
        cds_1_other = F.normalize(
            torch.stack(other_cds_1), dim=-1
        )  # (n_other_cat_support, n_layers, d)
        for layer_idx in range(n_layers):
            between_cos = (
                (cds_0_cat[:, layer_idx, :] @ cds_1_other[:, layer_idx, :].T)
                .mean()
                .item()
            )
            between_rows.append(
                {
                    "category": cat,
                    "layer": layer_idx,
                    "between_cat_cosine_sim": between_cos,
                }
            )

    between_df = pd.DataFrame(between_rows)

    ### Compute mean category context displacement vectors ###
    # mean_cds[cat] = normalize(
    #               mean over context displacement vectors from one list, so as not to introduce
    #               weird statistical biases from only including pairs of disjoint lists
    #               )
    mean_cds = {}
    for cat in categories:
        cat_cds = [cd for (c, s), cd in context_displacement_0.items() if c == cat]
        if cat_cds:
            mean_cd = torch.stack(cat_cds).mean(dim=0)  # (n_layers, d)
            mean_cds[cat] = F.normalize(mean_cd, dim=-1)

    ### RSA analysis between embedding and context displacement vectors ###
    embed_vecs = torch.stack([embed_hyps[cat] for cat in categories])  # (n_cat, d)
    embed_rsm = (embed_vecs @ embed_vecs.T).float().cpu().numpy()  # (n_cat, n_cat)
    n_cat = len(categories)
    triu = np.triu_indices(n_cat, k=1)
    embed_rsm_flat = embed_rsm[triu]

    ### RSA as a function of number of support sets ###
    all_support_ids = sorted({s for (c, s) in context_displacement_0})
    rsa_by_n_supports = []

    for k in tqdm(range(1, len(all_support_ids) + 1), desc="RSA by n_support_sets"):
        k_support_ids = set(all_support_ids[:k])

        mean_cds_k = {}
        for cat in categories:
            cat_cds = [
                cd
                for (c, s), cd in context_displacement_0.items()
                if c == cat and s in k_support_ids
            ]
            if cat_cds:
                mean_cd = torch.stack(cat_cds).mean(dim=0)  # (n_layers, d)
                mean_cds_k[cat] = F.normalize(mean_cd, dim=-1)

        for layer_idx in range(n_layers):
            cds_k = torch.stack(
                [mean_cds_k[cat][layer_idx] for cat in categories]
            )  # (n_cat, d)
            cd_rsm_k = (cds_k @ cds_k.T).float().cpu().numpy()  # (n_cat, n_cat)
            cd_rsm_k_flat = cd_rsm_k[triu]
            spearman_r, _ = spearmanr(embed_rsm_flat, cd_rsm_k_flat)
            pearson_r, _ = pearsonr(embed_rsm_flat, cd_rsm_k_flat)

            perm_spearman = np.empty(args.n_permutations)
            perm_pearson = np.empty(args.n_permutations)
            for i in range(args.n_permutations):
                perm_idx = rng.permutation(n_cat)
                cd_rsm_perm = cd_rsm_k[np.ix_(perm_idx, perm_idx)]
                flat_perm = cd_rsm_perm[triu]
                perm_spearman[i], _ = spearmanr(embed_rsm_flat, flat_perm)
                perm_pearson[i], _ = pearsonr(embed_rsm_flat, flat_perm)
            perm_p_spearman = float(np.mean(perm_spearman >= spearman_r))
            perm_p_pearson = float(np.mean(perm_pearson >= pearson_r))

            rsa_by_n_supports.append(
                {
                    "n_support_sets": k,
                    "layer": layer_idx,
                    "spearman_r": float(spearman_r),
                    "perm_p_spearman": perm_p_spearman,
                    "pearson_r": float(pearson_r),
                    "perm_p_pearson": perm_p_pearson,
                }
            )

    rsa_by_n_supports_df = pd.DataFrame(rsa_by_n_supports)

    ### Save ###
    model_name_safe = args.model_name.replace("/", "_")
    save_dir = os.path.join(args.save_root, model_name_safe, "context_displacement_analysis")
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "mean_disp.npy"), mean_disp.float().cpu().numpy())

    within_df.to_csv(os.path.join(save_dir, "within_consistency.csv"), index=False)
    between_df.to_csv(os.path.join(save_dir, "between_consistency.csv"), index=False)
    rsa_by_n_supports_df.to_csv(
        os.path.join(save_dir, "rsa_by_n_supports.csv"), index=False
    )

    # mean_cd: (n_cat, n_layers, d) — mean over support sets, unit-normed
    cd_arr = np.stack([mean_cds[cat].float().cpu().numpy() for cat in categories])
    np.save(os.path.join(save_dir, "mean_cd.npy"), cd_arr)

    # mean_cd_single: (n_cat, n_layers, d) — single support set (support_id=0), unit-normed
    # for RSM comparison
    single_support_id = sorted({s for (c, s) in context_displacement_0})[0]
    cd_single_arr = np.stack(
        [
            F.normalize(
                context_displacement_0[(cat, single_support_id)], dim=-1
            ).numpy()
            for cat in categories
        ]
    )
    np.save(os.path.join(save_dir, "mean_cd_single.npy"), cd_single_arr)

    # embed_hyp: (n_cat, d)
    embed_arr = np.stack([embed_hyps[cat].float().cpu().numpy() for cat in categories])
    np.save(os.path.join(save_dir, "embed_hyp.npy"), embed_arr)

    with open(os.path.join(save_dir, "categories.json"), "w") as f:
        json.dump(categories, f, indent=2)

    with open(os.path.join(save_dir, "hyperparams.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved results to {save_dir}")
