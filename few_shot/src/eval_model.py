import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from helpers import data
from helpers import compute_quantities

if __name__ == "__main__":
    """This script sanity checks whether language models perform basic concept induction.

    Given a model and set of data hyperparameters, this script generates a
    dataset of concept list tasks, then collects target log-probabilities.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2-2b",
        help="HF model id",
    )
    parser.add_argument("--n_categories", type=int, default=113)
    parser.add_argument("--n_supports", type=int, default=10)
    parser.add_argument("--n_targets", type=int, default=5)
    parser.add_argument("--support_len", type=int, default=4)
    parser.add_argument("--save_root", type=str, default="../results_test")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "../data/categories_data.csv"),
        help="Path to categories_data.csv",
    )

    args = parser.parse_args()

    ### Model Loading ###
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", trust_remote_code=True)
    model.eval()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available -- this script requires a GPU.")
    device = torch.device("cuda")
    model = model.to(device)
    print(f"Model loaded on {device}")

    ### Data Loading ###
    print("Loading and filtering data...")
    # keep_default_na=False: a category member that happens to spell a pandas NA sentinel
    # ("nan", "null", "NA", "None", ...) must survive as literal text, not silently become
    # a float NaN that then blows up downstream tokenization.
    raw_data = pd.read_csv(args.data_path, keep_default_na=False)
    filtered_data = data.filter_multitoken_data(raw_data, tokenizer)
    print(
        f"Filtered to {len(filtered_data)} single-token members across "
        f"{filtered_data['category'].nunique()} categories"
    )

    dataset = data.create_dataset(
        filtered_data,
        n_categories=args.n_categories,
        n_supports=args.n_supports,
        n_targets=args.n_targets,
        support_len=args.support_len,
        include_out_of_category=True,
        random_seed=args.seed,
    )
    print(f"Dataset created: {len(dataset)} concept lists")


    rows = []
    for _, row in tqdm(dataset.iterrows(), desc="Computing target log probabilities", total=len(dataset)):
        concept_list_str = row["concept_list"]
        category = row["category"]

        # The concept list is " s1 s2 ... sN target"; split on whitespace
        tokens = concept_list_str.strip().split()
        target_token = tokens[args.support_len]

        # Compute conditional target probability
        # concept_list_str already has a leading space (see data.create_dataset)
        log_p_target_given_S = compute_quantities.compute_conditional_probability(
            model, tokenizer, concept_list_str
        )

        base_log_prob = compute_quantities.compute_base_probability(
            model, tokenizer, target_token
        )

        rows.append(
            {
                **row.to_dict(),
                "log_p_target_given_S": log_p_target_given_S,
                "base_log_prob": base_log_prob,
            }
        )

    result_df = pd.DataFrame(rows)
    context_boost = result_df["log_p_target_given_S"] - result_df["base_log_prob"]
    result_df["context_boost"] = context_boost

    ### Save Data ###
    model_name_safe = args.model_name.replace("/", "_")
    save_dir = os.path.join(args.save_root, model_name_safe, "concept_list", "default")
    os.makedirs(save_dir, exist_ok=True)
    result_df.to_csv(os.path.join(save_dir, "concept_list_behavior.csv"), index=False)