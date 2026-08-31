import numpy as np
import pandas as pd

###
#   This file implements data preprocesing and dataset creation
#   for several of the analyses.
###

def filter_multitoken_data(data, tokenizer, min_tokens=8):
    """Filter out category members comprised of multiple tokens.
    Also drops categories with fewer than min_tokens single-token members remaining.

    Args:
        data (pd.DataFrame): dataframe with columns 'category' and 'category.member'
        tokenizer: HuggingFace tokenizer
        min_tokens (int): minimum single-token members required per category
    """
    def is_single_token(word):
        # Leading space so tokenization matches how the word appears mid-sequence
        return len(tokenizer.encode(" " + word, add_special_tokens=False)) == 1

    mask = data["category.member"].apply(is_single_token)
    filtered = data[mask].copy()

    category_counts = filtered.groupby("category")["category.member"].count()
    valid_categories = category_counts[category_counts >= min_tokens].index
    filtered = filtered[filtered["category"].isin(valid_categories)]

    return filtered.reset_index(drop=True)


def create_dataset(data, n_categories, n_supports, n_targets, support_len=4,
                   include_out_of_category=True, random_seed=19):
    """Create concept list dataset. Concept list format: " tok1 tok2 ... tokN target"

    For each of n_categories randomly selected categories, samples n_targets in-category
    targets. For each target, samples n_supports independent support sets from the
    remaining members. Optionally also samples out-of-category targets.

    Returns a DataFrame with columns:
        'concept_list', 'category', 'target', 'support_id', 'target_id', 'target_in_category'

    Args:
        data (pd.DataFrame): dataframe with columns 'category' and 'category.member'
        n_categories (int): number of categories to sample
        n_supports (int): number of support sets per (category, target) pair
        n_targets (int): number of targets per category
        support_len (int): number of items in each support set
        include_out_of_category (bool): whether to include out-of-category targets
        random_seed (int): random seed
    """
    rng = np.random.default_rng(random_seed)

    all_categories = data["category"].unique()
    selected_categories = rng.choice(
        all_categories, size=min(n_categories, len(all_categories)), replace=False
    )

    records = []
    for cat in selected_categories:
        members = data[data["category"] == cat]["category.member"].tolist()
        members_set = set(members)

        actual_n_targets = min(n_targets, max(0, len(members) - support_len))
        if actual_n_targets == 0:
            continue
        in_cat_targets = rng.choice(members, size=actual_n_targets, replace=False).tolist()

        # Reserve out of category items just in case we're including them as targets
        out_of_cat_pool = data[~data["category.member"].isin(members_set)]["category.member"].unique().tolist()
        n_out_of_cat_targets = min(actual_n_targets, len(out_of_cat_pool))
        out_cat_targets = rng.choice(out_of_cat_pool, size=n_out_of_cat_targets, replace=False).tolist()

        for target_id, target in enumerate(in_cat_targets):
            support_pool = [m for m in members if m != target]
            for support_id in range(n_supports):
                support = rng.choice(support_pool, size=support_len, replace=False).tolist()

                records.append({
                    "concept_list": " " + " ".join(support) + " " + target,
                    "category": cat,
                    "target": target,
                    "support_id": support_id,
                    "target_id": target_id,
                    "target_in_category": True,
                })

        if not include_out_of_category:
            continue

        # Remove targets from out of cat
        for target_id, target in enumerate(out_cat_targets):
            for support_id in range(n_supports):
                support = rng.choice(members, size=support_len, replace=False).tolist()
                records.append({
                    "concept_list": " " + " ".join(support) + " " + target,
                    "category": cat,
                    "target": target,
                    "support_id": support_id,
                    "target_id": target_id,
                    "target_in_category": False,
                })

    return pd.DataFrame(records)


def create_cdv_dataset(data, n_categories, n_supports, support_len=4, random_seed=19):
    """Create dataset for context displacement vector analysis.

    For each category, samples n_supports sets of 2*support_len tokens and splits them
    into two disjoint support lists, enabling clean consistency analysis. The analysis
    target is always the last item in each support list.

    Returns a DataFrame with columns:
        'category', 'support_id', 'analysis_target_0', 'analysis_target_1',
        'concept_list_0', 'concept_list_1'

    Args:
        data (pd.DataFrame): dataframe with columns 'category' and 'category.member'
        n_categories (int): number of categories to sample
        n_supports (int): number of support sets per category
        support_len (int): number of tokens in each support list
        random_seed (int): random seed
    """
    rng = np.random.default_rng(random_seed)

    all_categories = data["category"].unique()
    selected_categories = rng.choice(
        all_categories, size=min(n_categories, len(all_categories)), replace=False
    )

    records = []
    for cat in selected_categories:
        members = data[data["category"] == cat]["category.member"].tolist()
        if len(members) < (support_len * 2):
            continue

        for support_id in range(n_supports):
            support = rng.choice(members, size=2 * support_len, replace=False).tolist()
            support_list_0 = support[:support_len]
            support_list_1 = support[support_len:]

            records.append({
                "category": cat,
                "support_id": support_id,
                "analysis_target_0": support_list_0[-1],
                "analysis_target_1": support_list_1[-1],
                "concept_list_0": " " + " ".join(support_list_0),
                "concept_list_1": " " + " ".join(support_list_1),
            })

    return pd.DataFrame(records)
