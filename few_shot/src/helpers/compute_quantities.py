import torch

from helpers.data_mappings import NEGATIVE_EXCLUSIONS


def _get_token_embedding(model, tokenizer, word):
    """Return the static input embedding for a single-token word.

    A leading space is prepended so tokenization is consistent with how
    words appear mid-sequence (e.g. GPT-2 byte-pair encoding).
    """
    token_ids = tokenizer.encode(" " + word, add_special_tokens=False)
    id_tensor = torch.tensor([token_ids[0]], device=next(model.parameters()).device)
    with torch.no_grad():
        embedding = model.get_input_embeddings()(id_tensor)
    return embedding.squeeze(0)


def _get_negative_mean(model, tokenizer, data, category):
    """Compute the mean embedding of all tokens NOT in the target category
    and NOT in any category listed in NEGATIVE_EXCLUSIONS[category].
    Also drops any word that appears as a member of the target category,
    even when that word is listed under a different category.
    """
    excluded_categories = set(NEGATIVE_EXCLUSIONS[category] + [category])
    pos_members = set(data[data["category"] == category]["category.member"].tolist())
    neg_data = data[
        ~data["category"].isin(excluded_categories)
        & ~data["category.member"].isin(pos_members)
    ]
    neg_members = neg_data["category.member"].unique().tolist()
    neg_embeds = torch.stack(
        [_get_token_embedding(model, tokenizer, m) for m in neg_members]
    )
    return neg_embeds.mean(dim=0)


def compute_hypothesis_vector(model, tokenizer, data, category):
    """This function computes the hypothesis vector for a category
    given a model using the mean difference technique over the embedding space.

    First, retrieve the embedding for each positive example of a category, and
    compute the mean of those vectors. Next, retrieve the embedding for every
    other token in the dataset, excluding the categories present in NEGATIVE_EXCLUSIONS
    for the target cateogry. Also explicitly drop tokens that are in the category.
    Compute the mean of these negative examples.

    Returns the difference between these vectors

    Args:
        model (hf.model): model
        tokenizer (hf.tokenizer): tokenizer
        data (pd.DataFrame): dataframe with columns category and category.member
        category (str): string denoting the category
    """
    pos_members = data[data["category"] == category]["category.member"].tolist()
    pos_embeds = torch.stack(
        [_get_token_embedding(model, tokenizer, m) for m in pos_members]
    )
    pos_mean = pos_embeds.mean(dim=0)

    neg_mean = _get_negative_mean(model, tokenizer, data, category)

    diff = pos_mean - neg_mean
    return diff


def compute_base_probability(model, tokenizer, target=None):
    """Compute the probability of the target word without conditioning
    on any prior words

    Args:
        model (hf.model): model
        tokenizer (hf.tokenizer): tokenizer
        target (string): target string
    """
    # Build a minimal context: BOS, to match compute_conditional_probability's
    # context (which is also anchored on BOS).
    if tokenizer.bos_token_id is None:
        raise ValueError(
            "Tokenizer has no bos_token_id; compute_base_probability and "
            "compute_conditional_probability must share the same context "
            "convention to make context_boost comparisons meaningful."
        )
    context_ids = [tokenizer.bos_token_id]

    device = next(model.parameters()).device
    context_tensor = torch.tensor([context_ids], device=device)
    with torch.no_grad():
        outputs = model(context_tensor)

    # Logits at the last context position predict the next token
    logits = outputs.logits[0, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)

    if target:
        if not target.startswith(" "):
            target = " " + target
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        target_id = target_ids[0]  # single token after filtering
        return log_probs[target_id].item()
    return log_probs.detach().float().cpu()


def compute_conditional_probability(model, tokenizer, concept_list_str):
    """Compute the probability of the target word (last word of the concept list)
    in the context of the rest of the concept list

    Args:
        model (hf.model): model
        tokenizer (hf.tokenizer): tokenizer
        concept_list_str (string): string with the entire concept list + target word
    """
    # A leading space keeps the first support word's tokenization consistent with
    # how it would be tokenized mid-sequence; make sure we have one
    if not concept_list_str.startswith(" "):
        concept_list_str = " " + concept_list_str

    # All members are single-token after filtering, so each word maps to exactly
    # one token and the last token in the sequence is the target.
    input_ids = tokenizer.encode(concept_list_str, add_special_tokens=False)

    # Prepend BOS for models that expect it (LLaMA, Gemma, etc.) so this is
    # consistent with compute_base_probability and with how these models were trained.
    if tokenizer.bos_token_id is not None:
        input_ids = [tokenizer.bos_token_id] + input_ids

    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        outputs = model(input_tensor)

    logits = outputs.logits[0]  # (seq_len, vocab_size)
    log_probs = torch.log_softmax(logits, dim=-1)

    # Position -2 predicts position -1 (the target token)
    target_id = input_ids[-1]
    return log_probs[-2, target_id].item()
