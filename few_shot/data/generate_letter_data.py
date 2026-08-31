import argparse
import os
import re

import nltk
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

WORD_RE = re.compile(r"^[a-z]+$")

if __name__ == "__main__":
    """Builds letter_data.csv: A simple alternative dataset for orthographic features.
    This dataset will have the same structure as categoriges_data.csv. 
    Categories correspond to letters, and the entries within those categories
    are just random words from a model's tokenizer that 1) start with that letter and
    2) are in nltk's known English word list.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b", help="HF model id")
    parser.add_argument("--n_per_letter", type=int, default=50)
    parser.add_argument("--min_word_len", type=int, default=3,
                        help="Minimum characters, to avoid degenerate 1-2 letter tokens")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--out_path", type=str,
                        default=os.path.join(os.path.dirname(__file__), "letter_data.csv"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    vocab = tokenizer.get_vocab()

    # Restrict candidates to real English words (nltk's bundled 'words' corpus), so the
    # letters dataset stays somewhat comparable to categories_data.csv's English category members
    try:
        nltk.data.find("corpora/words")
    except LookupError:
        nltk.download("words")
    from nltk.corpus import words as nltk_words
    english_words = {w.lower() for w in nltk_words.words()}

    # SentencePiece marks word-initial tokens with a leading meta symbol (e.g. "▁"); read it
    # off a known example instead of hardcoding, so this works across tokenizer families.
    word_boundary = tokenizer.tokenize(" a")[0][:-1]

    letters = "abcdefghijklmnopqrstuvwxyz"
    candidates = {letter: [] for letter in letters}
    for token, token_id in vocab.items():
        if not token.startswith(word_boundary):
            continue
        word = token[len(word_boundary):]
        if len(word) < args.min_word_len or not WORD_RE.match(word):
            continue
        if word[0] not in candidates:
            continue
        if word not in english_words:
            continue
        # Round-trip check, matching helpers/data.py's filter_multitoken_data: the word must
        # tokenize back to exactly this single token when it appears mid-sequence.
        if tokenizer.encode(" " + word, add_special_tokens=False) != [token_id]:
            continue
        candidates[word[0]].append(word)

    rng = np.random.default_rng(args.seed)
    rows = []
    for letter in letters:
        words = sorted(set(candidates[letter]))
        if len(words) < args.n_per_letter:
            print(f"warning: only {len(words)} single-token candidates for '{letter}', "
                  f"wanted {args.n_per_letter}")
        chosen = rng.choice(words, size=min(args.n_per_letter, len(words)), replace=False)
        for word in chosen:
            rows.append({"category": letter, "category.member": word})

    pd.DataFrame(rows).to_csv(args.out_path, index=False)
    print(f"Saved {len(rows)} rows across {len(letters)} letters to {args.out_path}")
