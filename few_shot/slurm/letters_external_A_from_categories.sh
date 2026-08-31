#!/bin/bash
# Re-runs latent_regressions.py on the LETTERS domain for each model, adding
# --external_A_path pointing at the CATEGORIES all_A.npy.
#
# This evaluates how well a ridge map trained on semantic categories generalises
# to predicting first-letter displacements -- the reverse of the existing pipeline
# (which evaluates letters->categories). Adds cos_external_A / cos_external_A_shuffled
# columns to letters/*/displacement_regression/regressions.csv.
#
# Usage:
#   ./letters_external_A_from_categories.sh             # submit all 5 models
#   ./letters_external_A_from_categories.sh google/gemma-2-2b  # one model only
#   ./letters_external_A_from_categories.sh --dry-run   # print commands, no submit

set -euo pipefail
source "$(dirname "$0")/pipeline_common.sh"

DRY_RUN=0
REQUESTED_MODEL=""
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && DRY_RUN=1 || REQUESTED_MODEL="$arg"
done

MODELS=(
    "google/gemma-2-2b"
    "allenai/OLMo-2-0425-1B"
    "allenai/OLMo-2-1124-7B"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
)

for MODEL in "${MODELS[@]}"; do
    [[ -n "$REQUESTED_MODEL" && "$MODEL" != "$REQUESTED_MODEL" ]] && continue

    SAFE=$(echo "$MODEL" | tr '/' '_')
    GRES=$(gpu_gres_for_model "$MODEL")

    CAT_ALL_A="$CAT_ROOT/$SAFE/displacement_regression/all_A.npy"
    LET_REG_DIR="$LET_ROOT/$SAFE/displacement_regression"

    if [[ ! -f "$CAT_ALL_A" ]]; then
        echo "ERROR: $CAT_ALL_A not found — run categories regression with --save_all_A first"
        continue
    fi
    if [[ ! -f "$LET_REG_DIR/regressions.csv" ]]; then
        echo "ERROR: $LET_REG_DIR/regressions.csv not found — run letters regression first"
        continue
    fi

    WRAP="export HF_HOME='$HF_HOME_VAL' HF_HUB_OFFLINE=1 && cd '$SRC_DIR' && \
        uv run --frozen --project '$PROJECT_ROOT' python \
        '$SRC_DIR/latent_regressions.py' \
        --model_name '$MODEL' \
        --save_root '$LET_ROOT' \
        --save_all_A \
        --external_A_path '$CAT_ALL_A'"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "=== $MODEL ==="
        echo "sbatch --job-name=let_ext_cats-$SAFE --partition=$PARTITION --gres=$GRES \\"
        echo "  --cpus-per-task=$CPUS_PER_TASK --mem=${STAGE_MEM[regress]} --time=${STAGE_TIME[regress]} \\"
        echo "  --output=$LOG_DIR/let_ext_cats-$SAFE-%j.out --error=$LOG_DIR/let_ext_cats-$SAFE-%j.err \\"
        echo "  --wrap=\"$WRAP\""
        echo
    else
        JOB_ID=$(sbatch --parsable \
            --job-name="let_ext_cats-$SAFE" \
            --partition="$PARTITION" \
            --gres="$GRES" \
            --cpus-per-task="$CPUS_PER_TASK" \
            --mem="${STAGE_MEM[regress]}" \
            --time="${STAGE_TIME[regress]}" \
            --output="$LOG_DIR/let_ext_cats-$SAFE-%j.out" \
            --error="$LOG_DIR/let_ext_cats-$SAFE-%j.err" \
            --wrap="$WRAP")
        echo "Submitted $MODEL → job $JOB_ID"
    fi
done
