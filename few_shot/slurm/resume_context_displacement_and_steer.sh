#!/bin/bash
# Resubmits just the context_displacement -> steer branch for models where it failed,
# without redoing the already-completed regress/behavioral/stretch stages.
#
# Background: create_cdv_dataset (helpers/data.py) was missing the
# min(n_categories, len(all_categories)) clamp that create_dataset already had, so
# context_displacement_analysis.py crashed with "Cannot take a larger sample than
# population when replace is False" for every model whose filtered category count
# (min_tokens=8) is below the default --n_categories=113. Only google/gemma-2-2b (which
# happens to filter to exactly 113 categories) was spared. Now fixed.
#
# This script does NOT check for cats_regress output before submitting (unlike
# run_analysis_pipeline.sh's harvest check) -- only run it for models where cats_regress
# already completed successfully.
#
# Usage:
#   ./resume_context_displacement_and_steer.sh                          # the 4 models that failed
#   ./resume_context_displacement_and_steer.sh meta-llama/Llama-3.2-1B  # just this one
#   ./resume_context_displacement_and_steer.sh --dry-run                # print the sbatch calls, submit nothing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_common.sh"

DRY_RUN=0
MODELS=()
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN=1
    else
        MODELS+=("$arg")
    fi
done
if [ "${#MODELS[@]}" -eq 0 ]; then
    MODELS=(
        "allenai/OLMo-2-0425-1B"
        "allenai/OLMo-2-1124-7B"
        "meta-llama/Llama-3.2-1B"
        "meta-llama/Llama-3.2-3B"
    )
fi

for MODEL in "${MODELS[@]}"; do
    SAFE=$(echo "$MODEL" | tr '/' '_')
    GRES=$(gpu_gres_for_model "$MODEL")

    ALL_A="$CAT_ROOT/$SAFE/displacement_regression/all_A.npy"
    if [ ! -f "$ALL_A" ]; then
        echo "── Skipping $MODEL: $ALL_A not found (cats_regress hasn't completed) ──" >&2
        continue
    fi

    STEER_LAYERS=$(steer_layers_for_model "$MODEL")
    echo "── Submitting context_displacement + steer for $MODEL (safe name: $SAFE, gres: $GRES) ──"

    CDA=$(submit "context_displacement-$SAFE" context_displacement "$GRES" "" \
        "'$SRC_DIR/context_displacement_analysis.py' --model_name '$MODEL' --save_root '$CAT_ROOT' --data_path '$DATA_DIR/categories_data.csv'")
    submit "steer-$SAFE" steer "$GRES" "$CDA" \
        "'$SRC_DIR/feature_steering.py' --model_name '$MODEL' --save_root '$CAT_ROOT' --data_path '$DATA_DIR/categories_data.csv' --steer_layers '$STEER_LAYERS'" > /dev/null

    echo "  context_displacement=$CDA"
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run only -- nothing was submitted."
else
    echo "Resubmitted. Track with: squeue -u \$USER"
fi
