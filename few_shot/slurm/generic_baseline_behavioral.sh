#!/bin/bash
# Computes the generic-noun-list P_0(y) baseline (compute_generic_context_baseline.py) and
# reruns behavioral_modeling.py against it, for each model -- an additive companion to the
# existing BOS-baseline behavioral_modeling run, not a replacement.
#
# Requires cats_regress's all_A.npy to already exist (from run_full_pipeline.sh /
# run_analysis_pipeline.sh) -- this script does NOT redo harvest/regress/stretch/etc.,
# only the two new stages:
#
#   generic_baseline ─▶ behavioral (generic baseline, --leave_one_concept_out)
#
# Output lands in results_final/categories/<model>/behavioral_modeling_generic_context_log_probs/
# (see behavioral_modeling.py's --baseline_log_probs_path), alongside the untouched
# original results_final/categories/<model>/behavioral_modeling/ (BOS baseline).
#
# Usage:
#   ./generic_baseline_behavioral.sh                          # all 5 models below
#   ./generic_baseline_behavioral.sh meta-llama/Llama-3.2-1B   # just this one
#   ./generic_baseline_behavioral.sh --dry-run                 # print the sbatch calls, submit nothing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_common.sh"

# generic_baseline is a new, much cheaper stage (n_contexts=10 forward passes, no dataset
# iteration) than every existing stage; behavioral_generic reuses the existing "behavioral"
# budget below since it's the exact same workload (full behavioral_modeling.py + LOCO),
# just against a different baseline.
STAGE_TIME[generic_baseline]="00:30:00"
STAGE_MEM[generic_baseline]="32G"

parse_models_and_dry_run "$@"

for MODEL in "${MODELS[@]}"; do
    SAFE=$(echo "$MODEL" | tr '/' '_')
    GRES=$(gpu_gres_for_model "$MODEL")

    ALL_A="$CAT_ROOT/$SAFE/displacement_regression/all_A.npy"
    if [ ! -f "$ALL_A" ]; then
        echo "── Skipping $MODEL: $ALL_A not found (cats_regress hasn't completed) ──" >&2
        continue
    fi

    echo "── Submitting generic-baseline behavioral modeling for $MODEL (safe name: $SAFE, gres: $GRES) ──"

    GB=$(submit "generic-baseline-$SAFE" generic_baseline "$GRES" "" \
        "'$SRC_DIR/compute_generic_context_baseline.py' --model_name '$MODEL' --save_root '$CAT_ROOT' --data_path '$DATA_DIR/categories_data.csv'")
    submit "behavioral-generic-$SAFE" behavioral "$GRES" "$GB" \
        "'$SRC_DIR/behavioral_modeling.py' --model_name '$MODEL' --save_root '$CAT_ROOT' --leave_one_concept_out --baseline_log_probs_path '$CAT_ROOT/$SAFE/concept_list_data/generic_context_log_probs.pt'" > /dev/null

    echo "  generic_baseline=$GB"
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run only -- nothing was submitted."
else
    echo "Submitted. Track with: squeue -u \$USER"
fi
