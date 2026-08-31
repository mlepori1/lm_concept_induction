#!/bin/bash
# Submits just the data-harvesting half of the ADA pipeline (letter-data generation/sanity/
# collection + categories sanity/collection), once per model, into few_shot/results_final/ --
# a fresh tree that doesn't touch existing results/ runs.
#
# This is stage 0 of run_full_pipeline.sh, split out so it can run unattended (e.g.
# overnight) while the downstream analysis scripts (latent_regressions.py,
# behavioral_modeling.py, concept_stretching.py, context_displacement_analysis.py,
# feature_steering.py) get a final review. Once every job here finishes successfully and
# the analysis scripts are confirmed correct, run run_analysis_pipeline.sh to pick up from
# here.
#
# Per model, the DAG is:
#
#   letters_gen ─▶ letters_sanity ─▶ letters_collect
#
#                    cats_sanity ─▶ cats_collect        (independent branch)
#
# *_sanity runs eval_model.py -- a cheap early check that the model shows basic concept
# induction (positive context boost) on that dataset -- BEFORE the expensive collection
# stage runs, so a model that doesn't do the intended thing fails fast instead of burning
# GPU-hours downstream.
#
# Usage:
#   ./run_harvest_pipeline.sh                          # submit all 5 models below
#   ./run_harvest_pipeline.sh meta-llama/Llama-3.2-1B   # submit just this one
#   ./run_harvest_pipeline.sh --dry-run                 # print the sbatch calls, submit nothing
#
# Track jobs with: squeue -u $USER
# Cancel a model's harvest DAG: scancel --name='*-<safe_model_name>'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_common.sh"

parse_models_and_dry_run "$@"

for MODEL in "${MODELS[@]}"; do
    SAFE=$(echo "$MODEL" | tr '/' '_')
    GRES=$(gpu_gres_for_model "$MODEL")
    echo "── Submitting harvest pipeline for $MODEL (safe name: $SAFE, gres: $GRES) ──"

    LETTER_CSV="$DATA_DIR/letter_data_${SAFE}.csv"

    # Letters: per-model word list, then a sanity check that the model shows basic concept
    # induction on it, before collecting activations.
    GEN=$(submit "letters_gen-$SAFE" collect "$GRES" "" \
        "'$DATA_DIR/generate_letter_data.py' --model_name '$MODEL' --out_path '$LETTER_CSV'")
    LSANE=$(submit "letters_sanity-$SAFE" sanity "$GRES" "$GEN" \
        "'$SRC_DIR/eval_model.py' --model_name '$MODEL' --data_path '$LETTER_CSV' --save_root '$LET_ROOT'")
    LCOL=$(submit "letters_collect-$SAFE" collect "$GRES" "$LSANE" \
        "'$SRC_DIR/harvest_model_data.py' --model_name '$MODEL' --data_path '$LETTER_CSV' --save_root '$LET_ROOT'")

    # Categories: same sanity check, then activation collection (independent of the letters
    # branch).
    CSANE=$(submit "cats_sanity-$SAFE" sanity "$GRES" "" \
        "'$SRC_DIR/eval_model.py' --model_name '$MODEL' --data_path '$DATA_DIR/categories_data.csv' --save_root '$CAT_ROOT'")
    CCOL=$(submit "cats_collect-$SAFE" collect "$GRES" "$CSANE" \
        "'$SRC_DIR/harvest_model_data.py' --model_name '$MODEL' --data_path '$DATA_DIR/categories_data.csv' --save_root '$CAT_ROOT'")

    echo "  letters_gen=$GEN letters_sanity=$LSANE letters_collect=$LCOL cats_sanity=$CSANE cats_collect=$CCOL"
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run only -- nothing was submitted."
else
    echo "Harvest pipelines submitted. Track with: squeue -u \$USER"
    echo "Once these finish, review the analysis scripts and then run ./run_analysis_pipeline.sh"
fi
