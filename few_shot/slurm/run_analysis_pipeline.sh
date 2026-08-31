#!/bin/bash
# Submits the downstream-analysis half of the ADA pipeline (latent regressions through
# behavioral modeling, concept stretching, context displacement analysis, and steering),
# once per model, into few_shot/results_final/.
#
# Run this AFTER every job from run_harvest_pipeline.sh has finished successfully (check
# squeue -u $USER and the logs under few_shot/slurm/logs/) and after re-reviewing
# latent_regressions.py, behavioral_modeling.py, concept_stretching.py,
# context_displacement_analysis.py, and feature_steering.py.
#
# Unlike run_harvest_pipeline.sh, this script does NOT chain SLURM --dependency onto the
# harvest jobs -- those already completed in a separate sbatch session (possibly hours or
# days earlier), so there's no live job ID left to depend on. Instead, before submitting
# anything for a model, it checks that harvest actually produced records.csv for both
# datasets; if not, that model is skipped with an error instead of silently submitting
# regress jobs that are guaranteed to fail once they start.
#
# Per model, the DAG is:
#
#   letters_regress ─┐
#                     ├─▶ cats_regress ─┬─▶ behavioral
#   (harvest, already done) ───────────┘   └─▶ stretch
#
#   context_displacement ─▶ steer  (independent branch: context_displacement_analysis.py
#                                    reads categories_data.csv directly, no dependency on
#                                    the harvest/regression tree)
#
# cats_regress needs letters_regress's all_A.npy for --external_A_path (the "does a
# first-letter map predict semantic displacements" control) and produces its own all_A.npy /
# loco_A.npy, which behavioral_modeling.py and the LOCO analysis then consume.
#
# Usage:
#   ./run_analysis_pipeline.sh                          # submit all 5 models below
#   ./run_analysis_pipeline.sh meta-llama/Llama-3.2-1B   # submit just this one
#   ./run_analysis_pipeline.sh --dry-run                 # print the sbatch calls, submit nothing
#
# Track jobs with: squeue -u $USER
# Cancel a model's analysis DAG: scancel --name='*-<safe_model_name>'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pipeline_common.sh"

parse_models_and_dry_run "$@"

for MODEL in "${MODELS[@]}"; do
    SAFE=$(echo "$MODEL" | tr '/' '_')
    GRES=$(gpu_gres_for_model "$MODEL")

    LET_RECORDS="$LET_ROOT/$SAFE/concept_list_data/records.csv"
    CAT_RECORDS="$CAT_ROOT/$SAFE/concept_list_data/records.csv"
    if [ ! -f "$LET_RECORDS" ] || [ ! -f "$CAT_RECORDS" ]; then
        echo "── Skipping $MODEL: harvest output missing ──" >&2
        echo "   Expected $LET_RECORDS and $CAT_RECORDS" >&2
        echo "   Run run_harvest_pipeline.sh '$MODEL' first." >&2
        continue
    fi

    STEER_LAYERS=$(steer_layers_for_model "$MODEL")
    echo "── Submitting analysis pipeline for $MODEL (safe name: $SAFE, gres: $GRES) ──"

    # Regressions on the letters dataset -- --save_all_A so cats_regress can load it as an
    # external-A control below.
    LREG=$(submit "letters_regress-$SAFE" regress "$GRES" "" \
        "'$SRC_DIR/latent_regressions.py' --model_name '$MODEL' --save_root '$LET_ROOT' --save_all_A")

    # Regressions on the real categories dataset: --save_all_A (behavioral_modeling.py needs
    # every layer's A), --leave_one_concept_out (LOCO at the best layer), and
    # --external_A_path pointed at the letters run's all_A.npy (does a first-letter map
    # predict semantic displacements as well as one trained on semantic categories?).
    CREG=$(submit "cats_regress-$SAFE" regress "$GRES" "$LREG" \
        "'$SRC_DIR/latent_regressions.py' --model_name '$MODEL' --save_root '$CAT_ROOT' --save_all_A --leave_one_concept_out --external_A_path '$LET_ROOT/$SAFE/displacement_regression/all_A.npy'")

    # Behavioral modeling: stretched-direction fit at every layer (default now), LOCO at the
    # best layer only.
    submit "behavioral-$SAFE" behavioral "$GRES" "$CREG" \
        "'$SRC_DIR/behavioral_modeling.py' --model_name '$MODEL' --save_root '$CAT_ROOT' --leave_one_concept_out" > /dev/null

    # concept_stretching on the categories ridge maps -- CPU-only (no GPU requested): it
    # just loads an embedding table via get_input_embeddings().weight and never touches CUDA.
    submit "stretch-$SAFE" stretch "" "$CREG" \
        "'$SRC_DIR/concept_stretching.py' --model_name '$MODEL' --results_dir '$CAT_ROOT/$SAFE/displacement_regression'" > /dev/null

    # context_displacement_analysis, then feature_steering (reads its output dir,
    # auto-derived from --save_root since it matches this stage's --save_root here).
    # --steer_layers is an 8-layer subset proportionally scaled to this model's depth (see
    # steer_layers_for_model) -- feature_steering.py's own default is now "all" layers,
    # which is untested and far too expensive for the steer stage's time budget.
    CDA=$(submit "context_displacement-$SAFE" context_displacement "$GRES" "" \
        "'$SRC_DIR/context_displacement_analysis.py' --model_name '$MODEL' --save_root '$CAT_ROOT' --data_path '$DATA_DIR/categories_data.csv'")
    submit "steer-$SAFE" steer "$GRES" "$CDA" \
        "'$SRC_DIR/feature_steering.py' --model_name '$MODEL' --save_root '$CAT_ROOT' --data_path '$DATA_DIR/categories_data.csv' --steer_layers '$STEER_LAYERS'" > /dev/null

    echo "  letters_regress=$LREG cats_regress=$CREG context_displacement=$CDA"
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run only -- nothing was submitted."
else
    echo "Analysis pipelines submitted. Track with: squeue -u \$USER"
fi
