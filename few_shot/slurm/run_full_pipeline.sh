#!/bin/bash
# Submits the full ADA pipeline (letters + categories data collection through steering) as a
# chained SLURM job DAG, once per model, into few_shot/results_final/ -- a fresh tree that
# doesn't touch existing results/ runs.
#
# Per model, the DAG is:
#
#   letters_gen ─▶ letters_sanity ─▶ letters_collect ─▶ letters_regress ─┐
#                                                                        ├─▶ cats_regress ─┬─▶ behavioral
#                    cats_sanity ─▶ cats_collect ────────────────────── ┘                 └─▶ stretch
#
#   context_displacement ─▶ steer  (independent branch: context_displacement_analysis.py
#                                    reads categories_data.csv directly, no dependency on
#                                    the concept_list_data/displacement_regression tree)
#
# *_sanity runs concept_list_behavior.py -- a cheap early check that the model shows basic
# concept induction (positive context boost) on that dataset -- BEFORE the expensive
# collection/regression stages run, so a model that doesn't do the intended thing fails fast
# instead of burning GPU-hours downstream.
#
# cats_regress needs letters_regress's all_A.npy for --external_A_path (the "does a
# first-letter map predict semantic displacements" control) and produces its own all_A.npy /
# loco_A.npy, which behavioral_modeling.py and the LOCO analysis then consume.
#
# Usage:
#   ./run_full_pipeline.sh                          # submit all 5 models below
#   ./run_full_pipeline.sh meta-llama/Llama-3.2-1B   # submit just this one
#   ./run_full_pipeline.sh --dry-run                 # print the sbatch calls, submit nothing
#
# Track jobs with: squeue -u $USER
# Cancel a model's whole DAG: scancel --name='*-<safe_model_name>'

set -euo pipefail

# ── Config -- tune to taste ──────────────────────────────────────────────────
PROJECT_ROOT="/oscar/data/epavlick/mlepori/projects/rerepresentation"
FEW_SHOT_DIR="$PROJECT_ROOT/few_shot"
SRC_DIR="$FEW_SHOT_DIR/src"
DATA_DIR="$FEW_SHOT_DIR/data"
RESULTS_ROOT="$FEW_SHOT_DIR/results_final"
LOG_DIR="$FEW_SHOT_DIR/slurm/logs"

PARTITION="gpu"
CPUS_PER_TASK=8

# All 5 target models already have cached weights under $HF_HOME/hub -- HF_HUB_OFFLINE=1
# below keeps every job from touching the network. If you add a model that isn't cached
# yet, download it from a login node first (compute nodes here have no internet).
HF_HOME_VAL="${HF_HOME:-/gpfs/data/epavlick/mlepori}"

# Rough per-stage wall-clock/memory budgets -- these are estimates, not measurements.
# feature_steering.py in particular does O(n_categories * n_supports * layers * conditions *
# alphas) forward passes and is the single most expensive stage; adjust if it times out.
declare -A STAGE_TIME=(
    [sanity]="04:00:00"
    [collect]="08:00:00"
    [regress]="08:00:00"
    [behavioral]="08:00:00"
    [stretch]="01:00:00"
    [context_displacement]="04:00:00"
    [steer]="16:00:00"
)
declare -A STAGE_MEM=(
    [sanity]="32G"
    [collect]="64G"
    [regress]="64G"
    [behavioral]="64G"
    [stretch]="48G"
    [context_displacement]="64G"
    [steer]="64G"
)

# 7B models get a 48GB card (nvidia_a40 / nvidia_rtx_a6000 are the ones available in bulk
# here); everything else takes whatever GPU the "gpu" partition hands it.
gpu_gres_for_model () {
    case "$1" in
        *7B*|*7b*) echo "gpu:nvidia_a40:1" ;;
        *)         echo "gpu:1" ;;
    esac
}

# steer_layers_for_model <model_name> -- prints a comma-separated list of 8 layer indices
# for feature_steering.py's --steer_layers, proportionally scaled to this model's actual
# num_hidden_layers. The original hand-tuned list was [3,6,9,12,15,18,21,24], which is
# exactly linspace(3, 24, 8) for a 26-layer model (Gemma-2-2b) -- this reproduces that list
# exactly for Gemma-2-2b and scales the same 11.5%-92.3%-of-depth span to any other model,
# so every model gets a comparably-tuned 8-layer subset instead of all layers (which is
# both untested and, per STAGE_TIME[steer], likely to blow the time budget).
# Only reads the (already locally cached) config, no weights -- cheap enough to run here
# on the login node at submission time rather than as its own SLURM job.
steer_layers_for_model () {
    local model_name="$1"
    HF_HUB_OFFLINE=1 uv run --frozen --project "$PROJECT_ROOT" python -c "
import sys
import numpy as np
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
n_layers = cfg.num_hidden_layers
layers = sorted(set(int(round(x)) for x in np.linspace(3 / 26 * n_layers, 24 / 26 * n_layers, 8)))
print(','.join(str(l) for l in layers))
" "$model_name"
}

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
        "google/gemma-2-2b"
        "allenai/OLMo-2-0425-1B"
        "allenai/OLMo-2-1124-7B"
        "meta-llama/Llama-3.2-1B"
        "meta-llama/Llama-3.2-3B"
    )
fi

mkdir -p "$LOG_DIR"

# submit <jobname> <stage-key> <gres> <dep-jobids-colon-separated-or-empty> <python-cmd-string>
# Pass an empty string for <gres> to request no GPU at all (e.g. concept_stretching.py,
# which only loads an embedding table on CPU and never touches CUDA).
# Returns the new job's numeric ID on stdout (or "SKIPPED" in --dry-run mode).
submit () {
    local jobname="$1" stage="$2" gres="$3" deps="$4" cmd="$5"
    local dep_args=()
    if [ -n "$deps" ]; then
        dep_args=(--dependency="afterok:${deps}")
    fi
    local gres_args=()
    if [ -n "$gres" ]; then
        gres_args=(--gres="$gres")
    fi
    local wrap="export HF_HOME='$HF_HOME_VAL' HF_HUB_OFFLINE=1 && cd '$SRC_DIR' && uv run --frozen --project '$PROJECT_ROOT' python $cmd"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "[dry-run] $jobname (stage=$stage gres=${gres:-none} deps=${deps:-none})" >&2
        echo "    $wrap" >&2
        echo "DRYRUN-$jobname"
        return
    fi

    sbatch --parsable \
        --job-name="$jobname" \
        --output="$LOG_DIR/${jobname}-%j.out" \
        --error="$LOG_DIR/${jobname}-%j.err" \
        --partition="$PARTITION" \
        "${gres_args[@]}" \
        --cpus-per-task="$CPUS_PER_TASK" \
        --mem="${STAGE_MEM[$stage]}" \
        --time="${STAGE_TIME[$stage]}" \
        "${dep_args[@]}" \
        --wrap="$wrap"
}

for MODEL in "${MODELS[@]}"; do
    SAFE=$(echo "$MODEL" | tr '/' '_')
    GRES=$(gpu_gres_for_model "$MODEL")
    STEER_LAYERS=$(steer_layers_for_model "$MODEL")
    echo "── Submitting pipeline for $MODEL (safe name: $SAFE, gres: $GRES) ──"

    CAT_ROOT="$RESULTS_ROOT/categories"
    LET_ROOT="$RESULTS_ROOT/letters"
    LETTER_CSV="$DATA_DIR/letter_data_${SAFE}.csv"

    # (0a) Letters: per-model word list, then a sanity check that the model shows basic
    # concept induction on it, before collecting activations.
    GEN=$(submit "letters_gen-$SAFE" collect "$GRES" "" \
        "'$DATA_DIR/generate_letter_data.py' --model_name '$MODEL' --out_path '$LETTER_CSV'")
    LSANE=$(submit "letters_sanity-$SAFE" sanity "$GRES" "$GEN" \
        "'$SRC_DIR/eval_model.py' --model_name '$MODEL' --data_path '$LETTER_CSV' --save_root '$LET_ROOT'")
    LCOL=$(submit "letters_collect-$SAFE" collect "$GRES" "$LSANE" \
        "'$SRC_DIR/harvest_model_data.py' --model_name '$MODEL' --data_path '$LETTER_CSV' --save_root '$LET_ROOT'")

    # (0b) Categories: same sanity check, then activation collection (independent of the
    # letters branch)
    CSANE=$(submit "cats_sanity-$SAFE" sanity "$GRES" "" \
        "'$SRC_DIR/eval_model.py' --model_name '$MODEL' --data_path '$DATA_DIR/categories_data.csv' --save_root '$CAT_ROOT'")
    CCOL=$(submit "cats_collect-$SAFE" collect "$GRES" "$CSANE" \
        "'$SRC_DIR/harvest_model_data.py' --model_name '$MODEL' --data_path '$DATA_DIR/categories_data.csv' --save_root '$CAT_ROOT'")

    # (2) Regressions on the letters dataset -- --save_all_A so cats_regress can load it
    # as an external-A control below.
    LREG=$(submit "letters_regress-$SAFE" regress "$GRES" "$LCOL" \
        "'$SRC_DIR/latent_regressions.py' --model_name '$MODEL' --save_root '$LET_ROOT' --save_all_A")

    # (3) Regressions on the real categories dataset: --save_all_A (behavioral_modeling.py
    # needs every layer's A), --leave_one_concept_out (LOCO at the best layer), and
    # --external_A_path pointed at the letters run's all_A.npy (does a first-letter map
    # predict semantic displacements as well as one trained on semantic categories?).
    CREG=$(submit "cats_regress-$SAFE" regress "$GRES" "${CCOL}:${LREG}" \
        "'$SRC_DIR/latent_regressions.py' --model_name '$MODEL' --save_root '$CAT_ROOT' --save_all_A --leave_one_concept_out --external_A_path '$LET_ROOT/$SAFE/displacement_regression/all_A.npy'")

    # (4) Behavioral modeling: stretched-direction fit at every layer (default now), LOCO
    # at the best layer only.
    submit "behavioral-$SAFE" behavioral "$GRES" "$CREG" \
        "'$SRC_DIR/behavioral_modeling.py' --model_name '$MODEL' --save_root '$CAT_ROOT' --leave_one_concept_out" > /dev/null

    # (5) concept_stretching on the categories ridge maps -- CPU-only (no GPU requested):
    # it just loads an embedding table via get_input_embeddings().weight, never touches CUDA.
    submit "stretch-$SAFE" stretch "" "$CREG" \
        "'$SRC_DIR/concept_stretching.py' --model_name '$MODEL' --results_dir '$CAT_ROOT/$SAFE/displacement_regression'" > /dev/null

    # (6)+(7) context_displacement_analysis, then feature_steering (reads its output dir,
    # auto-derived from --save_root since it matches (6)'s --save_root here).
    # --steer_layers is an 8-layer subset proportionally scaled to this model's depth (see
    # steer_layers_for_model) -- feature_steering.py's own default is now "all" layers,
    # which is untested and far too expensive for the steer stage's time budget.
    CDA=$(submit "context_displacement-$SAFE" context_displacement "$GRES" "" \
        "'$SRC_DIR/context_displacement_analysis.py' --model_name '$MODEL' --save_root '$CAT_ROOT' --data_path '$DATA_DIR/categories_data.csv'")
    submit "steer-$SAFE" steer "$GRES" "$CDA" \
        "'$SRC_DIR/feature_steering.py' --model_name '$MODEL' --save_root '$CAT_ROOT' --data_path '$DATA_DIR/categories_data.csv' --steer_layers '$STEER_LAYERS'" > /dev/null

    echo "  letters_gen=$GEN letters_sanity=$LSANE letters_collect=$LCOL cats_sanity=$CSANE cats_collect=$CCOL letters_regress=$LREG cats_regress=$CREG context_displacement=$CDA"
done

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run only -- nothing was submitted."
else
    echo "All pipelines submitted. Track with: squeue -u $USER"
fi
