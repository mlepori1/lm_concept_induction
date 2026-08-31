#!/bin/bash
# Shared config + helpers for the harvest/analysis SLURM pipeline scripts. Source this from
# run_harvest_pipeline.sh / run_analysis_pipeline.sh -- it's not meant to be run directly.

PROJECT_ROOT="/oscar/data/epavlick/mlepori/projects/rerepresentation"
FEW_SHOT_DIR="$PROJECT_ROOT/few_shot"
SRC_DIR="$FEW_SHOT_DIR/src"
DATA_DIR="$FEW_SHOT_DIR/data"
RESULTS_ROOT="$FEW_SHOT_DIR/results_final"
LOG_DIR="$FEW_SHOT_DIR/slurm/logs"
CAT_ROOT="$RESULTS_ROOT/categories"
LET_ROOT="$RESULTS_ROOT/letters"

PARTITION="gpu"
CPUS_PER_TASK=8

# All 5 target models already have cached weights under $HF_HOME/hub -- HF_HUB_OFFLINE=1
# below keeps every job from touching the network. If you add a model that isn't cached
# yet, download it from a login node first (compute nodes here have no internet).
HF_HOME_VAL="${HF_HOME:-/gpfs/data/epavlick/mlepori}"

# Rough per-stage wall-clock/memory budgets -- these are estimates, not measurements.
# feature_steering.py in particular does O(n_categories * n_supports * layers * conditions *
# alphas) forward passes and is the single most expensive stage; adjust if it times out.
# stretch (concept_stretching.py) force-loads torch_dtype=float32 to grab the embedding
# table, which for the 7B model is ~28GB of weights alone -- 48G leaves headroom for
# whatever HF's loading path needs above the final resident size.
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

DEFAULT_MODELS=(
    "google/gemma-2-2b"
    "allenai/OLMo-2-0425-1B"
    "allenai/OLMo-2-1124-7B"
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-3B"
)

# parse_models_and_dry_run "$@" -- sets the DRY_RUN and MODELS globals from a script's argv.
parse_models_and_dry_run () {
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
        MODELS=("${DEFAULT_MODELS[@]}")
    fi
}

mkdir -p "$LOG_DIR"

# submit <jobname> <stage-key> <gres> <dep-jobids-colon-separated-or-empty> <python-cmd-string>
# Pass an empty string for <gres> to request no GPU at all (e.g. concept_stretching.py,
# which only loads an embedding table on CPU and never touches CUDA).
# Returns the new job's numeric ID on stdout (or "DRYRUN-<jobname>" in --dry-run mode).
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
