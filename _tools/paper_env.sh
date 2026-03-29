#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source ../_tools/paper_env.sh [PAPER_ID]
#
# Prefer setting PAPER_ID in the caller (e.g. each paper repo's env.sh).

if [[ -z "${PAPER_ID:-}" ]]; then
  PAPER_ID="${1:-$(basename "$(pwd)")}" 
fi

export PAPER_ID

: "${PAPER_ROOT:=/home/lnick/GitHub/Paper/${PAPER_ID}}"
: "${PAPER_DATA:=/home/lnick/DataSet/Hugging-Face/${PAPER_ID}}"

export PAPER_ROOT
export PAPER_DATA

# Strict isolation: always derive HF cache locations from PAPER_DATA, ignoring any
# globally-exported HF_* variables.
export HF_HOME="${PAPER_DATA}/hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# Output isolation: /home/lnick/Output/<paper>/output
export OUTPUT_ROOT="/home/lnick/Output/${PAPER_ID}"
export OUTPUT_DIR="${OUTPUT_ROOT}/output"

mkdir -p \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_MODULES_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${PAPER_DATA}/data" \
  "${OUTPUT_ROOT}/logs" \
  "${OUTPUT_DIR}"

# Optional: auto-activate conda env named as PAPER_ID when sourced in
# interactive shells. Disable with: export PAPER_AUTO_ACTIVATE_CONDA=0
if [[ "${PAPER_AUTO_ACTIVATE_CONDA:-1}" == "1" ]] && [[ $- == *i* ]]; then
  if command -v conda >/dev/null 2>&1; then
    if conda env list 2>/dev/null | awk '{print $1}' | grep -Fxq "${PAPER_ID}"; then
      if [[ "${CONDA_DEFAULT_ENV:-}" != "${PAPER_ID}" ]]; then
        conda activate "${PAPER_ID}" >/dev/null 2>&1 || conda activate "${PAPER_ID}"
      fi
    fi
  fi
fi
