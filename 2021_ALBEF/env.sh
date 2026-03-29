#!/usr/bin/env bash
set -euo pipefail

export ALBEF_PROJECT_ROOT="/home/lnick/GitHub/Paper/2021_ALBEF"
export ALBEF_CODE_ROOT="${ALBEF_PROJECT_ROOT}/code"
export ALBEF_DATA_ROOT="/home/lnick/DataSet/2021_ALBEF"
export ALBEF_OUTPUT_ROOT="/home/lnick/Output/2021_ALBEF"
export ALBEF_CONDA_ENV="2021_ALBEF"

# Avoid mixing ~/.local packages into this project runtime.
export PYTHONNOUSERSITE=1

# Optional: keep HF cache in the shared cache root.
export HF_HOME="/home/lnick/DataSet/Hugging-Face"
