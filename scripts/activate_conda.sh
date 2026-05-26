#!/usr/bin/env bash

# Activate the project-local conda environment for DRLA MVP experiments.
# Usage:
#   source /data1/luyifei/drla/scripts/activate_conda.sh

export DRLA_ROOT="/data1/luyifei/drla"
export DRLA_CONDA_ENV="${DRLA_ROOT}/.conda/drla-mvp"

if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/usr/local/anaconda3/etc/profile.d/conda.sh
  source "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
  echo "conda.sh not found under /usr/local/anaconda3" >&2
  return 1 2>/dev/null || exit 1
fi

conda activate "${DRLA_CONDA_ENV}"

# Keep project caches and model/data downloads inside the workspace.
# shellcheck source=/data1/luyifei/drla/scripts/env.sh
source "${DRLA_ROOT}/scripts/env.sh"
