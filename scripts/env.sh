#!/usr/bin/env bash

# Project-local personal Python user base for DRLA experiments.
# Usage:
#   source /data1/luyifei/drla/scripts/env.sh
#   python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --user some_package==1.0.0

export DRLA_ROOT="/data1/luyifei/drla"
export PIP_CACHE_DIR="${DRLA_ROOT}/.cache/pip"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export CONDA_PKGS_DIRS="${DRLA_ROOT}/.cache/conda/pkgs"
export HF_HOME="${DRLA_ROOT}/.cache/huggingface"
export HF_DATASETS_CACHE="${DRLA_ROOT}/.cache/huggingface/datasets"
export HF_HUB_CACHE="${DRLA_ROOT}/.cache/huggingface/hub"

_drla_project_conda="${DRLA_ROOT}/.conda/drla-mvp"

if [ "${CONDA_PREFIX:-}" = "${_drla_project_conda}" ]; then
  unset PYTHONUSERBASE
  export PYTHONNOUSERSITE=1
else
  export PYTHONUSERBASE="${DRLA_ROOT}/.pyuser"
  export PATH="${PYTHONUSERBASE}/bin:${PATH}"

  _drla_pyver="$(python3 - <<'PY'
import sys
print(f"python{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

  export PYTHONPATH="${PYTHONUSERBASE}/lib/${_drla_pyver}/site-packages:${PYTHONPATH:-}"
  unset _drla_pyver
fi

unset _drla_project_conda
