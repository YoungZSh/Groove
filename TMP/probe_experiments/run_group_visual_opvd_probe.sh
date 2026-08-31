#!/usr/bin/env bash
set -euo pipefail

PROBE_ROOT="/root/siton-tmp/yzs/mmcot_opsd"
PROBE_ENV="${PROBE_ENV:-/home/yzs/miniconda3/envs/vision-opd}"
PROBE_PYTHON="${PROBE_PYTHON:-${PROBE_ENV}/bin/python}"

# The environment has a newer libstdc++ than the host.  Prepending it avoids
# CXXABI import failures in causal_conv1d when Qwen3.5 is loaded.
export LD_LIBRARY_PATH="${PROBE_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec "${PROBE_PYTHON}" "${PROBE_ROOT}/group_visual_opvd_probe.py" "$@"
