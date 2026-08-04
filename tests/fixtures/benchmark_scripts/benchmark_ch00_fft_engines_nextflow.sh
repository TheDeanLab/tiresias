#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${OUT_DIR:-/tmp/tiresias_ch00_000000_fft_benchmark_nextflow}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/work/benchmark_ch00_fft_engines}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-/endosome/work/bioinformatics/s249154/deconvolution-gpu/workflow/scripts/compare_psfs.py}"
DECONVOLUTION_ROOT="${DECONVOLUTION_ROOT:-/endosome/work/bioinformatics/s249154/deconvolution-gpu}"
MATLAB_PSF_SCRIPT="${MATLAB_PSF_SCRIPT:-$DECONVOLUTION_ROOT/workflow/scripts/psf_estimation.py}"
NXF_HOME="${NXF_HOME:-/tmp/tiresias_nextflow_home}"
NXF_OFFLINE="${NXF_OFFLINE:-true}"
BENCHMARK_VERSION="${BENCHMARK_VERSION:-cupyx-scout-v1}"
QUEUE="${QUEUE:-GPUp40}"
RUN_MATLAB="${RUN_MATLAB:-0}"
GPU_POWER_LIMIT_WATTS="${GPU_POWER_LIMIT_WATTS:-0}"
MATLAB_QUEUE="${MATLAB_QUEUE:-super}"
COMPARE_QUEUE="${COMPARE_QUEUE:-$MATLAB_QUEUE}"
MATLAB_CPUS="${MATLAB_CPUS:-24}"
MATLAB_MEMORY="${MATLAB_MEMORY:-128 GB}"
MATLAB_WORKERS="${MATLAB_WORKERS:-24}"
MATLAB_THREADS="${MATLAB_THREADS:-1}"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
MATLAB_TIMEOUT="${MATLAB_TIMEOUT:-1800}"
CHUNK_XY="${CHUNK_XY:-128}"
BLIND_Z_SLICES="${BLIND_Z_SLICES:-128}"
PAD_Z="${PAD_Z:-20}"
BLIND_MAX_TILES="${BLIND_MAX_TILES:-16}"
CUPY_ENGINE="${CUPY_ENGINE:-${REFERENCE_ENGINE:-cupyx}}"
CUPY_FAST_ENGINE="${CUPY_FAST_ENGINE:-${CANDIDATE_ENGINE:-scout}}"
CUPY_N_ITERS="${CUPY_N_ITERS:-10}"
CUPY_FAST_N_ITERS="${CUPY_FAST_N_ITERS:-10}"
CUPY_BLIND_MAX_TILES="${CUPY_BLIND_MAX_TILES:-$BLIND_MAX_TILES}"
CUPY_FAST_BLIND_MAX_TILES="${CUPY_FAST_BLIND_MAX_TILES:-$BLIND_MAX_TILES}"
CUPY_BLIND_Z_SLICES="${CUPY_BLIND_Z_SLICES:-$BLIND_Z_SLICES}"
CUPY_FAST_BLIND_Z_SLICES="${CUPY_FAST_BLIND_Z_SLICES:-$BLIND_Z_SLICES}"
CUPY_TILE_SELECTION_STRATEGY="${CUPY_TILE_SELECTION_STRATEGY:-spatial_snr_v1}"
CUPY_FAST_TILE_SELECTION_STRATEGY="${CUPY_FAST_TILE_SELECTION_STRATEGY:-spatial_snr_v1}"
CUPY_COARSE_REGION_ROWS="${CUPY_COARSE_REGION_ROWS:-4}"
CUPY_COARSE_REGION_COLUMNS="${CUPY_COARSE_REGION_COLUMNS:-4}"
CUPY_COARSE_REGION_LIMIT="${CUPY_COARSE_REGION_LIMIT:-8}"
CUPY_FAST_COARSE_REGION_ROWS="${CUPY_FAST_COARSE_REGION_ROWS:-4}"
CUPY_FAST_COARSE_REGION_COLUMNS="${CUPY_FAST_COARSE_REGION_COLUMNS:-4}"
CUPY_FAST_COARSE_REGION_LIMIT="${CUPY_FAST_COARSE_REGION_LIMIT:-8}"
CUPY_ADAPTIVE_SCOUT_ITERS="${CUPY_ADAPTIVE_SCOUT_ITERS:-2}"
CUPY_ADAPTIVE_KEEP_TILES="${CUPY_ADAPTIVE_KEEP_TILES:-4}"
CUPY_FAST_ADAPTIVE_SCOUT_ITERS="${CUPY_FAST_ADAPTIVE_SCOUT_ITERS:-2}"
CUPY_FAST_ADAPTIVE_KEEP_TILES="${CUPY_FAST_ADAPTIVE_KEEP_TILES:-4}"

case "${RUN_MATLAB,,}" in
  1|true|yes|on)
    RUN_MATLAB="1"
    ;;
  0|false|no|off)
    RUN_MATLAB="0"
    ;;
  *)
    echo "RUN_MATLAB must be 0/1, false/true, no/yes, or off/on" >&2
    exit 2
    ;;
esac

export NXF_HOME NXF_OFFLINE

module load nextflow/24.10.0

mkdir -p "$OUT_DIR" "$WORK_DIR" "$NXF_HOME"
if [ ! -f "$NXF_HOME/framework/24.10.0/nextflow-24.10.0-one.jar" ] \
  && [ -f /home2/s249154/.nextflow/framework/24.10.0/nextflow-24.10.0-one.jar ]; then
  mkdir -p "$NXF_HOME/framework/24.10.0"
  cp /home2/s249154/.nextflow/framework/24.10.0/nextflow-24.10.0-one.jar \
    "$NXF_HOME/framework/24.10.0/"
fi

RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ]; then
  RESUME_ARGS=(-resume)
fi

nextflow run "$REPO_ROOT/tests/fixtures/benchmark_scripts/benchmark_ch00_fft_engines.nf" \
  -work-dir "$WORK_DIR" \
  "${RESUME_ARGS[@]}" \
  --repo_root "$REPO_ROOT" \
  --out_dir "$OUT_DIR" \
  --compare_script "$COMPARE_SCRIPT" \
  --deconvolution_root "$DECONVOLUTION_ROOT" \
  --matlab_psf_script "$MATLAB_PSF_SCRIPT" \
  --benchmark_version "$BENCHMARK_VERSION" \
  --queue "$QUEUE" \
  --run_matlab "$RUN_MATLAB" \
  --gpu_power_limit_watts "$GPU_POWER_LIMIT_WATTS" \
  --matlab_queue "$MATLAB_QUEUE" \
  --compare_queue "$COMPARE_QUEUE" \
  --matlab_cpus "$MATLAB_CPUS" \
  --matlab_memory "$MATLAB_MEMORY" \
  --matlab_workers "$MATLAB_WORKERS" \
  --matlab_threads "$MATLAB_THREADS" \
  --matlab_bin "$MATLAB_BIN" \
  --matlab_timeout "$MATLAB_TIMEOUT" \
  --chunk_xy "$CHUNK_XY" \
  --blind_z_slices "$BLIND_Z_SLICES" \
  --pad_z "$PAD_Z" \
  --blind_max_tiles "$BLIND_MAX_TILES" \
  --cupy_engine "$CUPY_ENGINE" \
  --cupy_fast_engine "$CUPY_FAST_ENGINE" \
  --cupy_n_iters "$CUPY_N_ITERS" \
  --cupy_fast_n_iters "$CUPY_FAST_N_ITERS" \
  --cupy_blind_max_tiles "$CUPY_BLIND_MAX_TILES" \
  --cupy_fast_blind_max_tiles "$CUPY_FAST_BLIND_MAX_TILES" \
  --cupy_blind_z_slices "$CUPY_BLIND_Z_SLICES" \
  --cupy_fast_blind_z_slices "$CUPY_FAST_BLIND_Z_SLICES" \
  --cupy_tile_selection_strategy "$CUPY_TILE_SELECTION_STRATEGY" \
  --cupy_fast_tile_selection_strategy "$CUPY_FAST_TILE_SELECTION_STRATEGY" \
  --cupy_coarse_region_rows "$CUPY_COARSE_REGION_ROWS" \
  --cupy_coarse_region_columns "$CUPY_COARSE_REGION_COLUMNS" \
  --cupy_coarse_region_limit "$CUPY_COARSE_REGION_LIMIT" \
  --cupy_fast_coarse_region_rows "$CUPY_FAST_COARSE_REGION_ROWS" \
  --cupy_fast_coarse_region_columns "$CUPY_FAST_COARSE_REGION_COLUMNS" \
  --cupy_fast_coarse_region_limit "$CUPY_FAST_COARSE_REGION_LIMIT" \
  --cupy_adaptive_scout_iters "$CUPY_ADAPTIVE_SCOUT_ITERS" \
  --cupy_adaptive_keep_tiles "$CUPY_ADAPTIVE_KEEP_TILES" \
  --cupy_fast_adaptive_scout_iters "$CUPY_FAST_ADAPTIVE_SCOUT_ITERS" \
  --cupy_fast_adaptive_keep_tiles "$CUPY_FAST_ADAPTIVE_KEEP_TILES" \
  "$@"

echo "Benchmark outputs: $OUT_DIR"
