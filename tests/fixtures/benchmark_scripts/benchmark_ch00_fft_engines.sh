#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-/endosome/work/bioinformatics/s249154/decon_env}"
IMAGE_PATH="${IMAGE_PATH:-/archive/bioinformatics/Danuser_lab/Dean/dean/2026-06-10-ricky/CH00_000000.tiff}"
OUT_DIR="${OUT_DIR:-/tmp/tiresias_ch00_000000_fft_benchmark}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-/endosome/work/bioinformatics/s249154/deconvolution-gpu/workflow/scripts/compare_psfs.py}"
CHUNK_XY="${CHUNK_XY:-128}"
BLIND_Z_SLICES="${BLIND_Z_SLICES:-128}"
PAD_Z="${PAD_Z:-20}"
BLIND_MAX_TILES="${BLIND_MAX_TILES:-16}"
REFERENCE_ENGINE="${REFERENCE_ENGINE:-cupyx}"
CANDIDATE_ENGINE="${CANDIDATE_ENGINE:-scout}"
ADAPTIVE_SCOUT_ITERS="${ADAPTIVE_SCOUT_ITERS:-2}"
ADAPTIVE_KEEP_TILES="${ADAPTIVE_KEEP_TILES:-4}"

mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --image-path "$IMAGE_PATH"
  --wavelength 0.515
  --na 0.7467
  --ni 1.56
  --ns 1.56
  --dxy 0.167
  --dz 0.2
  --n-iters 10
  --chunk-xy "$CHUNK_XY"
  --blind-z-slices "$BLIND_Z_SLICES"
  --pad-z "$PAD_Z"
  --blind-max-tiles "$BLIND_MAX_TILES"
  --no-psf-cache
)

run_estimate() {
  local engine="$1"
  local output_path="$OUT_DIR/estimated_psf_${engine}.tif"
  local log_path="$OUT_DIR/estimate_${engine}.log"
  local time_path="$OUT_DIR/estimate_${engine}.time"

  echo "Running cupy_fft_engine=${engine}"
  /usr/bin/time \
    -f "elapsed_seconds=%e"$'\n'"max_rss_kb=%M" \
    -o "$time_path" \
    conda run -p "$CONDA_ENV" env \
      PYTHONPATH="$REPO_ROOT/src" \
      python -c 'import sys; from tiresias.cli import estimate_psf_main; estimate_psf_main(sys.argv[1:])' \
      "${COMMON_ARGS[@]}" \
      --cupy-fft-engine "$engine" \
      --adaptive-scout-iters "$ADAPTIVE_SCOUT_ITERS" \
      --adaptive-keep-tiles "$ADAPTIVE_KEEP_TILES" \
      --output-path "$output_path" \
    2>&1 | tee "$log_path"

  echo "Wrote $output_path"
  cat "$time_path"
}

run_estimate "$REFERENCE_ENGINE"
run_estimate "$CANDIDATE_ENGINE"

conda run -p "$CONDA_ENV" python "$COMPARE_SCRIPT" \
  "$OUT_DIR/estimated_psf_${REFERENCE_ENGINE}.tif" \
  "$OUT_DIR/estimated_psf_${CANDIDATE_ENGINE}.tif" \
  --spacing 0.2 0.167 0.167 \
  --csv "$OUT_DIR/psf_comparison.csv" \
  --json "$OUT_DIR/psf_comparison.json" \
  2>&1 | tee "$OUT_DIR/psf_comparison.log"

{
  echo "output_dir=$OUT_DIR"
  echo "reference_engine=$REFERENCE_ENGINE"
  echo "candidate_engine=$CANDIDATE_ENGINE"
  echo "adaptive_scout_iters=$ADAPTIVE_SCOUT_ITERS"
  echo "adaptive_keep_tiles=$ADAPTIVE_KEEP_TILES"
  echo "${REFERENCE_ENGINE}_time_file=$OUT_DIR/estimate_${REFERENCE_ENGINE}.time"
  echo "${CANDIDATE_ENGINE}_time_file=$OUT_DIR/estimate_${CANDIDATE_ENGINE}.time"
  echo "comparison_csv=$OUT_DIR/psf_comparison.csv"
  echo "comparison_json=$OUT_DIR/psf_comparison.json"
} | tee "$OUT_DIR/benchmark_summary.txt"
