#!/usr/bin/env nextflow
nextflow.enable.dsl=2

params.repo_root = params.containsKey('repo_root') && params.repo_root
    ? params.repo_root
    : '/endosome/work/bioinformatics/s249154/tiresias'
params.conda_env = params.containsKey('conda_env') && params.conda_env
    ? params.conda_env
    : '/endosome/work/bioinformatics/s249154/decon_env'
params.image_path = params.containsKey('image_path') && params.image_path
    ? params.image_path
    : '/archive/bioinformatics/Danuser_lab/Dean/dean/2026-06-10-ricky/CH00_000000.tiff'
params.out_dir = params.containsKey('out_dir') && params.out_dir
    ? params.out_dir
    : '/tmp/tiresias_ch00_000000_fft_benchmark_nextflow'
params.compare_script = params.containsKey('compare_script') && params.compare_script
    ? params.compare_script
    : '/endosome/work/bioinformatics/s249154/deconvolution-gpu/workflow/scripts/compare_psfs.py'
params.deconvolution_root = params.containsKey('deconvolution_root') && params.deconvolution_root
    ? params.deconvolution_root
    : '/endosome/work/bioinformatics/s249154/deconvolution-gpu'
params.matlab_psf_script = params.containsKey('matlab_psf_script') && params.matlab_psf_script
    ? params.matlab_psf_script
    : "${params.deconvolution_root}/workflow/scripts/psf_estimation.py"
params.conda_exe = params.containsKey('conda_exe') && params.conda_exe
    ? params.conda_exe
    : '/usr/bin/conda'
params.benchmark_version = params.containsKey('benchmark_version') && params.benchmark_version
    ? params.benchmark_version
    : 'cupyx-scout-v1'
params.queue = params.containsKey('queue') && params.queue
    ? params.queue
    : 'GPU'
params.gpu_power_limit_watts = params.containsKey('gpu_power_limit_watts') && params.gpu_power_limit_watts
    ? params.gpu_power_limit_watts
    : 0
run_matlab = params.containsKey('run_matlab') && params.run_matlab
    ? params.run_matlab.toString().toLowerCase() in ['1', 'true', 'yes', 'on']
    : false
params.run_matlab = run_matlab
params.matlab_queue = params.containsKey('matlab_queue') && params.matlab_queue
    ? params.matlab_queue
    : 'super'
params.compare_queue = params.containsKey('compare_queue') && params.compare_queue
    ? params.compare_queue
    : params.matlab_queue
params.matlab_cpus = params.containsKey('matlab_cpus') && params.matlab_cpus
    ? params.matlab_cpus
    : 24
params.matlab_memory = params.containsKey('matlab_memory') && params.matlab_memory
    ? params.matlab_memory
    : '128 GB'
params.matlab_workers = params.containsKey('matlab_workers') && params.matlab_workers
    ? params.matlab_workers
    : 24
params.matlab_threads = params.containsKey('matlab_threads') && params.matlab_threads
    ? params.matlab_threads
    : 1
params.matlab_bin = params.containsKey('matlab_bin') && params.matlab_bin
    ? params.matlab_bin
    : 'matlab'
params.matlab_timeout = params.containsKey('matlab_timeout') && params.matlab_timeout
    ? params.matlab_timeout
    : 1800
params.chunk_xy = params.containsKey('chunk_xy') && params.chunk_xy
    ? params.chunk_xy
    : 128
params.blind_z_slices = params.containsKey('blind_z_slices') && params.blind_z_slices
    ? params.blind_z_slices
    : 128
params.pad_z = params.containsKey('pad_z') && params.pad_z
    ? params.pad_z
    : 20
params.blind_max_tiles = params.containsKey('blind_max_tiles') && params.blind_max_tiles
    ? params.blind_max_tiles
    : 16
params.cupy_engine = params.containsKey('cupy_engine') && params.cupy_engine
    ? params.cupy_engine
    : 'cupyx'
params.cupy_fast_engine = params.containsKey('cupy_fast_engine') && params.cupy_fast_engine
    ? params.cupy_fast_engine
    : 'scout'
params.cupy_n_iters = params.containsKey('cupy_n_iters') && params.cupy_n_iters
    ? params.cupy_n_iters
    : 10
params.cupy_fast_n_iters = params.containsKey('cupy_fast_n_iters') && params.cupy_fast_n_iters
    ? params.cupy_fast_n_iters
    : 10
params.cupy_blind_max_tiles = params.containsKey('cupy_blind_max_tiles') && params.cupy_blind_max_tiles
    ? params.cupy_blind_max_tiles
    : params.blind_max_tiles
params.cupy_fast_blind_max_tiles = params.containsKey('cupy_fast_blind_max_tiles') && params.cupy_fast_blind_max_tiles
    ? params.cupy_fast_blind_max_tiles
    : params.blind_max_tiles
params.cupy_blind_z_slices = params.containsKey('cupy_blind_z_slices') && params.cupy_blind_z_slices
    ? params.cupy_blind_z_slices
    : params.blind_z_slices
params.cupy_fast_blind_z_slices = params.containsKey('cupy_fast_blind_z_slices') && params.cupy_fast_blind_z_slices
    ? params.cupy_fast_blind_z_slices
    : params.blind_z_slices
params.cupy_tile_selection_strategy = params.containsKey('cupy_tile_selection_strategy') && params.cupy_tile_selection_strategy
    ? params.cupy_tile_selection_strategy
    : 'spatial_snr_v1'
params.cupy_fast_tile_selection_strategy = params.containsKey('cupy_fast_tile_selection_strategy') && params.cupy_fast_tile_selection_strategy
    ? params.cupy_fast_tile_selection_strategy
    : 'spatial_snr_v1'
params.cupy_coarse_region_rows = params.containsKey('cupy_coarse_region_rows') && params.cupy_coarse_region_rows
    ? params.cupy_coarse_region_rows
    : 4
params.cupy_coarse_region_columns = params.containsKey('cupy_coarse_region_columns') && params.cupy_coarse_region_columns
    ? params.cupy_coarse_region_columns
    : 4
params.cupy_coarse_region_limit = params.containsKey('cupy_coarse_region_limit') && params.cupy_coarse_region_limit
    ? params.cupy_coarse_region_limit
    : 8
params.cupy_fast_coarse_region_rows = params.containsKey('cupy_fast_coarse_region_rows') && params.cupy_fast_coarse_region_rows
    ? params.cupy_fast_coarse_region_rows
    : 4
params.cupy_fast_coarse_region_columns = params.containsKey('cupy_fast_coarse_region_columns') && params.cupy_fast_coarse_region_columns
    ? params.cupy_fast_coarse_region_columns
    : 4
params.cupy_fast_coarse_region_limit = params.containsKey('cupy_fast_coarse_region_limit') && params.cupy_fast_coarse_region_limit
    ? params.cupy_fast_coarse_region_limit
    : 8
params.cupy_adaptive_scout_iters = params.containsKey('cupy_adaptive_scout_iters') && params.cupy_adaptive_scout_iters
    ? params.cupy_adaptive_scout_iters
    : 2
params.cupy_adaptive_keep_tiles = params.containsKey('cupy_adaptive_keep_tiles') && params.cupy_adaptive_keep_tiles
    ? params.cupy_adaptive_keep_tiles
    : 4
params.cupy_fast_adaptive_scout_iters = params.containsKey('cupy_fast_adaptive_scout_iters') && params.cupy_fast_adaptive_scout_iters
    ? params.cupy_fast_adaptive_scout_iters
    : 2
params.cupy_fast_adaptive_keep_tiles = params.containsKey('cupy_fast_adaptive_keep_tiles') && params.cupy_fast_adaptive_keep_tiles
    ? params.cupy_fast_adaptive_keep_tiles
    : 4
process ESTIMATE_CUPY_PSF {
    tag "${label}"
    executor 'slurm'
    queue "${params.queue}"
    cpus 8
    memory '64 GB'
    time '04:00:00'
    clusterOptions '--gres=gpu:1'
    publishDir "${params.out_dir}", mode: 'copy'

    input:
    tuple val(label), val(engine), val(n_iters), val(max_tiles), val(z_slices), val(tile_strategy), val(coarse_rows), val(coarse_columns), val(coarse_limit), val(adaptive_scout_iters), val(adaptive_keep_tiles)

    output:
    path "estimated_psf_${label}.tif", emit: psf
    path "estimate_${label}.time", emit: time
    path "estimate_${label}.log", emit: log
    path "gpu_state_${label}.log", emit: gpu_log

    script:
    """
    set -euo pipefail
    echo "benchmark_version=${params.benchmark_version}"
    echo "label=${label}"
    echo "cupy_fft_engine=${engine}"
    echo "n_iters=${n_iters}"
    echo "blind_max_tiles=${max_tiles}"
    echo "blind_z_slices=${z_slices}"
    echo "tile_selection_strategy=${tile_strategy}"
    echo "coarse_region_rows=${coarse_rows}"
    echo "coarse_region_columns=${coarse_columns}"
    echo "coarse_region_limit=${coarse_limit}"
    echo "adaptive_scout_iters=${adaptive_scout_iters}"
    echo "adaptive_keep_tiles=${adaptive_keep_tiles}"
    echo "gpu_power_limit_watts=${params.gpu_power_limit_watts}"

    visible_devices="\${CUDA_VISIBLE_DEVICES:-}"
    gpu_power_device="\${visible_devices%%,*}"
    if [ -z "\$gpu_power_device" ]; then
        gpu_power_device=0
    fi
    gpu_state_log="gpu_state_${label}.log"
    original_power_limit=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "=== GPU state before ${label} ===" > "\$gpu_state_log"
        nvidia-smi -i "\$gpu_power_device" --query-gpu=index,name,power.limit,power.draw,clocks.sm,clocks.mem --format=csv >> "\$gpu_state_log" 2>&1 || true
        if [ "${params.gpu_power_limit_watts}" != "0" ]; then
            original_power_limit="\$(nvidia-smi -i "\$gpu_power_device" --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
            restore_power_limit() {
                if [ -n "\$original_power_limit" ]; then
                    nvidia-smi -i "\$gpu_power_device" -pl "\$original_power_limit" >> "\$gpu_state_log" 2>&1 || true
                fi
            }
            trap restore_power_limit EXIT
            if nvidia-smi -i "\$gpu_power_device" -pl "${params.gpu_power_limit_watts}" >> "\$gpu_state_log" 2>&1; then
                echo "Applied GPU power limit ${params.gpu_power_limit_watts}W on device \$gpu_power_device" >> "\$gpu_state_log"
            else
                echo "WARNING: unable to apply GPU power limit ${params.gpu_power_limit_watts}W on device \$gpu_power_device; continuing without failing." >> "\$gpu_state_log"
            fi
            nvidia-smi -i "\$gpu_power_device" --query-gpu=index,name,power.limit,power.draw,clocks.sm,clocks.mem --format=csv >> "\$gpu_state_log" 2>&1 || true
        fi
    else
        echo "nvidia-smi not found; skipping GPU power-limit setup." > "\$gpu_state_log"
    fi

    /usr/bin/time \\
      -f "elapsed_seconds=%e\\nmax_rss_kb=%M" \\
      -o estimate_${label}.time \\
      ${params.conda_exe} run -p ${params.conda_env} \\
      env PYTHONPATH=${params.repo_root}/src \\
        python -c 'import sys; from tiresias.cli import estimate_psf_main; estimate_psf_main(sys.argv[1:])' \\
        --image-path ${params.image_path} \\
        --wavelength 0.515 \\
        --na 0.7467 \\
        --ni 1.56 \\
        --ns 1.56 \\
        --dxy 0.167 \\
        --dz 0.2 \\
        --n-iters ${n_iters} \\
        --chunk-xy ${params.chunk_xy} \\
        --blind-z-slices ${z_slices} \\
        --pad-z ${params.pad_z} \\
        --blind-max-tiles ${max_tiles} \\
        --tile-selection-strategy ${tile_strategy} \\
        --coarse-region-rows ${coarse_rows} \\
        --coarse-region-columns ${coarse_columns} \\
        --coarse-region-limit ${coarse_limit} \\
        --no-psf-cache \\
        --cupy-fft-engine ${engine} \\
        --adaptive-scout-iters ${adaptive_scout_iters} \\
        --adaptive-keep-tiles ${adaptive_keep_tiles} \\
        --output-path estimated_psf_${label}.tif \\
      > estimate_${label}.log 2>&1

    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "=== GPU state after ${label} ===" >> "\$gpu_state_log"
        nvidia-smi -i "\$gpu_power_device" --query-gpu=index,name,power.limit,power.draw,clocks.sm,clocks.mem --format=csv >> "\$gpu_state_log" 2>&1 || true
    fi
    """

    stub:
    """
    printf 'stub psf for ${label}\\n' > estimated_psf_${label}.tif
    printf 'elapsed_seconds=0.00\\nmax_rss_kb=0\\n' > estimate_${label}.time
    printf 'stub log for ${label}\\n' > estimate_${label}.log
    printf 'stub gpu state for ${label}\\n' > gpu_state_${label}.log
    """
}

process ESTIMATE_MATLAB_PSF {
    tag "matlab"
    executor 'slurm'
    queue "${params.matlab_queue}"
    cpus params.matlab_cpus
    memory "${params.matlab_memory}"
    time '08:00:00'
    publishDir "${params.out_dir}", mode: 'copy'

    output:
    path "estimated_psf_matlab.tif", emit: psf
    path "estimate_matlab.time", emit: time
    path "estimate_matlab.log", emit: log

    script:
    """
    set -euo pipefail
    echo "benchmark_version=${params.benchmark_version}"
    echo "label=matlab"
    echo "matlab_queue=${params.matlab_queue}"
    echo "matlab_workers=${params.matlab_workers}"
    echo "matlab_threads=${params.matlab_threads}"

    matlab_bin="${params.matlab_bin ?: 'matlab'}"
    resolved_matlab_bin=""
    for candidate in "\${matlab_bin}" matlab /home1/apps/MATLAB/R2024a/bin/matlab; do
        if [ -n "\$candidate" ] && command -v "\$candidate" >/dev/null 2>&1; then
            resolved_matlab_bin="\$(command -v "\$candidate")"
            break
        elif [ -n "\$candidate" ] && [ -x "\$candidate" ]; then
            resolved_matlab_bin="\$candidate"
            break
        fi
    done
    if [ -z "\$resolved_matlab_bin" ]; then
        echo "ERROR: MATLAB executable not found. Checked requested matlab_bin='\$matlab_bin', PATH, and /home1/apps/MATLAB/R2024a/bin/matlab." >&2
        exit 127
    fi

    /usr/bin/time \\
      -f "elapsed_seconds=%e\\nmax_rss_kb=%M" \\
      -o estimate_matlab.time \\
      ${params.conda_exe} run -p ${params.conda_env} \\
        python ${params.matlab_psf_script} \\
        --image_path ${params.image_path} \\
        --output_path estimated_psf_matlab.tif \\
        --blind_backend matlab \\
        --matlab_bin "\$resolved_matlab_bin" \\
        --matlab_workers ${params.matlab_workers} \\
        --matlab_threads ${params.matlab_threads} \\
        --matlab_timeout ${params.matlab_timeout} \\
        --script_dir ${params.deconvolution_root}/workflow/scripts \\
        --wavelength 0.515 \\
        --na 0.7467 \\
        --ni 1.56 \\
        --ns 1.56 \\
        --dxy 0.167 \\
        --dz 0.2 \\
        --n_iters 10 \\
        --chunk_xy ${params.chunk_xy} \\
        --blind_z_slices ${params.blind_z_slices} \\
        --pad_z ${params.pad_z} \\
        --blind_max_tiles ${params.blind_max_tiles} \\
        --no_psf_cache \\
      > estimate_matlab.log 2>&1
    """

    stub:
    """
    printf 'stub psf for matlab\\n' > estimated_psf_matlab.tif
    printf 'elapsed_seconds=0.00\\nmax_rss_kb=0\\n' > estimate_matlab.time
    printf 'stub log for matlab\\n' > estimate_matlab.log
    """
}

process SUMMARIZE_CUPY_PSFS {
    tag "summarize_cupy_psfs"
    executor 'slurm'
    queue "${params.compare_queue}"
    cpus 2
    memory '16 GB'
    time '01:00:00'
    publishDir "${params.out_dir}", mode: 'copy'

    input:
    path psfs
    path time_files
    path log_files

    output:
    path "psf_comparison_cupy_vs_cupy-fast.csv"
    path "psf_comparison_cupy_vs_cupy-fast.json"
    path "psf_comparison_cupy_vs_cupy-fast.log"
    path "benchmark_summary.txt"

    script:
    """
    set -euo pipefail

    ${params.conda_exe} run -p ${params.conda_env} \\
      python ${params.compare_script} \\
        estimated_psf_cupy.tif \\
        estimated_psf_cupy-fast.tif \\
        --spacing 0.2 0.167 0.167 \\
        --csv psf_comparison_cupy_vs_cupy-fast.csv \\
        --json psf_comparison_cupy_vs_cupy-fast.json \\
      > psf_comparison_cupy_vs_cupy-fast.log 2>&1

    {
      echo "output_dir=${params.out_dir}"
      echo "run_matlab=${params.run_matlab}"
      echo "gpu_power_limit_watts=${params.gpu_power_limit_watts}"
      echo "reference_engine=cupy"
      echo "cupy_engine=${params.cupy_engine}"
      echo "cupy_fast_engine=${params.cupy_fast_engine}"
      echo "cupy_n_iters=${params.cupy_n_iters}"
      echo "cupy_fast_n_iters=${params.cupy_fast_n_iters}"
      echo "cupy_blind_max_tiles=${params.cupy_blind_max_tiles}"
      echo "cupy_fast_blind_max_tiles=${params.cupy_fast_blind_max_tiles}"
      echo "cupy_blind_z_slices=${params.cupy_blind_z_slices}"
      echo "cupy_fast_blind_z_slices=${params.cupy_fast_blind_z_slices}"
      echo "cupy_tile_selection_strategy=${params.cupy_tile_selection_strategy}"
      echo "cupy_fast_tile_selection_strategy=${params.cupy_fast_tile_selection_strategy}"
      echo "cupy_coarse_region_rows=${params.cupy_coarse_region_rows}"
      echo "cupy_coarse_region_columns=${params.cupy_coarse_region_columns}"
      echo "cupy_coarse_region_limit=${params.cupy_coarse_region_limit}"
      echo "cupy_fast_coarse_region_rows=${params.cupy_fast_coarse_region_rows}"
      echo "cupy_fast_coarse_region_columns=${params.cupy_fast_coarse_region_columns}"
      echo "cupy_fast_coarse_region_limit=${params.cupy_fast_coarse_region_limit}"
      echo "cupy_adaptive_scout_iters=${params.cupy_adaptive_scout_iters}"
      echo "cupy_adaptive_keep_tiles=${params.cupy_adaptive_keep_tiles}"
      echo "cupy_fast_adaptive_scout_iters=${params.cupy_fast_adaptive_scout_iters}"
      echo "cupy_fast_adaptive_keep_tiles=${params.cupy_fast_adaptive_keep_tiles}"
      echo "cupy_time_file=estimate_cupy.time"
      cat estimate_cupy.time
      echo "cupy-fast_time_file=estimate_cupy-fast.time"
      cat estimate_cupy-fast.time
      echo "comparison_cupy_vs_cupy-fast_csv=psf_comparison_cupy_vs_cupy-fast.csv"
      echo "comparison_cupy_vs_cupy-fast_json=psf_comparison_cupy_vs_cupy-fast.json"
    } > benchmark_summary.txt
    """

    stub:
    """
    printf 'reference,candidate,ncc,ssim\\n' > psf_comparison_cupy_vs_cupy-fast.csv
    printf '[]\\n' > psf_comparison_cupy_vs_cupy-fast.json
    printf 'stub comparison\\n' > psf_comparison_cupy_vs_cupy-fast.log
    printf 'output_dir=${params.out_dir}\\nrun_matlab=${params.run_matlab}\\ngpu_power_limit_watts=${params.gpu_power_limit_watts}\\nreference_engine=cupy\\ncupy_engine=${params.cupy_engine}\\ncupy_fast_engine=${params.cupy_fast_engine}\\ncupy_n_iters=${params.cupy_n_iters}\\ncupy_fast_n_iters=${params.cupy_fast_n_iters}\\ncupy_blind_max_tiles=${params.cupy_blind_max_tiles}\\ncupy_fast_blind_max_tiles=${params.cupy_fast_blind_max_tiles}\\ncupy_blind_z_slices=${params.cupy_blind_z_slices}\\ncupy_fast_blind_z_slices=${params.cupy_fast_blind_z_slices}\\ncupy_tile_selection_strategy=${params.cupy_tile_selection_strategy}\\ncupy_fast_tile_selection_strategy=${params.cupy_fast_tile_selection_strategy}\\ncupy_coarse_region_rows=${params.cupy_coarse_region_rows}\\ncupy_coarse_region_columns=${params.cupy_coarse_region_columns}\\ncupy_coarse_region_limit=${params.cupy_coarse_region_limit}\\ncupy_fast_coarse_region_rows=${params.cupy_fast_coarse_region_rows}\\ncupy_fast_coarse_region_columns=${params.cupy_fast_coarse_region_columns}\\ncupy_fast_coarse_region_limit=${params.cupy_fast_coarse_region_limit}\\ncupy_adaptive_scout_iters=${params.cupy_adaptive_scout_iters}\\ncupy_adaptive_keep_tiles=${params.cupy_adaptive_keep_tiles}\\ncupy_fast_adaptive_scout_iters=${params.cupy_fast_adaptive_scout_iters}\\ncupy_fast_adaptive_keep_tiles=${params.cupy_fast_adaptive_keep_tiles}\\n' > benchmark_summary.txt
    """
}

process COMPARE_MATLAB_PSFS {
    tag "compare_matlab_psfs"
    executor 'slurm'
    queue "${params.compare_queue}"
    cpus 2
    memory '16 GB'
    time '01:00:00'
    publishDir "${params.out_dir}", mode: 'copy'

    input:
    path psfs
    path time_files
    path log_files

    output:
    path "psf_comparison_matlab_vs_cupy.csv"
    path "psf_comparison_matlab_vs_cupy.json"
    path "psf_comparison_matlab_vs_cupy.log"
    path "psf_comparison_matlab_vs_cupy-fast.csv"
    path "psf_comparison_matlab_vs_cupy-fast.json"
    path "psf_comparison_matlab_vs_cupy-fast.log"
    path "benchmark_summary.txt"

    script:
    """
    set -euo pipefail

    ${params.conda_exe} run -p ${params.conda_env} \\
      python ${params.compare_script} \\
        estimated_psf_matlab.tif \\
        estimated_psf_cupy.tif \\
        --spacing 0.2 0.167 0.167 \\
        --csv psf_comparison_matlab_vs_cupy.csv \\
        --json psf_comparison_matlab_vs_cupy.json \\
      > psf_comparison_matlab_vs_cupy.log 2>&1

    ${params.conda_exe} run -p ${params.conda_env} \\
      python ${params.compare_script} \\
        estimated_psf_matlab.tif \\
        estimated_psf_cupy-fast.tif \\
        --spacing 0.2 0.167 0.167 \\
        --csv psf_comparison_matlab_vs_cupy-fast.csv \\
        --json psf_comparison_matlab_vs_cupy-fast.json \\
      > psf_comparison_matlab_vs_cupy-fast.log 2>&1

    {
      echo "output_dir=${params.out_dir}"
      echo "run_matlab=${params.run_matlab}"
      echo "gpu_power_limit_watts=${params.gpu_power_limit_watts}"
      echo "reference_engine=matlab"
      echo "cupy_engine=${params.cupy_engine}"
      echo "cupy_fast_engine=${params.cupy_fast_engine}"
      echo "cupy_n_iters=${params.cupy_n_iters}"
      echo "cupy_fast_n_iters=${params.cupy_fast_n_iters}"
      echo "cupy_blind_max_tiles=${params.cupy_blind_max_tiles}"
      echo "cupy_fast_blind_max_tiles=${params.cupy_fast_blind_max_tiles}"
      echo "cupy_blind_z_slices=${params.cupy_blind_z_slices}"
      echo "cupy_fast_blind_z_slices=${params.cupy_fast_blind_z_slices}"
      echo "cupy_tile_selection_strategy=${params.cupy_tile_selection_strategy}"
      echo "cupy_fast_tile_selection_strategy=${params.cupy_fast_tile_selection_strategy}"
      echo "cupy_coarse_region_rows=${params.cupy_coarse_region_rows}"
      echo "cupy_coarse_region_columns=${params.cupy_coarse_region_columns}"
      echo "cupy_coarse_region_limit=${params.cupy_coarse_region_limit}"
      echo "cupy_fast_coarse_region_rows=${params.cupy_fast_coarse_region_rows}"
      echo "cupy_fast_coarse_region_columns=${params.cupy_fast_coarse_region_columns}"
      echo "cupy_fast_coarse_region_limit=${params.cupy_fast_coarse_region_limit}"
      echo "cupy_adaptive_scout_iters=${params.cupy_adaptive_scout_iters}"
      echo "cupy_adaptive_keep_tiles=${params.cupy_adaptive_keep_tiles}"
      echo "cupy_fast_adaptive_scout_iters=${params.cupy_fast_adaptive_scout_iters}"
      echo "cupy_fast_adaptive_keep_tiles=${params.cupy_fast_adaptive_keep_tiles}"
      echo "matlab_time_file=estimate_matlab.time"
      cat estimate_matlab.time
      echo "cupy_time_file=estimate_cupy.time"
      cat estimate_cupy.time
      echo "cupy-fast_time_file=estimate_cupy-fast.time"
      cat estimate_cupy-fast.time
      echo "comparison_matlab_vs_cupy_csv=psf_comparison_matlab_vs_cupy.csv"
      echo "comparison_matlab_vs_cupy_json=psf_comparison_matlab_vs_cupy.json"
      echo "comparison_matlab_vs_cupy-fast_csv=psf_comparison_matlab_vs_cupy-fast.csv"
      echo "comparison_matlab_vs_cupy-fast_json=psf_comparison_matlab_vs_cupy-fast.json"
    } > benchmark_summary.txt
    """

    stub:
    """
    printf 'reference,candidate,ncc,ssim\\n' > psf_comparison_matlab_vs_cupy.csv
    printf '[]\\n' > psf_comparison_matlab_vs_cupy.json
    printf 'stub comparison\\n' > psf_comparison_matlab_vs_cupy.log
    printf 'reference,candidate,ncc,ssim\\n' > psf_comparison_matlab_vs_cupy-fast.csv
    printf '[]\\n' > psf_comparison_matlab_vs_cupy-fast.json
    printf 'stub comparison\\n' > psf_comparison_matlab_vs_cupy-fast.log
    printf 'output_dir=${params.out_dir}\\nrun_matlab=${params.run_matlab}\\ngpu_power_limit_watts=${params.gpu_power_limit_watts}\\nreference_engine=matlab\\ncupy_engine=${params.cupy_engine}\\ncupy_fast_engine=${params.cupy_fast_engine}\\ncupy_n_iters=${params.cupy_n_iters}\\ncupy_fast_n_iters=${params.cupy_fast_n_iters}\\ncupy_blind_max_tiles=${params.cupy_blind_max_tiles}\\ncupy_fast_blind_max_tiles=${params.cupy_fast_blind_max_tiles}\\ncupy_blind_z_slices=${params.cupy_blind_z_slices}\\ncupy_fast_blind_z_slices=${params.cupy_fast_blind_z_slices}\\ncupy_tile_selection_strategy=${params.cupy_tile_selection_strategy}\\ncupy_fast_tile_selection_strategy=${params.cupy_fast_tile_selection_strategy}\\ncupy_coarse_region_rows=${params.cupy_coarse_region_rows}\\ncupy_coarse_region_columns=${params.cupy_coarse_region_columns}\\ncupy_coarse_region_limit=${params.cupy_coarse_region_limit}\\ncupy_fast_coarse_region_rows=${params.cupy_fast_coarse_region_rows}\\ncupy_fast_coarse_region_columns=${params.cupy_fast_coarse_region_columns}\\ncupy_fast_coarse_region_limit=${params.cupy_fast_coarse_region_limit}\\ncupy_adaptive_scout_iters=${params.cupy_adaptive_scout_iters}\\ncupy_adaptive_keep_tiles=${params.cupy_adaptive_keep_tiles}\\ncupy_fast_adaptive_scout_iters=${params.cupy_fast_adaptive_scout_iters}\\ncupy_fast_adaptive_keep_tiles=${params.cupy_fast_adaptive_keep_tiles}\\n' > benchmark_summary.txt
    """
}

workflow {
    cupy_runs = Channel.of(
        tuple('cupy', params.cupy_engine, params.cupy_n_iters, params.cupy_blind_max_tiles, params.cupy_blind_z_slices, params.cupy_tile_selection_strategy, params.cupy_coarse_region_rows, params.cupy_coarse_region_columns, params.cupy_coarse_region_limit, params.cupy_adaptive_scout_iters, params.cupy_adaptive_keep_tiles),
        tuple('cupy-fast', params.cupy_fast_engine, params.cupy_fast_n_iters, params.cupy_fast_blind_max_tiles, params.cupy_fast_blind_z_slices, params.cupy_fast_tile_selection_strategy, params.cupy_fast_coarse_region_rows, params.cupy_fast_coarse_region_columns, params.cupy_fast_coarse_region_limit, params.cupy_fast_adaptive_scout_iters, params.cupy_fast_adaptive_keep_tiles),
    )
    ESTIMATE_CUPY_PSF(cupy_runs)
    if (params.run_matlab) {
        ESTIMATE_MATLAB_PSF()
        COMPARE_MATLAB_PSFS(
            ESTIMATE_MATLAB_PSF.out.psf.mix(ESTIMATE_CUPY_PSF.out.psf).collect(),
            ESTIMATE_MATLAB_PSF.out.time.mix(ESTIMATE_CUPY_PSF.out.time).collect(),
            ESTIMATE_MATLAB_PSF.out.log.mix(ESTIMATE_CUPY_PSF.out.log).collect(),
        )
    } else {
        SUMMARIZE_CUPY_PSFS(
            ESTIMATE_CUPY_PSF.out.psf.collect(),
            ESTIMATE_CUPY_PSF.out.time.collect(),
            ESTIMATE_CUPY_PSF.out.log.collect(),
        )
    }
}
