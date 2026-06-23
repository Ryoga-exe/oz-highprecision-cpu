#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

run_id="${OZ_EVAL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
results_root="${OZ_EVAL_RESULTS_DIR:-results/${run_id}}"

if [ "$#" -gt 0 ]; then
    backends=("$@")
else
    backends=(auto system openblas blis mkl)
fi

mkdir -p "${results_root}"
echo "results_dir=${results_root}"

run_csv() {
    local output_path="$1"
    shift

    "$@" | tee "${output_path}"
    return "${PIPESTATUS[0]}"
}

for backend in "${backends[@]}"; do
    echo "=== BLAS_BACKEND=${backend} ==="
    backend_dir="${results_root}/${backend}"
    mkdir -p "${backend_dir}"

    if ! make clean build/benchmark BLAS_BACKEND="${backend}" 2>&1 | tee "${backend_dir}/build.log"; then
        echo "skip: build failed for BLAS_BACKEND=${backend}"
        echo "build failed" > "${backend_dir}/SKIPPED"
        echo
        continue
    fi

    make blas-info BLAS_BACKEND="${backend}" | tee "${backend_dir}/blas-info.txt"
    if ! run_csv "${backend_dir}/quick.csv" build/benchmark --quick; then
        echo "skip: quick benchmark failed for BLAS_BACKEND=${backend}"
        echo "quick benchmark failed" > "${backend_dir}/SKIPPED"
        echo
        continue
    fi
    if ! run_csv "${backend_dir}/block_sweep_64x512x64.csv" build/benchmark --sweep-blocks 64 512 64; then
        echo "skip: block sweep failed for BLAS_BACKEND=${backend}"
        echo "block sweep failed" > "${backend_dir}/SKIPPED"
        echo
        continue
    fi
    if ! run_csv "${backend_dir}/crt_threads_64x512x64.csv" build/benchmark --sweep-crt-threads 64 512 64; then
        echo "skip: CRT thread sweep failed for BLAS_BACKEND=${backend}"
        echo "CRT thread sweep failed" > "${backend_dir}/SKIPPED"
        echo
        continue
    fi
    if ! run_csv "${backend_dir}/precision_64x64x64.csv" build/benchmark --sweep-precision 64 64 64; then
        echo "skip: precision sweep failed for BLAS_BACKEND=${backend}"
        echo "precision sweep failed" > "${backend_dir}/SKIPPED"
        echo
        continue
    fi
    echo
done
