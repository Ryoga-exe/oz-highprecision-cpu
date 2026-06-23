#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

if [ "$#" -gt 0 ]; then
    backends=("$@")
else
    backends=(auto system openblas blis mkl)
fi

for backend in "${backends[@]}"; do
    echo "=== BLAS_BACKEND=${backend} ==="
    if ! make clean build/benchmark BLAS_BACKEND="${backend}"; then
        echo "skip: build failed for BLAS_BACKEND=${backend}"
        echo
        continue
    fi

    make blas-info BLAS_BACKEND="${backend}"
    build/benchmark --quick
    build/benchmark --sweep-blocks 64 512 64
    echo
done
