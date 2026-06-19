# Evaluation Snapshot

Date: 2026-06-19

Environment:

- CPU: Intel Core i7-9700, 8 cores, AVX2/FMA available
- BLAS linked by default: `/lib/x86_64-linux-gnu/libblas.so.3`
- Compiler: `g++ 13.3.0`
- Build flags: `-O2 -std=c++17`

This is an early PoC measurement. The system did not expose OpenBLAS through
`ldconfig`, so the BLAS numbers below are from the default system BLAS.

## Correctness

Command:

```sh
make -C oz-highprecision-cpu test
```

Result:

```text
cancellation ok: high_precision=1 fp64_left_to_right=0 moduli=10
plan reject ok
wide exponent ok: max_abs=0 max_rel=0
parallel crt ok: max_abs=0 max_rel=0
blocked residue ok: max_abs=0 max_rel=0
nn ok: max_abs=0 max_rel=0
nn reusable plan ok: max_abs=0 max_rel=0
tn ok: max_abs=0 max_rel=0
tn reusable plan ok: max_abs=0 max_rel=0
nt ok: max_abs=0 max_rel=0
nt reusable plan ok: max_abs=0 max_rel=0
```

The cancellation case verifies the intended behavior for:

```text
[1e16, 1, -1e16] * [1, 1, 1]^T
```

Left-to-right FP64 summation returns `0`, while the modular/CRT path returns
the exact `double`-input result `1`.

## Timing

Command:

```sh
oz-highprecision-cpu/build/benchmark
```

CSV output:

```text
m,n,k,moduli,exact_required_bits,planned_bits,max_exact_modulus_bound,selected_max_modulus,fp64_seconds,oz_seconds,plan_seconds,crt_threads,residue_col_block,oz_reuse_seconds,oz_reuse_serial_seconds,naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs,reuse_vs_oz_max_abs
8,8,8,11,110,264,67108862,67108859,3.912e-06,0.000152725,3.471e-05,1,8,9.466e-05,8.7695e-05,0.000184829,0,3.71258675529e-16,0
16,16,16,11,111,264,47453131,47453111,2.188e-06,0.00048593,4.2554e-05,1,16,0.000381716,0.000375367,0.001480381,0,7.59906913097e-16,0
32,32,32,11,112,264,33554430,33554393,1.5228e-05,0.001732901,6.4305e-05,1,32,0.001683581,0.001635595,0.011774708,0,2.43739505418e-15,0
64,64,64,11,113,264,23726565,23726561,0.000119984,0.004425599,0.000147692,8,64,0.004151518,0.007516863,-1,-1,4.82793210353e-15,0
96,96,96,11,114,264,19372659,19372651,0.000494213,0.011346109,0.000293532,8,96,0.01106648,0.01903113,-1,-1,1.04819437783e-14,0
```

Additional cases with naive Boost multiprecision enabled:

```text
m,n,k,moduli,exact_required_bits,planned_bits,max_exact_modulus_bound,selected_max_modulus,fp64_seconds,oz_seconds,plan_seconds,crt_threads,residue_col_block,oz_reuse_seconds,oz_reuse_serial_seconds,naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs,reuse_vs_oz_max_abs
64,64,64,11,113,264,23726565,23726561,0.000130348,0.004438221,0.000146509,8,64,0.004635271,0.007600337,0.097663944,0,4.59062343515e-15,0
128,128,128,12,114,264,16777214,16777213,0.001029779,0.026310392,0.000560937,8,128,0.026003672,0.038597167,0.741642773,0,1.12859793792e-14,0
64,192,64,11,113,264,23726565,23726561,0.000380298,0.011928095,0.000319242,8,128,0.011388198,0.021605001,0.279688942,0,5.2808122754e-15,0
```

`max_exact_modulus_bound` is the largest integer allowed by the exact FP64
accumulation bound. It is not necessarily prime. `selected_max_modulus` is the
largest prime modulus actually used.

## Initial Read

- The modular/CRT PoC exactly matches the naive Boost `cpp_bin_float<256>`
  reference for tested random `double` inputs.
- One-shot execution is still slower than naive Boost at `8x8x8`, but
  reusable-plan execution is already faster in this run and the gap increases
  quickly with size.
- With CRT/Garner inverse precomputation, reusable plans, the int64 residue
  fast path, per-call input decomposition reuse, and precomputed powers of two
  modulo each prime, reusable-plan execution is about `21.1x` faster than
  naive Boost at `64x64x64` and about `28.5x` faster at `128x128x128`.
- Parallel CRT recovery uses 8 threads in this environment for outputs with at
  least 4096 entries. In the default run, reusable execution is about `1.8x`
  faster than serial CRT at `64x64x64` and about `1.7x` faster at `96x96x96`.
- Residue GEMM and CRT recovery now operate on output-column blocks. The
  default automatic block size is the full `n` for these default cases and 128
  columns for the `64x192x64` case, reducing temporary residue storage without
  changing numerical results.
- Reusable plans separate input scale/CRT setup from execution. For these
  random matrices, planning is small compared with modular GEMM and CRT at
  medium sizes. Reuse output exactly matches one-shot output in these tests.
- It is still far slower than ordinary FP64 BLAS, as expected. The point of the
  PoC is high-precision accumulation, not competing with FP64 accuracy/perf.

## Bottlenecks

Current implementation is intentionally simple:

- residue matrices are rebuilt for every modulus, though inputs are decomposed
  once per call and plans cache `2^shift mod p` tables for wide-exponent cases,
  and temporary output residues are now limited to a column block,
- CRT reconstruction is still scalar within each output element, though output
  elements are now split across worker threads and Garner inverses/full modulus
  products are precomputed once per plan,
- reusable plans currently store scale bounds from a reference input and reject
  later inputs outside those bounds,
- A-side residues are regenerated for each column block instead of being cached
  as panels across blocks,
- the BLAS backend here is system BLAS, not OpenBLAS/MKL/BLIS.

The next performance work should focus on caching A-side residue panels across
column blocks, richer compatibility policies for reused plans, and switching
from the system BLAS to a tuned OpenBLAS/MKL/BLIS build.
