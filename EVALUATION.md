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
m,n,k,moduli,exact_required_bits,planned_bits,max_exact_modulus_bound,selected_max_modulus,fp64_seconds,oz_seconds,plan_seconds,oz_reuse_seconds,naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs,reuse_vs_oz_max_abs
8,8,8,11,110,264,67108862,67108859,3.494e-06,0.000190125,3.49e-05,9.1269e-05,0.000174993,0,3.71258675529e-16,0
16,16,16,11,111,264,47453131,47453111,2.221e-06,0.000484531,4.2071e-05,0.000389942,0.001383371,0,7.59906913097e-16,0
32,32,32,11,112,264,33554430,33554393,1.5218e-05,0.001695343,6.3847e-05,0.001609434,0.011404631,0,2.43739505418e-15,0
64,64,64,11,113,264,23726565,23726561,0.000119996,0.007381413,0.000146432,0.007082254,-1,-1,4.82793210353e-15,0
96,96,96,11,114,264,19372659,19372651,0.000431526,0.018008616,0.000284762,0.017699432,-1,-1,1.04819437783e-14,0
```

Additional 64 and 128 cases with naive Boost multiprecision enabled:

```text
m,n,k,moduli,exact_required_bits,planned_bits,max_exact_modulus_bound,selected_max_modulus,fp64_seconds,oz_seconds,plan_seconds,oz_reuse_seconds,naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs,reuse_vs_oz_max_abs
64,64,64,11,113,264,23726565,23726561,0.000123062,0.007718111,0.000147322,0.007618626,0.0908848,0,4.59062343515e-15,0
128,128,128,12,114,264,16777214,16777213,0.00104144,0.039497311,0.000520466,0.038295078,0.711202457,0,1.12859793792e-14,0
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
  modulo each prime, reusable-plan execution is about `11.9x` faster than
  naive Boost at `64x64x64` and about `18.6x` faster at `128x128x128`.
- Reusable plans separate input scale/CRT setup from execution. For these
  random matrices, planning is small compared with modular GEMM and CRT at
  medium sizes. Reuse output exactly matches one-shot output in these tests.
- It is still far slower than ordinary FP64 BLAS, as expected. The point of the
  PoC is high-precision accumulation, not competing with FP64 accuracy/perf.

## Bottlenecks

Current implementation is intentionally simple:

- residue matrices are rebuilt for every modulus, though inputs are decomposed
  once per call and plans cache `2^shift mod p` tables for wide-exponent cases,
- CRT reconstruction is scalar and per element, though Garner inverses and the
  full modulus product are now precomputed once per plan,
- reusable plans currently store scale bounds from a reference input and reject
  later inputs outside those bounds,
- there is no blocking over output tiles,
- the BLAS backend here is system BLAS, not OpenBLAS/MKL/BLIS.

The next performance work should focus on blocked residue generation,
parallel CRT recovery, richer compatibility policies for reused plans, and
switching from the system BLAS to a tuned OpenBLAS/MKL/BLIS build.
