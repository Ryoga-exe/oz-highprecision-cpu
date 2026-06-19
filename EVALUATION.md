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
m,n,k,moduli,exact_required_bits,planned_bits,max_exact_modulus_bound,selected_max_modulus,fp64_seconds,oz_seconds,plan_seconds,crt_threads,oz_reuse_seconds,oz_reuse_serial_seconds,naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs,reuse_vs_oz_max_abs
8,8,8,11,110,264,67108862,67108859,3.715e-06,0.000146803,3.5219e-05,1,8.8651e-05,0.000122416,0.000260119,0,3.71258675529e-16,0
16,16,16,11,111,264,47453131,47453111,2.196e-06,0.000466073,4.2188e-05,1,0.000373377,0.000367291,0.001450812,0,7.59906913097e-16,0
32,32,32,11,112,264,33554430,33554393,1.5247e-05,0.001689433,6.4101e-05,1,0.001620712,0.00163788,0.012090181,0,2.43739505418e-15,0
64,64,64,11,113,264,23726565,23726561,0.000159838,0.004472427,0.00014751,8,0.004068542,0.007424622,-1,-1,4.82793210353e-15,0
96,96,96,11,114,264,19372659,19372651,0.00043357,0.011240363,0.000286838,8,0.01079301,0.01863612,-1,-1,1.04819437783e-14,0
```

Additional 64 and 128 cases with naive Boost multiprecision enabled:

```text
m,n,k,moduli,exact_required_bits,planned_bits,max_exact_modulus_bound,selected_max_modulus,fp64_seconds,oz_seconds,plan_seconds,crt_threads,oz_reuse_seconds,oz_reuse_serial_seconds,naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs,reuse_vs_oz_max_abs
64,64,64,11,113,264,23726565,23726561,0.000124706,0.004917421,0.000147444,8,0.004519972,0.007477203,0.102942572,0,4.59062343515e-15,0
128,128,128,12,114,264,16777214,16777213,0.001110352,0.026548837,0.000496881,8,0.025403401,0.037863467,0.745553726,0,1.12859793792e-14,0
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
  modulo each prime, reusable-plan execution is about `22.8x` faster than
  naive Boost at `64x64x64` and about `29.3x` faster at `128x128x128`.
- Parallel CRT recovery uses 8 threads in this environment for outputs with at
  least 4096 entries. In the default run, reusable execution is about `1.8x`
  faster than serial CRT at `64x64x64` and about `1.7x` faster at `96x96x96`.
- Reusable plans separate input scale/CRT setup from execution. For these
  random matrices, planning is small compared with modular GEMM and CRT at
  medium sizes. Reuse output exactly matches one-shot output in these tests.
- It is still far slower than ordinary FP64 BLAS, as expected. The point of the
  PoC is high-precision accumulation, not competing with FP64 accuracy/perf.

## Bottlenecks

Current implementation is intentionally simple:

- residue matrices are rebuilt for every modulus, though inputs are decomposed
  once per call and plans cache `2^shift mod p` tables for wide-exponent cases,
- CRT reconstruction is still scalar within each output element, though output
  elements are now split across worker threads and Garner inverses/full modulus
  products are precomputed once per plan,
- reusable plans currently store scale bounds from a reference input and reject
  later inputs outside those bounds,
- there is no blocking over output tiles,
- the BLAS backend here is system BLAS, not OpenBLAS/MKL/BLIS.

The next performance work should focus on blocked residue generation, richer
compatibility policies for reused plans, and switching from the system BLAS to
a tuned OpenBLAS/MKL/BLIS build.
