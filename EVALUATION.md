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
nn ok: max_abs=0 max_rel=0
tn ok: max_abs=0 max_rel=0
nt ok: max_abs=0 max_rel=0
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
m,n,k,moduli,exact_required_bits,planned_bits,max_modulus,fp64_seconds,oz_seconds,naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs
8,8,8,11,110,264,67108862,4.943e-06,0.000307421,0.000179994,0,3.71258675529e-16
16,16,16,11,111,264,47453131,2.185e-06,0.000973419,0.001452394,0,7.59906913097e-16
32,32,32,11,112,264,33554430,1.5205e-05,0.003884505,0.011718998,0,2.43739505418e-15
64,64,64,11,113,264,23726565,0.000124263,0.015401716,-1,-1,4.82793210353e-15
96,96,96,11,114,264,19372659,0.00042169,0.035941358,-1,-1,1.04819437783e-14
```

Additional 64 and 128 cases with naive Boost multiprecision enabled:

```text
m,n,k,moduli,exact_required_bits,planned_bits,max_modulus,fp64_seconds,oz_seconds,naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs
64,64,64,11,113,264,23726565,0.000120837,0.016083703,0.089275631,0,4.59062343515e-15
128,128,128,12,114,264,16777214,0.001241289,0.073740812,0.726132425,0,1.12859793792e-14
```

## Initial Read

- The modular/CRT PoC exactly matches the naive Boost `cpp_bin_float<256>`
  reference for tested random `double` inputs.
- It is slower than naive Boost at `8x8x8`, roughly comparable by `16x16x16`,
  and faster from `32x32x32` upward.
- On this machine, `64x64x64` is about `5.5x` faster than naive Boost, and
  `128x128x128` is about `9.8x` faster.
- It is still far slower than ordinary FP64 BLAS, as expected. The point of the
  PoC is high-precision accumulation, not competing with FP64 accuracy/perf.

## Bottlenecks

Current implementation is intentionally simple:

- residue matrices are rebuilt for every modulus,
- CRT reconstruction is scalar and per element,
- there is no plan reuse,
- there is no blocking over output tiles,
- the BLAS backend here is system BLAS, not OpenBLAS/MKL/BLIS.

The next performance work should focus on plan reuse, precomputed powers of
two modulo each prime, blocked residue generation, and parallel CRT recovery.
