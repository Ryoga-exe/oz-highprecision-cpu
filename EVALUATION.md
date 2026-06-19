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
m,n,k,moduli,exact_required_bits,planned_bits,max_exact_modulus_bound,selected_max_modulus,fp64_seconds,oz_seconds,naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs
8,8,8,11,110,264,67108862,67108859,4.946e-06,0.000155548,0.000177047,0,3.71258675529e-16
16,16,16,11,111,264,47453131,47453111,2.218e-06,0.000510305,0.001508071,0,7.59906913097e-16
32,32,32,11,112,264,33554430,33554393,1.6306e-05,0.002008568,0.011940601,0,2.43739505418e-15
64,64,64,11,113,264,23726565,23726561,0.000128265,0.008608349,-1,-1,4.82793210353e-15
96,96,96,11,114,264,19372659,19372651,0.000454313,0.019867353,-1,-1,1.04819437783e-14
```

Additional 64 and 128 cases with naive Boost multiprecision enabled:

```text
m,n,k,moduli,exact_required_bits,planned_bits,max_exact_modulus_bound,selected_max_modulus,fp64_seconds,oz_seconds,naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs
64,64,64,11,113,264,23726565,23726561,0.000375243,0.010063672,0.091333782,0,4.59062343515e-15
128,128,128,12,114,264,16777214,16777213,0.001089329,0.042543685,0.71644155,0,1.12859793792e-14
```

`max_exact_modulus_bound` is the largest integer allowed by the exact FP64
accumulation bound. It is not necessarily prime. `selected_max_modulus` is the
largest prime modulus actually used.

## Initial Read

- The modular/CRT PoC exactly matches the naive Boost `cpp_bin_float<256>`
  reference for tested random `double` inputs.
- It is slower than naive Boost at `8x8x8`, roughly comparable by `16x16x16`,
  and faster from `32x32x32` upward.
- With CRT/Garner inverse precomputation and the int64 residue fast path,
  `64x64x64` is about `9.1x` faster than naive Boost, and `128x128x128` is
  about `16.8x` faster.
- It is still far slower than ordinary FP64 BLAS, as expected. The point of the
  PoC is high-precision accumulation, not competing with FP64 accuracy/perf.

## Bottlenecks

Current implementation is intentionally simple:

- residue matrices are rebuilt for every modulus, though common scaled inputs
  that fit in `int64_t` now avoid modular exponentiation,
- CRT reconstruction is scalar and per element, though Garner inverses and the
  full modulus product are now precomputed once per plan,
- there is no plan reuse,
- there is no blocking over output tiles,
- the BLAS backend here is system BLAS, not OpenBLAS/MKL/BLIS.

The next performance work should focus on plan reuse, precomputed powers of
two modulo each prime, blocked residue generation, and parallel CRT recovery.
