# Evaluation Snapshot

Date: 2026-06-22

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
m,n,k,moduli,exact_required_bits,planned_bits,max_exact_modulus_bound,selected_max_modulus,fp64_seconds,oz_seconds,plan_seconds,crt_threads,residue_col_block,a_residue_cached,oz_reuse_seconds,oz_reuse_serial_seconds,naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs,reuse_vs_oz_max_abs
8,8,8,11,110,264,67108862,67108859,4.021e-06,0.000185068,3.4317e-05,1,8,0,8.9107e-05,8.5657e-05,0.000180515,0,3.71258675529e-16,0
16,16,16,11,111,264,47453131,47453111,2.188e-06,0.000480613,4.1919e-05,1,16,0,0.000372457,0.00036837,0.001504982,0,7.59906913097e-16,0
32,32,32,11,112,264,33554430,33554393,1.4947e-05,0.001901094,6.267e-05,1,32,0,0.001630618,0.00164414,0.011572459,0,2.43739505418e-15,0
64,64,64,11,113,264,23726565,23726561,0.000120316,0.004354512,0.000153513,8,64,0,0.004184076,0.007531077,-1,-1,4.82793210353e-15,0
96,96,96,11,114,264,19372659,19372651,0.000495212,0.011315923,0.000285711,8,96,0,0.011004105,0.018683673,-1,-1,1.04819437783e-14,0
```

Additional cases with naive Boost multiprecision enabled:

```text
m,n,k,moduli,exact_required_bits,planned_bits,max_exact_modulus_bound,selected_max_modulus,fp64_seconds,oz_seconds,plan_seconds,crt_threads,residue_col_block,a_residue_cached,oz_reuse_seconds,oz_reuse_serial_seconds,naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs,reuse_vs_oz_max_abs
64,64,64,11,113,264,23726565,23726561,0.000126079,0.004583461,0.00016651,8,64,0,0.004253937,0.007416298,0.098873591,0,4.59062343515e-15,0
128,128,128,12,114,264,16777214,16777213,0.001073747,0.027228722,0.000574065,8,128,0,0.026499239,0.038459068,0.761871991,0,1.12859793792e-14,0
64,192,64,11,113,264,23726565,23726561,0.000370778,0.013536225,0.000313499,8,128,0,0.013018314,0.021682731,0.289360712,0,5.2808122754e-15,0
64,512,64,11,113,264,23726565,23726561,0.001061378,0.033108772,0.000614148,8,128,1,0.032600126,0.052653167,0.7555242,0,6.25928911874e-15,0
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
  modulo each prime, reusable-plan execution is about `23.2x` faster than
  naive Boost at `64x64x64` and about `28.8x` faster at `128x128x128`.
- Parallel CRT recovery uses 8 threads in this environment for outputs with at
  least 4096 entries. In the default run, reusable execution is about `1.8x`
  faster than serial CRT at `64x64x64` and about `1.7x` faster at `96x96x96`.
- Residue GEMM and CRT recovery now operate on output-column blocks. The
  default automatic block size is the full `n` for these default cases and 128
  columns for the `64x192x64` case, reducing temporary residue storage without
  changing numerical results.
- A-side residue panels are cached when the output spans more than two column
  blocks. That path is active in the `64x512x64` case above and avoids
  regenerating the same A residues for each block.
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
- A-side residue panels are cached adaptively for wider outputs, but B-side
  residue panels are still rebuilt per block and modulus,
- CRT reconstruction is still scalar within each output element, though output
  elements are now split across worker threads and Garner inverses/full modulus
  products are precomputed once per plan,
- reusable plans currently store scale bounds from a reference input and reject
  later inputs outside those bounds,
- the BLAS backend here is system BLAS, not OpenBLAS/MKL/BLIS.

The next performance work should focus on richer compatibility policies for
reused plans, tuned residue block sizing, and switching from the system BLAS to
a tuned OpenBLAS/MKL/BLIS build.
