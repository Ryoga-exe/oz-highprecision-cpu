# Evaluation Snapshot

Date: 2026-06-23

Environment:

- CPU: Intel Core i7-9700, 8 cores, AVX2/FMA available
- BLAS linked by default: `/lib/x86_64-linux-gnu/libblas.so.3`
- Compiler: `g++ 13.3.0`
- Build flags: `-O2 -std=c++17`

This is an early PoC measurement. The system did not expose OpenBLAS, BLIS, or
MKL through `ldconfig`/`pkg-config`, so the BLAS numbers below are from the
default system BLAS.

## BLAS Backend Selection

The Makefile supports:

```sh
make -C oz-highprecision-cpu bench BLAS_BACKEND=auto
make -C oz-highprecision-cpu bench BLAS_BACKEND=system
make -C oz-highprecision-cpu bench BLAS_BACKEND=openblas
make -C oz-highprecision-cpu bench BLAS_BACKEND=blis
make -C oz-highprecision-cpu bench BLAS_BACKEND=mkl
make -C oz-highprecision-cpu bench BLAS_BACKEND=custom BLAS_LIBS="-L/path -lblas"
```

The helper script runs the quick benchmark and residue-block sweep for each
requested backend, skipping backends that fail to link:

```sh
oz-highprecision-cpu/scripts/evaluate_blas.sh auto system openblas blis mkl
```

On this machine, both `auto` and `system` resolve to
`/lib/x86_64-linux-gnu/libblas.so.3`. OpenBLAS/BLIS/MKL should be evaluated on
a machine where those libraries are installed.

## Correctness

Command:

```sh
make -C oz-highprecision-cpu test
```

Result:

```text
cancellation ok: high_precision=1 fp64_left_to_right=0 moduli=10
plan reject ok
plan slack ok: max_abs=0 max_rel=0
zero vector reuse ok: max_abs=0 max_rel=0
reuse policy ok: max_abs=0 max_rel=0
wide exponent ok: max_abs=0 max_rel=0
parallel crt ok: max_abs=0 max_rel=0
blocked residue ok: max_abs=0 max_rel=0
residue block sizing ok: default=256 small_budget=4
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
8,8,8,11,110,264,67108862,67108859,3.768e-06,0.000149423,3.4274e-05,1,8,0,9.1548e-05,9.0671e-05,0.000235943,0,3.71258675529e-16,0
16,16,16,11,111,264,47453131,47453111,2.164e-06,0.000432232,4.1162e-05,1,16,0,0.00037061,0.000420648,0.001490832,0,7.59906913097e-16,0
32,32,32,11,112,264,33554430,33554393,1.4941e-05,0.001733113,6.2743e-05,1,32,0,0.001657423,0.001674872,0.011964541,0,2.43739505418e-15,0
64,64,64,11,113,264,23726565,23726561,0.000124509,0.004549013,0.000148223,8,64,0,0.00419422,0.007561393,-1,-1,4.82793210353e-15,0
96,96,96,11,114,264,19372659,19372651,0.000497755,0.011549213,0.000333304,8,96,0,0.011132054,0.019072199,-1,-1,1.04819437783e-14,0
```

Additional cases with naive Boost multiprecision enabled:

```text
m,n,k,moduli,exact_required_bits,planned_bits,max_exact_modulus_bound,selected_max_modulus,fp64_seconds,oz_seconds,plan_seconds,crt_threads,residue_col_block,a_residue_cached,oz_reuse_seconds,oz_reuse_serial_seconds,naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs,reuse_vs_oz_max_abs
64,64,64,11,113,264,23726565,23726561,0.00012819,0.005075343,0.000151928,8,64,0,0.004769681,0.007632965,0.100856821,0,4.59062343515e-15,0
128,128,128,12,114,264,16777214,16777213,0.001077992,0.027387371,0.00051329,8,128,0,0.026315556,0.040008881,0.76524883,0,1.12859793792e-14,0
64,512,64,11,113,264,23726565,23726561,0.001026902,0.033664041,0.000623134,8,256,0,0.032676907,0.054333682,0.760407014,0,6.25928911874e-15,0
64,768,64,11,113,264,23726565,23726561,0.001511435,0.040968753,0.00088436,8,256,1,0.040509639,0.079773932,1.11972198,0,6.25928911874e-15,0
```

Residue block-size sweep:

```sh
oz-highprecision-cpu/build/benchmark --sweep-blocks 64 512 64
```

```text
m,n,k,requested_block,effective_block,a_residue_cached,moduli,exact_required_bits,planned_bits,plan_seconds,reuse_seconds,vs_auto_max_abs
64,512,64,0,256,0,11,113,264,0.000611599,0.02757801,0
64,512,64,32,32,1,11,113,264,0.000629869,0.053494286,0
64,512,64,64,64,1,11,113,264,0.000575506,0.027715016,0
64,512,64,128,128,1,11,113,264,0.000579423,0.026949108,0
64,512,64,256,256,0,11,113,264,0.000576571,0.027101911,0
64,512,64,512,512,0,11,113,264,0.000628058,0.026623022,0
```

`max_exact_modulus_bound` is the largest integer allowed by the exact FP64
accumulation bound. It is not necessarily prime. `selected_max_modulus` is the
largest prime modulus actually used.

## Phase Profile

Command:

```sh
oz-highprecision-cpu/build/benchmark --profile M N K
```

Representative reusable-plan phase breakdowns:

```text
m,n,k,moduli,total_seconds,input_prepare_seconds,a_residue_seconds,b_residue_seconds,blas_seconds,residue_store_seconds,crt_seconds,residue_fraction,blas_fraction,crt_fraction,residue_col_block,a_residue_cached,crt_threads,output_blocks,residue_gemm_calls
64,64,64,11,0.00480298,0.000113426,0.000643012,0.000680529,0.001417878,0.000484852,0.001369942,0.376514788735,0.295207975049,0.285227504591,64,0,8,1,11
128,128,128,12,0.025152896,0.000475307,0.002830023,0.002964616,0.013143027,0.002090398,0.003006175,0.313484260421,0.522525398268,0.119516058906,128,0,8,1,12
64,512,64,11,0.028009466,0.000530114,0.001251814,0.005756637,0.011008654,0.003681065,0.005177021,0.381639407192,0.393033340943,0.184831121022,256,0,8,2,22
64,768,64,11,0.040046608,0.00078069,0.000630203,0.008103371,0.016642813,0.005687695,0.007422375,0.360112122355,0.415586084095,0.185343412855,256,1,8,3,33
```

The profile records wall-clock time inside `gemm_with_plan`, so it includes the
currently linked system BLAS. At `64x64x64`, residue generation/storage,
residue GEMM, and CRT reconstruction are all large contributors. At
`128x128x128`, the BLAS calls dominate this system-BLAS run. For wider `n`,
B-side residue generation and residue storage grow visibly. CRT recovery now
reads the residue block with a stride instead of copying per-output residues to
a temporary vector first; this lowers the wider-case CRT cost, but it remains a
material fraction. A-side residue panel caching is active in the `64x768x64`
case and keeps A residue generation small.

## Initial Read

- The modular/CRT PoC exactly matches the naive Boost `cpp_bin_float<256>`
  reference for tested random `double` inputs.
- One-shot execution is still slower than naive Boost at `8x8x8`, but
  reusable-plan execution is already faster in this run and the gap increases
  quickly with size.
- With CRT/Garner inverse precomputation, reusable plans, the int64 residue
  fast path, per-call input decomposition reuse, and precomputed powers of two
  modulo each prime, reusable-plan execution is about `21.1x` faster than
  naive Boost at `64x64x64` and about `29.1x` faster at `128x128x128`.
- Parallel CRT recovery uses 8 threads in this environment for outputs with at
  least 4096 entries. In the default run, reusable execution is about `1.8x`
  faster than serial CRT at `64x64x64` and about `1.7x` faster at `96x96x96`.
- Residue GEMM and CRT recovery now operate on output-column blocks. The
  automatic selector uses a temporary-memory target and picks the full `n` for
  these default cases and 256 columns for the wider `64x512x64` and
  `64x768x64` cases.
- The block-size sweep shows that the memory-budget selector is conservative:
  with the system BLAS in this environment, `64x512x64` is fastest with a
  larger explicit block, while the auto block remains close and bounds
  temporary storage.
- A-side residue panels are cached when the output spans more than two column
  blocks. That path is active in the `64x768x64` case above and avoids
  regenerating the same A residues for each block.
- Reusable plans can reserve scale and magnitude slack. This lets a plan accept
  later inputs with finer exponents or larger scaled integers, while increasing
  the exact CRT width up front.
- Rows/columns that are all zero at plan creation are zero-only by default.
  They can now reserve explicit scale and bit bounds so a later compatible
  nonzero row/column can reuse the same plan.
- Reuse-policy presets provide `Strict`, `Moderate`, and `Wide` defaults for
  scale slack, magnitude slack, and zero-vector reserve bounds while preserving
  unrelated options such as precision target and thread settings.
- Reusable plans separate input scale/CRT setup from execution. For these
  random matrices, planning is small compared with modular GEMM and CRT at
  medium sizes. Reuse output exactly matches one-shot output in these tests.
- Phase profiling is now available through `benchmark --profile M N K`.
  Current system-BLAS measurements show that tuned BLAS should help most at
  larger square shapes, while wide-output cases still need residue/CRT-side
  work.
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
  elements are now split across worker threads, Garner inverses/full modulus
  products are precomputed once per plan, and per-output residue copies have
  been removed,
- reusable plans currently store scale bounds from a reference input and reject
  later inputs outside those bounds unless slack was reserved,
- automatic residue block sizing uses a simple memory-budget heuristic rather
  than empirical tuning for the active BLAS backend,
- the BLAS backend here is system BLAS, not OpenBLAS/MKL/BLIS.

The next performance work should focus on:

- evaluating the same profile with tuned OpenBLAS/MKL/BLIS,
- reducing B-side residue rebuild cost for repeated or wide-output workloads,
- optimizing CRT reconstruction and residue storage, which remain material at
  `64x64x64` and wide-output cases,
- replacing the simple memory-budget block-size heuristic with empirical
  backend-aware tuning.
