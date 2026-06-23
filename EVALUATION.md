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

The helper script runs the quick benchmark, residue-block sweep, CRT
thread-count sweep, and precision sweep for each requested backend, skipping
backends that fail to link. It mirrors output to stdout and saves artifacts
under `results/<UTC timestamp>/<backend>/` unless `OZ_EVAL_RUN_ID` or
`OZ_EVAL_RESULTS_DIR` is set:

```sh
oz-highprecision-cpu/scripts/evaluate_blas.sh auto system openblas blis mkl
```

Saved runs can be summarized with:

```sh
oz-highprecision-cpu/scripts/summarize_results.py oz-highprecision-cpu/results
oz-highprecision-cpu/scripts/summarize_results.py oz-highprecision-cpu/results/<run-id> --csv-dir /tmp/oz-summary
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

CRT thread-count sweep:

```sh
oz-highprecision-cpu/build/benchmark --sweep-crt-threads 64 512 64
```

```text
m,n,k,requested_threads,auto_max_threads,effective_threads,moduli,residue_col_block,a_residue_cached,total_seconds,crt_seconds,crt_fraction,vs_auto_max_abs
64,64,64,0,0,8,11,64,0,0.004097165,0.000721131,0.176007312373,0
64,64,64,0,4,4,11,64,0,0.004451496,0.001119927,0.251584411173,0
64,64,64,1,0,1,11,64,0,0.007353013,0.00402745,0.547727849794,0
64,64,64,2,0,2,11,64,0,0.00509708,0.001991825,0.390777660935,0
64,64,64,4,0,4,11,64,0,0.004303522,0.001121462,0.260591673518,0
64,64,64,8,0,8,11,64,0,0.004030805,0.000737303,0.182917059992,0
128,128,128,0,0,8,12,128,0,0.025621384,0.004489891,0.175239987036,0
128,128,128,0,4,4,12,128,0,0.025527293,0.004609765,0.18058181884,0
128,128,128,1,0,1,12,128,0,0.037060134,0.017107054,0.461602594313,0
128,128,128,2,0,2,12,128,0,0.028631299,0.008678002,0.303094945151,0
128,128,128,4,0,4,12,128,0,0.024909504,0.004533987,0.182018357331,0
128,128,128,8,0,8,12,128,0,0.026373604,0.005771004,0.21881742063,0
64,512,64,0,0,8,11,256,0,0.030703972,0.008567004,0.279019405046,0
64,512,64,0,4,4,11,256,0,0.029964782,0.008416906,0.280893283322,0
64,512,64,1,0,1,11,256,0,0.051668202,0.031064523,0.601230966001,0
64,512,64,2,0,2,11,256,0,0.037426507,0.016144984,0.431378327665,0
64,512,64,4,0,4,11,256,0,0.029256891,0.008235927,0.28150383443,0
64,512,64,8,0,8,11,256,0,0.026312428,0.004836155,0.183797367541,0
```

On this 8-core machine, serial CRT is consistently much slower for these
medium shapes. The sweep now includes both uncapped auto
(`requested_threads=0,auto_max_threads=0`) and capped auto
(`requested_threads=0,auto_max_threads=4`). The best setting is still somewhat
shape- and run-dependent, so the default remains uncapped while
`Options::crt_auto_max_threads` allows applications to cap automatic CRT
threading explicitly.

Precision target sweep:

```sh
oz-highprecision-cpu/build/benchmark --sweep-precision 64 64 64
oz-highprecision-cpu/build/benchmark --sweep-precision 64 512 64
```

```text
m,n,k,target_bits,guard_bits,moduli,exact_required_bits,planned_bits,max_exact_modulus_bound,selected_max_modulus,total_seconds,residue_seconds,blas_seconds,crt_seconds,residue_fraction,blas_fraction,crt_fraction,residue_gemm_calls,vs_target256_max_abs
64,64,64,64,8,5,113,121,23726565,23726561,0.002411131,0.000793276,0.000673788,0.000714507,0.329005765344,0.279448939108,0.296336864318,5,0
64,64,64,96,8,5,113,121,23726565,23726561,0.002305344,0.000797306,0.00062809,0.000748356,0.345851204853,0.272449578024,0.324617931207,5,0
64,64,64,128,8,6,113,136,23726565,23726561,0.002723354,0.000906463,0.000777647,0.000889877,0.332848024899,0.285547527057,0.326757740639,6,0
64,64,64,160,8,7,113,168,23726565,23726561,0.003037612,0.001030535,0.00090905,0.000878546,0.339258272617,0.299264685549,0.28922258669,7,0
64,64,64,192,8,9,113,200,23726565,23726561,0.003794765,0.001354002,0.001183915,0.001040783,0.356807865573,0.311986381238,0.274268103558,9,0
64,64,64,224,8,10,113,232,23726565,23726561,0.004286012,0.001568144,0.001280871,0.001198095,0.36587485056,0.298849139946,0.27953608156,10,0
64,64,64,256,8,11,113,264,23726565,23726561,0.004540114,0.001645102,0.001489497,0.001205154,0.362348170112,0.328074801646,0.265445757529,11,0
64,64,64,320,8,14,113,328,23726565,23726561,0.005769208,0.002124526,0.001848766,0.001551933,0.368252626704,0.320454038059,0.269002781664,14,0
64,64,64,384,8,17,113,392,23726565,23726561,0.006851081,0.002586231,0.002210953,0.001833345,0.377492398645,0.322715933442,0.267599375923,17,0
64,512,64,64,8,5,113,121,23726565,23726561,0.014754284,0.004210191,0.00526058,0.004485439,0.285353799615,0.356545936082,0.304009262666,10,0
64,512,64,96,8,5,113,121,23726565,23726561,0.013858499,0.00435496,0.005062157,0.003564265,0.314244710051,0.365274551017,0.257189829865,10,0
64,512,64,128,8,6,113,136,23726565,23726561,0.015060795,0.005353888,0.005949143,0.003102936,0.355485085615,0.395008563625,0.206027371065,12,0
64,512,64,160,8,7,113,168,23726565,23726561,0.017182799,0.006076343,0.007190111,0.003373042,0.35362940578,0.418448181813,0.196303407844,14,0
64,512,64,192,8,9,113,200,23726565,23726561,0.021483639,0.007747255,0.009204717,0.00383342,0.360611859099,0.428452414416,0.178434389072,18,0
64,512,64,224,8,10,113,232,23726565,23726561,0.024252307,0.008612826,0.010276053,0.004348626,0.355134297121,0.423714453227,0.17930772524,20,0
64,512,64,256,8,11,113,264,23726565,23726561,0.026147268,0.009480303,0.011334784,0.004636233,0.36257336713,0.433497832355,0.177312329533,22,0
64,512,64,320,8,14,113,328,23726565,23726561,0.03297742,0.012179422,0.014297449,0.005825642,0.369326102527,0.433552685444,0.176655481235,28,0
64,512,64,384,8,17,113,392,23726565,23726561,0.039928946,0.014665322,0.017508036,0.007038686,0.367285477558,0.438479793581,0.176280285485,34,0
```

For these random `[-1, 1]` inputs, `exact_required_bits` is `113`, so target
settings below that are floored by exactness and use the same `planned_bits`.
Above that point, runtime grows roughly with the number of selected moduli and
therefore with the number of residue GEMM calls.

## Phase Profile

Command:

```sh
oz-highprecision-cpu/build/benchmark --profile M N K
```

Representative reusable-plan phase breakdowns:

```text
m,n,k,moduli,total_seconds,input_prepare_seconds,a_residue_seconds,b_residue_seconds,blas_seconds,residue_store_seconds,crt_seconds,residue_fraction,blas_fraction,crt_fraction,residue_col_block,a_residue_cached,crt_threads,output_blocks,residue_gemm_calls
64,64,64,11,0.004233301,0.000113229,0.000655418,0.000664162,0.00138225,0.000336328,0.000857498,0.391162357697,0.326518241911,0.20256012979,64,0,8,1,11
128,128,128,12,0.023577905,0.000497548,0.002780687,0.002904021,0.012912858,0.00156443,0.002637363,0.307454712367,0.547667742321,0.111857393606,128,0,8,1,12
64,512,64,11,0.026265695,0.000507152,0.001312932,0.005345011,0.011043315,0.002763849,0.004796501,0.358710934548,0.420446327424,0.182614661443,256,0,8,2,22
64,768,64,11,0.038836659,0.000749566,0.000658218,0.008116234,0.016901606,0.00422854,0.007424126,0.334812322553,0.435197219205,0.191162839213,256,1,8,3,33
```

The profile records wall-clock time inside `gemm_with_plan`, so it includes the
currently linked system BLAS. At `64x64x64`, residue generation/storage,
residue GEMM, and CRT reconstruction are all large contributors. At
`128x128x128`, the BLAS calls dominate this system-BLAS run. For wider `n`,
B-side residue generation and residue storage grow visibly. CRT recovery now
reads the residue block with a stride instead of copying per-output residues to
a temporary vector first; this lowers the wider-case CRT cost, but it remains a
material fraction. Residue GEMM outputs are also cast directly to `int64_t`
because the exact-modular bound makes them exact FP64 integers, avoiding a
per-output `llround` call. A-side residue panel caching is active in the
`64x768x64` case and keeps A residue generation small.

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
  temporary output residues are now limited to a column block, and exact
  residue GEMM outputs avoid libm rounding during residue storage,
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
