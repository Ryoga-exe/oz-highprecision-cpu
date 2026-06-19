# oz-highprecision-cpu

PoC for high-precision GEMM emulation on CPUs. It uses ordinary FP64 BLAS
`dgemm` calls as exact modular GEMM kernels, then reconstructs the integer
product with CRT and converts it to a Boost binary multiprecision result.

The current implementation targets:

- `double` input matrices,
- column-major BLAS layout,
- `NoTrans` and `Trans`,
- high-precision output such as `cpp_bin_float<256, digit_base_2>`.

For this first PoC, "high precision" means exact accumulation of the supplied
`double` inputs into a high-precision output type. It is not yet a GEMM for
arbitrary FP256 input matrices.

## Why this differs from `oz-cpu`

`oz-cpu` mirrors the INT8-moduli flow used by GEMMul8. This PoC instead chooses
larger prime moduli automatically. For a given inner dimension `k`, each modulus
`p` is constrained so that

```text
k * ((p - 1) / 2)^2 < 2^53
```

That keeps each modular GEMM exact when it is executed by FP64 `dgemm`.

The implementation scales every row of `op(A)` and every column of `op(B)` by
a power of two so that all input `double` values become exact integers. It then
uses enough CRT width for the requested target precision and for the exact
integer product bound implied by the input exponents.

## Build and test

```sh
make -C oz-highprecision-cpu test
make -C oz-highprecision-cpu example
make -C oz-highprecision-cpu bench
```

To force a BLAS library:

```sh
make -C oz-highprecision-cpu test BLAS_LIBS="-lopenblas"
make -C oz-highprecision-cpu test BLAS_LIBS="-lblas"
```

The benchmark target runs `build/benchmark --quick` and prints CSV columns for
FP64 BLAS time, modular/CRT time, optional naive Boost multiprecision time, and
maximum absolute differences. Run the default benchmark set with:

```sh
make -C oz-highprecision-cpu build/benchmark
oz-highprecision-cpu/build/benchmark
```

The `max_exact_modulus_bound` column is the exact-FP64-accumulation upper bound
for a modulus; it is not necessarily prime. The actual largest selected prime
modulus is reported separately as `selected_max_modulus`.

## Example

The example computes the dot product

```text
[1e16, 1, -1e16] * [1, 1, 1]^T
```

Left-to-right FP64 summation produces `0`, while the high-precision modular
path reconstructs the exact `double` input result `1`.

## Scope and limitations

This is a correctness-oriented PoC, not a tuned implementation. It currently
rebuilds residue matrices for each modulus and performs scalar CRT recovery per
output element. Obvious next steps are residue blocking, batched/threaded CRT
reconstruction, plan reuse across calls, and CPU matrix-extension backends
such as AMX/VNNI for smaller residues.
