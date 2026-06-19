#pragma once

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
void dgemm_(const char *transa, const char *transb,
            const int *m, const int *n, const int *k,
            const double *alpha,
            const double *a, const int *lda,
            const double *b, const int *ldb,
            const double *beta,
            double *c, const int *ldc);
}

namespace oz_hp_cpu {

template <unsigned Bits>
using binary_float = boost::multiprecision::number<
    boost::multiprecision::cpp_bin_float<Bits, boost::multiprecision::digit_base_2>>;

using float256 = binary_float<256>;

enum class Operation {
    NoTrans,
    Trans
};

struct Options {
    int target_bits = 256;
    int guard_bits = 8;
    int max_moduli = 256;
};

struct Plan {
    std::vector<int> moduli;
    std::vector<int> garner_inverses;
    boost::multiprecision::cpp_int modulus_product = 1;
    int exact_required_bits = 0;
    int planned_bits = 0;
    // Upper bound imposed by exact FP64 accumulation; not necessarily prime.
    int max_exact_modulus = 0;
};

namespace detail {

using boost::multiprecision::cpp_int;

struct DoubleParts {
    bool negative = false;
    bool zero = true;
    std::uint64_t mantissa = 0;
    int exponent = 0;
};

inline void validate_options(const Options &options) {
    if (options.target_bits < 2) {
        throw std::invalid_argument("target_bits must be >= 2");
    }
    if (options.guard_bits < 0) {
        throw std::invalid_argument("guard_bits must be >= 0");
    }
    if (options.max_moduli < 1) {
        throw std::invalid_argument("max_moduli must be >= 1");
    }
}

inline void validate_gemm_args(Operation op_a, Operation op_b,
                               int m, int n, int k,
                               const double *a, int lda,
                               const double *b, int ldb) {
    if (m < 0 || n < 0 || k < 0) {
        throw std::invalid_argument("matrix dimensions must be non-negative");
    }
    if (a == nullptr || b == nullptr) {
        throw std::invalid_argument("A and B must be non-null");
    }

    const int a_rows = (op_a == Operation::NoTrans) ? m : k;
    const int b_rows = (op_b == Operation::NoTrans) ? k : n;
    if (lda < std::max(1, a_rows)) {
        throw std::invalid_argument("lda is too small for A");
    }
    if (ldb < std::max(1, b_rows)) {
        throw std::invalid_argument("ldb is too small for B");
    }
}

inline double read_a(Operation op, const double *a, int lda, int row, int col) {
    return (op == Operation::NoTrans) ? a[row + col * lda] : a[col + row * lda];
}

inline double read_b(Operation op, const double *b, int ldb, int row, int col) {
    return (op == Operation::NoTrans) ? b[row + col * ldb] : b[col + row * ldb];
}

inline int bit_length_u64(std::uint64_t x) {
    int bits = 0;
    while (x != 0) {
        ++bits;
        x >>= 1;
    }
    return bits;
}

inline int ceil_log2_int(int x) {
    if (x <= 1) {
        return 0;
    }
    int bits = 0;
    int v = x - 1;
    while (v != 0) {
        ++bits;
        v >>= 1;
    }
    return bits;
}

inline DoubleParts decompose_double(double value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const std::uint64_t sign = bits >> 63;
    const std::uint64_t exp_bits = (bits >> 52) & 0x7ffULL;
    const std::uint64_t frac = bits & ((1ULL << 52) - 1ULL);

    if (exp_bits == 0x7ffULL) {
        throw std::invalid_argument("inputs must be finite");
    }
    if (exp_bits == 0 && frac == 0) {
        return DoubleParts{};
    }

    DoubleParts out;
    out.negative = (sign != 0);
    out.zero = false;
    if (exp_bits == 0) {
        out.mantissa = frac;
        out.exponent = -1022 - 52;
    } else {
        out.mantissa = (1ULL << 52) | frac;
        out.exponent = static_cast<int>(exp_bits) - 1023 - 52;
    }

    while ((out.mantissa & 1ULL) == 0ULL) {
        out.mantissa >>= 1;
        ++out.exponent;
    }
    return out;
}

struct VectorScale {
    int scale_exp = 0;
    int max_scaled_bits = 0;
};

template <typename Reader>
inline VectorScale analyze_vector(int len, Reader reader) {
    bool any = false;
    int min_exp = 0;
    std::vector<DoubleParts> parts;
    parts.reserve(static_cast<std::size_t>(len));

    for (int i = 0; i < len; ++i) {
        DoubleParts p = decompose_double(reader(i));
        parts.push_back(p);
        if (!p.zero) {
            if (!any || p.exponent < min_exp) {
                min_exp = p.exponent;
            }
            any = true;
        }
    }

    if (!any) {
        return VectorScale{};
    }

    VectorScale out;
    out.scale_exp = std::max(0, -min_exp);
    for (const DoubleParts &p : parts) {
        if (p.zero) {
            continue;
        }
        const int shift = p.exponent + out.scale_exp;
        if (shift < 0) {
            throw std::logic_error("internal scaling error");
        }
        out.max_scaled_bits = std::max(out.max_scaled_bits,
                                       bit_length_u64(p.mantissa) + shift);
    }
    return out;
}

inline std::uint64_t pow_mod_u64(std::uint64_t base,
                                 std::uint64_t exp,
                                 std::uint64_t mod) {
    std::uint64_t result = 1 % mod;
    base %= mod;
    while (exp != 0) {
        if ((exp & 1ULL) != 0ULL) {
            result = static_cast<std::uint64_t>((__uint128_t(result) * base) % mod);
        }
        base = static_cast<std::uint64_t>((__uint128_t(base) * base) % mod);
        exp >>= 1;
    }
    return result;
}

inline bool miller_rabin_witness(std::uint64_t n,
                                 std::uint64_t d,
                                 int s,
                                 std::uint64_t a) {
    if (a % n == 0) {
        return false;
    }
    std::uint64_t x = pow_mod_u64(a, d, n);
    if (x == 1 || x == n - 1) {
        return false;
    }
    for (int r = 1; r < s; ++r) {
        x = static_cast<std::uint64_t>((__uint128_t(x) * x) % n);
        if (x == n - 1) {
            return false;
        }
    }
    return true;
}

inline bool is_prime_u32(std::uint64_t n) {
    if (n < 2) {
        return false;
    }
    static constexpr std::uint64_t small_primes[] = {
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    };
    for (std::uint64_t p : small_primes) {
        if (n == p) {
            return true;
        }
        if (n % p == 0) {
            return false;
        }
    }

    std::uint64_t d = n - 1;
    int s = 0;
    while ((d & 1ULL) == 0ULL) {
        d >>= 1;
        ++s;
    }

    static constexpr std::uint64_t witnesses[] = {2, 3, 5, 7, 11};
    for (std::uint64_t a : witnesses) {
        if (miller_rabin_witness(n, d, s, a)) {
            return false;
        }
    }
    return true;
}

inline int max_exact_modulus_for_dgemm(int k) {
    const long double exact_limit = std::ldexp(1.0L, 53) - 1.0L;
    const long double root = std::sqrt(exact_limit / std::max(1, k));
    long double p = 2.0L * root - 1.0L;
    p = std::min<long double>(p, static_cast<long double>(std::numeric_limits<int>::max()));
    return static_cast<int>(std::floor(p));
}

inline std::vector<int> choose_moduli(int k,
                                      int required_bits,
                                      const Options &options,
                                      int *max_exact_modulus_out) {
    const int max_modulus = max_exact_modulus_for_dgemm(k);
    if (max_exact_modulus_out != nullptr) {
        *max_exact_modulus_out = max_modulus;
    }
    if (max_modulus < 3) {
        throw std::invalid_argument("k is too large for exact FP64 modular dgemm");
    }

    std::vector<int> moduli;
    moduli.reserve(static_cast<std::size_t>(std::min(options.max_moduli, 32)));

    double bits = 0.0;
    int candidate = max_modulus;
    if ((candidate & 1) == 0) {
        --candidate;
    }

    while (candidate >= 3 && bits < required_bits) {
        if (is_prime_u32(static_cast<std::uint64_t>(candidate))) {
            moduli.push_back(candidate);
            bits += std::log2(static_cast<double>(candidate));
            if (static_cast<int>(moduli.size()) > options.max_moduli) {
                throw std::runtime_error("max_moduli is too small for requested precision");
            }
        }
        candidate -= 2;
    }

    if (bits < required_bits) {
        throw std::runtime_error("could not find enough moduli for requested precision");
    }
    return moduli;
}

inline int positive_mod_int(const cpp_int &value, int modulus) {
    cpp_int rem = value % modulus;
    if (rem < 0) {
        rem += modulus;
    }
    return rem.convert_to<int>();
}

inline int positive_mod_int(int value, int modulus) {
    int rem = value % modulus;
    if (rem < 0) {
        rem += modulus;
    }
    return rem;
}

inline int centered_from_positive(int rem, int modulus) {
    const int half = modulus / 2;
    return (rem > half) ? (rem - modulus) : rem;
}

inline int centered_mod_i64(std::int64_t value, int modulus) {
    std::int64_t rem = value % modulus;
    if (rem < 0) {
        rem += modulus;
    }
    return centered_from_positive(static_cast<int>(rem), modulus);
}

inline int mod_inverse(int value, int modulus) {
    int t = 0;
    int new_t = 1;
    int r = modulus;
    int new_r = positive_mod_int(value, modulus);

    while (new_r != 0) {
        const int q = r / new_r;
        const int next_t = t - q * new_t;
        t = new_t;
        new_t = next_t;
        const int next_r = r - q * new_r;
        r = new_r;
        new_r = next_r;
    }

    if (r != 1) {
        throw std::runtime_error("moduli are not pairwise coprime");
    }
    return positive_mod_int(t, modulus);
}

inline void finalize_crt_plan(Plan &plan) {
    plan.garner_inverses.clear();
    plan.garner_inverses.reserve(plan.moduli.size());
    plan.modulus_product = 1;

    cpp_int product = 1;
    for (int p : plan.moduli) {
        const int product_mod = positive_mod_int(product, p);
        plan.garner_inverses.push_back(mod_inverse(product_mod, p));
        product *= p;
    }
    plan.modulus_product = product;
}

inline cpp_int reconstruct_crt_centered(const std::vector<int> &residues,
                                        const Plan &plan) {
    cpp_int x = 0;
    cpp_int product = 1;

    for (std::size_t i = 0; i < plan.moduli.size(); ++i) {
        const int p = plan.moduli[i];
        const int target = positive_mod_int(residues[i], p);
        const int x_mod = positive_mod_int(x, p);
        const int inv = plan.garner_inverses[i];
        const int delta = positive_mod_int(target - x_mod, p);
        const int coeff = static_cast<int>((static_cast<long long>(delta) * inv) % p);
        x += product * coeff;
        product *= p;
    }

    if (x * 2 > plan.modulus_product) {
        x -= plan.modulus_product;
    }
    return x;
}

inline int scaled_double_centered_mod(double value, int scale_exp, int modulus) {
    const DoubleParts p = decompose_double(value);
    if (p.zero) {
        return 0;
    }

    const int shift = p.exponent + scale_exp;
    if (shift < 0) {
        throw std::logic_error("scale exponent does not make input integral");
    }

    const int scaled_bits = bit_length_u64(p.mantissa) + shift;
    if (scaled_bits <= 63) {
        const std::uint64_t magnitude = p.mantissa << shift;
        const std::int64_t signed_value =
            p.negative ? -static_cast<std::int64_t>(magnitude)
                       : static_cast<std::int64_t>(magnitude);
        return centered_mod_i64(signed_value, modulus);
    }

    const std::uint64_t m = static_cast<std::uint64_t>(modulus);
    std::uint64_t rem = (p.mantissa % m) * pow_mod_u64(2, static_cast<std::uint64_t>(shift), m);
    rem %= m;
    if (p.negative && rem != 0) {
        rem = m - rem;
    }
    return centered_from_positive(static_cast<int>(rem), modulus);
}

inline void blas_dgemm_nn(int m, int n, int k,
                          const double *a, int lda,
                          const double *b, int ldb,
                          double *c, int ldc) {
    const char trans = 'N';
    const double one = 1.0;
    const double zero = 0.0;
    dgemm_(&trans, &trans, &m, &n, &k,
           &one, a, &lda, b, &ldb, &zero, c, &ldc);
}

template <typename HighPrec>
inline HighPrec scale_cpp_int_pow2(const cpp_int &value, int exponent) {
    using boost::multiprecision::ldexp;
    HighPrec out(value);
    return ldexp(out, exponent);
}

} // namespace detail

inline Plan make_plan(Operation op_a, Operation op_b,
                      int m, int n, int k,
                      const double *a, int lda,
                      const double *b, int ldb,
                      const Options &options = Options{}) {
    detail::validate_options(options);
    detail::validate_gemm_args(op_a, op_b, m, n, k, a, lda, b, ldb);

    std::vector<detail::VectorScale> rows(static_cast<std::size_t>(m));
    std::vector<detail::VectorScale> cols(static_cast<std::size_t>(n));

    int max_row_bits = 0;
    int max_col_bits = 0;
    for (int row = 0; row < m; ++row) {
        rows[row] = detail::analyze_vector(k, [&](int col) {
            return detail::read_a(op_a, a, lda, row, col);
        });
        max_row_bits = std::max(max_row_bits, rows[row].max_scaled_bits);
    }
    for (int col = 0; col < n; ++col) {
        cols[col] = detail::analyze_vector(k, [&](int row) {
            return detail::read_b(op_b, b, ldb, row, col);
        });
        max_col_bits = std::max(max_col_bits, cols[col].max_scaled_bits);
    }

    Plan plan;
    plan.exact_required_bits = max_row_bits + max_col_bits +
                               detail::ceil_log2_int(std::max(1, k)) + 1;
    plan.planned_bits = std::max(options.target_bits, plan.exact_required_bits) +
                        options.guard_bits;
    plan.moduli = detail::choose_moduli(k, plan.planned_bits, options,
                                        &plan.max_exact_modulus);
    detail::finalize_crt_plan(plan);
    return plan;
}

template <typename HighPrec>
inline void gemm(Operation op_a, Operation op_b,
                 int m, int n, int k,
                 const HighPrec &alpha,
                 const double *a, int lda,
                 const double *b, int ldb,
                 const HighPrec &beta,
                 HighPrec *c, int ldc,
                 const Options &options = Options{},
                 Plan *plan_out = nullptr) {
    detail::validate_options(options);
    detail::validate_gemm_args(op_a, op_b, m, n, k, a, lda, b, ldb);
    if (c == nullptr) {
        throw std::invalid_argument("C must be non-null");
    }
    if (ldc < std::max(1, m)) {
        throw std::invalid_argument("ldc is too small for C");
    }

    if (m == 0 || n == 0) {
        return;
    }
    if (k == 0) {
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                c[row + col * ldc] *= beta;
            }
        }
        return;
    }

    std::vector<detail::VectorScale> rows(static_cast<std::size_t>(m));
    std::vector<detail::VectorScale> cols(static_cast<std::size_t>(n));

    int max_row_bits = 0;
    int max_col_bits = 0;
    for (int row = 0; row < m; ++row) {
        rows[row] = detail::analyze_vector(k, [&](int col) {
            return detail::read_a(op_a, a, lda, row, col);
        });
        max_row_bits = std::max(max_row_bits, rows[row].max_scaled_bits);
    }
    for (int col = 0; col < n; ++col) {
        cols[col] = detail::analyze_vector(k, [&](int row) {
            return detail::read_b(op_b, b, ldb, row, col);
        });
        max_col_bits = std::max(max_col_bits, cols[col].max_scaled_bits);
    }

    Plan plan;
    plan.exact_required_bits = max_row_bits + max_col_bits +
                               detail::ceil_log2_int(std::max(1, k)) + 1;
    plan.planned_bits = std::max(options.target_bits, plan.exact_required_bits) +
                        options.guard_bits;
    plan.moduli = detail::choose_moduli(k, plan.planned_bits, options,
                                        &plan.max_exact_modulus);
    detail::finalize_crt_plan(plan);
    if (plan_out != nullptr) {
        *plan_out = plan;
    }

    std::vector<double> a_mod(static_cast<std::size_t>(m) * k);
    std::vector<double> b_mod(static_cast<std::size_t>(k) * n);
    std::vector<double> c_mod(static_cast<std::size_t>(m) * n);
    std::vector<int> residues(plan.moduli.size());
    std::vector<int> all_residues(plan.moduli.size() * static_cast<std::size_t>(m) * n);

    for (std::size_t imod = 0; imod < plan.moduli.size(); ++imod) {
        const int p = plan.moduli[imod];

        for (int col = 0; col < k; ++col) {
            for (int row = 0; row < m; ++row) {
                const double value = detail::read_a(op_a, a, lda, row, col);
                a_mod[row + col * m] = static_cast<double>(
                    detail::scaled_double_centered_mod(value, rows[row].scale_exp, p));
            }
        }

        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < k; ++row) {
                const double value = detail::read_b(op_b, b, ldb, row, col);
                b_mod[row + col * k] = static_cast<double>(
                    detail::scaled_double_centered_mod(value, cols[col].scale_exp, p));
            }
        }

        detail::blas_dgemm_nn(m, n, k,
                              a_mod.data(), m,
                              b_mod.data(), k,
                              c_mod.data(), m);

        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                const std::size_t idx =
                    static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * m;
                const auto rounded = static_cast<std::int64_t>(std::llround(c_mod[idx]));
                all_residues[imod * static_cast<std::size_t>(m) * n + idx] =
                    detail::centered_mod_i64(rounded, p);
            }
        }
    }

    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            const std::size_t idx =
                static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * m;
            for (std::size_t imod = 0; imod < plan.moduli.size(); ++imod) {
                residues[imod] = all_residues[imod * static_cast<std::size_t>(m) * n + idx];
            }

            const detail::cpp_int crt =
                detail::reconstruct_crt_centered(residues, plan);
            const int restore_exp = -(rows[row].scale_exp + cols[col].scale_exp);
            const HighPrec ab = detail::scale_cpp_int_pow2<HighPrec>(crt, restore_exp);
            const std::size_t out_idx =
                static_cast<std::size_t>(row) + static_cast<std::size_t>(col) * ldc;
            c[out_idx] = alpha * ab + beta * c[out_idx];
        }
    }
}

template <typename HighPrec>
inline std::vector<HighPrec> gemm(Operation op_a, Operation op_b,
                                  int m, int n, int k,
                                  const double *a, int lda,
                                  const double *b, int ldb,
                                  const Options &options = Options{},
                                  Plan *plan_out = nullptr) {
    std::vector<HighPrec> c(static_cast<std::size_t>(m) * n, HighPrec(0));
    gemm(op_a, op_b, m, n, k,
         HighPrec(1), a, lda, b, ldb, HighPrec(0), c.data(), m,
         options, plan_out);
    return c;
}

} // namespace oz_hp_cpu
