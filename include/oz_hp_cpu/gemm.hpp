#pragma once

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
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

enum class PlanReusePolicy {
    Strict,
    Moderate,
    Wide
};

struct Options {
    int target_bits = 256;
    int guard_bits = 8;
    int max_moduli = 256;
    // Extra power-of-two scale bits reserved for reusing a plan with finer inputs.
    int reuse_scale_slack_bits = 0;
    // Extra scaled-integer magnitude bits reserved for reusing a plan.
    int reuse_magnitude_slack_bits = 0;
    // Bounds assigned to rows/columns that are all zero at plan creation.
    int zero_vector_scale_exp = 0;
    int zero_vector_max_scaled_bits = 0;
    // 0 selects an automatic thread count; 1 forces serial CRT recovery.
    int crt_threads = 0;
    // 0 selects an automatic column block size for residue GEMM/CRT.
    int residue_col_block = 0;
    // Target temporary bytes for automatic residue column block sizing.
    int residue_target_bytes = 8 * 1024 * 1024;
};

inline Options with_reuse_policy(Options options, PlanReusePolicy policy) {
    switch (policy) {
    case PlanReusePolicy::Strict:
        options.reuse_scale_slack_bits = 0;
        options.reuse_magnitude_slack_bits = 0;
        options.zero_vector_scale_exp = 0;
        options.zero_vector_max_scaled_bits = 0;
        break;
    case PlanReusePolicy::Moderate:
        options.reuse_scale_slack_bits = 8;
        options.reuse_magnitude_slack_bits = 8;
        options.zero_vector_scale_exp = 64;
        options.zero_vector_max_scaled_bits = 64;
        break;
    case PlanReusePolicy::Wide:
        options.reuse_scale_slack_bits = 32;
        options.reuse_magnitude_slack_bits = 32;
        options.zero_vector_scale_exp = 128;
        options.zero_vector_max_scaled_bits = 128;
        break;
    }
    return options;
}

struct Plan {
    std::vector<int> moduli;
    std::vector<int> garner_inverses;
    boost::multiprecision::cpp_int modulus_product = 1;
    int exact_required_bits = 0;
    int planned_bits = 0;
    // Upper bound imposed by exact FP64 accumulation; not necessarily prime.
    int max_exact_modulus = 0;
};

struct GemmPlan {
    Operation op_a = Operation::NoTrans;
    Operation op_b = Operation::NoTrans;
    int m = 0;
    int n = 0;
    int k = 0;
    Plan crt;
    std::vector<int> row_scale_exp;
    std::vector<int> col_scale_exp;
    std::vector<int> row_max_scaled_bits;
    std::vector<int> col_max_scaled_bits;
    int max_pow2_shift = 0;
    std::vector<std::vector<int>> pow2_residues;
    int crt_threads = 0;
    int residue_col_block = 0;
    int residue_target_bytes = 8 * 1024 * 1024;
};

struct GemmExecutionStats {
    double total_seconds = 0.0;
    double input_prepare_seconds = 0.0;
    double a_residue_seconds = 0.0;
    double b_residue_seconds = 0.0;
    double blas_seconds = 0.0;
    double residue_store_seconds = 0.0;
    double crt_seconds = 0.0;
    int residue_col_block = 0;
    int crt_threads = 0;
    int residue_gemm_calls = 0;
    std::size_t output_blocks = 0;
    std::size_t moduli = 0;
    bool a_residue_cached = false;
};

namespace detail {

using boost::multiprecision::cpp_int;

struct DoubleParts {
    bool negative = false;
    bool zero = true;
    std::uint64_t mantissa = 0;
    int exponent = 0;
};

struct ScaledDouble {
    DoubleParts parts;
    int shift = 0;
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
    if (options.reuse_scale_slack_bits < 0) {
        throw std::invalid_argument("reuse_scale_slack_bits must be >= 0");
    }
    if (options.reuse_magnitude_slack_bits < 0) {
        throw std::invalid_argument("reuse_magnitude_slack_bits must be >= 0");
    }
    if (options.zero_vector_scale_exp < 0) {
        throw std::invalid_argument("zero_vector_scale_exp must be >= 0");
    }
    if (options.zero_vector_max_scaled_bits < 0) {
        throw std::invalid_argument("zero_vector_max_scaled_bits must be >= 0");
    }
    if (options.crt_threads < 0) {
        throw std::invalid_argument("crt_threads must be >= 0");
    }
    if (options.residue_col_block < 0) {
        throw std::invalid_argument("residue_col_block must be >= 0");
    }
    if (options.residue_target_bytes < 1) {
        throw std::invalid_argument("residue_target_bytes must be >= 1");
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

inline cpp_int reconstruct_crt_centered_strided(const int *residues,
                                                std::size_t stride,
                                                const Plan &plan) {
    cpp_int x = 0;
    cpp_int product = 1;

    for (std::size_t i = 0; i < plan.moduli.size(); ++i) {
        const int p = plan.moduli[i];
        const int target = positive_mod_int(residues[i * stride], p);
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

inline cpp_int reconstruct_crt_centered(const std::vector<int> &residues,
                                        const Plan &plan) {
    return reconstruct_crt_centered_strided(residues.data(), 1, plan);
}

inline int scaled_parts_centered_mod(const DoubleParts &p, int shift, int modulus) {
    if (p.zero) {
        return 0;
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

inline int scaled_parts_centered_mod(const DoubleParts &p,
                                     int shift,
                                     int modulus,
                                     const std::vector<int> &pow2_mod) {
    if (p.zero) {
        return 0;
    }

    const int scaled_bits = bit_length_u64(p.mantissa) + shift;
    if (scaled_bits <= 63) {
        const std::uint64_t magnitude = p.mantissa << shift;
        const std::int64_t signed_value =
            p.negative ? -static_cast<std::int64_t>(magnitude)
                       : static_cast<std::int64_t>(magnitude);
        return centered_mod_i64(signed_value, modulus);
    }

    std::uint64_t pow2 = 0;
    if (shift >= 0 && static_cast<std::size_t>(shift) < pow2_mod.size()) {
        pow2 = static_cast<std::uint64_t>(pow2_mod[static_cast<std::size_t>(shift)]);
    } else {
        pow2 = pow_mod_u64(2, static_cast<std::uint64_t>(shift),
                           static_cast<std::uint64_t>(modulus));
    }
    const std::uint64_t m = static_cast<std::uint64_t>(modulus);
    std::uint64_t rem = (p.mantissa % m) * pow2;
    rem %= m;
    if (p.negative && rem != 0) {
        rem = m - rem;
    }
    return centered_from_positive(static_cast<int>(rem), modulus);
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
    return scaled_parts_centered_mod(p, shift, modulus);
}

inline ScaledDouble prepare_scaled_double_checked(double value,
                                                  int scale_exp,
                                                  int max_scaled_bits) {
    const DoubleParts p = decompose_double(value);
    if (p.zero) {
        return ScaledDouble{};
    }

    const int shift = p.exponent + scale_exp;
    if (shift < 0) {
        throw std::invalid_argument("input requires a finer scale than GemmPlan provides");
    }
    const int scaled_bits = bit_length_u64(p.mantissa) + shift;
    if (scaled_bits > max_scaled_bits) {
        throw std::invalid_argument("input exceeds GemmPlan scaled integer bound");
    }
    return ScaledDouble{p, shift};
}

inline int scaled_double_centered_mod_checked(double value,
                                              int scale_exp,
                                              int max_scaled_bits,
                                              int modulus) {
    const ScaledDouble scaled =
        prepare_scaled_double_checked(value, scale_exp, max_scaled_bits);
    return scaled_parts_centered_mod(scaled.parts, scaled.shift, modulus);
}

inline void build_pow2_residue_tables(GemmPlan &plan) {
    plan.max_pow2_shift = 0;
    for (int bits : plan.row_max_scaled_bits) {
        if (bits > 0) {
            plan.max_pow2_shift = std::max(plan.max_pow2_shift, bits - 1);
        }
    }
    for (int bits : plan.col_max_scaled_bits) {
        if (bits > 0) {
            plan.max_pow2_shift = std::max(plan.max_pow2_shift, bits - 1);
        }
    }

    plan.pow2_residues.assign(plan.crt.moduli.size(), {});
    for (std::size_t imod = 0; imod < plan.crt.moduli.size(); ++imod) {
        const int p = plan.crt.moduli[imod];
        std::vector<int> table(static_cast<std::size_t>(plan.max_pow2_shift) + 1);
        table[0] = 1 % p;
        for (int shift = 1; shift <= plan.max_pow2_shift; ++shift) {
            table[static_cast<std::size_t>(shift)] =
                static_cast<int>((2LL * table[static_cast<std::size_t>(shift - 1)]) % p);
        }
        plan.pow2_residues[imod] = std::move(table);
    }
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

inline void validate_gemm_plan(const GemmPlan &plan,
                               Operation op_a,
                               Operation op_b,
                               int m,
                               int n,
                               int k) {
    if (plan.op_a != op_a || plan.op_b != op_b ||
        plan.m != m || plan.n != n || plan.k != k) {
        throw std::invalid_argument("GemmPlan does not match requested GEMM shape");
    }
    if (plan.row_scale_exp.size() != static_cast<std::size_t>(m) ||
        plan.col_scale_exp.size() != static_cast<std::size_t>(n) ||
        plan.row_max_scaled_bits.size() != static_cast<std::size_t>(m) ||
        plan.col_max_scaled_bits.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument("GemmPlan has invalid scale vector sizes");
    }
    if (plan.crt.moduli.empty() ||
        plan.crt.garner_inverses.size() != plan.crt.moduli.size()) {
        throw std::invalid_argument("GemmPlan has invalid CRT data");
    }
    if (plan.pow2_residues.size() != plan.crt.moduli.size()) {
        throw std::invalid_argument("GemmPlan has invalid pow2 residue table count");
    }
    if (plan.crt_threads < 0) {
        throw std::invalid_argument("GemmPlan has invalid CRT thread count");
    }
    if (plan.residue_col_block < 0) {
        throw std::invalid_argument("GemmPlan has invalid residue column block size");
    }
    if (plan.residue_target_bytes < 1) {
        throw std::invalid_argument("GemmPlan has invalid residue target byte count");
    }
    for (const std::vector<int> &table : plan.pow2_residues) {
        if (table.size() != static_cast<std::size_t>(plan.max_pow2_shift) + 1) {
            throw std::invalid_argument("GemmPlan has invalid pow2 residue table size");
        }
    }
}

inline int choose_crt_threads(std::size_t output_size, int requested_threads) {
    if (output_size < 4096) {
        return 1;
    }

    int threads = requested_threads;
    if (threads == 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        threads = (hw == 0) ? 1 : static_cast<int>(hw);
    }
    threads = std::max(1, threads);
    threads = std::min<std::size_t>(static_cast<std::size_t>(threads), output_size);
    return threads;
}

inline int choose_residue_col_block(int m,
                                    int n,
                                    int k,
                                    std::size_t moduli_count,
                                    int requested_block,
                                    int target_bytes) {
    if (n <= 0) {
        return 0;
    }
    if (requested_block > 0) {
        return std::min(n, requested_block);
    }

    const std::size_t safe_m = static_cast<std::size_t>(std::max(1, m));
    const std::size_t safe_k = static_cast<std::size_t>(std::max(1, k));
    const std::size_t safe_moduli = std::max<std::size_t>(1, moduli_count);
    const std::size_t bytes_per_col =
        sizeof(double) * safe_k + sizeof(double) * safe_m + sizeof(int) * safe_moduli * safe_m;
    const std::size_t budget = static_cast<std::size_t>(std::max(1, target_bytes));
    const std::size_t max_cols_by_budget = std::max<std::size_t>(1, budget / bytes_per_col);

    if (static_cast<std::size_t>(n) <= max_cols_by_budget && n <= 256) {
        return n;
    }

    static constexpr int candidates[] = {256, 128, 64, 32, 16, 8, 4, 2, 1};
    for (int candidate : candidates) {
        if (candidate <= n && static_cast<std::size_t>(candidate) <= max_cols_by_budget) {
            return candidate;
        }
    }
    return 1;
}

inline bool should_cache_a_residue_panels(int n, int col_block) {
    return col_block > 0 && n > 2 * col_block;
}

inline double elapsed_seconds(std::chrono::steady_clock::time_point begin,
                              std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

template <typename HighPrec>
inline HighPrec scale_cpp_int_pow2(const cpp_int &value, int exponent) {
    using boost::multiprecision::ldexp;
    HighPrec out(value);
    return ldexp(out, exponent);
}

} // namespace detail

inline GemmPlan make_gemm_plan(Operation op_a, Operation op_b,
                               int m, int n, int k,
                               const double *a, int lda,
                               const double *b, int ldb,
                               const Options &options = Options{}) {
    detail::validate_options(options);
    detail::validate_gemm_args(op_a, op_b, m, n, k, a, lda, b, ldb);

    std::vector<detail::VectorScale> rows(static_cast<std::size_t>(m));
    std::vector<detail::VectorScale> cols(static_cast<std::size_t>(n));

    for (int row = 0; row < m; ++row) {
        rows[row] = detail::analyze_vector(k, [&](int col) {
            return detail::read_a(op_a, a, lda, row, col);
        });
    }
    for (int col = 0; col < n; ++col) {
        cols[col] = detail::analyze_vector(k, [&](int row) {
            return detail::read_b(op_b, b, ldb, row, col);
        });
    }

    GemmPlan plan;
    plan.op_a = op_a;
    plan.op_b = op_b;
    plan.m = m;
    plan.n = n;
    plan.k = k;
    plan.crt_threads = options.crt_threads;
    plan.residue_col_block = options.residue_col_block;
    plan.residue_target_bytes = options.residue_target_bytes;
    plan.row_scale_exp.resize(static_cast<std::size_t>(m));
    plan.col_scale_exp.resize(static_cast<std::size_t>(n));
    plan.row_max_scaled_bits.resize(static_cast<std::size_t>(m));
    plan.col_max_scaled_bits.resize(static_cast<std::size_t>(n));
    int max_row_bits = 0;
    int max_col_bits = 0;
    for (int row = 0; row < m; ++row) {
        plan.row_scale_exp[row] = rows[row].scale_exp;
        plan.row_max_scaled_bits[row] = rows[row].max_scaled_bits;
        if (rows[row].max_scaled_bits > 0) {
            plan.row_scale_exp[row] += options.reuse_scale_slack_bits;
            plan.row_max_scaled_bits[row] += options.reuse_scale_slack_bits +
                                             options.reuse_magnitude_slack_bits;
        } else if (options.zero_vector_max_scaled_bits > 0) {
            plan.row_scale_exp[row] = options.zero_vector_scale_exp;
            plan.row_max_scaled_bits[row] = options.zero_vector_max_scaled_bits;
        }
        max_row_bits = std::max(max_row_bits, plan.row_max_scaled_bits[row]);
    }
    for (int col = 0; col < n; ++col) {
        plan.col_scale_exp[col] = cols[col].scale_exp;
        plan.col_max_scaled_bits[col] = cols[col].max_scaled_bits;
        if (cols[col].max_scaled_bits > 0) {
            plan.col_scale_exp[col] += options.reuse_scale_slack_bits;
            plan.col_max_scaled_bits[col] += options.reuse_scale_slack_bits +
                                             options.reuse_magnitude_slack_bits;
        } else if (options.zero_vector_max_scaled_bits > 0) {
            plan.col_scale_exp[col] = options.zero_vector_scale_exp;
            plan.col_max_scaled_bits[col] = options.zero_vector_max_scaled_bits;
        }
        max_col_bits = std::max(max_col_bits, plan.col_max_scaled_bits[col]);
    }

    plan.crt.exact_required_bits = max_row_bits + max_col_bits +
                                   detail::ceil_log2_int(std::max(1, k)) + 1;
    plan.crt.planned_bits = std::max(options.target_bits, plan.crt.exact_required_bits) +
                            options.guard_bits;
    plan.crt.moduli = detail::choose_moduli(k, plan.crt.planned_bits, options,
                                            &plan.crt.max_exact_modulus);
    detail::finalize_crt_plan(plan.crt);
    detail::build_pow2_residue_tables(plan);
    return plan;
}

inline Plan make_plan(Operation op_a, Operation op_b,
                      int m, int n, int k,
                      const double *a, int lda,
                      const double *b, int ldb,
                      const Options &options = Options{}) {
    return make_gemm_plan(op_a, op_b, m, n, k, a, lda, b, ldb, options).crt;
}

inline int effective_crt_threads(const GemmPlan &plan) {
    const int col_block =
        detail::choose_residue_col_block(plan.m,
                                         plan.n,
                                         plan.k,
                                         plan.crt.moduli.size(),
                                         plan.residue_col_block,
                                         plan.residue_target_bytes);
    return detail::choose_crt_threads(static_cast<std::size_t>(plan.m) * col_block,
                                      plan.crt_threads);
}

inline int effective_residue_col_block(const GemmPlan &plan) {
    return detail::choose_residue_col_block(plan.m,
                                            plan.n,
                                            plan.k,
                                            plan.crt.moduli.size(),
                                            plan.residue_col_block,
                                            plan.residue_target_bytes);
}

inline bool effective_a_residue_panel_cache(const GemmPlan &plan) {
    return detail::should_cache_a_residue_panels(plan.n, effective_residue_col_block(plan));
}

template <typename HighPrec>
inline void gemm_with_plan(const GemmPlan &plan,
                           const HighPrec &alpha,
                           const double *a, int lda,
                           const double *b, int ldb,
                           const HighPrec &beta,
                           HighPrec *c, int ldc,
                           GemmExecutionStats *stats = nullptr) {
    const auto total_begin = std::chrono::steady_clock::now();
    if (stats != nullptr) {
        *stats = GemmExecutionStats{};
        stats->moduli = plan.crt.moduli.size();
    }

    const Operation op_a = plan.op_a;
    const Operation op_b = plan.op_b;
    const int m = plan.m;
    const int n = plan.n;
    const int k = plan.k;

    detail::validate_gemm_args(op_a, op_b, m, n, k, a, lda, b, ldb);
    detail::validate_gemm_plan(plan, op_a, op_b, m, n, k);
    if (c == nullptr) {
        throw std::invalid_argument("C must be non-null");
    }
    if (ldc < std::max(1, m)) {
        throw std::invalid_argument("ldc is too small for C");
    }

    if (m == 0 || n == 0) {
        if (stats != nullptr) {
            stats->total_seconds =
                detail::elapsed_seconds(total_begin, std::chrono::steady_clock::now());
        }
        return;
    }
    if (k == 0) {
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                c[row + col * ldc] *= beta;
            }
        }
        if (stats != nullptr) {
            stats->total_seconds =
                detail::elapsed_seconds(total_begin, std::chrono::steady_clock::now());
        }
        return;
    }

    const int residue_col_block =
        detail::choose_residue_col_block(m,
                                         n,
                                         k,
                                         plan.crt.moduli.size(),
                                         plan.residue_col_block,
                                         plan.residue_target_bytes);
    const bool cache_a_residue_panels =
        detail::should_cache_a_residue_panels(n, residue_col_block);
    if (stats != nullptr) {
        stats->residue_col_block = residue_col_block;
        stats->a_residue_cached = cache_a_residue_panels;
    }
    std::vector<double> a_mod_scratch(static_cast<std::size_t>(m) * k);
    std::vector<double> b_mod(static_cast<std::size_t>(k) * residue_col_block);
    std::vector<double> c_mod(static_cast<std::size_t>(m) * residue_col_block);
    std::vector<detail::ScaledDouble> a_scaled(static_cast<std::size_t>(m) * k);
    std::vector<detail::ScaledDouble> b_scaled(static_cast<std::size_t>(k) * n);

    const auto input_prepare_begin = std::chrono::steady_clock::now();
    for (int col = 0; col < k; ++col) {
        for (int row = 0; row < m; ++row) {
            const double value = detail::read_a(op_a, a, lda, row, col);
            a_scaled[row + col * m] =
                detail::prepare_scaled_double_checked(
                    value, plan.row_scale_exp[row], plan.row_max_scaled_bits[row]);
        }
    }

    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < k; ++row) {
            const double value = detail::read_b(op_b, b, ldb, row, col);
            b_scaled[row + col * k] =
                detail::prepare_scaled_double_checked(
                    value, plan.col_scale_exp[col], plan.col_max_scaled_bits[col]);
        }
    }
    if (stats != nullptr) {
        stats->input_prepare_seconds +=
            detail::elapsed_seconds(input_prepare_begin, std::chrono::steady_clock::now());
    }

    auto build_a_mod_panel = [&](std::size_t imod, double *a_mod) {
        const int p = plan.crt.moduli[imod];
        const std::vector<int> &pow2_mod = plan.pow2_residues[imod];
        for (int col = 0; col < k; ++col) {
            for (int row = 0; row < m; ++row) {
                a_mod[row + col * m] = static_cast<double>(
                    detail::scaled_parts_centered_mod(
                        a_scaled[row + col * m].parts,
                        a_scaled[row + col * m].shift,
                        p,
                        pow2_mod));
            }
        }
    };

    std::vector<double> a_mod_panels;
    if (cache_a_residue_panels) {
        a_mod_panels.resize(plan.crt.moduli.size() * static_cast<std::size_t>(m) * k);
        const auto a_begin = std::chrono::steady_clock::now();
        for (std::size_t imod = 0; imod < plan.crt.moduli.size(); ++imod) {
            build_a_mod_panel(
                imod, a_mod_panels.data() + imod * static_cast<std::size_t>(m) * k);
        }
        if (stats != nullptr) {
            stats->a_residue_seconds +=
                detail::elapsed_seconds(a_begin, std::chrono::steady_clock::now());
        }
    }

    std::vector<int> all_residues(plan.crt.moduli.size() *
                                  static_cast<std::size_t>(m) * residue_col_block);
    for (int col_begin = 0; col_begin < n; col_begin += residue_col_block) {
        const int nb = std::min(residue_col_block, n - col_begin);
        const std::size_t block_output_size = static_cast<std::size_t>(m) * nb;
        if (stats != nullptr) {
            ++stats->output_blocks;
        }

        for (std::size_t imod = 0; imod < plan.crt.moduli.size(); ++imod) {
            const int p = plan.crt.moduli[imod];
            const std::vector<int> &pow2_mod = plan.pow2_residues[imod];
            const double *a_mod = nullptr;
            if (cache_a_residue_panels) {
                a_mod = a_mod_panels.data() + imod * static_cast<std::size_t>(m) * k;
            } else {
                const auto a_begin = std::chrono::steady_clock::now();
                build_a_mod_panel(imod, a_mod_scratch.data());
                if (stats != nullptr) {
                    stats->a_residue_seconds +=
                        detail::elapsed_seconds(a_begin, std::chrono::steady_clock::now());
                }
                a_mod = a_mod_scratch.data();
            }

            const auto b_begin = std::chrono::steady_clock::now();
            for (int local_col = 0; local_col < nb; ++local_col) {
                const int global_col = col_begin + local_col;
                for (int row = 0; row < k; ++row) {
                    b_mod[row + local_col * k] = static_cast<double>(
                        detail::scaled_parts_centered_mod(
                            b_scaled[row + global_col * k].parts,
                            b_scaled[row + global_col * k].shift,
                            p,
                            pow2_mod));
                }
            }
            if (stats != nullptr) {
                stats->b_residue_seconds +=
                    detail::elapsed_seconds(b_begin, std::chrono::steady_clock::now());
            }

            const auto blas_begin = std::chrono::steady_clock::now();
            detail::blas_dgemm_nn(m, nb, k,
                                  a_mod, m,
                                  b_mod.data(), k,
                                  c_mod.data(), m);
            if (stats != nullptr) {
                stats->blas_seconds +=
                    detail::elapsed_seconds(blas_begin, std::chrono::steady_clock::now());
                ++stats->residue_gemm_calls;
            }

            const auto store_begin = std::chrono::steady_clock::now();
            for (int local_col = 0; local_col < nb; ++local_col) {
                for (int row = 0; row < m; ++row) {
                    const std::size_t idx =
                        static_cast<std::size_t>(row) +
                        static_cast<std::size_t>(local_col) * m;
                    // The modulus bound makes every residue GEMM output an exact FP64 integer.
                    const auto rounded = static_cast<std::int64_t>(c_mod[idx]);
                    all_residues[imod * block_output_size + idx] =
                        detail::centered_mod_i64(rounded, p);
                }
            }
            if (stats != nullptr) {
                stats->residue_store_seconds +=
                    detail::elapsed_seconds(store_begin, std::chrono::steady_clock::now());
            }
        }

        const int crt_threads =
            detail::choose_crt_threads(block_output_size, plan.crt_threads);
        if (stats != nullptr) {
            stats->crt_threads = std::max(stats->crt_threads, crt_threads);
        }
        auto reconstruct_range = [&](std::size_t begin, std::size_t end) {
            for (std::size_t idx = begin; idx < end; ++idx) {
                const int row = static_cast<int>(idx % static_cast<std::size_t>(m));
                const int local_col = static_cast<int>(idx / static_cast<std::size_t>(m));
                const int global_col = col_begin + local_col;

                const detail::cpp_int crt =
                    detail::reconstruct_crt_centered_strided(
                        all_residues.data() + idx, block_output_size, plan.crt);
                const int restore_exp =
                    -(plan.row_scale_exp[row] + plan.col_scale_exp[global_col]);
                const HighPrec ab = detail::scale_cpp_int_pow2<HighPrec>(crt, restore_exp);
                const std::size_t out_idx =
                    static_cast<std::size_t>(row) +
                    static_cast<std::size_t>(global_col) * ldc;
                c[out_idx] = alpha * ab + beta * c[out_idx];
            }
        };

        const auto crt_begin = std::chrono::steady_clock::now();
        if (crt_threads == 1) {
            reconstruct_range(0, block_output_size);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(static_cast<std::size_t>(crt_threads - 1));
            for (int tid = 1; tid < crt_threads; ++tid) {
                const std::size_t begin = block_output_size * static_cast<std::size_t>(tid) /
                                          static_cast<std::size_t>(crt_threads);
                const std::size_t end =
                    block_output_size * static_cast<std::size_t>(tid + 1) /
                    static_cast<std::size_t>(crt_threads);
                workers.emplace_back(reconstruct_range, begin, end);
            }
            const std::size_t first_end =
                block_output_size / static_cast<std::size_t>(crt_threads);
            reconstruct_range(0, first_end);
            for (std::thread &worker : workers) {
                worker.join();
            }
        }
        if (stats != nullptr) {
            stats->crt_seconds +=
                detail::elapsed_seconds(crt_begin, std::chrono::steady_clock::now());
        }
    }
    if (stats != nullptr) {
        stats->total_seconds =
            detail::elapsed_seconds(total_begin, std::chrono::steady_clock::now());
    }
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
                 Plan *plan_out = nullptr,
                 GemmExecutionStats *stats = nullptr) {
    detail::validate_options(options);
    GemmPlan plan = make_gemm_plan(op_a, op_b, m, n, k, a, lda, b, ldb, options);
    if (plan_out != nullptr) {
        *plan_out = plan.crt;
    }
    gemm_with_plan(plan, alpha, a, lda, b, ldb, beta, c, ldc, stats);
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
