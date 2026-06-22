#include <oz_hp_cpu/gemm.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using hp_t = oz_hp_cpu::binary_float<256>;

double read_matrix(oz_hp_cpu::Operation op,
                   const std::vector<double> &a,
                   int lda,
                   int row,
                   int col) {
    return (op == oz_hp_cpu::Operation::NoTrans) ? a[row + col * lda] : a[col + row * lda];
}

void reference_gemm(oz_hp_cpu::Operation op_a,
                    oz_hp_cpu::Operation op_b,
                    int m,
                    int n,
                    int k,
                    const hp_t &alpha,
                    const std::vector<double> &a,
                    int lda,
                    const std::vector<double> &b,
                    int ldb,
                    const hp_t &beta,
                    std::vector<hp_t> &c,
                    int ldc) {
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            hp_t sum = 0;
            for (int kk = 0; kk < k; ++kk) {
                sum += hp_t(read_matrix(op_a, a, lda, row, kk)) *
                       hp_t(read_matrix(op_b, b, ldb, kk, col));
            }
            c[row + col * ldc] = alpha * sum + beta * c[row + col * ldc];
        }
    }
}

std::vector<double> random_matrix(int rows,
                                  int cols,
                                  int ld,
                                  std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> out(static_cast<std::size_t>(ld) * cols, 0.0);
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            out[row + col * ld] = dist(rng);
        }
    }
    return out;
}

void check_close(const std::string &name,
                 const std::vector<hp_t> &got,
                 const std::vector<hp_t> &want,
                 int m,
                 int n,
                 int ldc) {
    hp_t max_abs = 0;
    hp_t max_rel = 0;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            const hp_t abs_err = abs(got[row + col * ldc] - want[row + col * ldc]);
            const hp_t want_abs = abs(want[row + col * ldc]);
            const hp_t denom = (want_abs > hp_t(1)) ? want_abs : hp_t(1);
            const hp_t rel_err = abs_err / denom;
            if (abs_err > max_abs) {
                max_abs = abs_err;
            }
            if (rel_err > max_rel) {
                max_rel = rel_err;
            }
        }
    }

    const hp_t tol("1e-60");
    if (max_abs > tol && max_rel > tol) {
        std::ostringstream oss;
        oss << name << " failed: max_abs=" << max_abs << " max_rel=" << max_rel;
        throw std::runtime_error(oss.str());
    }
    std::cout << name << " ok: max_abs=" << max_abs
              << " max_rel=" << max_rel << '\n';
}

void run_random_case(const std::string &name,
                     oz_hp_cpu::Operation op_a,
                     oz_hp_cpu::Operation op_b,
                     int m,
                     int n,
                     int k,
                     std::mt19937_64 &rng) {
    const int a_rows = (op_a == oz_hp_cpu::Operation::NoTrans) ? m : k;
    const int a_cols = (op_a == oz_hp_cpu::Operation::NoTrans) ? k : m;
    const int b_rows = (op_b == oz_hp_cpu::Operation::NoTrans) ? k : n;
    const int b_cols = (op_b == oz_hp_cpu::Operation::NoTrans) ? n : k;
    const int lda = a_rows + 2;
    const int ldb = b_rows + 3;
    const int ldc = m + 1;

    std::vector<double> a = random_matrix(a_rows, a_cols, lda, rng);
    std::vector<double> b = random_matrix(b_rows, b_cols, ldb, rng);
    std::vector<hp_t> c(static_cast<std::size_t>(ldc) * n, hp_t(0));
    std::vector<hp_t> cref = c;

    oz_hp_cpu::Options options;
    options.target_bits = 256;
    oz_hp_cpu::Plan plan;
    oz_hp_cpu::gemm(op_a, op_b,
                    m, n, k,
                    hp_t(1), a.data(), lda,
                    b.data(), ldb,
                    hp_t(0), c.data(), ldc,
                    options, &plan);

    const oz_hp_cpu::GemmPlan reusable_plan =
        oz_hp_cpu::make_gemm_plan(op_a, op_b,
                                  m, n, k,
                                  a.data(), lda,
                                  b.data(), ldb,
                                  options);
    std::vector<hp_t> c_reuse(static_cast<std::size_t>(ldc) * n, hp_t(0));
    oz_hp_cpu::gemm_with_plan(reusable_plan,
                              hp_t(1), a.data(), lda,
                              b.data(), ldb,
                              hp_t(0), c_reuse.data(), ldc);

    reference_gemm(op_a, op_b,
                   m, n, k,
                   hp_t(1), a, lda,
                   b, ldb,
                   hp_t(0), cref, ldc);
    std::cout << name << " plan: moduli=" << plan.moduli.size()
              << " exact_required_bits=" << plan.exact_required_bits
              << " planned_bits=" << plan.planned_bits
              << " max_exact_modulus_bound=" << plan.max_exact_modulus << '\n';
    check_close(name, c, cref, m, n, ldc);
    check_close(name + " reusable plan", c_reuse, cref, m, n, ldc);
}

void run_cancellation_case() {
    constexpr int m = 1;
    constexpr int n = 1;
    constexpr int k = 3;
    constexpr int lda = m;
    constexpr int ldb = k;
    constexpr int ldc = m;

    const std::vector<double> a = {1.0e16, 1.0, -1.0e16};
    const std::vector<double> b = {1.0, 1.0, 1.0};
    std::vector<hp_t> c(1, hp_t(0));

    oz_hp_cpu::Options options;
    options.target_bits = 256;
    oz_hp_cpu::Plan plan;
    oz_hp_cpu::gemm(oz_hp_cpu::Operation::NoTrans,
                    oz_hp_cpu::Operation::NoTrans,
                    m, n, k,
                    hp_t(1), a.data(), lda,
                    b.data(), ldb,
                    hp_t(0), c.data(), ldc,
                    options, &plan);

    const volatile double fp64_left = (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];
    if (c[0] != hp_t(1)) {
        std::ostringstream oss;
        oss << "cancellation failed: got " << c[0];
        throw std::runtime_error(oss.str());
    }

    std::cout << "cancellation ok: high_precision=" << c[0]
              << " fp64_left_to_right=" << fp64_left
              << " moduli=" << plan.moduli.size() << '\n';
}

void run_plan_reject_case() {
    constexpr int m = 1;
    constexpr int n = 1;
    constexpr int k = 1;
    constexpr int lda = m;
    constexpr int ldb = k;
    constexpr int ldc = m;

    const std::vector<double> a = {1.0};
    const std::vector<double> b = {1.0};
    const oz_hp_cpu::GemmPlan plan =
        oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                  oz_hp_cpu::Operation::NoTrans,
                                  m, n, k,
                                  a.data(), lda,
                                  b.data(), ldb);

    const std::vector<double> finer_a = {0.5};
    std::vector<hp_t> c(1, hp_t(0));
    bool rejected = false;
    try {
        oz_hp_cpu::gemm_with_plan(plan,
                                  hp_t(1), finer_a.data(), lda,
                                  b.data(), ldb,
                                  hp_t(0), c.data(), ldc);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("plan reuse accepted incompatible input");
    }
    std::cout << "plan reject ok\n";
}

void run_plan_slack_case() {
    constexpr int m = 1;
    constexpr int n = 1;
    constexpr int k = 1;
    constexpr int lda = m;
    constexpr int ldb = k;
    constexpr int ldc = m;

    const std::vector<double> a = {1.0};
    const std::vector<double> b = {1.0};
    oz_hp_cpu::Options options;
    options.reuse_scale_slack_bits = 1;
    options.reuse_magnitude_slack_bits = 1;
    const oz_hp_cpu::GemmPlan plan =
        oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                  oz_hp_cpu::Operation::NoTrans,
                                  m, n, k,
                                  a.data(), lda,
                                  b.data(), ldb,
                                  options);

    const std::vector<double> reuse_a = {0.5};
    const std::vector<double> reuse_b = {2.0};
    std::vector<hp_t> c(1, hp_t(0));
    std::vector<hp_t> cref(1, hp_t(0));
    oz_hp_cpu::gemm_with_plan(plan,
                              hp_t(1), reuse_a.data(), lda,
                              reuse_b.data(), ldb,
                              hp_t(0), c.data(), ldc);
    reference_gemm(oz_hp_cpu::Operation::NoTrans,
                   oz_hp_cpu::Operation::NoTrans,
                   m, n, k,
                   hp_t(1), reuse_a, lda,
                   reuse_b, ldb,
                   hp_t(0), cref, ldc);
    check_close("plan slack", c, cref, m, n, ldc);
}

void run_zero_vector_reuse_case() {
    constexpr int m = 1;
    constexpr int n = 1;
    constexpr int k = 1;
    constexpr int lda = m;
    constexpr int ldb = k;
    constexpr int ldc = m;

    const std::vector<double> zero_a = {0.0};
    const std::vector<double> zero_b = {0.0};
    const std::vector<double> reuse_a = {0.5};
    const std::vector<double> reuse_b = {1.0};

    const oz_hp_cpu::GemmPlan strict_plan =
        oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                  oz_hp_cpu::Operation::NoTrans,
                                  m, n, k,
                                  zero_a.data(), lda,
                                  zero_b.data(), ldb);

    std::vector<hp_t> c(1, hp_t(0));
    bool rejected = false;
    try {
        oz_hp_cpu::gemm_with_plan(strict_plan,
                                  hp_t(1), reuse_a.data(), lda,
                                  reuse_b.data(), ldb,
                                  hp_t(0), c.data(), ldc);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error("zero-vector plan accepted nonzero input without reserve");
    }

    oz_hp_cpu::Options options;
    options.zero_vector_scale_exp = 1;
    options.zero_vector_max_scaled_bits = 2;
    const oz_hp_cpu::GemmPlan reserved_plan =
        oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                  oz_hp_cpu::Operation::NoTrans,
                                  m, n, k,
                                  zero_a.data(), lda,
                                  zero_b.data(), ldb,
                                  options);

    std::vector<hp_t> cref(1, hp_t(0));
    c[0] = hp_t(0);
    oz_hp_cpu::gemm_with_plan(reserved_plan,
                              hp_t(1), reuse_a.data(), lda,
                              reuse_b.data(), ldb,
                              hp_t(0), c.data(), ldc);
    reference_gemm(oz_hp_cpu::Operation::NoTrans,
                   oz_hp_cpu::Operation::NoTrans,
                   m, n, k,
                   hp_t(1), reuse_a, lda,
                   reuse_b, ldb,
                   hp_t(0), cref, ldc);
    check_close("zero vector reuse", c, cref, m, n, ldc);
}

void run_reuse_policy_case() {
    constexpr int m = 1;
    constexpr int n = 1;
    constexpr int k = 1;
    constexpr int lda = m;
    constexpr int ldb = k;
    constexpr int ldc = m;

    oz_hp_cpu::Options options;
    options.target_bits = 192;
    options.crt_threads = 1;
    options = oz_hp_cpu::with_reuse_policy(options, oz_hp_cpu::PlanReusePolicy::Moderate);
    if (options.target_bits != 192 || options.crt_threads != 1 ||
        options.reuse_scale_slack_bits <= 0 ||
        options.reuse_magnitude_slack_bits <= 0 ||
        options.zero_vector_scale_exp <= 0 ||
        options.zero_vector_max_scaled_bits <= 0) {
        throw std::runtime_error("reuse policy did not set expected options");
    }

    const std::vector<double> zero_a = {0.0};
    const std::vector<double> b = {1.0};
    const oz_hp_cpu::GemmPlan plan =
        oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                  oz_hp_cpu::Operation::NoTrans,
                                  m, n, k,
                                  zero_a.data(), lda,
                                  b.data(), ldb,
                                  options);

    const std::vector<double> reuse_a = {std::ldexp(1.0, -8)};
    std::vector<hp_t> c(1, hp_t(0));
    std::vector<hp_t> cref(1, hp_t(0));
    oz_hp_cpu::gemm_with_plan(plan,
                              hp_t(1), reuse_a.data(), lda,
                              b.data(), ldb,
                              hp_t(0), c.data(), ldc);
    reference_gemm(oz_hp_cpu::Operation::NoTrans,
                   oz_hp_cpu::Operation::NoTrans,
                   m, n, k,
                   hp_t(1), reuse_a, lda,
                   b, ldb,
                   hp_t(0), cref, ldc);
    check_close("reuse policy", c, cref, m, n, ldc);
}

void run_wide_exponent_case() {
    constexpr int m = 1;
    constexpr int n = 1;
    constexpr int k = 2;
    constexpr int lda = m;
    constexpr int ldb = k;
    constexpr int ldc = m;

    const std::vector<double> a = {std::ldexp(1.0, 80), std::ldexp(1.0, -80)};
    const std::vector<double> b = {1.0, 1.0};
    std::vector<hp_t> c(1, hp_t(0));
    std::vector<hp_t> cref(1, hp_t(0));

    const oz_hp_cpu::GemmPlan plan =
        oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                  oz_hp_cpu::Operation::NoTrans,
                                  m, n, k,
                                  a.data(), lda,
                                  b.data(), ldb);
    if (plan.max_pow2_shift < 160 || plan.pow2_residues.empty()) {
        throw std::runtime_error("wide exponent case did not build pow2 residue tables");
    }

    oz_hp_cpu::gemm_with_plan(plan,
                              hp_t(1), a.data(), lda,
                              b.data(), ldb,
                              hp_t(0), c.data(), ldc);
    reference_gemm(oz_hp_cpu::Operation::NoTrans,
                   oz_hp_cpu::Operation::NoTrans,
                   m, n, k,
                   hp_t(1), a, lda,
                   b, ldb,
                   hp_t(0), cref, ldc);
    check_close("wide exponent", c, cref, m, n, ldc);
}

void run_parallel_crt_case(std::mt19937_64 &rng) {
    constexpr int m = 65;
    constexpr int n = 64;
    constexpr int k = 3;
    constexpr int lda = m;
    constexpr int ldb = k;
    constexpr int ldc = m;

    std::vector<double> a = random_matrix(m, k, lda, rng);
    std::vector<double> b = random_matrix(k, n, ldb, rng);
    std::vector<hp_t> c(static_cast<std::size_t>(ldc) * n, hp_t(0));
    std::vector<hp_t> cref = c;

    oz_hp_cpu::Options options;
    options.crt_threads = 4;
    const oz_hp_cpu::GemmPlan plan =
        oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                  oz_hp_cpu::Operation::NoTrans,
                                  m, n, k,
                                  a.data(), lda,
                                  b.data(), ldb,
                                  options);

    oz_hp_cpu::gemm_with_plan(plan,
                              hp_t(1), a.data(), lda,
                              b.data(), ldb,
                              hp_t(0), c.data(), ldc);
    reference_gemm(oz_hp_cpu::Operation::NoTrans,
                   oz_hp_cpu::Operation::NoTrans,
                   m, n, k,
                   hp_t(1), a, lda,
                   b, ldb,
                   hp_t(0), cref, ldc);
    check_close("parallel crt", c, cref, m, n, ldc);
}

void run_blocked_residue_case(std::mt19937_64 &rng) {
    constexpr int m = 7;
    constexpr int n = 5;
    constexpr int k = 6;
    constexpr int lda = m;
    constexpr int ldb = k;
    constexpr int ldc = m;

    std::vector<double> a = random_matrix(m, k, lda, rng);
    std::vector<double> b = random_matrix(k, n, ldb, rng);
    std::vector<hp_t> c(static_cast<std::size_t>(ldc) * n, hp_t(0));
    std::vector<hp_t> cref = c;

    oz_hp_cpu::Options options;
    options.residue_col_block = 2;
    const oz_hp_cpu::GemmPlan plan =
        oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                  oz_hp_cpu::Operation::NoTrans,
                                  m, n, k,
                                  a.data(), lda,
                                  b.data(), ldb,
                                  options);
    if (oz_hp_cpu::effective_residue_col_block(plan) != 2) {
        throw std::runtime_error("blocked residue case did not use requested block size");
    }
    if (!oz_hp_cpu::effective_a_residue_panel_cache(plan)) {
        throw std::runtime_error("blocked residue case did not enable A residue panel cache");
    }

    oz_hp_cpu::gemm_with_plan(plan,
                              hp_t(1), a.data(), lda,
                              b.data(), ldb,
                              hp_t(0), c.data(), ldc);
    reference_gemm(oz_hp_cpu::Operation::NoTrans,
                   oz_hp_cpu::Operation::NoTrans,
                   m, n, k,
                   hp_t(1), a, lda,
                   b, ldb,
                   hp_t(0), cref, ldc);
    check_close("blocked residue", c, cref, m, n, ldc);
}

} // namespace

int main() {
    std::mt19937_64 rng(0x515151);

    run_cancellation_case();
    run_plan_reject_case();
    run_plan_slack_case();
    run_zero_vector_reuse_case();
    run_reuse_policy_case();
    run_wide_exponent_case();
    run_parallel_crt_case(rng);
    run_blocked_residue_case(rng);
    run_random_case("nn", oz_hp_cpu::Operation::NoTrans,
                    oz_hp_cpu::Operation::NoTrans, 5, 4, 7, rng);
    run_random_case("tn", oz_hp_cpu::Operation::Trans,
                    oz_hp_cpu::Operation::NoTrans, 4, 6, 5, rng);
    run_random_case("nt", oz_hp_cpu::Operation::NoTrans,
                    oz_hp_cpu::Operation::Trans, 6, 5, 4, rng);

    return 0;
}
