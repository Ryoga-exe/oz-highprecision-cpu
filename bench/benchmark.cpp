#include <oz_hp_cpu/gemm.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using hp_t = oz_hp_cpu::binary_float<256>;
using steady_clock_t = std::chrono::steady_clock;

struct Case {
    int m;
    int n;
    int k;
    bool run_naive;
};

double seconds_since(steady_clock_t::time_point begin, steady_clock_t::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

std::vector<double> random_matrix(int rows, int cols, int ld, std::mt19937_64 &rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> out(static_cast<std::size_t>(ld) * cols, 0.0);
    for (int col = 0; col < cols; ++col) {
        for (int row = 0; row < rows; ++row) {
            out[row + col * ld] = dist(rng);
        }
    }
    return out;
}

void blas_dgemm(int m, int n, int k,
                const double *a, int lda,
                const double *b, int ldb,
                double *c, int ldc) {
    const char trans = 'N';
    const double one = 1.0;
    const double zero = 0.0;
    dgemm_(&trans, &trans, &m, &n, &k,
           &one, a, &lda, b, &ldb, &zero, c, &ldc);
}

std::vector<hp_t> naive_hp_gemm(int m, int n, int k,
                                const std::vector<double> &a, int lda,
                                const std::vector<double> &b, int ldb) {
    std::vector<hp_t> c(static_cast<std::size_t>(m) * n, hp_t(0));
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            hp_t sum = 0;
            for (int kk = 0; kk < k; ++kk) {
                sum += hp_t(a[row + kk * lda]) * hp_t(b[kk + col * ldb]);
            }
            c[row + col * m] = sum;
        }
    }
    return c;
}

hp_t max_abs_diff_hp(const std::vector<hp_t> &x, const std::vector<hp_t> &y) {
    hp_t out = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const hp_t diff = abs(x[i] - y[i]);
        if (diff > out) {
            out = diff;
        }
    }
    return out;
}

hp_t max_abs_diff_fp64(const std::vector<double> &x, const std::vector<hp_t> &y) {
    hp_t out = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const hp_t diff = abs(hp_t(x[i]) - y[i]);
        if (diff > out) {
            out = diff;
        }
    }
    return out;
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    return values[mid];
}

double fraction_or_zero(double numerator, double denominator) {
    return denominator > 0.0 ? numerator / denominator : 0.0;
}

std::vector<Case> default_cases() {
    return {
        {8, 8, 8, true},
        {16, 16, 16, true},
        {32, 32, 32, true},
        {64, 64, 64, false},
        {96, 96, 96, false}
    };
}

std::vector<Case> quick_cases() {
    return {
        {8, 8, 8, true},
        {16, 16, 16, true},
        {32, 32, 32, true}
    };
}

std::vector<Case> parse_cases(int argc, char **argv) {
    if (argc <= 1) {
        return default_cases();
    }

    const std::string mode = argv[1];
    if (mode == "--quick") {
        return quick_cases();
    }
    if (mode == "--one") {
        if (argc != 5) {
            throw std::invalid_argument("usage: benchmark --one M N K");
        }
        return {{std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), true}};
    }
    throw std::invalid_argument(
        "usage: benchmark [--quick|--one M N K|--sweep-blocks M N K|"
        "--sweep-crt-threads M N K|--profile M N K]");
}

void run_block_sweep(int m, int n, int k) {
    std::mt19937_64 rng(0x61282024);
    oz_hp_cpu::Options base_options;
    base_options.target_bits = 256;
    base_options.guard_bits = 8;

    const int lda = m;
    const int ldb = k;
    const int ldc = m;
    std::vector<double> a = random_matrix(m, k, lda, rng);
    std::vector<double> b = random_matrix(k, n, ldb, rng);

    oz_hp_cpu::GemmPlan auto_plan =
        oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                  oz_hp_cpu::Operation::NoTrans,
                                  m, n, k,
                                  a.data(), lda,
                                  b.data(), ldb,
                                  base_options);
    std::vector<hp_t> c_auto(static_cast<std::size_t>(ldc) * n, hp_t(0));
    oz_hp_cpu::gemm_with_plan(auto_plan,
                              hp_t(1), a.data(), lda,
                              b.data(), ldb,
                              hp_t(0), c_auto.data(), ldc);

    std::cout << "m,n,k,requested_block,effective_block,a_residue_cached,"
              << "moduli,exact_required_bits,planned_bits,plan_seconds,"
              << "reuse_seconds,vs_auto_max_abs\n";
    std::cout << std::setprecision(12);

    const int requested_blocks[] = {0, 32, 64, 128, 256, 512};
    const int repeats = (m * n <= 4096) ? 3 : 1;
    for (int requested_block : requested_blocks) {
        if (requested_block > 0 && requested_block > n) {
            continue;
        }

        oz_hp_cpu::Options options = base_options;
        options.residue_col_block = requested_block;

        const auto plan_begin = steady_clock_t::now();
        oz_hp_cpu::GemmPlan plan =
            oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                      oz_hp_cpu::Operation::NoTrans,
                                      m, n, k,
                                      a.data(), lda,
                                      b.data(), ldb,
                                      options);
        const auto plan_end = steady_clock_t::now();

        std::vector<hp_t> c(static_cast<std::size_t>(ldc) * n, hp_t(0));
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            std::fill(c.begin(), c.end(), hp_t(0));
            const auto begin = steady_clock_t::now();
            oz_hp_cpu::gemm_with_plan(plan,
                                      hp_t(1), a.data(), lda,
                                      b.data(), ldb,
                                      hp_t(0), c.data(), ldc);
            const auto end = steady_clock_t::now();
            samples.push_back(seconds_since(begin, end));
        }

        std::cout << m << ','
                  << n << ','
                  << k << ','
                  << requested_block << ','
                  << oz_hp_cpu::effective_residue_col_block(plan) << ','
                  << (oz_hp_cpu::effective_a_residue_panel_cache(plan) ? 1 : 0) << ','
                  << plan.crt.moduli.size() << ','
                  << plan.crt.exact_required_bits << ','
                  << plan.crt.planned_bits << ','
                  << seconds_since(plan_begin, plan_end) << ','
                  << median(samples) << ','
                  << max_abs_diff_hp(c, c_auto) << '\n';
    }
}

void run_crt_thread_sweep(int m, int n, int k) {
    std::mt19937_64 rng(0x61282024);
    oz_hp_cpu::Options base_options;
    base_options.target_bits = 256;
    base_options.guard_bits = 8;

    const int lda = m;
    const int ldb = k;
    const int ldc = m;
    std::vector<double> a = random_matrix(m, k, lda, rng);
    std::vector<double> b = random_matrix(k, n, ldb, rng);

    oz_hp_cpu::GemmPlan auto_plan =
        oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                  oz_hp_cpu::Operation::NoTrans,
                                  m, n, k,
                                  a.data(), lda,
                                  b.data(), ldb,
                                  base_options);
    std::vector<hp_t> c_auto(static_cast<std::size_t>(ldc) * n, hp_t(0));
    oz_hp_cpu::gemm_with_plan(auto_plan,
                              hp_t(1), a.data(), lda,
                              b.data(), ldb,
                              hp_t(0), c_auto.data(), ldc);

    std::cout << "m,n,k,requested_threads,effective_threads,moduli,"
              << "residue_col_block,a_residue_cached,total_seconds,crt_seconds,"
              << "crt_fraction,vs_auto_max_abs\n";
    std::cout << std::setprecision(12);

    const int requested_threads[] = {0, 1, 2, 4, 8};
    const int repeats = (m * n <= 65536) ? 3 : 1;
    for (int requested_thread : requested_threads) {
        oz_hp_cpu::Options options = base_options;
        options.crt_threads = requested_thread;
        oz_hp_cpu::GemmPlan plan =
            oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                      oz_hp_cpu::Operation::NoTrans,
                                      m, n, k,
                                      a.data(), lda,
                                      b.data(), ldb,
                                      options);

        std::vector<hp_t> c(static_cast<std::size_t>(ldc) * n, hp_t(0));
        std::vector<oz_hp_cpu::GemmExecutionStats> stats_samples;
        stats_samples.reserve(static_cast<std::size_t>(repeats));
        for (int repeat = 0; repeat < repeats; ++repeat) {
            std::fill(c.begin(), c.end(), hp_t(0));
            oz_hp_cpu::GemmExecutionStats stats;
            oz_hp_cpu::gemm_with_plan(plan,
                                      hp_t(1), a.data(), lda,
                                      b.data(), ldb,
                                      hp_t(0), c.data(), ldc,
                                      &stats);
            stats_samples.push_back(stats);
        }

        auto median_stat = [&](double oz_hp_cpu::GemmExecutionStats::*field) {
            std::vector<double> values;
            values.reserve(stats_samples.size());
            for (const oz_hp_cpu::GemmExecutionStats &stats : stats_samples) {
                values.push_back(stats.*field);
            }
            return median(values);
        };

        const double total_seconds = median_stat(&oz_hp_cpu::GemmExecutionStats::total_seconds);
        const double crt_seconds = median_stat(&oz_hp_cpu::GemmExecutionStats::crt_seconds);
        const oz_hp_cpu::GemmExecutionStats &last = stats_samples.back();

        std::cout << m << ','
                  << n << ','
                  << k << ','
                  << requested_thread << ','
                  << last.crt_threads << ','
                  << plan.crt.moduli.size() << ','
                  << last.residue_col_block << ','
                  << (last.a_residue_cached ? 1 : 0) << ','
                  << total_seconds << ','
                  << crt_seconds << ','
                  << fraction_or_zero(crt_seconds, total_seconds) << ','
                  << max_abs_diff_hp(c, c_auto) << '\n';
    }
}

void run_profile(int m, int n, int k) {
    std::mt19937_64 rng(0x61282024);
    oz_hp_cpu::Options options;
    options.target_bits = 256;
    options.guard_bits = 8;

    const int lda = m;
    const int ldb = k;
    const int ldc = m;
    std::vector<double> a = random_matrix(m, k, lda, rng);
    std::vector<double> b = random_matrix(k, n, ldb, rng);

    const auto plan_begin = steady_clock_t::now();
    const oz_hp_cpu::GemmPlan plan =
        oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                  oz_hp_cpu::Operation::NoTrans,
                                  m, n, k,
                                  a.data(), lda,
                                  b.data(), ldb,
                                  options);
    const auto plan_end = steady_clock_t::now();

    const int repeats = (m * n <= 65536) ? 3 : 1;
    std::vector<oz_hp_cpu::GemmExecutionStats> stats_samples;
    stats_samples.reserve(static_cast<std::size_t>(repeats));
    std::vector<hp_t> c(static_cast<std::size_t>(ldc) * n, hp_t(0));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        std::fill(c.begin(), c.end(), hp_t(0));
        oz_hp_cpu::GemmExecutionStats stats;
        oz_hp_cpu::gemm_with_plan(plan,
                                  hp_t(1), a.data(), lda,
                                  b.data(), ldb,
                                  hp_t(0), c.data(), ldc,
                                  &stats);
        stats_samples.push_back(stats);
    }

    auto median_stat = [&](double oz_hp_cpu::GemmExecutionStats::*field) {
        std::vector<double> values;
        values.reserve(stats_samples.size());
        for (const oz_hp_cpu::GemmExecutionStats &stats : stats_samples) {
            values.push_back(stats.*field);
        }
        return median(values);
    };

    const oz_hp_cpu::GemmExecutionStats &last = stats_samples.back();
    const double total_seconds = median_stat(&oz_hp_cpu::GemmExecutionStats::total_seconds);
    const double input_prepare_seconds =
        median_stat(&oz_hp_cpu::GemmExecutionStats::input_prepare_seconds);
    const double a_residue_seconds =
        median_stat(&oz_hp_cpu::GemmExecutionStats::a_residue_seconds);
    const double b_residue_seconds =
        median_stat(&oz_hp_cpu::GemmExecutionStats::b_residue_seconds);
    const double blas_seconds = median_stat(&oz_hp_cpu::GemmExecutionStats::blas_seconds);
    const double residue_store_seconds =
        median_stat(&oz_hp_cpu::GemmExecutionStats::residue_store_seconds);
    const double crt_seconds = median_stat(&oz_hp_cpu::GemmExecutionStats::crt_seconds);
    const double residue_seconds =
        a_residue_seconds + b_residue_seconds + residue_store_seconds;
    const double attributed_seconds =
        input_prepare_seconds + residue_seconds + blas_seconds + crt_seconds;
    const double unattributed_seconds = std::max(0.0, total_seconds - attributed_seconds);

    std::cout << "m,n,k,moduli,exact_required_bits,planned_bits,plan_seconds,"
              << "total_seconds,input_prepare_seconds,a_residue_seconds,"
              << "b_residue_seconds,blas_seconds,residue_store_seconds,crt_seconds,"
              << "unattributed_seconds,residue_fraction,blas_fraction,crt_fraction,"
              << "residue_col_block,a_residue_cached,crt_threads,output_blocks,"
              << "residue_gemm_calls\n";
    std::cout << std::setprecision(12);
    std::cout << m << ','
              << n << ','
              << k << ','
              << plan.crt.moduli.size() << ','
              << plan.crt.exact_required_bits << ','
              << plan.crt.planned_bits << ','
              << seconds_since(plan_begin, plan_end) << ','
              << total_seconds << ','
              << input_prepare_seconds << ','
              << a_residue_seconds << ','
              << b_residue_seconds << ','
              << blas_seconds << ','
              << residue_store_seconds << ','
              << crt_seconds << ','
              << unattributed_seconds << ','
              << fraction_or_zero(residue_seconds, total_seconds) << ','
              << fraction_or_zero(blas_seconds, total_seconds) << ','
              << fraction_or_zero(crt_seconds, total_seconds) << ','
              << last.residue_col_block << ','
              << (last.a_residue_cached ? 1 : 0) << ','
              << last.crt_threads << ','
              << last.output_blocks << ','
              << last.residue_gemm_calls << '\n';
}

} // namespace

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "--sweep-blocks") {
        if (argc != 5) {
            throw std::invalid_argument("usage: benchmark --sweep-blocks M N K");
        }
        run_block_sweep(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]));
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--sweep-crt-threads") {
        if (argc != 5) {
            throw std::invalid_argument("usage: benchmark --sweep-crt-threads M N K");
        }
        run_crt_thread_sweep(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]));
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--profile") {
        if (argc != 5) {
            throw std::invalid_argument("usage: benchmark --profile M N K");
        }
        run_profile(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]));
        return 0;
    }

    const std::vector<Case> cases = parse_cases(argc, argv);
    std::mt19937_64 rng(0x61282024);

    oz_hp_cpu::Options options;
    options.target_bits = 256;
    options.guard_bits = 8;

    std::cout << "m,n,k,moduli,exact_required_bits,planned_bits,"
              << "max_exact_modulus_bound,selected_max_modulus,"
              << "fp64_seconds,oz_seconds,plan_seconds,crt_threads,residue_col_block,"
              << "a_residue_cached,"
              << "oz_reuse_seconds,oz_reuse_serial_seconds,"
              << "naive_hp_seconds,oz_vs_naive_max_abs,fp64_vs_oz_max_abs,"
              << "reuse_vs_oz_max_abs\n";
    std::cout << std::setprecision(12);

    for (const Case &tc : cases) {
        const int lda = tc.m;
        const int ldb = tc.k;
        const int ldc = tc.m;
        std::vector<double> a = random_matrix(tc.m, tc.k, lda, rng);
        std::vector<double> b = random_matrix(tc.k, tc.n, ldb, rng);
        std::vector<double> c_fp64(static_cast<std::size_t>(ldc) * tc.n, 0.0);
        std::vector<hp_t> c_oz(static_cast<std::size_t>(ldc) * tc.n, hp_t(0));
        std::vector<hp_t> c_naive;

        const auto fp64_begin = steady_clock_t::now();
        blas_dgemm(tc.m, tc.n, tc.k,
                   a.data(), lda,
                   b.data(), ldb,
                   c_fp64.data(), ldc);
        const auto fp64_end = steady_clock_t::now();

        oz_hp_cpu::Plan plan;
        const auto oz_begin = steady_clock_t::now();
        oz_hp_cpu::gemm(oz_hp_cpu::Operation::NoTrans,
                        oz_hp_cpu::Operation::NoTrans,
                        tc.m, tc.n, tc.k,
                        hp_t(1), a.data(), lda,
                        b.data(), ldb,
                        hp_t(0), c_oz.data(), ldc,
                        options, &plan);
        const auto oz_end = steady_clock_t::now();

        const auto plan_begin = steady_clock_t::now();
        const oz_hp_cpu::GemmPlan reusable_plan =
            oz_hp_cpu::make_gemm_plan(oz_hp_cpu::Operation::NoTrans,
                                      oz_hp_cpu::Operation::NoTrans,
                                      tc.m, tc.n, tc.k,
                                      a.data(), lda,
                                      b.data(), ldb,
                                      options);
        const auto plan_end = steady_clock_t::now();

        std::vector<hp_t> c_reuse(static_cast<std::size_t>(ldc) * tc.n, hp_t(0));
        std::vector<double> reuse_samples;
        const int reuse_repeats = (tc.m <= 32) ? 3 : 1;
        reuse_samples.reserve(static_cast<std::size_t>(reuse_repeats));
        for (int repeat = 0; repeat < reuse_repeats; ++repeat) {
            std::fill(c_reuse.begin(), c_reuse.end(), hp_t(0));
            const auto reuse_begin = steady_clock_t::now();
            oz_hp_cpu::gemm_with_plan(reusable_plan,
                                      hp_t(1), a.data(), lda,
                                      b.data(), ldb,
                                      hp_t(0), c_reuse.data(), ldc);
            const auto reuse_end = steady_clock_t::now();
            reuse_samples.push_back(seconds_since(reuse_begin, reuse_end));
        }
        const double reuse_seconds = median(reuse_samples);

        oz_hp_cpu::GemmPlan serial_plan = reusable_plan;
        serial_plan.crt_threads = 1;
        std::vector<hp_t> c_reuse_serial(static_cast<std::size_t>(ldc) * tc.n, hp_t(0));
        std::vector<double> reuse_serial_samples;
        reuse_serial_samples.reserve(static_cast<std::size_t>(reuse_repeats));
        for (int repeat = 0; repeat < reuse_repeats; ++repeat) {
            std::fill(c_reuse_serial.begin(), c_reuse_serial.end(), hp_t(0));
            const auto reuse_begin = steady_clock_t::now();
            oz_hp_cpu::gemm_with_plan(serial_plan,
                                      hp_t(1), a.data(), lda,
                                      b.data(), ldb,
                                      hp_t(0), c_reuse_serial.data(), ldc);
            const auto reuse_end = steady_clock_t::now();
            reuse_serial_samples.push_back(seconds_since(reuse_begin, reuse_end));
        }
        const double reuse_serial_seconds = median(reuse_serial_samples);

        double naive_seconds = -1.0;
        hp_t oz_vs_naive = -1;
        if (tc.run_naive) {
            const auto naive_begin = steady_clock_t::now();
            c_naive = naive_hp_gemm(tc.m, tc.n, tc.k, a, lda, b, ldb);
            const auto naive_end = steady_clock_t::now();
            naive_seconds = seconds_since(naive_begin, naive_end);
            oz_vs_naive = max_abs_diff_hp(c_oz, c_naive);
        }

        const hp_t fp64_vs_oz = max_abs_diff_fp64(c_fp64, c_oz);
        const hp_t reuse_vs_oz = max_abs_diff_hp(c_reuse, c_oz);

        const int selected_max_modulus = plan.moduli.empty() ? 0 : plan.moduli.front();
        const int crt_threads = oz_hp_cpu::effective_crt_threads(reusable_plan);
        const int residue_col_block = oz_hp_cpu::effective_residue_col_block(reusable_plan);
        const int a_residue_cached =
            oz_hp_cpu::effective_a_residue_panel_cache(reusable_plan) ? 1 : 0;

        std::cout << tc.m << ','
                  << tc.n << ','
                  << tc.k << ','
                  << plan.moduli.size() << ','
                  << plan.exact_required_bits << ','
                  << plan.planned_bits << ','
                  << plan.max_exact_modulus << ','
                  << selected_max_modulus << ','
                  << seconds_since(fp64_begin, fp64_end) << ','
                  << seconds_since(oz_begin, oz_end) << ','
                  << seconds_since(plan_begin, plan_end) << ','
                  << crt_threads << ','
                  << residue_col_block << ','
                  << a_residue_cached << ','
                  << reuse_seconds << ','
                  << reuse_serial_seconds << ','
                  << naive_seconds << ','
                  << oz_vs_naive << ','
                  << fp64_vs_oz << ','
                  << reuse_vs_oz << '\n';
    }

    return 0;
}
