#include <oz_hp_cpu/gemm.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

int main() {
    using hp_t = oz_hp_cpu::binary_float<256>;

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

    const double fp64_left_to_right = (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2];

    std::cout << std::setprecision(80);
    std::cout << "high precision: " << c[0] << '\n';
    std::cout << "fp64 left-to-right: " << fp64_left_to_right << '\n';
    std::cout << "moduli: " << plan.moduli.size()
              << ", exact_required_bits: " << plan.exact_required_bits
              << ", planned_bits: " << plan.planned_bits << '\n';
}
