CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS ?= -Iinclude
BLAS_LIBS ?= $(shell if ldconfig -p 2>/dev/null | grep -q libopenblas.so; then echo -lopenblas; elif [ -e /usr/lib/x86_64-linux-gnu/libblas.so ]; then echo -lblas; elif [ -e /lib/x86_64-linux-gnu/libblas.so.3 ]; then echo /lib/x86_64-linux-gnu/libblas.so.3; else echo -lblas; fi)

BUILD_DIR := build
TEST_BIN := $(BUILD_DIR)/test_gemm
EXAMPLE_BIN := $(BUILD_DIR)/cancellation
BENCH_BIN := $(BUILD_DIR)/benchmark

.PHONY: all test example bench clean

all: test example

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_BIN): tests/test_gemm.cpp include/oz_hp_cpu/gemm.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BLAS_LIBS) -o $@

$(EXAMPLE_BIN): examples/cancellation.cpp include/oz_hp_cpu/gemm.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BLAS_LIBS) -o $@

$(BENCH_BIN): bench/benchmark.cpp include/oz_hp_cpu/gemm.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< $(BLAS_LIBS) -o $@

test: $(TEST_BIN)
	$(TEST_BIN)

example: $(EXAMPLE_BIN)
	$(EXAMPLE_BIN)

bench: $(BENCH_BIN)
	$(BENCH_BIN) --quick

clean:
	rm -rf $(BUILD_DIR)
