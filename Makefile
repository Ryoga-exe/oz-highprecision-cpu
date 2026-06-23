CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -pedantic
CPPFLAGS ?= -Iinclude
THREAD_FLAGS ?= -pthread
BLAS_BACKEND ?= auto
BLAS_LIBS_AUTO := $(shell if ldconfig -p 2>/dev/null | grep -q libopenblas.so; then echo -lopenblas; elif [ -e /usr/lib/x86_64-linux-gnu/libblas.so ]; then echo -lblas; elif [ -e /lib/x86_64-linux-gnu/libblas.so.3 ]; then echo /lib/x86_64-linux-gnu/libblas.so.3; else echo -lblas; fi)
BLAS_LIBS_SYSTEM := $(shell if [ -e /usr/lib/x86_64-linux-gnu/libblas.so ]; then echo -lblas; elif [ -e /lib/x86_64-linux-gnu/libblas.so.3 ]; then echo /lib/x86_64-linux-gnu/libblas.so.3; else echo -lblas; fi)
BLAS_LIBS_OPENBLAS := $(shell if pkg-config --exists openblas 2>/dev/null; then pkg-config --libs openblas; else echo -lopenblas; fi)
BLAS_LIBS_BLIS := $(shell if pkg-config --exists blis 2>/dev/null; then pkg-config --libs blis; else echo -lblis; fi)
BLAS_LIBS_MKL := -lmkl_rt

ifeq ($(origin BLAS_LIBS), undefined)
ifeq ($(BLAS_BACKEND),auto)
BLAS_LIBS := $(BLAS_LIBS_AUTO)
else ifeq ($(BLAS_BACKEND),system)
BLAS_LIBS := $(BLAS_LIBS_SYSTEM)
else ifeq ($(BLAS_BACKEND),openblas)
BLAS_LIBS := $(BLAS_LIBS_OPENBLAS)
else ifeq ($(BLAS_BACKEND),blis)
BLAS_LIBS := $(BLAS_LIBS_BLIS)
else ifeq ($(BLAS_BACKEND),mkl)
BLAS_LIBS := $(BLAS_LIBS_MKL)
else ifeq ($(BLAS_BACKEND),custom)
$(error BLAS_BACKEND=custom requires BLAS_LIBS="...")
else
$(error unknown BLAS_BACKEND '$(BLAS_BACKEND)')
endif
endif

BUILD_DIR := build
TEST_BIN := $(BUILD_DIR)/test_gemm
EXAMPLE_BIN := $(BUILD_DIR)/cancellation
BENCH_BIN := $(BUILD_DIR)/benchmark

.PHONY: all test example bench blas-info clean

all: test example

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TEST_BIN): tests/test_gemm.cpp include/oz_hp_cpu/gemm.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $< $(BLAS_LIBS) -o $@

$(EXAMPLE_BIN): examples/cancellation.cpp include/oz_hp_cpu/gemm.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $< $(BLAS_LIBS) -o $@

$(BENCH_BIN): bench/benchmark.cpp include/oz_hp_cpu/gemm.hpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(THREAD_FLAGS) $< $(BLAS_LIBS) -o $@

test: $(TEST_BIN)
	$(TEST_BIN)

example: $(EXAMPLE_BIN)
	$(EXAMPLE_BIN)

bench: $(BENCH_BIN)
	$(BENCH_BIN) --quick

blas-info:
	@echo "BLAS_BACKEND=$(BLAS_BACKEND)"
	@echo "BLAS_LIBS=$(BLAS_LIBS)"

clean:
	rm -rf $(BUILD_DIR)
