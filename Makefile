export SHELL := /bin/bash

INPUT_FILE_NAME=matrix

CFLAGS = -g -Wall -march=native -std=gnu99 -fopenmp -m64
NVCCFLAGS = -g -c -m64 -O3

LDFLAGS := -lrt -lm -lpthread
CUDA_LDFLAGS := -L/opt/cuda/lib64 -lcudart

# workaround for cip since symlinks without version do not exist
atlas := $(shell echo 'int main() { cblas_sgemm(0,0,0,0,0,0,0,0,0,0); return 0;}' | gcc -x c -o /dev/null - -latlas -lcblas 2>&1 > /dev/null; echo $$?)
atlas_explicit := $(shell echo 'int main() { cblas_sgemm(0,0,0,0,0,0,0,0,0,0); return 0;}' | gcc -x c -o /dev/null - -l:libatlas.so.3 -l:libcblas.so.3 2>&1 > /dev/null; echo $$?)
blas := $(shell echo 'int main() { cblas_sgemm(0,0,0,0,0,0,0,0,0,0); return 0;}' | gcc -x c -o /dev/null - -lblas 2>&1 > /dev/null; echo $$?)

ifeq ($(atlas),0)
LDFLAGS += -latlas
CFLAGS += -DBLAS
else
ifeq ($(atlas_explicit),0)
LDFLAGS += -l:libblas.so.3 -l:libatlas.so.3
CFLAGS += -DBLAS
else
endif
endif

ifeq ($(clas),0)
LDFLAGS += -lblas
CFLAGS += -DBLAS
endif

clang-exists := $(shell clang --version 2>&1 > /dev/null; echo $$?)

gcc-exists := $(shell gcc --version 2>&1 > /dev/null; echo $$?)
gcc-4.4-exists := $(shell gcc-4.4 --version 2>&1 > /dev/null; echo $$?)
gcc-4.5-exists := $(shell gcc-4.5 --version 2>&1 > /dev/null; echo $$?)
gcc-4.6-exists := $(shell gcc-4.6 --version 2>&1 > /dev/null; echo $$?)
gcc-4.7-exists := $(shell gcc-4.7 --version 2>&1 > /dev/null; echo $$?)
gcc-4.8-exists := $(shell gcc-4.8 --version 2>&1 > /dev/null; echo $$?)

cuda-exists := $(shell nvcc --version 2>&1 > /dev/null; echo $$?)
cudart-exists := $(shell echo 'int main() { return 0;}' | gcc -x c -o /dev/null - $(CUDA_LDFLAGS) 2>&1 > /dev/null; echo $$?)

C_COMPILER := clang gcc gcc-4.4 gcc-4.5 gcc-4.6 gcc-4.7 gcc-4.8

$(info selected: $(C_COMPILER))

ifneq ($(clang-exists),0)
C_COMPILER := $(filter-out clang,$(C_COMPILER))
endif

ifneq ($(gcc-exists),0)
C_COMPILER := $(filter-out gcc,$(C_COMPILER))
endif

ifneq ($(gcc-4.4-exists),0)
C_COMPILER := $(filter-out gcc-4.4,$(C_COMPILER))
endif

ifneq ($(gcc-4.5-exists),0)
C_COMPILER := $(filter-out gcc-4.5,$(C_COMPILER))
endif

ifneq ($(gcc-4.6-exists),0)
C_COMPILER := $(filter-out gcc-4.6,$(C_COMPILER))
endif

ifneq ($(gcc-4.7-exists),0)
C_COMPILER := $(filter-out gcc-4.7,$(C_COMPILER))
endif

ifneq ($(gcc-4.8-exists),0)
C_COMPILER := $(filter-out gcc-4.8,$(C_COMPILER))
endif

binaries := $(patsubst %,bin/matrix-%-O0, $(C_COMPILER))
binaries += $(patsubst %,bin/matrix-%-O3, $(C_COMPILER))
binaries += $(patsubst %,bin/matrix-%-O3-unroll, $(C_COMPILER))


ifeq ($(cuda-exists),0)
ifeq ($(cudart-exists),0)
binaries += matrix-cuda matrix-cuda-unroll
endif
endif

$(info build: $(binaries))

all: bin ${binaries}

bin:
	@mkdir -p bin

bin/matrix-clang-O0: ${INPUT_FILE_NAME}.c
	clang $(CFLAGS) -O0 -S $< -o $@.s
	clang $(CFLAGS) -O0 -o $@ $< $(LDFLAGS) -funroll-loops
bin/matrix-clang-O3: ${INPUT_FILE_NAME}.c
	clang $(CFLAGS) -O3 -S $< -o $@.s
	clang $(CFLAGS) -O3 -o $@ $< $(LDFLAGS)
bin/matrix-clang-O3-unroll: ${INPUT_FILE_NAME}.c
	clang $(CFLAGS) -O3 -S $< -funroll-loops -o $@.s
	clang $(CFLAGS) -O3 -o $@ $< $(LDFLAGS) -funroll-loops

bin/matrix-%-O0: ${INPUT_FILE_NAME}.c
	$* $(CFLAGS) -O0 -S $< -o $@.s
	$* $(CFLAGS) -O0 -Wa,-ahlsm=$@.mixed.s -o $@ $< $(LDFLAGS)

bin/matrix-%-O3: ${INPUT_FILE_NAME}.c
	$* $(CFLAGS) -O3 -S $< -o $@.s
	$* $(CFLAGS) -O3 -Wa,-ahlsm=$@.mixed.s -o $@ $< $(LDFLAGS)

bin/matrix-%-O3-unroll: ${INPUT_FILE_NAME}.c
	$* $(CFLAGS) -O3 -S $< -funroll-loops -o $@.s
	$* $(CFLAGS) -O3 -Wa,-ahlsm=$@.mixed.s -o $@ $< $(LDFLAGS) -funroll-loops

bin/matrix-cuda: ${INPUT_FILE_NAME}.c ${INPUT_FILE_NAME}.cu
	nvcc ${NVCCFLAGS} -o $@.cu.o ${INPUT_FILE_NAME}.cu
	gcc -O3 -DCUDA ${CFLAGS} -o $@ ${INPUT_FILE_NAME}.c $@.cu.o ${CUDA_LDFLAGS}

bin/matrix-cuda-unroll: ${INPUT_FILE_NAME}.c
	nvcc ${NVCCFLAGS} -o $@.cu.o ${INPUT_FILE_NAME}.cu
	gcc -O3 -DCUDA -funroll-loops ${CFLAGS} -o $@ ${INPUT_FILE_NAME}.c $@.cu.o $(CUDA_LDFLAGS)

clean:
	rm -f *.pdf *.s *.o
	rm -rf bin

distclean:
	rm -f *.dat
