#include <stdio.h>
#include "matrix.h"

/* 
*http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api 
*/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %d %s %s %d\n", code, cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



__global__ void vanilla(matrix_t A, matrix_t B, matrix_t C) {
  TYPE c = 0;

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row > A.r || col > B.c) return;

  for (int i = 0; i < A.c; ++i) {
    c += A.data[ row * A.c + i] * B.data [ i * B.c + col ];
  }
  C.data[ row * C.c + col ] = c;
}

__global__ void coal() {
}

extern "C" {
  void *mult_vanilla_cuda(arg_t *args) {

    int devices;
    gpuErrchk(cudaGetDeviceCount( &devices ));
    printf("%d devices\n", devices);

    cudaSetDevice(1);
    struct cudaDeviceProp properties;
    gpuErrchk( cudaGetDeviceProperties( &properties, 1 ));

    printf("%s\n", properties.name);


    matrix_t A = {NULL, args->A.r, args->A.c};
    matrix_t B = {NULL, args->B.r, args->B.c};
    matrix_t C = {NULL, args->C.r, args->C.c};

    int sizeA = args->A.r * args->A.c * sizeof(TYPE);
    int sizeB = args->B.r * args->B.c * sizeof(TYPE);
    int sizeC = args->C.r * args->C.c * sizeof(TYPE);

    gpuErrchk( cudaMalloc( &A.data, sizeA ));
    gpuErrchk( cudaMemcpy( A.data, args->A.data, sizeA, cudaMemcpyHostToDevice ));
    
    gpuErrchk( cudaMalloc( &B.data, sizeB ));
    gpuErrchk( cudaMemcpy( B.data, args->B.data, sizeB, cudaMemcpyHostToDevice ));

    gpuErrchk( cudaMalloc( &C.data, sizeC ));


    dim3 dimBlock(16,16);
    dim3 dimGrid(args->C.c, args->C.r);
    vanilla<<<dimGrid,dimBlock>>>(A,B,C);

    gpuErrchk(cudaThreadSynchronize());

    gpuErrchk(cudaMemcpy(args->C.data, C.data, sizeC, cudaMemcpyDeviceToHost ));

    cudaFree(A.data);
    cudaFree(B.data);
    cudaFree(C.data);

    return NULL;
  }

  void *mult_coal_cuda(arg_t *args) {
    coal<<<1,1>>>();
    return NULL;
  }
}

