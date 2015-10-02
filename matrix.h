#ifndef MATRIX_H
#define MATRIX_H

#ifdef __cplusplus
#define RESTRICT
#else
#define RESTRICT restrict
#endif


#endif

//#define IACA

#define S 0

#if S == 0
#define TYPE float
#else
#define TYPE double
#endif


typedef struct {
  TYPE * RESTRICT data;
  size_t r;
  size_t c;
} matrix_t;

typedef struct {
  matrix_t A;
  matrix_t B;
  matrix_t C;
  size_t block_r;
  size_t block_c;
  int tid;
  int num_threads;
} arg_t;

typedef struct {
  char name[256];
  void * (* func)( arg_t *);
} function_t;

/*
 * kernel functions
 */

#ifdef CUDA
#ifdef __cplusplus
extern "C" {
#endif

void *mult_vanilla_cuda(arg_t *);
void *mult_coal_cuda(arg_t *);

#ifdef __cplusplus
}
#endif

#endif //MATRIX_H
