#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sched.h>
#include <sys/sysinfo.h>

#ifdef BLAS
#include <cblas.h>
#endif

#include <pthread.h>

#include <immintrin.h>
#include "iacaMarks.h"

#include "matrix.h"

static int num_threads = 1;

static int use_omp = 0;

double get_time() {
  struct timespec t;

  if ( clock_gettime( CLOCK_MONOTONIC, &t ) ) {
    perror( "Failed to get time" );
    exit(EXIT_FAILURE);
  }

  return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
}

#ifdef __SSE__
void print_sse_ps(const char *s, __m128 v) {
  float buf[4];
  _mm_storeu_ps(buf, v);
  printf("%s %f %f %f %f\n", s, buf[0], buf[1], buf[2], buf[3]);
}

void print_sse_pd(const char *s, __m128d v) {
  double buf[2];
  _mm_storeu_pd(buf, v);
  printf("%s %lf %lf\n", s, buf[0], buf[1]);
}
#endif

#ifdef __AVX__
void print_avx_ps(const char *s, __m256 v) {
  float buf[8];
  _mm256_storeu_ps(buf, v);
  printf("%s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", s, buf[7], buf[6], buf[5], buf[4], buf[3], buf[2], buf[1], buf[0]);
}

void print_avx_pd(const char *s, __m256d v) {
  double buf[4];
  _mm256_storeu_pd(buf, v);
  printf("%s %.2lf %.2lf %.2lf %.2lf\n", s, buf[3], buf[2], buf[1], buf[0]);
}
#endif

#ifdef BLAS
void *mult_blas(arg_t *args) {
#if S == 0 //float
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
      args->A.r, args->B.c, args->A.c, 1.0f, 
      args->A.data, args->A.r, 
      args->B.data, args->B.c, 
      0.0f,
      args->C.data, args->A.r);
#else
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
      args->A.r, args->C.c, args->A.c, 1.0f, 
      args->A.data, args->A.r, 
      args->B.data, args->A.c, 
      0.0f,
      args->C.data, args->A.r);
#endif
  return NULL;
}
#endif


void * mult_vanilla( arg_t *args) {
  int i,j,k;

  TYPE * RESTRICT c_p = args->C.data;

  for ( i = 0; i < args->A.r; ++i ) {
    size_t arow = i * args->A.c;
    size_t crow = i * args->C.c;

#pragma omp parallel for if(use_omp) private(j,k)

    for ( j = 0; j < args->B.c; ++j ) {
      TYPE sum = 0.;
      for ( k = 0; k < args->B.r; ++k ) {
        sum += args->A.data[arow + k] * args->B.data[k * args->B.c + j];
      }
      c_p[crow + j ] = sum;
    }
  }
  return NULL;
}

void * mult_vanilla2( arg_t *args) {
  int i,j,k;

  TYPE * RESTRICT c_p = args->C.data;

  for ( i = 0; i < args->A.r; ++i ) {
  #pragma omp parallel for if(use_omp) private(j,k)
    for ( j = 0; j < args->B.c; ++j ) {
      TYPE c0 = c_p[j*args->C.c+i];

      for ( k = 0; k < args->B.r; ++k ) {
        TYPE a = args->B.data[ k*args->B.c+i];
        TYPE b = args->A.data[ j * args->A.c + k];
        TYPE mul = a*b;
        c0 += mul;
      }
      c_p[j*args->C.c + i] = c0;
    }
  }
  return NULL;
}

/*
void * mult_transposed( arg_t *args) {
  int i,j,k;
  for ( i = 0; i < args->A.r; ++i ) {
    size_t arow = i * args->A.c;
    size_t crow = i * args->C.c;
    for ( j = 0; j < args->B.c; ++j ) {
      size_t brow = j * args->B.c;
      TYPE sum = 0.;
      for ( k = 0; k < args->B.r; ++k ) {
        sum += args->A.data[arow + k] * args->B.data[brow + k];
      }
      args->C.data[crow + j ] = sum;
    }
  }
}
*/

//CIP 1/4
void * mult_unroll2a( arg_t *args) {
  int i,j,k;

  TYPE * RESTRICT c_p = args->C.data;

  for ( i = 0; i < args->A.r; ++i ) {
    size_t arow = i * args->A.c;
    size_t crow = i * args->C.c;

#pragma omp parallel for if(use_omp) private(j,k)
    for ( j = 0; j < args->B.c; ++j ) {
      TYPE sum = 0.;
      for ( k = 0; (k+1) < args->B.r; k+=2 ) {
         sum += args->A.data[arow + k+0] * args->B.data[(k+0) * args->B.c + j];
         sum += args->A.data[arow + k+1] * args->B.data[(k+1) * args->B.c + j];
      }
      c_p[crow + j] = sum;
    }
  }
  return NULL;
}
void * mult_unroll4a( arg_t *args) {
  int i,j,k;

  TYPE * RESTRICT c_p = args->C.data;

  for ( i = 0; i < args->A.r; ++i ) {
    size_t arow = i * args->A.c;
    size_t crow = i * args->C.c;

#pragma omp parallel for if(use_omp) private(j,k)
    for ( j = 0; j < args->B.c; ++j ) {
      TYPE sum = 0.;
      for ( k = 0; (k+3) < args->B.r; k+=4 ) {
        sum += args->A.data[arow + k+0] * args->B.data[(k+0) * args->B.c + j];
        sum += args->A.data[arow + k+1] * args->B.data[(k+1) * args->B.c + j];
        sum += args->A.data[arow + k+2] * args->B.data[(k+2) * args->B.c + j];
        sum += args->A.data[arow + k+3] * args->B.data[(k+3) * args->B.c + j];
      }
      c_p[crow + j + 0] = sum;
    }
  }
  return NULL;
}

void * mult_unroll2b( arg_t *args) {
  int i,j,k;

  TYPE * RESTRICT c_p = args->C.data;

  for ( i = 0; i < args->A.r; ++i ) {
    size_t arow = i * args->A.c;
    size_t crow = i * args->C.c;

#pragma omp parallel for if(use_omp) private(j,k)
    for ( j = 0; j < args->B.c; ++j ) {
      TYPE sum = 0.;
      for ( k = 0; (k+1) < args->B.r; k+=2 ) {
        TYPE a0 = args->A.data[arow + k+0];
        TYPE a1 = args->A.data[arow + k+1];

        TYPE b0 = args->B.data[(k+0) * args->B.c + j];
        TYPE b1 = args->B.data[(k+1) * args->B.c + j];

        sum += a0*b0;
        sum += a1*b1;
      }
      c_p[crow + j] = sum;
    }
  }
  return NULL;
}
void * mult_unroll4b( arg_t *args ) {
  int i,j,k;

  TYPE * RESTRICT c_p = args->C.data;

  for ( i = 0; i < args->A.r; ++i ) {
    size_t arow = i * args->A.c;
    size_t crow = i * args->C.c;

#pragma omp parallel for if(use_omp) private(j,k)
    for ( j = 0; j < args->B.c; ++j ) {
      TYPE sum = 0.;
      for ( k = 0; (k+3) < args->B.r; k+=4 ) {
        TYPE a0 = args->A.data[arow + k+0];
        TYPE a1 = args->A.data[arow + k+1];
        TYPE a2 = args->A.data[arow + k+2];
        TYPE a3 = args->A.data[arow + k+3];

        TYPE b0 = args->B.data[(k+0) * args->B.c + j];
        TYPE b1 = args->B.data[(k+1) * args->B.c + j];
        TYPE b2 = args->B.data[(k+2) * args->B.c + j];
        TYPE b3 = args->B.data[(k+3) * args->B.c + j];

         sum += a0*b0;
         sum += a1*b1;
         sum += a2*b2;
         sum += a3*b3;
      }
      c_p[crow + j] = sum;
    }
  }
  return NULL;
}

//CIP 2/2
void * mult_stride( arg_t *args ) {
  int i,j,k,s;

  TYPE * RESTRICT c_p = args->C.data;

  for ( i = 0; i < args->A.r; ++i ) { //Zeilen A
    size_t arow = i * args->A.c;
    size_t crow = i * args->C.c;
    for ( s = 0; s < args->A.c; s+=args->block_r ) { //Spalten in A

#pragma omp parallel for if(use_omp) private (j,k) shared(c_p)
      for ( j = 0; j < args->B.c; ++j ) { // Spalte C
        TYPE sum = 0;
        for ( k = s; (k+3) < s+args->block_r; k+=4 ) {
          TYPE a0 = args->A.data[arow + k+0];
          TYPE a1 = args->A.data[arow + k+1];
          TYPE a2 = args->A.data[arow + k+2];
          TYPE a3 = args->A.data[arow + k+3];

          TYPE b0 = args->B.data[(k+0) * args->B.c + j];
          TYPE b1 = args->B.data[(k+1) * args->B.c + j];
          TYPE b2 = args->B.data[(k+2) * args->B.c + j];
          TYPE b3 = args->B.data[(k+3) * args->B.c + j];

          TYPE sum0 = a0*b0;
          TYPE sum1 = a1*b1;
          TYPE sum2 = a2*b2;
          TYPE sum3 = a3*b3;

          TYPE sum4 = sum0 + sum1;
          TYPE sum5 = sum2 + sum3;

          sum4 += sum5;

          sum += sum4;
        }
        TYPE prev = c_p[crow + j];
        prev     += sum;
        c_p[crow + j ] = prev;
      }
    }
  }
  return NULL;
}

//CIP 2/3
void * mult_block( arg_t *args ) {
  int i,j,k,s,t,u;

  TYPE * RESTRICT c_p = args->C.data;

  for ( s = 0; s < args->A.r; s+=args->block_r ) { //Zeilen A
    for ( t = 0; t < args->B.c; t+=args->block_r ) { //Spalten B
      for ( u = 0; u < args->B.r; u+=args->block_r ) { 

#pragma omp parallel for if(use_omp) private(i,j,k) shared(c_p)
        for ( i = s; i < s+args->block_r; ++i ) { //Block in C
          size_t arow = i * args->A.c;
          size_t crow = i * args->C.c;


          for ( j = t; j < t+args->block_r; ++j ) { //Block in A
            TYPE sum = 0;

            for ( k = u; (k+3) < u+args->block_r; k+=4 ) { //Block in A
              TYPE a0 = args->A.data[arow + k+0];
              TYPE a1 = args->A.data[arow + k+1];


              TYPE a2 = args->A.data[arow + k+2];
              TYPE a3 = args->A.data[arow + k+3];

              TYPE b0 = args->B.data[(k+0) * args->B.c + j];
              TYPE b1 = args->B.data[(k+1) * args->B.c + j];
              TYPE b2 = args->B.data[(k+2) * args->B.c + j];
              TYPE b3 = args->B.data[(k+3) * args->B.c + j];

              sum += a0*b0;
              sum += a1*b1;
              sum += a2*b2;
              sum += a3*b3;
            }
            TYPE prev = c_p[crow + j];
            prev     += sum;
            c_p[crow + j ] = prev;
          }
        }
      }
    }
  }
  return NULL;
}

#ifdef __SSE__
//CIP 3/1
void * mult_block_sse( arg_t *args ) {
  int i,j,k,s,t,u;

  const TYPE * RESTRICT a_p = args->A.data;
  const TYPE * RESTRICT b_p = args->B.data;
  TYPE * RESTRICT c_p = args->C.data;

  for ( s = 0; s < args->A.r; s+=args->block_r ) { //Zeilen A
    for ( t = 0; t < args->B.c; t+=args->block_r ) { //Spalten B
      for ( u = 0; u < args->B.r; u+=args->block_r ) { 

#pragma omp parallel for if(use_omp) private(i,j,k) shared(c_p)
        for ( i = s; i < s+args->block_r; ++i ) { //Block in A
          size_t arow = i * args->A.c;
          size_t crow = i * args->C.c;

#if S == 0 //float
          for ( j = t; (j+3) < t+args->block_r; j+=4 ) { //Block in B
            __m128 sum = _mm_setzero_ps();
            for ( k = u; (k+3) < u+args->block_r; k+=4 ) { //Block in A
#ifdef IACA
              IACA_START //1
#endif

              __m128 a = _mm_load_ps( &( a_p[arow + k]) );

              __m128 b0 = _mm_load_ps( &(b_p[ (k+0) * args->B.c + j ]) );
              __m128 b1 = _mm_load_ps( &(b_p[ (k+1) * args->B.c + j ]) );
              __m128 b2 = _mm_load_ps( &(b_p[ (k+2) * args->B.c + j ]) );
              __m128 b3 = _mm_load_ps( &(b_p[ (k+3) * args->B.c + j ]) );

              __m128 blow0 = _mm_unpacklo_ps( b0, b1 );
              __m128 blow1 = _mm_unpacklo_ps( b2, b3 );

              __m128 bhigh0 = _mm_unpackhi_ps( b0, b1 );
              __m128 bhigh1 = _mm_unpackhi_ps( b2, b3 );

              b0 = _mm_shuffle_ps( blow0, blow1, 0x44 );
              b1 = _mm_shuffle_ps( blow0, blow1, 0xee );
              b2 = _mm_shuffle_ps( bhigh0, bhigh1, 0x44 );
              b3 = _mm_shuffle_ps( bhigh0, bhigh1, 0xee );

              __m128 res0 = _mm_mul_ps( a, b0 );
              __m128 res1 = _mm_mul_ps( a, b1 );
              __m128 res2 = _mm_mul_ps( a, b2 );
              __m128 res3 = _mm_mul_ps( a, b3 );

              __m128 res4 = _mm_shuffle_ps( res0, res1, 0x44 );
              __m128 res5 = _mm_shuffle_ps( res0, res1, 0xee );
              __m128 res6 = _mm_shuffle_ps( res2, res3, 0x44 );
              __m128 res7 = _mm_shuffle_ps( res2, res3, 0xee );

              __m128 res8 = _mm_shuffle_ps( res4, res6, 0x88 );
              __m128 res9 = _mm_shuffle_ps( res4, res6, 0xdd );
              __m128 res10 = _mm_shuffle_ps( res5, res7, 0x88 );
              __m128 res11 = _mm_shuffle_ps( res5, res7, 0xdd );

              __m128 res12 = _mm_add_ps(res8, res9);
              __m128 res13 = _mm_add_ps(res10, res11);

              __m128 res14 = _mm_add_ps(res12, res13);

              sum = _mm_add_ps( sum, res14 );
            }
#ifdef IACA
            IACA_END // 1
#endif

            __m128 prev = _mm_load_ps( &( c_p[crow + j]) );
            prev = _mm_add_ps( prev, sum);
            _mm_store_ps( &( c_p[crow + j]) , prev );
          }
#elif S == 1 //double
          for ( j = t; (j+1) < t+args->block_r; j+=2 ) { //Block in B
            __m128d sum = _mm_setzero_pd( );

            for ( k = u; (k+1) < u+args->block_r; k+=2 ) { //Block in A
#ifdef IACA
              IACA_START // 1
#endif

              __m128d a = _mm_load_pd( &( a_p[arow + k]) );

              __m128d b0 = _mm_load_pd( &(b_p[ (k+0) * args->B.c + j ]) );
              __m128d b1 = _mm_load_pd( &(b_p[ (k+1) * args->B.c + j ]) );

              __m128d blow = _mm_unpacklo_pd( b0, b1 );
              __m128d bhigh = _mm_unpackhi_pd( b0, b1 );

              __m128d res0 = _mm_mul_pd( a, blow );
              __m128d res1 = _mm_mul_pd( a, bhigh );

              __m128d res2 = _mm_shuffle_pd(res0, res1, 0x0 );
              __m128d res3 = _mm_shuffle_pd(res0, res1, 0x3 );

              __m128d res4 = _mm_add_pd(res2, res3);

              sum = _mm_add_pd( sum, res4 );
            }
#ifdef IACA
            IACA_END // 1
#endif

            __m128d prev = _mm_load_pd( &( c_p[crow + j]) );
            prev = _mm_add_pd ( prev, sum);
            _mm_store_pd( &( c_p[crow + j]), prev );
          }
#else
          fprintf(stderr,"block_sse only applicable for float/double\n");
          return;
#endif
        }
      }
    }
  }
  return NULL;
}
#endif //__SSE__

#ifdef __AVX__
//CIP 3/1

void * mult_avx( arg_t *args) {
  int i,j,k;

  TYPE * RESTRICT c_p = args->C.data;

#if S == 0
  for ( i = 0; i < args->A.r; i+=8 ) {

#pragma omp parallel for if(use_omp) private(j,k) shared(c_p)
    for ( j = 0; j < args->B.c; ++j ) {
      __m256 sum = _mm256_setzero_ps( );

      for ( k = 0; k < args->B.r; ++k ) {
        __m256 a = _mm256_load_ps( &(args->B.data[ k*args->B.c+i]) );
        __m256 b = _mm256_broadcast_ss( &( args->A.data[ j * args->A.c + k] ) );
        __m256 mul = _mm256_mul_ps( a,b );
        sum = _mm256_add_ps(sum, mul );
      }
      __m256 prev = _mm256_load_ps( &(c_p[j*args->C.c+i]) );
      prev = _mm256_add_ps( prev, sum );
      _mm256_store_ps( &(c_p[j*args->C.c + i]), prev);
    }
  }
#elif S == 1
  for ( i = 0; i < args->A.r; i+=4 ) {
#pragma omp parallel for if(use_omp) private(j,k) shared(c_p)
    for ( j = 0; j < args->B.c; ++j ) {
      __m256d sum = _mm256_setzero( );

      for ( k = 0; k < args->B.r; ++k ) {
        __m256d a = _mm256_load_pd( &(args->B.data[ k*B->c+i]) );
        __m256d b = _mm256_broadcast_sd( &( args->A.data[ j * args->A->c + k] ) );
        __m256d mul = _mm256_mul_pd( a,b );
        sum = _mm256_add_pd(c0, mul );
      }
      __m256d prev = _mm256_load_pd( &(c_p[j*args->C.c+i]) );
      prev = _mm256_sum( prev, sum );
      _mm256_store_pd( &(c_p[j*args->C.c + i]), prev );
    }
  }
#endif
  return NULL;
}

void * mult_block_avx( arg_t *args ) {
  int i,j,k,s,t,u;

  const TYPE * RESTRICT a_p = args->A.data;
  const TYPE * RESTRICT b_p = args->B.data;
  TYPE * RESTRICT c_p = args->C.data;


  for ( s = 0; s < args->A.r; s+=args->block_r ) { //Zeilen A
    for ( t = 0; t < args->B.c; t+=args->block_r ) { //Spalten B
      for ( u = 0; u < args->B.r; u+=args->block_r ) { 

#pragma omp parallel for if(use_omp) private(i,j,k) shared(c_p)

        for ( i = s; i < s+args->block_r; ++i ) { //Block in A
          size_t arow = i * args->A.c;
          size_t crow = i * args->C.c;
#if S == 0 //float
          for ( j = t; (j+7) < t+args->block_r; j+=8 ) { //Block in B
            __m256 sum = _mm256_setzero_ps( );

            for ( k = u; (k+7) < u+args->block_r; k+=8 ) { //Block in A
#ifdef IACA
              IACA_START
#endif

              __m256 a = _mm256_load_ps( &( a_p[arow + k]) );

              __m256 b0 = _mm256_load_ps( &(b_p[ (k+0) * args->B.c + j ]) );
              __m256 b1 = _mm256_load_ps( &(b_p[ (k+1) * args->B.c + j ]) );
              __m256 b2 = _mm256_load_ps( &(b_p[ (k+2) * args->B.c + j ]) );
              __m256 b3 = _mm256_load_ps( &(b_p[ (k+3) * args->B.c + j ]) );
              __m256 b4 = _mm256_load_ps( &(b_p[ (k+4) * args->B.c + j ]) );
              __m256 b5 = _mm256_load_ps( &(b_p[ (k+5) * args->B.c + j ]) );
              __m256 b6 = _mm256_load_ps( &(b_p[ (k+6) * args->B.c + j ]) );
              __m256 b7 = _mm256_load_ps( &(b_p[ (k+7) * args->B.c + j ]) );

              __m256 blo0 = _mm256_unpacklo_ps(b0, b1);
              __m256 bhi0 = _mm256_unpackhi_ps(b0, b1);
              __m256 blo1 = _mm256_unpacklo_ps(b2, b3);
              __m256 bhi1 = _mm256_unpackhi_ps(b2, b3);
              __m256 blo2 = _mm256_unpacklo_ps(b4, b5);
              __m256 bhi2 = _mm256_unpackhi_ps(b4, b5);
              __m256 blo3 = _mm256_unpacklo_ps(b6, b7);
              __m256 bhi3 = _mm256_unpackhi_ps(b6, b7);

              __m256 tt0 = _mm256_shuffle_ps( blo0, blo1, _MM_SHUFFLE(1,0,1,0) );
              __m256 tt1 = _mm256_shuffle_ps( blo0, blo1, _MM_SHUFFLE(3,2,3,2) );
              __m256 tt2 = _mm256_shuffle_ps( bhi0, bhi1, _MM_SHUFFLE(1,0,1,0) );
              __m256 tt3 = _mm256_shuffle_ps( bhi0, bhi1, _MM_SHUFFLE(3,2,3,2) );
              __m256 tt4 = _mm256_shuffle_ps( blo2, blo3, _MM_SHUFFLE(1,0,1,0) );
              __m256 tt5 = _mm256_shuffle_ps( blo2, blo3, _MM_SHUFFLE(3,2,3,2) );
              __m256 tt6 = _mm256_shuffle_ps( bhi2, bhi3, _MM_SHUFFLE(1,0,1,0) );
              __m256 tt7 = _mm256_shuffle_ps( bhi2, bhi3, _MM_SHUFFLE(3,2,3,2) );

              __m256 row0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
              __m256 row1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
              __m256 row2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
              __m256 row3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
              __m256 row4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
              __m256 row5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
              __m256 row6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
              __m256 row7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);


              /* 
               * mul
               */

              __m256 res0 = _mm256_mul_ps( a, row0 );
              __m256 res1 = _mm256_mul_ps( a, row1 );
              __m256 res2 = _mm256_mul_ps( a, row2 );
              __m256 res3 = _mm256_mul_ps( a, row3 );
              __m256 res4 = _mm256_mul_ps( a, row4 );
              __m256 res5 = _mm256_mul_ps( a, row5 );
              __m256 res6 = _mm256_mul_ps( a, row6 );
              __m256 res7 = _mm256_mul_ps( a, row7 );

              /* 
               * shuffle back
               */

              __m256 res8 = _mm256_permute2f128_ps(res0, res4, 0x20);
              __m256 res9 = _mm256_permute2f128_ps(res1, res5, 0x20);
              __m256 res10 = _mm256_permute2f128_ps(res2, res6, 0x20);
              __m256 res11 = _mm256_permute2f128_ps(res3, res7, 0x20);
              __m256 res12 = _mm256_permute2f128_ps(res0, res4, 0x31);
              __m256 res13 = _mm256_permute2f128_ps(res1, res5, 0x31);
              __m256 res14 = _mm256_permute2f128_ps(res2, res6, 0x31);
              __m256 res15 = _mm256_permute2f128_ps(res3, res7, 0x31);

              __m256 res16 = _mm256_shuffle_ps( res8, res9, _MM_SHUFFLE(1,0,1,0) );
              __m256 res17 = _mm256_shuffle_ps( res8, res9, _MM_SHUFFLE(3,2,3,2) );
              __m256 res18 = _mm256_shuffle_ps( res10, res11, _MM_SHUFFLE(1,0,1,0) );
              __m256 res19 = _mm256_shuffle_ps( res10, res11, _MM_SHUFFLE(3,2,3,2) );
              __m256 res20 = _mm256_shuffle_ps( res12, res13, _MM_SHUFFLE(1,0,1,0) );
              __m256 res21 = _mm256_shuffle_ps( res12, res13, _MM_SHUFFLE(3,2,3,2) );
              __m256 res22 = _mm256_shuffle_ps( res14, res15, _MM_SHUFFLE(1,0,1,0) );
              __m256 res23 = _mm256_shuffle_ps( res14, res15, _MM_SHUFFLE(3,2,3,2) );

              __m256 res24 = _mm256_shuffle_ps(res16, res18, 0x88);
              __m256 res25 = _mm256_shuffle_ps(res16, res18, 0xDD);
              __m256 res26 = _mm256_shuffle_ps(res17, res19, 0x88);
              __m256 res27 = _mm256_shuffle_ps(res17, res19, 0xDD);
              __m256 res28 = _mm256_shuffle_ps(res20, res22, 0x88);
              __m256 res29 = _mm256_shuffle_ps(res20, res22, 0xDD);
              __m256 res30 = _mm256_shuffle_ps(res21, res23, 0x88);
              __m256 res31 = _mm256_shuffle_ps(res21, res23, 0xDD);

              /* 
               * add columns
               */

              __m256 sum0 = _mm256_add_ps(res24, res25); 
              __m256 sum1 = _mm256_add_ps(res26, res27); 
              __m256 sum2 = _mm256_add_ps(res28, res29); 
              __m256 sum3 = _mm256_add_ps(res30, res31); 

              __m256 sum4 = _mm256_add_ps(sum0, sum1); 
              __m256 sum5 = _mm256_add_ps(sum2, sum3); 

              __m256 sum6 = _mm256_add_ps(sum4, sum5); 

              sum = _mm256_add_ps( sum, sum6 );
            }
#ifdef IACA
            IACA_END
#endif
            __m256 prev = _mm256_load_ps( &( c_p[crow + j]) );
            prev = _mm256_add_ps( prev, sum );
            _mm256_stream_ps( &( c_p[crow + j]), prev );
          }


#elif S == 1 //double
          for ( j = t; (j+3) < t+args->block_r; j+=4 ) { //Block in B
            __m256d sum = _mm256_setzero_pd( );

            for ( k = u; (k+3) < u+args->block_r; k+=4 ) { //Block in A
#ifdef IACA
              IACA_START
#endif
              __m256d a = _mm256_load_pd( &( a_p[arow + k]) );

              __m256d b0 = _mm256_load_pd( &(b_p[ (k+0) * args->B.c + j ]) );
              __m256d b1 = _mm256_load_pd( &(b_p[ (k+1) * args->B.c + j ]) );
              __m256d b2 = _mm256_load_pd( &(b_p[ (k+2) * args->B.c + j ]) );
              __m256d b3 = _mm256_load_pd( &(b_p[ (k+3) * args->B.c + j ]) );

              /* 
               * transpose B 
               * */

              __m256d blo0 = _mm256_unpacklo_pd(b0, b1);
              __m256d bhi0 = _mm256_unpackhi_pd(b0, b1);
              __m256d blo1 = _mm256_unpacklo_pd(b2, b3);
              __m256d bhi1 = _mm256_unpackhi_pd(b2, b3);

              __m256d bperm0 = _mm256_permute2f128_pd(blo0, blo1, 0x20);
              __m256d bperm1 = _mm256_permute2f128_pd(bhi0, bhi1, 0x20);
              __m256d bperm2 = _mm256_permute2f128_pd(blo0, blo1, 0x31);
              __m256d bperm3 = _mm256_permute2f128_pd(bhi0, bhi1, 0x31);

              __m256d res0 = _mm256_mul_pd( a, bperm0 );
              __m256d res1 = _mm256_mul_pd( a, bperm1 );
              __m256d res2 = _mm256_mul_pd( a, bperm2 );
              __m256d res3 = _mm256_mul_pd( a, bperm3 );

              /* 
               * shuffle back
               */

              __m256d res4 = _mm256_shuffle_pd(res0, res1, 0x0);
              __m256d res5 = _mm256_shuffle_pd(res2, res3, 0x0);
              __m256d res6 = _mm256_shuffle_pd(res0, res1, 0xF);
              __m256d res7 = _mm256_shuffle_pd(res2, res3, 0xF);

              __m256d res8  = _mm256_permute2f128_pd(res4, res5, 0x20);
              __m256d res9  = _mm256_permute2f128_pd(res6, res7, 0x20);
              __m256d res10 = _mm256_permute2f128_pd(res4, res5, 0x31);
              __m256d res11 = _mm256_permute2f128_pd(res6, res7, 0x31);

              /* 
               * add columns
               */

              __m256d res12 = _mm256_add_pd(res8, res9); 
              __m256d res13 = _mm256_add_pd(res10, res11); 

              __m256d res14 = _mm256_add_pd(res12, res13); 

              sum = _mm256_add_pd( sum, res14 );
            }

            __m256d prev = _mm256_load_pd( &( c_p[crow + j]) );
            prev = __m256d_add_pd( prev, sum );
            _mm256_stream_pd( &( c_p[crow + j]), prev );
          }
#ifdef IACA
          IACA_END
#endif

#else
          fprintf(stderr,"block_avx only applicable for float/double\n");
          return;
#endif
        }
      }
    }
  }
  return NULL;
}
#endif //__AVX__

/*
 * Threads compute columnwise, e.g.
 *
 * 1 2 3 4
 * 1 2 3 4
 * 1 2 3 4
 * 1 2 3 4
 */
void *mult_vanilla_mt( arg_t *args ) {
  int i,j,k;

  TYPE * RESTRICT c_p = args->C.data;

  for ( i = args->tid; i < args->C.c; i+=num_threads ) {

    for ( j = 0; j < args->A.r; ++j ) {

      TYPE c0 = c_p[j*args->C.c+i];

      for ( k = 0; k < args->B.r; ++k ) {
        TYPE a = args->B.data[ k*args->B.c+i];
        TYPE b = args->A.data[ j * args->A.c + k];
        TYPE mul = a*b;
        c0 += mul;
      }
      c_p[j*args->C.c + i] = c0;
    }
  }

  return NULL;
}
void *mult_stride_mt( arg_t *args ) {
  int i,j,k;

  TYPE * RESTRICT c_p = args->C.data;

  int per_block = args->C.c / num_threads;
  int c_start = args->tid * per_block;
  int c_next   = c_start + per_block;

  for ( i = 0; i < args->C.r; ++i ) {

    for ( j = c_start; j < c_next; ++j ) {

      TYPE c0 = c_p[ i * args->C.c + j];

      for ( k = 0; k < args->B.r; ++k ) {
        TYPE a = args->A.data[ i * args->A.c + k];
        TYPE b = args->B.data[ k * args->B.c + j];
        TYPE mul = a*b;
        c0 += mul;
      }
      c_p[ i * args->C.c + j ] = c0;
    }
  }

  return NULL;
}

void transpose_square( matrix_t *A ) {
  int i,j;

  for (i=0; i < A->r; ++i) {
    size_t arow = i * A->c;
    for (j=i; j < A->c; ++j) {
      TYPE *a_ptr = A->data + arow + j;
      TYPE *b_ptr = A->data + j*A->c + i;

      TYPE a      = *a_ptr;
      *a_ptr      = *b_ptr;
      *b_ptr      = a;
    }
  }
}
void transpose( matrix_t *A, matrix_t *B ) {
  int i,j;

  B->r     = A->c;
  B->c     = A->r;

  for (i=0; i < A->r; ++i) {
    size_t arow = i * A->c;
    for (j=0; j < A->c; ++j) {
      B->data[ j*B->c+i ] = A->data[ arow + j ];
    }
  }
}

void print_matrix( matrix_t *A) {
  int i,j;

  for (i = 0; i < A->r; ++i) {
    for (j = 0; j < A->c; ++j) {
      printf( "%5.2f\t", A->data[ i*A->c + j] );
    }
    printf("\n");
  }
}

int compare_matrix(matrix_t *A, matrix_t *B) {
  if (A->r != B->r || A->c != B->c) {
    fprintf(stderr, "sizes do not match\n");
    return -2;
  }

  int i,j;
  for (i = 0; i < A->r; ++i) {
    for (j = 0; j < A->c; ++j) {
      TYPE a = A->data[ i*A->c + j]; 
      TYPE b = B->data[ i*A->c + j]; 

      if ( ( fabs(a - b) ) > 1e-5 ) {
        fprintf(stderr, "(%d,%d) %f != %f\n", i,j,a,b);
        return -1;
      }
    }
  }
  return 0;
}

void init_seq(matrix_t *A) {
  int i,j;

  for (i = 0; i < A->r; ++i) {
    for (j = 0; j < A->c; ++j) {
      A->data[ i*A->c + j] = j+1;
    }
  }
}

void init_rand(matrix_t *A) {
  int i,j;
  srand( get_time() );

  for (i = 0; i < A->r; ++i) {
    for (j = 0; j < A->c; ++j) {
      A->data[ i*A->c + j] = (int) (( ((float) rand())/RAND_MAX)*16.);
    }
  }
}

void m_malloc( matrix_t *A, size_t r, size_t c ) {
  A->r = r;
  A->c = c;
  //A->data = (TYPE *) calloc( r * c, sizeof(TYPE) );
  if ( posix_memalign( (void **) &(A->data), sysconf(_SC_PAGESIZE), r*c*sizeof(TYPE) ) ) {
    fprintf(stderr, "%d: Failed to alloc data\n", __LINE__);
    exit(EXIT_FAILURE);
  }
  memset(A->data, 0, r*c*sizeof(TYPE) );
}

void m_free( matrix_t *A) {
  free(A->data);
}

static function_t funcs[] =
{
  {"vanilla",mult_vanilla},
  {"vanilla2",mult_vanilla2},
#ifdef BLAS
  {"blas",mult_blas},
#endif
  {"unroll2a",mult_unroll2a},
  {"unroll4a",mult_unroll4a},
  {"unroll2b",mult_unroll2b},
  {"unroll4b",mult_unroll4b},
  {"stride",mult_stride},
  {"block",mult_block},
#ifdef __SSE__
  {"block_sse",mult_block_sse},
#endif
#ifdef __AVX__
  {"avx",mult_avx},
  {"block_avx",mult_block_avx},
#endif
  {"vanilla_mt",mult_vanilla_mt},
  {"stride_mt",mult_stride_mt},
#ifdef CUDA
  {"vanilla_cuda", mult_vanilla_cuda},
  {"coal_cuda", mult_vanilla_cuda},
#endif
  {"null",NULL}
};


int main(int argc, char **argv) {

  int ret = 0;
  if ( argc < 5 ) {
    fprintf(stderr, "Usage: %s A_r A_c, B_c <", argv[0]);
    int i=0;
    while ( funcs[i].func != NULL ) {
      fprintf( stderr, "%s", funcs[i].name );
      if (funcs[i+1].func != NULL ) fprintf( stderr, "|" );
      ++i;
    }
    fprintf(stderr, ">\n");
    exit( 1 );
  }

  /*
   * identify correct function to be called
   */

  int i = 0;
  void * (*mult)(arg_t *)=NULL;

  while ( funcs[i].func != NULL ) {
    char with_omp[256];
    snprintf(with_omp, 256, "%s_omp", funcs[i].name);

    if ( strncmp (funcs[i].name, argv[4], 256 ) == 0 ) {
      mult = funcs[i].func;

      /*
       * set affinity to a specific cpu only for non-parallel implementations
       */
      cpu_set_t  mask;
      CPU_ZERO(&mask);
      CPU_SET(0, &mask);
      int result = sched_setaffinity(0, sizeof(mask), &mask);
      if (result) {
        perror("Could not set affinity\n");
      }
      break;

    } else if ( strncmp ( with_omp, argv[4], 256 ) == 0 ) {
      mult = funcs[i].func;
      use_omp = 1;
      break;
    }
    ++i;
  }

  if (mult == NULL) {
    fprintf(stderr, "invalid function name\n");
    exit(EXIT_FAILURE);
  }

  /* 
   * special handling of functions using variable block/stride size
   *
   * TODO: make blocking variable in 2D
   */
  int stride = 0;
  if ( strncmp (funcs[i].name, "stride", 256 ) == 0  ||
       strncmp (funcs[i].name, "block", 256 ) == 0 ||
       strncmp (funcs[i].name, "block_sse", 256 ) == 0 ||
       strncmp (funcs[i].name, "block_avx", 256 ) == 0 ) {
    if ( argc != 6 ) {
      fprintf(stderr, "Missing size for stride/block\n");
      exit(EXIT_FAILURE);
    } else {
      stride = atoi( argv[5] );
    }
  }

  /* 
   * handle creation of threads
   */
  if ( strncmp (funcs[i].name, "vanilla_mt", 256 ) == 0  ||
       strncmp (funcs[i].name, "stride_mt", 256 ) == 0 ) {
    if ( argc != 6 ) {
      fprintf(stderr, "Missing thread count\n");
      exit(EXIT_FAILURE);
    } else {
      int cores     = get_nprocs();
      int requested = atoi(argv[5]);
      num_threads = cores > requested ? requested : cores;
    }
  }

  /*
   * print current build- and runtime config 
   * */
#ifdef __SSE__
  fprintf(stderr,"sse, ");
#endif
#ifdef __AVX__
  fprintf(stderr,"avx, ");
#endif
  fprintf(stderr,"%s, %s, %d threads\n", S==0?"float":"double", use_omp?"omp":"no_omp", num_threads);

  //
  // allocate matrices
  // 
  matrix_t A,B,C,Ref;

  m_malloc( &A, atoi(argv[1]), atoi(argv[2]) );
  m_malloc( &B, atoi(argv[2]), atoi(argv[3]) );
  
  //targets
  m_malloc( &C, atoi(argv[1]), atoi(argv[3]) );
  m_malloc( &Ref, atoi(argv[1]), atoi(argv[3]) );

  init_rand(&A);
  init_rand(&B);

  double start, end, delta, total=0.f;

  size_t iters=1;

  if ( num_threads != 1 ) {
    int i;

    /* 
     * create threads
     */
    pthread_t threads[num_threads];
    arg_t args[num_threads];

    for ( i = 0; i < num_threads; ++i) {
      args[i].A       = A;
      args[i].B       = B;
      args[i].C       = C;
      args[i].block_r = stride;
      args[i].block_c = stride;
      args[i].tid = i;
    }

    start = get_time();
    for ( i = 0; i < num_threads; ++i) {
      pthread_attr_t attr;
      pthread_attr_init( &attr );

      cpu_set_t cpus;
      CPU_ZERO( &cpus );
      CPU_SET( i, &cpus );
      pthread_attr_setaffinity_np( &attr, sizeof(cpu_set_t), &cpus );


      if ( pthread_create( &threads[i], &attr, (void * (*)(void *)) mult_vanilla_mt, &args[i] ) ) {
        fprintf(stderr, "failed to start thread %d\n", i);
        exit(EXIT_FAILURE);
      }
      pthread_attr_destroy( &attr );
    }

    /*
     * clean up
     */
    for ( i = 0; i < num_threads; ++i) {
      pthread_join( threads[i], NULL );
    }
    end = get_time();

    delta = end - start;
    total += delta;

      /* clean up */
  } else {
//#define ITERS
#ifdef ITERS
  for (; iters < 10 && total < 1.0f; ++iters) {
#endif
    memset(C.data, 0, C.r * C.c * sizeof(TYPE) );
    start = get_time();

    arg_t args = {A,B,C,stride,stride};

    mult( &args );

    end = get_time();

    delta = end - start;
    total += delta;

#ifdef ITERS
  }
#endif
  }

  double gup = (double) (A.r * B.c * B.r) / (1024*1024*1024) / (total/iters);
  double gb = (double) (A.r * A.c + B.r * B.c + C.r * C.c) * sizeof(TYPE ) / (1024*1024*1024) / (total/iters);


  fprintf(stderr,"Runtime: %f, gup/s: %f, GB/s: %f\n", delta, gup, gb);
  printf("%f %f %f\n", (total/iters), gup, gb);
  
#define COMPARE
#ifdef COMPARE

  arg_t args = {A,B,Ref,stride, stride};
#ifdef BLAS
  mult_blas( &args );
#else
  mult_vanilla ( &args );
#endif

  if ( compare_matrix(&Ref, &C) ) {
    fprintf(stderr, "\033[31mCompare failed\033[0m\n");
    ret = -1;
  } else {
    fprintf(stderr, "\033[32mCompare successfull\033[0m\n");
  }
#endif


//#define PRINT
#ifdef PRINT
  printf("A\n");
  print_matrix( &A );
  printf("B\n");
  print_matrix( &B );
  printf("C\n");
  print_matrix( &C );
  printf("Ref\n");
  print_matrix( &Ref );
#endif


  m_free( &A );
  m_free( &B );
  m_free( &C );
  m_free( &Ref );

  exit(ret);
}
