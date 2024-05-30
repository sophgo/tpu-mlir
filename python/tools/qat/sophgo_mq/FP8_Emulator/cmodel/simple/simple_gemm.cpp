/*----------------------------------------------------------------------------* 
 * Copyright (c) 2023, Intel Corporation - All rights reserved.               
 * This file is part of FP8-Emulation-Toolkit
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *----------------------------------------------------------------------------*
 * Naveen Mellempudi (Intel Corporation)                                      
 *----------------------------------------------------------------------------*/

#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <time.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <omp.h>
#if 10
//extern "C" {
extern int simple_sgemm_impl( char* transa, char* transb, int m, int n, int k,
                float alpha, float* a, int lda, float* b, int ldb,
                float beta, float* c, int ldc );
//}
#endif
#define gettid() ((int)syscall(SYS_gettid))

using namespace torch::autograd::profiler;
using namespace torch;
using namespace torch::autograd;
using at::Tensor;

double get_time() {
  static bool init_done = false;
  static struct timespec stp = {0,0};
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);

  if(!init_done) {
    init_done = true;
    stp = tp;
  }
  double ret = (tp.tv_sec - stp.tv_sec) * 1e3 + (tp.tv_nsec - stp.tv_nsec)*1e-6;
  return ret;
}

at::Tensor simple_gemm(torch::Tensor& C, torch::Tensor A, torch::Tensor B, float alpha, bool a_trans, bool b_trans)
{
  RECORD_FUNCTION("simple_gemm", std::vector<c10::IValue>({A, B, alpha}));

  const char *aT = a_trans ? "T" : "N";
  const char *bT = b_trans ? "T" : "N";

  auto M = C.size(0);
  auto N = C.size(1);
  auto K = a_trans ? A.size(0) : A.size(1);
  auto lda = A.size(1);
  auto ldb = B.size(1);
  auto ldc = C.size(1);

  float beta = 0.0;

  float *Aptr = A.data_ptr<float>();
  float *Bptr = B.data_ptr<float>();
  float *Cptr = C.data_ptr<float>();

  simple_sgemm_impl((char*)bT, (char*)aT, N, M, K, alpha, Bptr, ldb, Aptr, lda, beta, Cptr, ldc);
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm", &simple_gemm, "TMUL GEMM Implementation");
}
