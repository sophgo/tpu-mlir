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

extern int simple_conv2d_impl_fp(float* outputs, float *inputs, float *weights, float* bias, int N, int C, int iH, int iW,
        int K, int R, int S, int stride, int padding, int dilation, int groups);
extern int simple_conv2d_impl_bp(float* inputs, float *outputs, float *weights, int N, int C, int iH, int iW, 
	int K, int R, int S, int stride, int padding, int dilation, int groups); 
extern int simple_conv2d_impl_wu(float *weights, float *outputs, float *inputs, int N, int C, int iH, int iW, 
	int K, int R, int S, int stride, int padding, int dilation, int groups); 

#define gettid() ((int)syscall(SYS_gettid))

using namespace torch::autograd::profiler;


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

at::Tensor simple_conv2d_fp(torch::Tensor& output, torch::Tensor input, torch::Tensor weight, torch::Tensor bias, 
		int stride, int padding, int dilation, int groups)
{
  RECORD_FUNCTION("simple_conv2d_fp", std::vector<c10::IValue>({input, weight, bias}));

  auto N = input.size(0);
  auto C = input.size(1);
  auto H = input.size(2);
  auto W = input.size(3);

  auto K = weight.size(0);
  //auto C1 = weight.size(1);
  auto R = weight.size(2);
  auto S = weight.size(3);

  float *input_ptr = input.data_ptr<float>();
  float *weight_ptr = weight.data_ptr<float>();
  float *output_ptr = output.data_ptr<float>();
  float *bias_ptr = bias.data_ptr<float>();
  
  simple_conv2d_impl_fp(output_ptr, input_ptr, weight_ptr, bias_ptr, N, C, H, W,
        K, R, S, stride, padding, dilation, groups);

  //thnn_conv2d_out(output, input, weight, 
  return output;
}

at::Tensor simple_conv2d_bp(torch::Tensor& input, torch::Tensor output, torch::Tensor weight,  
		int stride, int padding, int dilation, int groups)
{
  RECORD_FUNCTION("simple_conv2d_bp", std::vector<c10::IValue>({output, weight}));

  auto N = input.size(0);
  auto C = input.size(1);
  auto H = input.size(2);
  auto W = input.size(3);

  auto K = weight.size(0);
  //auto C1 = weight.size(1);
  auto R = weight.size(2);
  auto S = weight.size(3);

  float *input_ptr = input.data_ptr<float>();
  float *weight_ptr = weight.data_ptr<float>();
  float *output_ptr = output.data_ptr<float>();
  
  simple_conv2d_impl_bp(input_ptr, output_ptr, weight_ptr, N, C, H, W,
        K, R, S, stride, padding, dilation, groups);

  //thnn_conv2d_out(output, input, weight, 
  return input;
}

at::Tensor simple_conv2d_wu(torch::Tensor& weight, torch::Tensor output, torch::Tensor input,  
		int stride, int padding, int dilation, int groups)
{
  RECORD_FUNCTION("simple_conv2d_wu", std::vector<c10::IValue>({output, input}));

  auto N = input.size(0);
  auto C = input.size(1);
  auto H = input.size(2);
  auto W = input.size(3);

  auto K = weight.size(0);
  //auto C1 = weight.size(1);
  auto R = weight.size(2);
  auto S = weight.size(3);

  float *input_ptr = input.data_ptr<float>();
  float *weight_ptr = weight.data_ptr<float>();
  float *output_ptr = output.data_ptr<float>();
  
  simple_conv2d_impl_wu(weight_ptr, output_ptr, input_ptr, N, C, H, W,
        K, R, S, stride, padding, dilation, groups);

  //thnn_conv2d_out(output, input, weight, 
  return weight;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv2d_fp", &simple_conv2d_fp, "simple conv_fp implementation");
  m.def("conv2d_bp", &simple_conv2d_bp, "simple conv_bp implementation");
  m.def("conv2d_wu", &simple_conv2d_wu, "simple conv_wu implementation");
}
