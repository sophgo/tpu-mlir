//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaDeconvOp(tpu::DeconvOp op) {
  if (!module::isUniformQuantized(op.getInput())) {
    UNREACHABLE_OP("Not Implemented", op);
  }
  auto num_out = module::getBytes(op.getOutput());
  auto out_stype = module::getStorageType(op.getOutput());
  auto p = op.parseParam();
  auto output = getCudaData(op.getOutput());
  // --------------------------------------------------------------------------
  // 1. inference int8 => float
  auto in_f32 = newCudaData(op.getInput(), cuda::DT_F32);
  auto kernel_f32 = newCudaData(op.getFilter(), cuda::DT_F32);
  cudnnTensorDescriptor_t input_desc;
  cudnnCreateTensorDescriptor(&input_desc);
  cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             p.n, p.ic, p.ih, p.iw);
  cudnnFilterDescriptor_t kernel_desc;
  cudnnCreateFilterDescriptor(&kernel_desc);
  cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                             p.oc, p.ic / p.g, p.kh, p.kw);
  cudnnTensorDescriptor_t outf32_desc;
  cudnnCreateTensorDescriptor(&outf32_desc);
  cudnnSetTensor4dDescriptor(outf32_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             p.n, p.oc, p.oh, p.ow);
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnCreateConvolutionDescriptor(&conv_desc);
  ASSERT_OP(p.pad_h == p.pad_h_after, op); // other not supported
  ASSERT_OP(p.pad_w == p.pad_w_after, op); // other not supported
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
      conv_desc, p.pad_h, p.pad_w, p.sh, p.sw, p.dh, p.dw,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  if (p.g > 1) {
    CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv_desc, p.g));
  }
  // prepare input output memory
  auto out_f32 = cuda_malloc(num_out * sizeof(float));

  cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  // forward conv
  size_t worksize = 0;
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn_, kernel_desc, input_desc, conv_desc, outf32_desc, algo,
      &worksize));
  auto conv_buffer = cuda_malloc(worksize);
  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnConvolutionBackwardData(
      cudnn_, &alpha, kernel_desc, kernel_f32.get(), input_desc, in_f32.get(),
      conv_desc, algo, conv_buffer.get(), worksize, &beta, outf32_desc,
      out_f32.get()));
  conv_buffer.reset();
  in_f32.reset();
  kernel_f32.reset();
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyFilterDescriptor(kernel_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  // 2. + bias
  if (p.with_bias) {
    auto bias = newCudaData(op.getBias(), cuda::DT_F32);
    cudnnTensorDescriptor_t bias_desc;
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, p.oc, 1, 1);
    alpha = 1.0f, beta = 1.0f;
    CHECK_CUDNN(cudnnAddTensor(cudnn_, &alpha, bias_desc, bias.get(), &beta,
                               outf32_desc, out_f32.get()));
    cudnnDestroyTensorDescriptor(bias_desc);
  }
  cudnnDestroyTensorDescriptor(outf32_desc);
  if (out_stype.isInteger(32)) {
    auto output = getCudaData(op.getOutput());
    cuda::convertType(out_f32.get(), output, num_out, cuda::DT_F32,
                      cuda::DT_INT32);
    if (p.do_relu) {
      cuda::doRelu(output, num_out, cuda::DT_INT32);
    }
    return;
  }
  auto out_i32 =
      newCudaData(out_f32.get(), num_out, cuda::DT_F32, cuda::DT_INT32);
  out_f32.reset();
  //-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  // 3. multiplier + shift i32 => i8
  auto cudaMults = cuda_malloc(p.oc * sizeof(int32_t));
  auto cudaShifts = cuda_malloc(p.oc * sizeof(int32_t));
  auto rshift_v = module::getI64Array(op.getRshift().value());
  auto multiplier_v =
      module::getI64Array(op.getMultiplier(), rshift_v->size(), 1);
  std::vector<int32_t> m(multiplier_v->begin(), multiplier_v->end());
  std::vector<int32_t> rs(rshift_v->begin(), rshift_v->end());
  CHECK_CUDA(cudaMemcpy(cudaMults.get(), m.data(), m.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(cudaShifts.get(), rs.data(),
                        rs.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  bool sign = !out_stype.isUnsignedInteger(8);
  bool qdm = module::isCV18xx();
  bool relu = sign && p.do_relu;
  cuda::requantInt8Perchannel(out_i32.get(), output, cudaMults.get(),
                              cudaShifts.get(), p.n, p.oc, p.oh, p.ow, sign,
                              qdm, relu);
}
