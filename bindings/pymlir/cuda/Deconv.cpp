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
  auto num_out = module::getBytes(op.getOutput());
  auto out_stype = module::getStorageType(op.getOutput());
  auto p = op.parseParam();
  auto num_elem = p.n * p.oc * p.oh * p.ow;
  auto output = getCudaData(op.getOutput());
  int32_t izp = 0;
  if (module::isUniformQuantized(op.getInput())) {
    izp = module::getUniformQuantizedType(op.getInput()).getZeroPoint();
  }
  // --------------------------------------------------------------------------
  // 1. inference int8 => float
  cuda_ptr in_f32_wrapper, kernel_f32_wrapper;
  void *in_f32 = nullptr, *kernel_f32 = nullptr;
  if (!module::getStorageType(op.getInput()).isF32()) {
    in_f32_wrapper = newCudaData(op.getInput(), cuda::DT_F32);
    kernel_f32_wrapper = newCudaData(op.getFilter(), cuda::DT_F32);
    in_f32 = in_f32_wrapper.get();
    kernel_f32 = kernel_f32_wrapper.get();
  } else {
    in_f32 = getCudaData(op.getInput());
    kernel_f32 = getCudaData(op.getFilter());
  }

  float *kernel_f32_transpose = nullptr;
  float *input_padded = nullptr;
  int padded_ih = p.ih;
  int padded_iw = p.iw;
  int pad_h_conv = p.pad_h;
  int pad_w_conv = p.pad_w;

  // When izp != 0, convert deconv to forward conv like CPU implementation
  if (izp != 0) {
    // Calculate padded input size for deconv->conv transformation
    padded_ih = (p.ih - 1) * p.sh + 1 + p.dh * (2 * p.kh - 2 - p.pad_h - p.pad_h_after) + p.output_pad_h;
    padded_iw = (p.iw - 1) * p.sw + 1 + p.dw * (2 * p.kw - 2 - p.pad_w - p.pad_w_after) + p.output_pad_w;

    // Allocate padded input
    cudaMalloc(&input_padded, p.n * p.ic * padded_ih * padded_iw * sizeof(float));

    // Pad and insert zeros for deconv (stride insertion + padding)
    cuda::padTensorForDeconv(input_padded, in_f32, p.n, p.ic, p.ih, p.iw,
                             p.kh, p.kw, p.dh, p.dw, p.sh, p.sw,
                             p.pad_h, p.pad_h_after, p.pad_w, p.pad_w_after,
                             p.output_pad_h, p.output_pad_w, (float)izp, sizeof(float));

    // Rotate kernel weights (flip spatially for deconv->conv conversion)
    cudaMalloc(&kernel_f32_transpose, p.oc * p.ic * p.kh * p.kw * sizeof(float) / p.g);
    cuda::rotateKernelWeight(kernel_f32, kernel_f32_transpose,
                            p.oc, p.ic, p.kh, p.kw, p.g, sizeof(float));

    pad_h_conv = 0;
    pad_w_conv = 0;
  } else {
    // Original logic for izp == 0
    cudaMalloc(&kernel_f32_transpose, p.ic * p.oc * p.kh * p.kw * sizeof(float) / p.g);
    cuda::permute6D(kernel_f32, kernel_f32_transpose,
      p.g, p.oc / p.g, p.ic / p.g, p.kh, p.kw, 1,
      0, 2, 1, 3, 4, 5, sizeof(float));
  }

  cudnnTensorDescriptor_t input_desc;
  cudnnCreateTensorDescriptor(&input_desc);
  cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             p.n, p.ic, izp != 0 ? padded_ih : p.ih, izp != 0 ? padded_iw : p.iw);
  cudnnFilterDescriptor_t kernel_desc;
  cudnnCreateFilterDescriptor(&kernel_desc);
  cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                              izp != 0 ? p.oc : p.ic, izp != 0 ? (p.ic / p.g) : (p.oc / p.g), p.kh, p.kw);
  cudnnTensorDescriptor_t outf32_desc;
  cudnnCreateTensorDescriptor(&outf32_desc);
  cudnnSetTensor4dDescriptor(outf32_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             p.n, p.oc, p.oh, p.ow);
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnCreateConvolutionDescriptor(&conv_desc);
  ASSERT_OP(p.pad_h == p.pad_h_after, op); // other not supported
  ASSERT_OP(p.pad_w == p.pad_w_after, op); // other not supported
  if (izp != 0) {
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc, pad_h_conv, pad_w_conv, 1, 1, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  } else {
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc, pad_h_conv, pad_w_conv, p.sh, p.sw, p.dh, p.dw,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  }
  if (p.g > 1) {
    CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv_desc, p.g));
  }
  // prepare input output memory
  auto out_f32 = cuda_malloc(num_out * sizeof(float));

  float alpha = 1.0f, beta = 0.0f;
  if (izp != 0) {
    // Use forward convolution for izp != 0
    cudnnConvolutionFwdAlgo_t fwd_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    size_t worksize = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_, input_desc, kernel_desc, conv_desc, outf32_desc, fwd_algo, &worksize));
    auto conv_buffer = cuda_malloc(worksize);
    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn_, &alpha, input_desc, input_padded, kernel_desc, kernel_f32_transpose,
        conv_desc, fwd_algo, conv_buffer.get(), worksize, &beta, outf32_desc, out_f32.get()));
    conv_buffer.reset();
  } else {
    // Use backward data (deconvolution) for izp == 0
    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    size_t worksize = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_, kernel_desc, input_desc, conv_desc, outf32_desc, algo, &worksize));
    auto conv_buffer = cuda_malloc(worksize);
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        cudnn_, &alpha, kernel_desc, kernel_f32_transpose, input_desc, in_f32,
        conv_desc, algo, conv_buffer.get(), worksize, &beta, outf32_desc, out_f32.get()));
    conv_buffer.reset();
  }

  if (!module::getStorageType(op.getInput()).isF32()) {
    in_f32_wrapper.reset();
    kernel_f32_wrapper.reset();
  }
  if (input_padded != nullptr) {
    cudaFree(input_padded);
  }
  if (kernel_f32_transpose != nullptr) {
    cudaFree(kernel_f32_transpose);
  }
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyFilterDescriptor(kernel_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  // -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
  // 2. + bias
  if (p.with_bias) {
    cuda_ptr bias_wrapper;
    void *bias = nullptr;
    if (!module::getStorageType(op.getBias()).isF32()) {
      bias_wrapper = newCudaData(op.getBias(), cuda::DT_F32);
      bias = bias_wrapper.get();
    } else {
      bias = getCudaData(op.getBias());
    }
    cudnnTensorDescriptor_t bias_desc;
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               1, p.oc, 1, 1);
    alpha = 1.0f, beta = 1.0f;
    CHECK_CUDNN(cudnnAddTensor(cudnn_, &alpha, bias_desc, bias, &beta,
                               outf32_desc, out_f32.get()));
    cudnnDestroyTensorDescriptor(bias_desc);
    if (!module::getStorageType(op.getBias()).isF32()) {
      bias_wrapper.reset();
    }
  }
  cudnnDestroyTensorDescriptor(outf32_desc);
  if (out_stype.isInteger(32)) {
    auto output = getCudaData(op.getOutput());
    cuda::convertType(out_f32.get(), output, num_elem, cuda::DT_F32,
                      cuda::DT_INT32);
    if (p.do_relu) {
      cuda::doRelu(output, num_elem, cuda::DT_INT32);
    }
    return;
  }
  if (out_stype.isa<FloatType>()) {
    if (p.do_relu) {
      cuda::doRelu(out_f32.get(), num_elem, cuda::DT_F32);
    }
    if (out_stype.isBF16() || out_stype.isF16()) {
      cuda::convertType(out_f32.get(), output, num_elem, cuda::DT_F32,
                        getCudaType(op.getOutput()));
    } else {
      cudaMemcpy(output, out_f32.get(), num_elem * sizeof(float), cudaMemcpyDeviceToDevice);
    }
  } else {
    auto out_i32 =
        newCudaData(out_f32.get(), num_out, cuda::DT_F32, cuda::DT_INT32);
    //-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
    // 3. multiplier + shift i32 => i8
    auto zero_point = module::getUniformQuantizedType(op.getOutput()).getZeroPoint();
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
                                qdm, relu, zero_point);
  }
  out_f32.reset();
}
