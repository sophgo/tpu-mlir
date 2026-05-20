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

void py_cuda::cudaConvBwdWeightOp(tpu::ConvBwdWeightOp op) {
  auto p = op.parseParam();
  bool need_pad = p.pht != p.phb || p.pwl != p.pwr;

  // cuDNN only supports symmetric padding — pre-pad input if needed
  int ih = p.ih, iw = p.iw;
  int pad_h = p.phb, pad_w = p.pwr;
  cuda_ptr in_f32;
  if (need_pad) {
    ih = p.ih + p.pht + p.phb;
    iw = p.iw + p.pwl + p.pwr;
    pad_h = 0;
    pad_w = 0;
    int num = p.n * p.ic * ih * iw;
    in_f32 = cuda_malloc(num * sizeof(float));
    auto input = getCudaData(op.getInput());
    if (module::getStorageType(op.getInput()).isF32()) {
      cuda::pad4D(input, in_f32.get(), p.n, p.ic, p.ih, p.iw, p.pht, p.phb,
                  p.pwl, p.pwr, sizeof(float));
    } else {
      auto in_tmp = newCudaData(op.getInput(), cuda::DT_F32);
      cuda::pad4D(in_tmp.get(), in_f32.get(), p.n, p.ic, p.ih, p.iw, p.pht,
                  p.phb, p.pwl, p.pwr, sizeof(float));
      in_tmp.reset();
    }
  }

  // input descriptor [N, IC, IH, IW]
  cudnnTensorDescriptor_t input_desc;
  cudnnCreateTensorDescriptor(&input_desc);
  cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             p.n, p.ic, ih, iw);

  // gradout (dy) descriptor [N, OC, OH, OW]
  cudnnTensorDescriptor_t dy_desc;
  cudnnCreateTensorDescriptor(&dy_desc);
  cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             p.n, p.oc, p.oh, p.ow);

  // dw filter descriptor [OC, IC/G, KH, KW]
  cudnnFilterDescriptor_t dw_desc;
  cudnnCreateFilterDescriptor(&dw_desc);
  cudnnSetFilter4dDescriptor(dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                             p.oc, p.ic / p.groups, p.kh, p.kw);

  // convolution descriptor
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnCreateConvolutionDescriptor(&conv_desc);
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, p.sh, p.sw, p.dh, p.dw,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  if (p.groups > 1) {
    CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv_desc, p.groups));
  }

  // prepare input and gradout pointers
  cuda_ptr in_nonpad_f32;
  if (!need_pad && !module::getStorageType(op.getInput()).isF32()) {
    in_nonpad_f32 = newCudaData(op.getInput(), cuda::DT_F32);
  }
  void *input_ptr = need_pad
      ? in_f32.get()
      : (module::getStorageType(op.getInput()).isF32()
             ? getCudaData(op.getInput())
             : in_nonpad_f32.get());

  auto gradout = getCudaData(op.getGradout());
  cuda_ptr gradout_f32;
  if (!module::getStorageType(op.getGradout()).isF32()) {
    gradout_f32 = newCudaData(op.getGradout(), cuda::DT_F32);
    gradout = gradout_f32.get();
  }

  // output weight gradient
  int dw_num = p.oc * (p.ic / p.groups) * p.kh * p.kw;
  auto dw_f32 = cuda_malloc(dw_num * sizeof(float));

  // get algorithm and workspace
  cudnnConvolutionBwdFilterAlgo_t algo =
      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  size_t worksize = 0;
  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnn_, input_desc, dy_desc, conv_desc, dw_desc, algo, &worksize));
  auto workspace = cuda_malloc(worksize);

  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnConvolutionBackwardFilter(
      cudnn_, &alpha, input_desc, input_ptr, dy_desc, gradout, conv_desc,
      algo, workspace.get(), worksize, &beta, dw_desc, dw_f32.get()));

  // copy dw to output
  if (!module::getStorageType(op.getOutput()).isF32()) {
    cuda::convertType(dw_f32.get(), getCudaData(op.getOutput()), dw_num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  } else {
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), dw_f32.get(),
                          dw_num * sizeof(float), cudaMemcpyDeviceToDevice));
  }

  // cleanup
  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(dy_desc);
  cudnnDestroyFilterDescriptor(dw_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  dw_f32.reset();
  workspace.reset();
  if (in_f32) in_f32.reset();
  if (gradout_f32) gradout_f32.reset();
}

void py_cuda::cudaConvBwdWeightOp(top::ConvBwdWeightOp op) {
  // top::ConvBwdWeightOp has no parseParam() — extract attributes manually
  auto input_shape = module::getI64Array(op.getInputShape());
  auto grad_out_shape = module::getI64Array(op.getGradOutShape());
  auto kernel_shape = module::getI64Array(op.getKernelShape());
  auto stride = module::getI64Array(op.getStride());
  auto dilations = module::getI64Array(op.getDilations());
  auto padding = module::getI64Array(op.getPadding());

  int64_t n = input_shape->at(0), ic = input_shape->at(1);
  int64_t ih = input_shape->at(2), iw = input_shape->at(3);
  int64_t oc = grad_out_shape->at(1), oh = grad_out_shape->at(2), ow = grad_out_shape->at(3);
  int64_t kh = kernel_shape->at(0), kw = kernel_shape->at(1);
  int64_t sh = stride->at(0), sw = stride->at(1);
  int64_t dh = dilations->at(0), dw = dilations->at(1);
  int64_t pht = padding->at(0), phb = padding->at(2);
  int64_t pwl = padding->at(1), pwr = padding->at(3);
  int64_t groups = op.getGroups();

  bool need_pad = pht != phb || pwl != pwr;
  int cur_ih = ih, cur_iw = iw;
  int pad_h = phb, pad_w = pwr;
  cuda_ptr in_f32;
  if (need_pad) {
    cur_ih = ih + pht + phb;
    cur_iw = iw + pwl + pwr;
    pad_h = 0;
    pad_w = 0;
    int num = n * ic * cur_ih * cur_iw;
    in_f32 = cuda_malloc(num * sizeof(float));
    cuda::pad4D(getCudaData(op.getInput()), in_f32.get(), n, ic, ih, iw,
                pht, phb, pwl, pwr, sizeof(float));
  }

  cudnnTensorDescriptor_t input_desc;
  cudnnCreateTensorDescriptor(&input_desc);
  cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             n, ic, cur_ih, cur_iw);

  cudnnTensorDescriptor_t dy_desc;
  cudnnCreateTensorDescriptor(&dy_desc);
  cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                             n, oc, oh, ow);

  cudnnFilterDescriptor_t dw_desc;
  cudnnCreateFilterDescriptor(&dw_desc);
  cudnnSetFilter4dDescriptor(dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                             oc, ic / groups, kh, kw);

  cudnnConvolutionDescriptor_t conv_desc;
  cudnnCreateConvolutionDescriptor(&conv_desc);
  CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, sh, sw, dh, dw,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  if (groups > 1) {
    CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv_desc, groups));
  }

  cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  size_t worksize = 0;
  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnn_, input_desc, dy_desc, conv_desc, dw_desc, algo, &worksize));
  auto workspace = cuda_malloc(worksize);

  void *in_ptr = need_pad ? in_f32.get() : getCudaData(op.getInput());

  int dw_num = oc * (ic / groups) * kh * kw;
  auto dw_f32 = cuda_malloc(dw_num * sizeof(float));

  float alpha = 1.0f, beta = 0.0f;
  CHECK_CUDNN(cudnnConvolutionBackwardFilter(
      cudnn_, &alpha, input_desc, in_ptr, dy_desc,
      getCudaData(op.getGradout()), conv_desc, algo, workspace.get(),
      worksize, &beta, dw_desc, dw_f32.get()));

  CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), dw_f32.get(),
                        dw_num * sizeof(float), cudaMemcpyDeviceToDevice));

  cudnnDestroyTensorDescriptor(input_desc);
  cudnnDestroyTensorDescriptor(dy_desc);
  cudnnDestroyFilterDescriptor(dw_desc);
  cudnnDestroyConvolutionDescriptor(conv_desc);
  dw_f32.reset();
  workspace.reset();
  if (in_f32) in_f32.reset();
}
