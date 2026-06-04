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

namespace {

int64_t getSpatialSize(mlir::Value input) {
  auto shape = module::getShape(input);
  int64_t spatial = 1;
  for (size_t i = 2; i < shape.size(); ++i) {
    spatial *= shape[i];
  }
  return spatial;
}

bool hasTensor(mlir::Value value) {
  return !value.getType().isa<mlir::NoneType>();
}

} // namespace


void py_cuda::cudaBatchNormOp(top::BatchNormOp op) {
  auto input_shape = module::getShape(op.getInput());
  int n = input_shape[0], c = input_shape[1];
  int spatial = getSpatialSize(op.getInput());
  int total = module::getNumElements(op.getInput());
  float eps = op.getEpsilon().convertToDouble();
  bool do_relu = op.getDoRelu();
  bool have_gamma = hasTensor(op.getGamma());
  bool have_beta = hasTensor(op.getBeta());

  auto input_f32 = getCudaType(op.getInput()) == cuda::DT_F32
                       ? cuda_ptr()
                       : newCudaData(op.getInput(), cuda::DT_F32);
  auto mean_f32 = getCudaType(op.getMean()) == cuda::DT_F32
                      ? cuda_ptr()
                      : newCudaData(op.getMean(), cuda::DT_F32);
  auto var_f32 = getCudaType(op.getVariance()) == cuda::DT_F32
                     ? cuda_ptr()
                     : newCudaData(op.getVariance(), cuda::DT_F32);
  auto gamma_f32 = have_gamma && getCudaType(op.getGamma()) != cuda::DT_F32
                       ? newCudaData(op.getGamma(), cuda::DT_F32) : cuda_ptr();
  auto beta_f32  = have_beta && getCudaType(op.getBeta()) != cuda::DT_F32
                       ? newCudaData(op.getBeta(), cuda::DT_F32) : cuda_ptr();
  auto output_f32 = getCudaType(op.getOutput()) == cuda::DT_F32
                        ? cuda_ptr() : cuda_malloc(total * sizeof(float));

  void *input  = input_f32  ? input_f32.get()  : getCudaData(op.getInput());
  void *mean   = mean_f32   ? mean_f32.get()   : getCudaData(op.getMean());
  void *var    = var_f32    ? var_f32.get()    : getCudaData(op.getVariance());
  void *gamma  = have_gamma ? (gamma_f32 ? gamma_f32.get() : getCudaData(op.getGamma())) : nullptr;
  void *beta   = have_beta  ? (beta_f32  ? beta_f32.get()  : getCudaData(op.getBeta()))  : nullptr;
  void *output = output_f32 ? output_f32.get() : getCudaData(op.getOutput());

  cuda::bmBatchNorm(input, output, n, c, spatial, gamma, beta, mean, var, eps, do_relu);

  if (output_f32) {
    cuda::convertType(output_f32.get(), getCudaData(op.getOutput()), total,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  }
}



void py_cuda::cudaBatchNormOp(tpu::BatchNormTrainOp op) {
  auto input_shape = module::getShape(op.getInput());
  int n = input_shape[0], c = input_shape[1];
  int spatial = getSpatialSize(op.getInput());
  int total = module::getNumElements(op.getInput());
  float eps = op.getEpsilon().convertToDouble();
  float momentum = op.getMomentum().convertToDouble();
  bool do_relu = op.getDoRelu();
  bool have_gamma = hasTensor(op.getGamma());
  bool have_beta = hasTensor(op.getBeta());

  auto input_f32  = getCudaType(op.getInput())  == cuda::DT_F32 ? cuda_ptr() : newCudaData(op.getInput(), cuda::DT_F32);
  auto mean_f32   = getCudaType(op.getMean())   == cuda::DT_F32 ? cuda_ptr() : newCudaData(op.getMean(), cuda::DT_F32);
  auto var_f32    = getCudaType(op.getVar())    == cuda::DT_F32 ? cuda_ptr() : newCudaData(op.getVar(), cuda::DT_F32);
  auto gamma_f32  = have_gamma && getCudaType(op.getGamma()) != cuda::DT_F32 ? newCudaData(op.getGamma(), cuda::DT_F32) : cuda_ptr();
  auto beta_f32   = have_beta  && getCudaType(op.getBeta())  != cuda::DT_F32 ? newCudaData(op.getBeta(),  cuda::DT_F32) : cuda_ptr();
  auto output_f32       = getCudaType(op.getOutput())       == cuda::DT_F32 ? cuda_ptr() : cuda_malloc(total * sizeof(float));
  auto mean_out_f32     = getCudaType(op.getMeanOut())      == cuda::DT_F32 ? cuda_ptr() : cuda_malloc(c * sizeof(float));
  auto saved_invstd_f32 = getCudaType(op.getSavedInvstd())  == cuda::DT_F32 ? cuda_ptr() : cuda_malloc(c * sizeof(float));
  auto running_mean_f32 = getCudaType(op.getRunningMean())  == cuda::DT_F32 ? cuda_ptr() : cuda_malloc(c * sizeof(float));
  auto running_var_f32  = getCudaType(op.getRunningVar())   == cuda::DT_F32 ? cuda_ptr() : cuda_malloc(c * sizeof(float));

  void *input   = input_f32  ? input_f32.get()  : getCudaData(op.getInput());
  void *mean    = mean_f32   ? mean_f32.get()   : getCudaData(op.getMean());
  void *var     = var_f32    ? var_f32.get()    : getCudaData(op.getVar());
  void *gamma   = have_gamma ? (gamma_f32 ? gamma_f32.get() : getCudaData(op.getGamma())) : nullptr;
  void *beta    = have_beta  ? (beta_f32  ? beta_f32.get()  : getCudaData(op.getBeta()))  : nullptr;
  void *output  = output_f32 ? output_f32.get() : getCudaData(op.getOutput());
  void *mean_out     = mean_out_f32     ? mean_out_f32.get()     : getCudaData(op.getMeanOut());
  void *saved_invstd = saved_invstd_f32 ? saved_invstd_f32.get() : getCudaData(op.getSavedInvstd());
  void *running_mean = running_mean_f32 ? running_mean_f32.get() : getCudaData(op.getRunningMean());
  void *running_var  = running_var_f32  ? running_var_f32.get()  : getCudaData(op.getRunningVar());

  cuda::bmBatchNormTrain(input, mean, var, gamma, beta, output, mean_out,
                         saved_invstd, running_mean, running_var,
                         n, c, spatial, eps, momentum, do_relu);

  if (output_f32)       cuda::convertType(output_f32.get(), getCudaData(op.getOutput()), total, cuda::DT_F32, getCudaType(op.getOutput()));
  if (mean_out_f32)     cuda::convertType(mean_out_f32.get(), getCudaData(op.getMeanOut()), c, cuda::DT_F32, getCudaType(op.getMeanOut()));
  if (saved_invstd_f32) cuda::convertType(saved_invstd_f32.get(), getCudaData(op.getSavedInvstd()), c, cuda::DT_F32, getCudaType(op.getSavedInvstd()));
  if (running_mean_f32) cuda::convertType(running_mean_f32.get(), getCudaData(op.getRunningMean()), c, cuda::DT_F32, getCudaType(op.getRunningMean()));
  if (running_var_f32)  cuda::convertType(running_var_f32.get(), getCudaData(op.getRunningVar()), c, cuda::DT_F32, getCudaType(op.getRunningVar()));
}


void py_cuda::cudaBatchNormBwdOp(tpu::BatchNormBwdOp op) {
  auto input_shape = module::getShape(op.getInput());
  int n = input_shape[0], c = input_shape[1];
  int spatial = getSpatialSize(op.getInput());
  int total = n * c * spatial;

  bool have_gamma = hasTensor(op.getWeightOpt());
  float *gamma_ptr = have_gamma ? (float *)getCudaData(op.getWeightOpt()) : nullptr;

  auto dxhut    = cuda_malloc(total * sizeof(float));
  auto dgamma   = cuda_malloc(c * sizeof(float));
  auto dbeta    = cuda_malloc(c * sizeof(float));
  auto dx2_tmp  = cuda_malloc(c * sizeof(float));
  auto dx3      = cuda_malloc(c * sizeof(float));

  CHECK_CUDA(cudaMemset(dgamma.get(),  0, c * sizeof(float)));
  CHECK_CUDA(cudaMemset(dbeta.get(),   0, c * sizeof(float)));
  CHECK_CUDA(cudaMemset(dx2_tmp.get(), 0, c * sizeof(float)));
  CHECK_CUDA(cudaMemset(dx3.get(),     0, c * sizeof(float)));

  cuda::bmBatchNormBwd(getCudaData(op.getGradOut()), getCudaData(op.getInput()),
                        gamma_ptr,
                        getCudaData(op.getSavedMean()), getCudaData(op.getSavedInvstd()),
                        dxhut.get(), dgamma.get(), dbeta.get(),
                        dx2_tmp.get(), dx3.get(),
                        getCudaData(op.getGradIn()),
                        n, c, spatial);


  CHECK_CUDA(cudaMemcpy(getCudaData(op.getWeightGrad()), dgamma.get(),
                        c * sizeof(float), cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemcpy(getCudaData(op.getBiasGrad()), dbeta.get(),
                        c * sizeof(float), cudaMemcpyDeviceToDevice));
}

void py_cuda::cudaBatchNormBwdOp(top::BatchNormBwdOp op) {
  auto input_shape = module::getShape(op.getInput());
  int n = input_shape[0], c = input_shape[1];
  int spatial = getSpatialSize(op.getInput());
  int total = n * c * spatial;

  bool have_gamma = hasTensor(op.getWeightOpt());
  float *gamma_ptr = have_gamma ? (float *)getCudaData(op.getWeightOpt()) : nullptr;

  auto dxhut    = cuda_malloc(total * sizeof(float));
  auto dgamma   = cuda_malloc(c * sizeof(float));
  auto dbeta    = cuda_malloc(c * sizeof(float));
  auto dx2_tmp  = cuda_malloc(c * sizeof(float));
  auto dx3      = cuda_malloc(c * sizeof(float));

  CHECK_CUDA(cudaMemset(dgamma.get(),  0, c * sizeof(float)));
  CHECK_CUDA(cudaMemset(dbeta.get(),   0, c * sizeof(float)));
  CHECK_CUDA(cudaMemset(dx2_tmp.get(), 0, c * sizeof(float)));
  CHECK_CUDA(cudaMemset(dx3.get(),     0, c * sizeof(float)));

  cuda::bmBatchNormBwd(getCudaData(op.getGradOut()), getCudaData(op.getInput()),
                        gamma_ptr,
                        getCudaData(op.getSavedMean()), getCudaData(op.getSavedInvstd()),
                        dxhut.get(), dgamma.get(), dbeta.get(),
                        dx2_tmp.get(), dx3.get(),
                        getCudaData(op.getGradIn()),
                        n, c, spatial);

  CHECK_CUDA(cudaMemcpy(getCudaData(op.getWeightGrad()), dgamma.get(),
                        c * sizeof(float), cudaMemcpyDeviceToDevice));
  CHECK_CUDA(cudaMemcpy(getCudaData(op.getBiasGrad()), dbeta.get(),
                        c * sizeof(float), cudaMemcpyDeviceToDevice));
}
