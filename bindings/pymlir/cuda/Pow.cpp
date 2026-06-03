//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
#include <vector>
#include <cmath>

void py_cuda::cudaPowOp(top::PowOp op) {
  int num = module::getNumElements(op.getInput());
  float exponent = op.getExponent().convertToDouble();
  std::vector<float> in(num);
  if (getCudaType(op.getInput()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(in.data(), getCudaData(op.getInput()),
                          num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getInput(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(in.data(), tmp.get(), num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  for (int i = 0; i < num; i++)
    in[i] = powf(in[i], exponent);
  auto out = cuda_malloc(num * sizeof(float));
  CHECK_CUDA(cudaMemcpy(out.get(), in.data(), num * sizeof(float),
                        cudaMemcpyHostToDevice));
  if (getCudaType(op.getOutput()) != cuda::DT_F32)
    cuda::convertType(out.get(), getCudaData(op.getOutput()), num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  else
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out.get(),
                          num * sizeof(float), cudaMemcpyDeviceToDevice));
}

void py_cuda::cudaPow2Op(top::Pow2Op op) {
  int num = module::getNumElements(op.getInput());
  float const_val = op.getConstVal().convertToDouble();
  float log_c = logf(const_val);
  std::vector<float> in(num);
  if (getCudaType(op.getInput()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(in.data(), getCudaData(op.getInput()),
                          num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getInput(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(in.data(), tmp.get(), num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  for (int i = 0; i < num; i++)
    in[i] = expf(in[i] * log_c);
  auto out = cuda_malloc(num * sizeof(float));
  CHECK_CUDA(cudaMemcpy(out.get(), in.data(), num * sizeof(float),
                        cudaMemcpyHostToDevice));
  if (getCudaType(op.getOutput()) != cuda::DT_F32)
    cuda::convertType(out.get(), getCudaData(op.getOutput()), num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  else
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out.get(),
                          num * sizeof(float), cudaMemcpyDeviceToDevice));
}

void py_cuda::cudaPow3Op(top::Pow3Op op) {
  auto in0 = op.getInputs()[0];
  auto in1 = op.getInputs()[1];
  int num = module::getNumElements(in0);
  std::vector<float> d0(num), d1(num);
  if (getCudaType(in0) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(d0.data(), getCudaData(in0),
                          num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(in0, cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(d0.data(), tmp.get(), num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  if (getCudaType(in1) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(d1.data(), getCudaData(in1),
                          num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(in1, cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(d1.data(), tmp.get(), num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  for (int i = 0; i < num; i++)
    d0[i] = powf(d0[i], d1[i]);
  auto out = cuda_malloc(num * sizeof(float));
  CHECK_CUDA(cudaMemcpy(out.get(), d0.data(), num * sizeof(float),
                        cudaMemcpyHostToDevice));
  if (getCudaType(op.getOutput()) != cuda::DT_F32)
    cuda::convertType(out.get(), getCudaData(op.getOutput()), num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  else
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out.get(),
                          num * sizeof(float), cudaMemcpyDeviceToDevice));
}
