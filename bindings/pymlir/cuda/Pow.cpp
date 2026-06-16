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
