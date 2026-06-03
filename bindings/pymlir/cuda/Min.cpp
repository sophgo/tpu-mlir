//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaMinOp(top::MinOp op) {
  auto inputs = op.getInputs();
  auto a = getCudaData(inputs[0]), b = getCudaData(inputs[1]);
  auto output = getCudaData(op.getOutput());
  int num = module::getNumElements(op.getOutput());
  cuda::bmMin(a, b, output, num);
}
void py_cuda::cudaMinOp(tpu::MinOp op) {
  auto inputs = op.getInputs();
  auto a = getCudaData(inputs[0]), b = getCudaData(inputs[1]);
  auto output = getCudaData(op.getOutput());
  int num = module::getNumElements(op.getOutput());
  auto stype = module::getStorageType(op.getOutput());
  if (stype.isF32()) { cuda::bmMin(a, b, output, num); }
  else {
    auto af = newCudaData(inputs[0], cuda::DT_F32);
    auto bf = newCudaData(inputs[1], cuda::DT_F32);
    auto of = cuda_malloc(num * sizeof(float));
    cuda::bmMin(af.get(), bf.get(), of.get(), num);
    cuda::convertType(of.get(), output, num, cuda::DT_F32, getCudaType(op.getOutput()));
  }
}
