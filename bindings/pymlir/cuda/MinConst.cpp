//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaMinConstOp(top::MinConstOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  float const_val = op.getConstVal().convertToDouble();
  int num = module::getNumElements(op.getOutput());
  cuda::bmMinConst(input, output, const_val, num);
}
void py_cuda::cudaMinConstOp(tpu::MinConstOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());
  float const_val = op.getConstVal().convertToDouble();
  int num = module::getNumElements(op.getOutput());
  auto stype = module::getStorageType(op.getOutput());
  if (stype.isF32()) { cuda::bmMinConst(input, output, const_val, num); }
  else {
    auto inf = newCudaData(input, num, getCudaType(op.getInput()), cuda::DT_F32);
    auto of = cuda_malloc(num * sizeof(float));
    cuda::bmMinConst(inf.get(), of.get(), const_val, num);
    cuda::convertType(of.get(), output, num, cuda::DT_F32, getCudaType(op.getOutput()));
  }
}
