//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaMeshGridOp(top::MeshGridOp op) {
  auto inputs = op.getInputs();
  auto outputs = op.getOutputs();
  int num = inputs.size();
  bool is_rev = op.getIsReverse();

  auto out_shape = module::getShape(outputs[0]);
  int64_t total = 1;
  for (auto d : out_shape) total *= d;

  for (int j = 0; j < num; j++) {
    int in_j = is_rev ? (num - 1 - j) : j;
    // stride for this dim: product of dims after this one
    int64_t inner = 1;
    for (int k = j + 1; k < num; k++) inner *= out_shape[k];
    cuda::meshGrid(getCudaData(inputs[in_j]),
                   getCudaData(outputs[in_j]), total, inner, out_shape[j]);
  }
}
