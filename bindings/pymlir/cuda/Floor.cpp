#include "../pycuda.h"
#include "cuda_helper.h"


void py_cuda::cudaFloorOp(top::FloorOp op) {
  void *input = getCudaData(op.getInput());
  void *output = getCudaData(op.getOutput());
  auto num_element = module::getNumElements(op.getInput());
  cuda::bmActive(input, output, num_element, cuda::ACTIVE_FLOOR);
}