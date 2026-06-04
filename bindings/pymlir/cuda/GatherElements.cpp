#include "../pycuda.h"
#include "cuda_helper.h"


void py_cuda::cudaGatherElementsOp(tpu::GatherElementsOp op) {
  void *input = getCudaData(op.getInput());
  void *indices = getCudaData(op.getIndices());
  void *output = getCudaData(op.getOutput());
  int axis = op.getAxis();
  auto input_shape = module::getShape(op.getInput());
  auto indices_shape = module::getShape(op.getIndices());
  if (axis < 0) {
    axis += input_shape.size();
  }
  auto input_type = getCudaType(op.getInput());
  auto index_type = getCudaType(op.getIndices());
  cuda::gatherElements(indices, input, output,
             input_shape.data(), indices_shape.data(),
             input_shape.size(), axis, index_type, input_type);
}


void py_cuda::cudaGatherElementsOp(top::GatherElementsOp op) {
  void *input = getCudaData(op.getInput());
  void *indices = getCudaData(op.getIndices());
  void *output = getCudaData(op.getOutput());
  int axis = op.getAxis();
  auto input_shape = module::getShape(op.getInput());
  auto indices_shape = module::getShape(op.getIndices());
  if (axis < 0) {
    axis += input_shape.size();
  }
  auto input_type = getCudaType(op.getInput());
  auto index_type = getCudaType(op.getIndices());
  cuda::gatherElements(indices, input, output,
             input_shape.data(), indices_shape.data(),
             input_shape.size(), axis, index_type, input_type);
}
