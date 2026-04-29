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
  int outer_dims = 1;
  for (int i = 0; i < axis; i++) {
    outer_dims *= input_shape[i];
  }
  int inner_dims = 1;
  for (int i = axis + 1; i < input_shape.size(); i++) {
    inner_dims *= input_shape[i];
  }
  auto input_type = getCudaType(op.getInput());
  auto index_type = getCudaType(op.getIndices());
  cuda::gatherElements(indices, input, output,
             indices_shape[axis], input_shape[axis],
             outer_dims, inner_dims, index_type, input_type);
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
  int outer_dims = 1;
  for (int i = 0; i < axis; i++) {
    outer_dims *= input_shape[i];
  }
  int inner_dims = 1;
  for (int i = axis + 1; i < input_shape.size(); i++) {
    inner_dims *= input_shape[i];
  }
  auto input_type = getCudaType(op.getInput());
  auto index_type = getCudaType(op.getIndices());
  cuda::gatherElements(indices, input, output,
             indices_shape[axis], input_shape[axis],
             outer_dims, inner_dims, index_type, input_type);
}