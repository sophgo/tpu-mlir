#include "../pycuda.h"
#include "cuda_helper.h"


void py_cuda::cudaArgOp(tpu::ArgOp op) {
  bool need_val = !module::isNone(op.getValues());
  bool is_argmax = op.getMode().str() == "ArgMax";
  int axis = op.getAxis();
  void *input = getCudaData(op.getInput());
  auto output_idx = newCudaData(op.getIndices(), cuda::DT_F32);
  void *output_val = need_val ? getCudaData(op.getValues()) : nullptr;
  auto input_shape = module::getShape(op.getInput());
  const int input_dims = input_shape.size();
  if (axis < 0) axis += input_dims;
  int outer_dims = 1, inner_dims = 1;
  for (int i = 0; i < axis; i++) {
    outer_dims *= input_shape[i];
  }
  for (int i = axis + 1; i < input_dims; i++) {
    inner_dims *= input_shape[i];
  }
  if (module::isCV18xx() && need_val) {
    llvm_unreachable("cv18xx does not support ArgMin/ArgMax with values output");
  }
  if (module::isCV18xx()) {
    auto input_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    cuda::argIndex(input_f32.get(), output_idx.get(), nullptr,
                    outer_dims, input_shape[axis], inner_dims, is_argmax, true);
    input_f32.reset();
  } else {
    cuda::argIndex(input, output_idx.get(), output_val, outer_dims, input_shape[axis], inner_dims, is_argmax);
  }
  cuda::convertType(output_idx.get(), getCudaData(op.getIndices()),
                    module::getNumElements(op.getIndices()), cuda::DT_F32,
                    getCudaType(op.getIndices()));
  output_idx.reset();
}

void py_cuda::cudaArgOp(top::ArgOp op) {
  bool need_val = !module::isNone(op.getValues());
  bool is_argmax = op.getMode().str() == "ArgMax";
  int axis = op.getAxis();
  void *input = getCudaData(op.getInput());
  void *output_idx = getCudaData(op.getIndices());
  void *output_val = need_val ? getCudaData(op.getValues()) : nullptr;
  auto input_shape = module::getShape(op.getInput());
  const int input_dims = input_shape.size();
  if (axis < 0) axis += input_dims;
  int outer_dims = 1, inner_dims = 1;
  for (int i = 0; i < axis; i++) {
    outer_dims *= input_shape[i];
  }
  for (int i = axis + 1; i < input_dims; i++) {
    inner_dims *= input_shape[i];
  }
  cuda::argIndex(input, output_idx, output_val, outer_dims, input_shape[axis], inner_dims, is_argmax);
}