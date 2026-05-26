//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaUnpackOp(top::UnpackOp op) {
  auto input = getCudaData(op.getInput());
  auto in_shape = module::getShape(op.getInput());
  int axis = op.getAxis();
  int dims = in_shape.size();
  if (axis < 0) axis += dims;

  int outer = 1, inner = 1;
  for (int i = 0; i < axis; i++) outer *= in_shape[i];
  int axis_len = in_shape[axis];
  for (int i = axis + 1; i < dims; i++) inner *= in_shape[i];
  int chunk_size = axis_len / op.getOutputs().size();
  int elem_bytes = module::getDtypeSize(op.getInput());

  for (size_t idx = 0; idx < op.getOutputs().size(); idx++) {
    auto output = getCudaData(op.getOutputs()[idx]);
    for (int o = 0; o < outer; o++) {
      int src_off = (o * axis_len + idx * chunk_size) * inner;
      int dst_off = (o * chunk_size) * inner;
      CHECK_CUDA(cudaMemcpy((char*)output + dst_off * elem_bytes,
                            (char*)input + src_off * elem_bytes,
                            chunk_size * inner * elem_bytes,
                            cudaMemcpyDeviceToDevice));
    }
  }
}
