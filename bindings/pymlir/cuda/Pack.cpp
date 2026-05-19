//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaPackOp(top::PackOp op) {
  int axis = op.getAxis();
  auto in0_shape = module::getShape(op.getInputs()[0]);
  int dims = in0_shape.size();
  if (axis < 0) axis += dims + 1;
  int num_inputs = op.getInputs().size();
  int chunk = 1;
  for (int i = 0; i < dims; i++) chunk *= in0_shape[i];
  int elem_bytes = module::getDtypeSize(op.getInputs()[0]);

  int pre = 1;
  for (int i = 0; i < axis; i++) pre *= in0_shape[i];
  int post = 1;
  for (int i = axis; i < dims; i++) post *= in0_shape[i];

  for (int n = 0; n < num_inputs; n++) {
    auto input = getCudaData(op.getInputs()[n]);
    auto output = getCudaData(op.getOutput());
    for (int p = 0; p < pre; p++) {
      int src_off = p * post;
      int dst_off = (p * num_inputs + n) * post;
      CHECK_CUDA(cudaMemcpy((char*)output + dst_off * elem_bytes,
                            (char*)input + src_off * elem_bytes,
                            post * elem_bytes, cudaMemcpyDeviceToDevice));
    }
  }
}
