//===----------------------------------------------------------------------===//
#include "../pycuda.h"

void py_cuda::cudaSplitOp(top::SplitOp op) {
  auto in_shape = module::getShape(op.getInput());
  int out_num = op.getNum();
  int axis = op.getAxis();
  if (axis < 0) axis += in_shape.size();
  auto split_size = module::getI64Array(op.getSplitSizeAttr());
  auto input = getCudaData(op.getInput());

  int64_t outer = 1, inner = 1;
  for (int i = 0; i < axis; i++) outer *= in_shape[i];
  for (int i = axis + 1; i < (int)in_shape.size(); i++) inner *= in_shape[i];
  int64_t copy_bytes = inner * sizeof(float);

  int64_t split_sum = 0;
  for (int o = 0; o < outer; o++) {
    int64_t src_base = o * in_shape[axis] * inner;
    for (int j = 0; j < out_num; j++) {
      auto dst = getCudaData(op.getOutputs()[j]);
      for (int s = 0; s < split_size->at(j); s++) {
        int64_t src_offset = src_base + (split_sum + s) * inner;
        int64_t dst_offset = (o * split_size->at(j) + s) * inner;
        CHECK_CUDA(cudaMemcpy((float*)dst + dst_offset, (float*)input + src_offset,
                              copy_bytes, cudaMemcpyDeviceToDevice));
      }
      split_sum += split_size->at(j);
    }
    split_sum = 0;
  }
}
