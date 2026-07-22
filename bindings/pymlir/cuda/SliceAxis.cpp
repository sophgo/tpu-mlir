//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
#include <cstdio>
#include <vector>

void py_cuda::cudaSliceAxisOp(top::SliceAxisOp op) {
  auto input = getCudaData(op.getInput());
  auto output = getCudaData(op.getOutput());

  auto in_shape = module::getShape(op.getInput());
  int dims = in_shape.size();

  // Read slice parameters from GPU (set via top.Input / set_tensor)
  cudaDeviceSynchronize();
  std::vector<float> axis_buf(1), start_buf(1), end_buf(1), step_buf(1);
  CHECK_CUDA(cudaMemcpy(axis_buf.data(), getCudaData(op.getAxis()), sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(start_buf.data(), getCudaData(op.getStart()), sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(end_buf.data(), getCudaData(op.getEnd()), sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(step_buf.data(), getCudaData(op.getStep()), sizeof(float), cudaMemcpyDeviceToHost));

  int axis = (int)axis_buf[0];
  if (axis < 0) axis += dims;
  int start = (int)start_buf[0];
  if (start < 0) start += in_shape[axis];
  int end = (int)end_buf[0];
  if (end < 0) end += in_shape[axis];
  if (end > in_shape[axis]) end = in_shape[axis];
  int step = (int)step_buf[0];
  int64_t slice_count = (end - start + step - 1) / step;

  int64_t total_out = 1;
  auto out_shape = module::getShape(op.getOutput());
  for (auto d : out_shape) total_out *= d;

  // Debug sentinel: if slice_count==0, write known values to output
  if (slice_count <= 0 || start >= end) {
    std::vector<float> sentinel(total_out);
    for (int64_t i = 0; i < total_out; i++) sentinel[i] = float(i);
    CHECK_CUDA(cudaMemcpy(output, sentinel.data(), total_out * sizeof(float), cudaMemcpyHostToDevice));
    return;
  }

  int64_t outer_size = 1;
  for (int i = 0; i < axis; i++) outer_size *= in_shape[i];
  int64_t inner_size = 1;
  for (int i = axis + 1; i < dims; i++) inner_size *= in_shape[i];
  int64_t copy_size = inner_size * sizeof(float);

  for (int64_t i = 0; i < outer_size; i++) {
    int64_t out_offset = i * slice_count * inner_size;
    for (int j = start, k = 0; j < end && k < slice_count; j += step, k++) {
      int64_t in_offset = (i * in_shape[axis] + j) * inner_size;
      CHECK_CUDA(cudaMemcpy((float*)output + out_offset + k * inner_size,
                            (float*)input + in_offset, copy_size,
                            cudaMemcpyDeviceToDevice));
    }
  }
}

