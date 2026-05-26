//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include <vector>
#include <algorithm>

void py_cuda::cudaSortOp(top::SortOp op) {
  auto in_shape = module::getShape(op.getInput());
  int dims = in_shape.size();
  int axis = op.getAxis();
  if (axis < 0) axis += dims;
  bool desc = op.getDescending();

  int in_num = module::getNumElements(op.getInput());
  std::vector<float> data(in_num), idx(in_num);
  CHECK_CUDA(cudaMemcpy(data.data(), getCudaData(op.getInput()), in_num * sizeof(float), cudaMemcpyDeviceToHost));

  // Compute strides
  int outer = 1, inner = 1;
  for (int i = 0; i < axis; i++) outer *= in_shape[i];
  for (int i = axis + 1; i < dims; i++) inner *= in_shape[i];
  int axis_len = in_shape[axis];

  // Initialize indices
  for (int i = 0; i < in_num; i++) idx[i] = (float)(i % (axis_len * inner) / inner);

  std::vector<float> sorted(in_num), sorted_idx(in_num);
  for (int o = 0; o < outer; o++) {
    int base = o * axis_len * inner;
    // Create index array and sort
    std::vector<int> order(axis_len);
    for (int a = 0; a < axis_len; a++) order[a] = a;
    if (desc)
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        for (int k = 0; k < inner; k++)
          if (data[base + a * inner + k] != data[base + b * inner + k])
            return data[base + a * inner + k] > data[base + b * inner + k];
        return a > b; });
    else
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        for (int k = 0; k < inner; k++)
          if (data[base + a * inner + k] != data[base + b * inner + k])
            return data[base + a * inner + k] < data[base + b * inner + k];
        return a < b; });

    for (int a = 0; a < axis_len; a++) {
      for (int k = 0; k < inner; k++) {
        sorted[base + a * inner + k] = data[base + order[a] * inner + k];
        sorted_idx[base + a * inner + k] = (float)order[a];
      }
    }
  }

  CHECK_CUDA(cudaMemcpy(getCudaData(op.getValues()), sorted.data(), in_num * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(getCudaData(op.getIndices()), sorted_idx.data(), in_num * sizeof(float), cudaMemcpyHostToDevice));
}

void py_cuda::cudaSortOp(tpu::SortOp op) {
  auto in_shape = module::getShape(op.getInput());
  int in_num = module::getNumElements(op.getInput());
  int dims = in_shape.size();
  int axis = op.getAxis();
  if (axis < 0) axis += dims;
  bool desc = op.getDescending();

  auto stype = module::getStorageType(op.getValues());
  std::vector<float> data(in_num);
  if (stype.isF32()) {
    CHECK_CUDA(cudaMemcpy(data.data(), getCudaData(op.getInput()), in_num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto in_f32 = cuda_malloc(in_num * sizeof(float));
    cuda::convertType(getCudaData(op.getInput()), in_f32.get(), in_num,
                      getCudaType(op.getInput()), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(data.data(), in_f32.get(), in_num * sizeof(float), cudaMemcpyDeviceToHost));
  }

  int outer = 1, inner = 1;
  for (int i = 0; i < axis; i++) outer *= in_shape[i];
  for (int i = axis + 1; i < dims; i++) inner *= in_shape[i];
  int axis_len = in_shape[axis];

  std::vector<float> sorted(in_num), sorted_idx(in_num);
  for (int o = 0; o < outer; o++) {
    int base = o * axis_len * inner;
    std::vector<int> order(axis_len);
    for (int a = 0; a < axis_len; a++) order[a] = a;
    if (desc)
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        for (int k = 0; k < inner; k++)
          if (data[base + a * inner + k] != data[base + b * inner + k])
            return data[base + a * inner + k] > data[base + b * inner + k];
        return a > b; });
    else
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        for (int k = 0; k < inner; k++)
          if (data[base + a * inner + k] != data[base + b * inner + k])
            return data[base + a * inner + k] < data[base + b * inner + k];
        return a < b; });
    for (int a = 0; a < axis_len; a++) {
      for (int k = 0; k < inner; k++) {
        sorted[base + a * inner + k] = data[base + order[a] * inner + k];
        sorted_idx[base + a * inner + k] = (float)order[a];
      }
    }
  }

  if (stype.isF32()) {
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getValues()), sorted.data(), in_num * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getIndices()), sorted_idx.data(), in_num * sizeof(float), cudaMemcpyHostToDevice));
  } else {
    auto val_f32 = cuda_malloc(in_num * sizeof(float));
    auto idx_f32 = cuda_malloc(in_num * sizeof(float));
    CHECK_CUDA(cudaMemcpy(val_f32.get(), sorted.data(), in_num * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(idx_f32.get(), sorted_idx.data(), in_num * sizeof(float), cudaMemcpyHostToDevice));
    cuda::convertType(val_f32.get(), getCudaData(op.getValues()), in_num,
                      cuda::DT_F32, getCudaType(op.getValues()));
    cuda::convertType(idx_f32.get(), getCudaData(op.getIndices()), in_num,
                      cuda::DT_F32, getCudaType(op.getIndices()));
  }
}
