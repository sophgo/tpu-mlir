//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
#include <vector>

void py_cuda::cudaScatterElementsOp(top::ScatterElementsOp op) {
  auto in_shape = module::getShape(op.getInput());
  int r = in_shape.size();
  int axis = op.getAxis();
  if (axis < 0) axis += r;
  int in_num = module::getNumElements(op.getInput());
  int idx_num = module::getNumElements(op.getIndices());
  int upd_num = module::getNumElements(op.getUpdates());

  // Read indices as F32
  std::vector<float> idx_h(idx_num), upd_h(upd_num);
  if (getCudaType(op.getIndices()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(idx_h.data(), getCudaData(op.getIndices()),
                          idx_num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getIndices(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(idx_h.data(), tmp.get(), idx_num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  if (getCudaType(op.getUpdates()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(upd_h.data(), getCudaData(op.getUpdates()),
                          upd_num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getUpdates(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(upd_h.data(), tmp.get(), upd_num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  // Prepare output buffer (F32)
  std::shared_ptr<void> out_guard;
  void *out_buf = getCudaData(op.getOutput());
  if (getCudaType(op.getOutput()) != cuda::DT_F32) {
    out_guard = cuda_malloc(in_num * sizeof(float));
    out_buf = out_guard.get();
  }

  // Copy input -> output
  if (getCudaType(op.getInput()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(out_buf, getCudaData(op.getInput()),
                          in_num * sizeof(float), cudaMemcpyDeviceToDevice));
  } else {
    auto data_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(out_buf, data_f32.get(),
                          in_num * sizeof(float), cudaMemcpyDeviceToDevice));
  }

  // Compute flat indices
  std::vector<int64_t> in_stride(r, 1);
  for (int i = r - 2; i >= 0; i--)
    in_stride[i] = in_stride[i + 1] * in_shape[i + 1];
  std::vector<int> flat_idx(upd_num);
  for (int n = 0; n < upd_num; n++) {
    int rem = n;
    int64_t off = 0;
    for (int d = 0; d < r; d++) {
      int coord = rem / in_stride[d];
      rem %= in_stride[d];
      if (d == axis) coord = (int)idx_h[n];
      off += coord * in_stride[d];
    }
    flat_idx[n] = off;
  }

  auto fi_d = cuda_malloc(upd_num * sizeof(int));
  CHECK_CUDA(cudaMemcpy(fi_d.get(), flat_idx.data(),
                        upd_num * sizeof(int), cudaMemcpyHostToDevice));
  auto upd_gpu = cuda_malloc(upd_num * sizeof(float));
  CHECK_CUDA(cudaMemcpy(upd_gpu.get(), upd_h.data(),
                        upd_num * sizeof(float), cudaMemcpyHostToDevice));
  cuda::scatterElements(out_buf, upd_gpu.get(), fi_d.get(), upd_num, false);

  if (out_guard)
    cuda::convertType(out_buf, getCudaData(op.getOutput()), in_num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
}

void py_cuda::cudaScatterElementsOp(tpu::ScatterElementsOp op) {
  auto in_shape = module::getShape(op.getInput());
  int r = in_shape.size();
  int axis = op.getAxis();
  if (axis < 0) axis += r;
  int in_num = module::getNumElements(op.getInput());
  int idx_num = module::getNumElements(op.getIndices());
  int upd_num = module::getNumElements(op.getUpdates());

  std::vector<float> idx_h(idx_num), upd_h(upd_num);
  if (getCudaType(op.getIndices()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(idx_h.data(), getCudaData(op.getIndices()),
                          idx_num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getIndices(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(idx_h.data(), tmp.get(), idx_num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  if (getCudaType(op.getUpdates()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(upd_h.data(), getCudaData(op.getUpdates()),
                          upd_num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getUpdates(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(upd_h.data(), tmp.get(), upd_num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  std::shared_ptr<void> out_guard;
  void *out_buf = getCudaData(op.getOutput());
  if (getCudaType(op.getOutput()) != cuda::DT_F32) {
    out_guard = cuda_malloc(in_num * sizeof(float));
    out_buf = out_guard.get();
  }

  if (getCudaType(op.getInput()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(out_buf, getCudaData(op.getInput()),
                          in_num * sizeof(float), cudaMemcpyDeviceToDevice));
  } else {
    auto data_f32 = newCudaData(op.getInput(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(out_buf, data_f32.get(),
                          in_num * sizeof(float), cudaMemcpyDeviceToDevice));
  }

  std::vector<int64_t> in_stride(r, 1);
  for (int i = r - 2; i >= 0; i--)
    in_stride[i] = in_stride[i + 1] * in_shape[i + 1];
  std::vector<int> flat_idx(upd_num);
  for (int n = 0; n < upd_num; n++) {
    int rem = n;
    int64_t off = 0;
    for (int d = 0; d < r; d++) {
      int coord = rem / in_stride[d];
      rem %= in_stride[d];
      if (d == axis) coord = (int)idx_h[n];
      off += coord * in_stride[d];
    }
    flat_idx[n] = off;
  }

  auto fi_d = cuda_malloc(upd_num * sizeof(int));
  CHECK_CUDA(cudaMemcpy(fi_d.get(), flat_idx.data(),
                        upd_num * sizeof(int), cudaMemcpyHostToDevice));
  auto upd_gpu = cuda_malloc(upd_num * sizeof(float));
  CHECK_CUDA(cudaMemcpy(upd_gpu.get(), upd_h.data(),
                        upd_num * sizeof(float), cudaMemcpyHostToDevice));
  cuda::scatterElements(out_buf, upd_gpu.get(), fi_d.get(), upd_num, false);

  if (out_guard)
    cuda::convertType(out_buf, getCudaData(op.getOutput()), in_num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
}
