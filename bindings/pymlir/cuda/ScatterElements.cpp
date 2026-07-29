//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
#include <unordered_map>
#include <vector>

void py_cuda::cudaScatterElementsOp(top::ScatterElementsOp op) {
  auto in_shape = module::getShape(op.getInput());
  int r = in_shape.size();
  int axis = op.getAxis();
  if (axis < 0) axis += r;
  int in_num = module::getNumElements(op.getInput());
  int idx_num = module::getNumElements(op.getIndices());
  int upd_num = module::getNumElements(op.getUpdates());

  // Read indices based on GPU storage type
  std::vector<float> idx_f(idx_num), upd_h(upd_num);
  auto idx_type = getCudaType(op.getIndices());
  if (idx_type == cuda::DT_INT32 || idx_type == cuda::DT_UINT32) {
    std::vector<int32_t> tmp(idx_num);
    CHECK_CUDA(cudaMemcpy(tmp.data(), getCudaData(op.getIndices()),
                          idx_num * sizeof(int32_t), cudaMemcpyDeviceToHost));
    for (int j = 0; j < idx_num; j++) idx_f[j] = (float)tmp[j];
  } else {
    CHECK_CUDA(cudaMemcpy(idx_f.data(), getCudaData(op.getIndices()),
                          idx_num * sizeof(float), cudaMemcpyDeviceToHost));
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

  // Compute flat indices using UPDATES shape (matching CPU idx_to_list)
  auto upd_shape = module::getShape(op.getUpdates());
  std::vector<int64_t> upd_stride(r, 1);
  for (int i = r - 2; i >= 0; i--)
    upd_stride[i] = upd_stride[i + 1] * upd_shape[i + 1];
  std::vector<int64_t> in_stride(r, 1);
  for (int i = r - 2; i >= 0; i--)
    in_stride[i] = in_stride[i + 1] * in_shape[i + 1];
  std::vector<int> flat_idx(upd_num);
  for (int n = 0; n < upd_num; n++) {
    int rem = n;
    int64_t off = 0;
    for (int d = 0; d < r; d++) {
      int coord = rem / upd_stride[d];
      rem %= upd_stride[d];
      if (d == axis) coord = (int)idx_f[n];
      off += coord * in_stride[d];
    }
    flat_idx[n] = off;
  }

  // Dedup: keep last occurrence for duplicate flat_idx (matches CPU sequential)
  std::unordered_map<int, int> last_writer;
  for (int n = 0; n < upd_num; n++) last_writer[flat_idx[n]] = n;
  std::vector<int> dedup_fi; dedup_fi.reserve(last_writer.size());
  std::vector<float> dedup_upd; dedup_upd.reserve(last_writer.size());
  for (auto &kv : last_writer) {
    dedup_fi.push_back(kv.first);
    dedup_upd.push_back(upd_h[kv.second]);
  }
  int dedup_num = (int)dedup_fi.size();

  auto fi_d = cuda_malloc(dedup_num * sizeof(int));
  CHECK_CUDA(cudaMemcpy(fi_d.get(), dedup_fi.data(),
                        dedup_num * sizeof(int), cudaMemcpyHostToDevice));
  auto upd_gpu = cuda_malloc(dedup_num * sizeof(float));
  CHECK_CUDA(cudaMemcpy(upd_gpu.get(), dedup_upd.data(),
                        dedup_num * sizeof(float), cudaMemcpyHostToDevice));
  cuda::scatterElements(out_buf, upd_gpu.get(), fi_d.get(), dedup_num, false);

  if (out_guard)
    cuda::convertType(out_buf, getCudaData(op.getOutput()), in_num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
}
