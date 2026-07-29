//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
#include <vector>

void py_cuda::cudaScatterNDOp(top::ScatterNDOp op) {
  auto in_shape = module::getShape(op.getInputData());
  auto idx_shape = module::getShape(op.getIndices());
  int in_num = module::getNumElements(op.getInputData());
  int upd_num = module::getNumElements(op.getUpdates());
  int index_depth = idx_shape.back();
  int input_dim = in_shape.size();

  // Copy input to output on GPU
  CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()),
                        getCudaData(op.getInputData()),
                        in_num * sizeof(float), cudaMemcpyDeviceToDevice));

  // Compute updates_elems, slice_elems, outer_strides on host
  int updates_elems = 1;
  for (int i = 0; i < (int)idx_shape.size() - 1; i++)
    updates_elems *= idx_shape[i];

  int slice_elems = 1;
  for (int i = index_depth; i < input_dim; i++)
    slice_elems *= in_shape[i];

  std::vector<int> outer_shape(in_shape.begin(), in_shape.begin() + index_depth);
  std::vector<int> outer_stride(outer_shape.size(), slice_elems);
  for (int i = outer_stride.size() - 2; i >= 0; i--)
    outer_stride[i] = outer_stride[i+1] * outer_shape[i+1];

  // Read indices and updates from GPU
  // Read indices matching GPU storage type.
  // Weight/initializer may be INT32, InputOp may be F32.
  std::vector<int> idx_h(updates_elems * index_depth);
  auto idx_dtype = getCudaType(op.getIndices());
  if (idx_dtype == cuda::DT_INT32 || idx_dtype == cuda::DT_UINT32) {
    CHECK_CUDA(cudaMemcpy(idx_h.data(), getCudaData(op.getIndices()),
                          idx_h.size() * sizeof(int), cudaMemcpyDeviceToHost));
  } else {
    std::vector<float> tmp(updates_elems * index_depth);
    CHECK_CUDA(cudaMemcpy(tmp.data(), getCudaData(op.getIndices()),
                          tmp.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < tmp.size(); i++) idx_h[i] = (int)tmp[i];
  }

  std::vector<float> upd_h(upd_num);
  CHECK_CUDA(cudaMemcpy(upd_h.data(), getCudaData(op.getUpdates()),
                        upd_num * sizeof(float), cudaMemcpyDeviceToHost));

  // Pre-compute flat output indices on host (CPU does the complex ND indexing)
  std::vector<int> flat_idx(upd_num);
  for (int n = 0; n < updates_elems; n++) {
    int out_offset = 0;
    for (int i = 0; i < (int)outer_stride.size(); i++) {
      int coord = (int)idx_h[n * index_depth + i];
      if (coord < 0) coord += in_shape[i];
      out_offset += coord * outer_stride[i];
    }
    for (int s = 0; s < slice_elems; s++) {
      flat_idx[n * slice_elems + s] = out_offset + s;
    }
  }

  // Upload flat indices and launch GPU kernel for the scatter
  auto fi_d = cuda_malloc(upd_num * sizeof(int));
  CHECK_CUDA(cudaMemcpy(fi_d.get(), flat_idx.data(),
                        upd_num * sizeof(int), cudaMemcpyHostToDevice));
  bool add = (op.getReduction() != 0);
  cuda::scatterND(getCudaData(op.getOutput()), getCudaData(op.getUpdates()),
                  fi_d.get(), upd_num, add);
}
