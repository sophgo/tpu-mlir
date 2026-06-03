//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
#include <vector>

void py_cuda::cudaStridedSliceOp(top::StridedSliceOp op) {
  auto in_shape = module::getShape(op.getInput());
  auto out_shape = module::getShape(op.getOutput());
  int dims = in_shape.size();
  int out_num = module::getNumElements(op.getOutput());

  // Read starts/ends/strides from GPU
  int s_num = module::getNumElements(op.getStarts());
  std::vector<float> starts_h(s_num), ends_h(s_num), strides_h(s_num);
  CHECK_CUDA(cudaMemcpy(starts_h.data(), getCudaData(op.getStarts()), s_num * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ends_h.data(), getCudaData(op.getEnds()), s_num * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(strides_h.data(), getCudaData(op.getStrides()), s_num * sizeof(float), cudaMemcpyDeviceToHost));

  int b_mask = op.getBeginMask(), e_mask = op.getEndMask();

  std::vector<int> begin(dims), end(dims), step(dims, 1);
  for (int i = 0; i < dims; i++) {
    begin[i] = (b_mask & (1 << i)) ? 0 : (int)starts_h[i];
    end[i]   = (e_mask & (1 << i)) ? in_shape[i] : (int)ends_h[i];
    if (i < s_num) step[i] = (int)strides_h[i];
    if (begin[i] < 0) begin[i] += in_shape[i];
    if (end[i] < 0) end[i] += in_shape[i];
    if (step[i] == 0) step[i] = 1;
  }

  // Compute strides
  std::vector<int> in_stride(dims, 1), out_stride(dims, 1);
  for (int i = dims - 2; i >= 0; i--) {
    in_stride[i] = in_stride[i+1] * in_shape[i+1];
    out_stride[i] = out_stride[i+1] * out_shape[i+1];
  }

  // Pre-compute flat input indices on host
  std::vector<int> flat_idx(out_num);
  for (int oi = 0; oi < out_num; oi++) {
    int remain = oi, in_idx = 0;
    for (int d = 0; d < dims; d++) {
      int coord = remain / out_stride[d];
      remain %= out_stride[d];
      in_idx += (begin[d] + coord * step[d]) * in_stride[d];
    }
    flat_idx[oi] = in_idx;
  }

  // GPU kernel: parallel lookup
  auto fi_d = cuda_malloc(out_num * sizeof(int));
  CHECK_CUDA(cudaMemcpy(fi_d.get(), flat_idx.data(), out_num * sizeof(int), cudaMemcpyHostToDevice));
  cuda::stridedSlice(getCudaData(op.getInput()), getCudaData(op.getOutput()),
                     fi_d.get(), out_num);
}

void py_cuda::cudaStridedSliceOp(tpu::StridedSliceOp op) {
  auto stype = module::getStorageType(op.getOutput());
  if (stype.isF32()) {
    // Re-use top handler via shared logic — but accessors differ, so inline
    auto in_shape = module::getShape(op.getInput());
    auto out_shape = module::getShape(op.getOutput());
    int dims = in_shape.size(), out_num = module::getNumElements(op.getOutput());
    int s_num = module::getNumElements(op.getStarts());
    std::vector<float> sh(s_num), eh(s_num), sth(s_num);
    CHECK_CUDA(cudaMemcpy(sh.data(), getCudaData(op.getStarts()), s_num * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(eh.data(), getCudaData(op.getEnds()), s_num * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(sth.data(), getCudaData(op.getStrides()), s_num * sizeof(float), cudaMemcpyDeviceToHost));
    int bm = op.getBeginMask(), em = op.getEndMask();
    std::vector<int> b(dims), e(dims), s(dims, 1);
    for (int i = 0; i < dims; i++) {
      b[i] = (bm & (1 << i)) ? 0 : (int)sh[i];
      e[i] = (em & (1 << i)) ? in_shape[i] : (int)eh[i];
      if (i < s_num) s[i] = (int)sth[i];
      if (b[i] < 0) b[i] += in_shape[i];
      if (e[i] < 0) e[i] += in_shape[i];
      if (s[i] == 0) s[i] = 1;
    }
    std::vector<int> istr(dims, 1), ostr(dims, 1);
    for (int i = dims-2; i >= 0; i--) { istr[i]=istr[i+1]*in_shape[i+1]; ostr[i]=ostr[i+1]*out_shape[i+1]; }
    std::vector<int> fi(out_num);
    for (int oi=0; oi<out_num; oi++) { int r=oi, in=0; for(int d=0;d<dims;d++){int c=r/ostr[d]; r%=ostr[d]; in+=(b[d]+c*s[d])*istr[d];} fi[oi]=in; }
    auto fid = cuda_malloc(out_num * sizeof(int));
    CHECK_CUDA(cudaMemcpy(fid.get(), fi.data(), out_num * sizeof(int), cudaMemcpyHostToDevice));
    cuda::stridedSlice(getCudaData(op.getInput()), getCudaData(op.getOutput()), fid.get(), out_num);
  } else {
    auto in_shape = module::getShape(op.getInput());
    auto out_shape = module::getShape(op.getOutput());
    int dims = in_shape.size(), out_num = module::getNumElements(op.getOutput());
    int in_num = module::getNumElements(op.getInput());
    int s_num = module::getNumElements(op.getStarts());
    std::vector<float> sh(s_num), eh(s_num), sth(s_num);
    CHECK_CUDA(cudaMemcpy(sh.data(), getCudaData(op.getStarts()), s_num * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(eh.data(), getCudaData(op.getEnds()), s_num * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(sth.data(), getCudaData(op.getStrides()), s_num * sizeof(float), cudaMemcpyDeviceToHost));
    int bm = op.getBeginMask(), em = op.getEndMask();
    std::vector<int> b(dims), e(dims), s(dims, 1);
    for (int i = 0; i < dims; i++) {
      b[i] = (bm & (1 << i)) ? 0 : (int)sh[i];
      e[i] = (em & (1 << i)) ? in_shape[i] : (int)eh[i];
      if (i < s_num) s[i] = (int)sth[i];
      if (b[i] < 0) b[i] += in_shape[i];
      if (e[i] < 0) e[i] += in_shape[i];
      if (s[i] == 0) s[i] = 1;
    }
    std::vector<int> istr(dims, 1), ostr(dims, 1);
    for (int i = dims-2; i >= 0; i--) { istr[i]=istr[i+1]*in_shape[i+1]; ostr[i]=ostr[i+1]*out_shape[i+1]; }
    std::vector<int> fi(out_num);
    for (int oi=0; oi<out_num; oi++) { int r=oi, in=0; for(int d=0;d<dims;d++){int c=r/ostr[d]; r%=ostr[d]; in+=(b[d]+c*s[d])*istr[d];} fi[oi]=in; }
    auto fid = cuda_malloc(out_num * sizeof(int));
    CHECK_CUDA(cudaMemcpy(fid.get(), fi.data(), out_num * sizeof(int), cudaMemcpyHostToDevice));
    auto in_f32 = cuda_malloc(in_num * sizeof(float));
    auto out_f32 = cuda_malloc(out_num * sizeof(float));
    cuda::convertType(getCudaData(op.getInput()), in_f32.get(), in_num,
                      getCudaType(op.getInput()), cuda::DT_F32);
    cuda::stridedSlice(in_f32.get(), out_f32.get(), fid.get(), out_num);
    cuda::convertType(out_f32.get(), getCudaData(op.getOutput()), out_num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  }
}
