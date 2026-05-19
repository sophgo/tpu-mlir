//===----------------------------------------------------------------------===//
#include "../pycuda.h"
#include "cuda_helper.h"
#include <vector>
#include <cstring>

void py_cuda::cudaPadOp(top::PadOp op) {
  auto in_shape = module::getShape(op.getInput());
  int dims = in_shape.size();
  int in_num = module::getNumElements(op.getInput());
  auto pads_v = module::getI64Array(op.getPaddings());
  float pad_val = op.getVal().convertToDouble();
  std::string mode = op.getMode().str();

  // Read input
  std::vector<float> in_h(in_num);
  if (getCudaType(op.getInput()) == cuda::DT_F32) {
    CHECK_CUDA(cudaMemcpy(in_h.data(), getCudaData(op.getInput()),
                          in_num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getInput(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(in_h.data(), tmp.get(), in_num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  // Compute output shape and strides
  std::vector<int64_t> pads(pads_v->begin(), pads_v->end());
  std::vector<int64_t> out_shape(dims);
  std::vector<int> in_stride(dims, 1), out_stride(dims, 1);
  int out_num = 1;
  for (int d = 0; d < dims; d++) {
    out_shape[d] = in_shape[d] + pads[d] + pads[d + dims];
    out_num *= out_shape[d];
  }
  for (int d = dims - 2; d >= 0; d--) {
    in_stride[d] = in_stride[d + 1] * in_shape[d + 1];
    out_stride[d] = out_stride[d + 1] * out_shape[d + 1];
  }

  std::vector<float> out_h(out_num, pad_val);

  if (mode == "constant") {
    for (int i = 0; i < out_num; i++) {
      int rem = i, in_idx = 0;
      bool inside = true;
      for (int d = 0; d < dims; d++) {
        int coord = rem / out_stride[d];
        rem %= out_stride[d];
        int in_coord = coord - pads[d];
        if (in_coord < 0 || in_coord >= in_shape[d]) { inside = false; break; }
        in_idx += in_coord * in_stride[d];
      }
      if (inside) out_h[i] = in_h[in_idx];
    }
  } else if (mode == "edge") {
    for (int i = 0; i < out_num; i++) {
      int rem = i, in_idx = 0;
      for (int d = 0; d < dims; d++) {
        int coord = rem / out_stride[d];
        rem %= out_stride[d];
        int in_coord = coord - pads[d];
        if (in_coord < 0) in_coord = 0;
        else if (in_coord >= in_shape[d]) in_coord = in_shape[d] - 1;
        in_idx += in_coord * in_stride[d];
      }
      out_h[i] = in_h[in_idx];
    }
  } else if (mode == "reflect") {
    for (int i = 0; i < out_num; i++) {
      int rem = i, in_idx = 0;
      for (int d = 0; d < dims; d++) {
        int coord = rem / out_stride[d];
        rem %= out_stride[d];
        int in_coord = coord - pads[d];
        int max_idx = in_shape[d] - 1;
        if (in_coord < 0) in_coord = -in_coord;
        else if (in_coord > max_idx) in_coord = 2 * max_idx - in_coord;
        if (in_coord < 0) in_coord = 0;
        if (in_coord > max_idx) in_coord = max_idx;
        in_idx += in_coord * in_stride[d];
      }
      out_h[i] = in_h[in_idx];
    }
  }

  auto out_f32 = cuda_malloc(out_num * sizeof(float));
  CHECK_CUDA(cudaMemcpy(out_f32.get(), out_h.data(), out_num * sizeof(float),
                        cudaMemcpyHostToDevice));
  if (getCudaType(op.getOutput()) != cuda::DT_F32) {
    cuda::convertType(out_f32.get(), getCudaData(op.getOutput()), out_num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  } else {
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out_f32.get(),
                          out_num * sizeof(float), cudaMemcpyDeviceToDevice));
  }
}

void py_cuda::cudaPadOp(tpu::PadOp op) {
  auto in_shape = module::getShape(op.getInput());
  int dims = in_shape.size();
  int in_num = module::getNumElements(op.getInput());
  auto pads_v = module::getI64Array(op.getPaddings());
  float pad_val = op.getVal().convertToDouble();
  std::string mode = stringifyPaddingMode(op.getMode()).str();
  auto stype = module::getStorageType(op.getInput());

  std::vector<float> in_h(in_num);
  if (stype.isF32()) {
    CHECK_CUDA(cudaMemcpy(in_h.data(), getCudaData(op.getInput()),
                          in_num * sizeof(float), cudaMemcpyDeviceToHost));
  } else {
    auto tmp = newCudaData(op.getInput(), cuda::DT_F32);
    CHECK_CUDA(cudaMemcpy(in_h.data(), tmp.get(), in_num * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  std::vector<int64_t> pads(pads_v->begin(), pads_v->end());
  std::vector<int64_t> out_shape(dims);
  std::vector<int> in_stride(dims, 1), out_stride(dims, 1);
  int out_num = 1;
  for (int d = 0; d < dims; d++) {
    out_shape[d] = in_shape[d] + pads[d] + pads[d + dims];
    out_num *= out_shape[d];
  }
  for (int d = dims - 2; d >= 0; d--) {
    in_stride[d] = in_stride[d + 1] * in_shape[d + 1];
    out_stride[d] = out_stride[d + 1] * out_shape[d + 1];
  }

  std::vector<float> out_h(out_num, pad_val);
  if (mode == "constant") {
    for (int i = 0; i < out_num; i++) {
      int rem = i, in_idx = 0;
      bool inside = true;
      for (int d = 0; d < dims; d++) {
        int coord = rem / out_stride[d]; rem %= out_stride[d];
        int in_coord = coord - pads[d];
        if (in_coord < 0 || in_coord >= in_shape[d]) { inside = false; break; }
        in_idx += in_coord * in_stride[d];
      }
      if (inside) out_h[i] = in_h[in_idx];
    }
  } else if (mode == "edge") {
    for (int i = 0; i < out_num; i++) {
      int rem = i, in_idx = 0;
      for (int d = 0; d < dims; d++) {
        int coord = rem / out_stride[d]; rem %= out_stride[d];
        int in_coord = coord - pads[d];
        if (in_coord < 0) in_coord = 0;
        else if (in_coord >= in_shape[d]) in_coord = in_shape[d] - 1;
        in_idx += in_coord * in_stride[d];
      }
      out_h[i] = in_h[in_idx];
    }
  } else if (mode == "reflect") {
    for (int i = 0; i < out_num; i++) {
      int rem = i, in_idx = 0;
      for (int d = 0; d < dims; d++) {
        int coord = rem / out_stride[d]; rem %= out_stride[d];
        int in_coord = coord - pads[d], max_idx = in_shape[d] - 1;
        if (in_coord < 0) in_coord = -in_coord;
        else if (in_coord > max_idx) in_coord = 2 * max_idx - in_coord;
        if (in_coord < 0) in_coord = 0;
        if (in_coord > max_idx) in_coord = max_idx;
        in_idx += in_coord * in_stride[d];
      }
      out_h[i] = in_h[in_idx];
    }
  }

  auto out_f32 = cuda_malloc(out_num * sizeof(float));
  CHECK_CUDA(cudaMemcpy(out_f32.get(), out_h.data(), out_num * sizeof(float),
                        cudaMemcpyHostToDevice));
  if (getCudaType(op.getOutput()) != cuda::DT_F32) {
    cuda::convertType(out_f32.get(), getCudaData(op.getOutput()), out_num,
                      cuda::DT_F32, getCudaType(op.getOutput()));
  } else {
    CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out_f32.get(),
                          out_num * sizeof(float), cudaMemcpyDeviceToDevice));
  }
}
