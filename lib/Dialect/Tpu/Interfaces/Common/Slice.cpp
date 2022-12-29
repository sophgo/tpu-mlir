//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"
#include <valarray>



template <typename T>
static int remove_value(std::vector<T> &v, T value) {
  int idx = 0;
  for (auto iter = v.begin(); iter != v.end(); iter++, idx++) {
    if (*iter == value) {
      v.erase(iter);
      return idx;
    }
  }
  return -1;
}

void tpu::SliceOp::parseParam(std::vector<int64_t> &is_4,
                              std::vector<int64_t> &os_4,
                              std::vector<int> &offset_4,
                              std::vector<int> &step_4, bool &fusible) {
  auto is = input().getType().cast<RankedTensorType>().getShape().vec();
  auto os = output().getType().cast<RankedTensorType>().getShape().vec();
  int num_dims = is.size();
  auto crop_offset = module::getI64Array(offset());
  auto crop_steps = module::getI64Array(steps());

  assert(crop_offset->size() == crop_steps->size());
  assert(is.size() == crop_steps->size());
  assert(is.size() == os.size());

  if (num_dims > 4) {
    // remove dims = 1
    while (num_dims > 4) {
      int idx = remove_value<int64_t>(is, 1);
      if (idx < 0) {
        break;
      }
      crop_offset->erase(crop_offset->begin() + idx);
      crop_steps->erase(crop_steps->begin() + idx);
      os.erase(os.begin() + idx);
      num_dims--;
    }
    // remove continous
    while (num_dims > 4) {
      bool done = false;
      for (int i = 0; i < num_dims - 1; i++) {
        if (is[i] == os[i] && is[i + 1] == os[i + 1]) {
          is[i] *= is[i + 1];
          os[i] *= os[i + 1];
          is.erase(is.begin() + i + 1);
          os.erase(os.begin() + i + 1);
          crop_steps->erase(crop_steps->begin() + i + 1);
          crop_offset->erase(crop_offset->begin() + i + 1);
          num_dims--;
          done = true;
          break;
        }
      }
      if (done == false) {
        break;
      }
    }
    if (num_dims > 4) {
      llvm_unreachable("permute shape not support");
    }
  }
  is_4 = {1, 1, 1, 1};
  os_4 = {1, 1, 1, 1};
  step_4 = {1, 1, 1, 1};
  offset_4 = {0, 0, 0, 0};
  std::vector<int> real_axes;
  bool no_step = true;
  for (int idx = 0; idx < num_dims; idx++) {
    is_4[idx] = is[idx];
    os_4[idx] = os[idx];
    step_4[idx] = crop_steps->at(idx);
    offset_4[idx] = crop_offset->at(idx);
    if (no_step && crop_steps->at(idx) != 1) {
      no_step = false;
    }
    if (is_4[idx] != os_4[idx]) {
      real_axes.push_back(idx);
    }
  }
  fusible = false;
  if (no_step && real_axes.size() == 1) {
    int axis = real_axes[0];
    int outer_dim = std::accumulate(is_4.begin(), is_4.begin() + axis, 1,
                                    std::multiplies<int64_t>());
    if (outer_dim == 1) {
      fusible = true;
    }
  }
}

LogicalResult tpu::SliceOp::init(InferenceParameter &p) { return success(); }
void tpu::SliceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SliceOp::inference(InferenceParameter &p) {
  auto out_num_elem = module::getNumElements(output());
  auto offset_v = module::getI64Array(offset());
  auto steps_v = module::getI64Array(steps());
  auto out_shape = module::getShape(output());
  auto in_shape = module::getShape(input());
  auto in_dims = in_shape.size();
  auto out_dims = out_shape.size();
  // just support the dims of input & input is equal.
  assert(in_dims == out_dims);

  // slice[range] -> (offset + stride)
  std::valarray<int64_t> in_stride_v(1, in_dims);
  std::valarray<int64_t> out_stride_v(1, out_dims);
  for (int i = in_stride_v.size() - 2; i >= 0; --i) {
    in_stride_v[i] *= in_stride_v[i + 1] * in_shape[i + 1];
    out_stride_v[i] *= out_stride_v[i + 1] * out_shape[i + 1];
  }
  auto in_offset_v = std::valarray<int64_t>(offset_v->data(), offset_v->size());
  auto in_offset = (in_offset_v * in_stride_v).sum();
  auto out_in_stride_v =
      std::valarray<int64_t>(steps_v->data(), steps_v->size());
  out_in_stride_v *= in_stride_v;

#pragma omp parallel for schedule(static, omp_schedule(out_num_elem))
  for (int64_t i = 0; i < out_num_elem; ++i) {
    std::valarray<int64_t> out_it(1, out_dims);
    int64_t tmp = i;
    for (int j = 0; j < out_dims; j++) {
      out_it[j] = tmp / out_stride_v[j];
      tmp = tmp % out_stride_v[j];
    }
    p.outputs[0][i] = p.inputs[0][(out_it * out_in_stride_v).sum() + in_offset];
  }

  return success();
}
