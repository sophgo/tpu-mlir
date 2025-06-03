//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"
#include <valarray>

using namespace tpu_mlir::backend;

template <typename T>
static int remove_value(std::vector<T> &v, T value, bool is_int8) {
  int idx = 0;
  for (auto iter = v.begin(); iter != v.end(); iter++, idx++) {
    if (idx == 0 && is_int8)
      continue;
    if (*iter == value) {
      v.erase(iter);
      return idx;
    }
  }
  return -1;
}

slice_attr_t tpu::SliceOp::parseParam() {
  slice_attr_t attr;
  std::vector<int64_t> is = module::getShape(getInput());
  std::vector<int64_t> os = module::getShape(getOutput());
  int num_dims = is.size();
  auto crop_offset = module::getI64Array(getOffset());
  auto crop_steps = module::getI64Array(getSteps());

  assert(crop_offset->size() == crop_steps->size());
  assert(is.size() == crop_steps->size());
  if (is.size() > os.size()) {
    for (int out_dims = os.size(); out_dims < num_dims; out_dims++) {
      os.insert(os.begin(), 1);
    }
  }

  auto input_dtype = BM1684::getDataType(getInput());
  bool is_int8 = (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8);
  if (num_dims > 4) {
    // remove dims = 1
    while (num_dims > 4) {
      int idx = remove_value<int64_t>(is, 1, is_int8);
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
        if (i == 0 && is_int8)
          continue;
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
    if (num_dims > 4 && !module::isBM1684XFamily()) {
      llvm_unreachable("Slice shape not support");
    }
  }

  if (num_dims > 4 && module::isBM1684XFamily()) {
    attr.is_4.assign(is.begin(), is.end());
    attr.os_4.assign(os.begin(), os.end());
    attr.step_4.assign(crop_steps->begin(), crop_steps->end());
    attr.offset_4.assign(crop_offset->begin(), crop_offset->end());
  } else {
    attr.is_4 = {1, 1, 1, 1};
    attr.os_4 = {1, 1, 1, 1};
    attr.step_4 = {1, 1, 1, 1};
    attr.offset_4 = {0, 0, 0, 0};
  }
  std::vector<int> real_axes;
  attr.no_step = true;
  for (int idx = 0; idx < num_dims; idx++) {
    if (num_dims <= 4) {
      attr.is_4[idx] = is[idx];
      attr.os_4[idx] = os[idx];
      attr.step_4[idx] = crop_steps->at(idx);
      attr.offset_4[idx] = crop_offset->at(idx);
    }

    if (attr.no_step && crop_steps->at(idx) != 1) {
      attr.no_step = false;
    }
    if (attr.is_4[idx] != attr.os_4[idx]) {
      real_axes.push_back(idx);
    }
  }

  attr.fusible = false;
  if (!module::isBM1684Family()) {
    if (attr.no_step && real_axes.size() == 1) {
      int axis = real_axes[0];
      int outer_dim =
          std::accumulate(attr.is_4.begin(), attr.is_4.begin() + axis, 1,
                          std::multiplies<int64_t>());
      if (outer_dim == 1) {
        attr.fusible = true;
      }
    }
  }
  return attr;
}

LogicalResult tpu::SliceOp::init(InferenceParameter &p) { return success(); }
void tpu::SliceOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SliceOp::inference(InferenceParameter &p) {
  auto out_num_elem = module::getNumElements(getOutput());
  auto offset_v = module::getI64Array(getOffset());
  auto steps_v = module::getI64Array(getSteps());
  auto ends_v_old = module::getI64Array(getEnds());
  const size_t slice_dims = offset_v->size();
  std::vector<int64_t> in_shape = module::getShape(getInput());
  auto in_dims = in_shape.size();
  auto axes = module::getI64Array(getHasparamConvertAxesAttr());
  auto slice_n = axes->size();
  auto ends_v = ends_v_old;
  if (slice_n) {
    ends_v = std::make_shared<std::vector<int64_t>>(in_shape);
  }
  for (int i = 0; i < slice_n; ++i) {
    int axis = axes->at(i);
    if (axis < 0) {
      axis += in_dims;
    }
    int step = steps_v->at(axis);
    int64_t end = ends_v_old->at(axis);
    int64_t offset = offset_v->at(axis);
    offset_v->at(axis) = offset;
    ends_v->at(axis) = end;
    steps_v->at(axis) = step;
  }
  for (int i = 0; i < slice_dims; ++i) {
    if (offset_v->at(i) < 0) {
      offset_v->at(i) += in_shape[i];
    }
  }
  std::vector<int64_t> output_shape(in_dims);
  for (size_t i = 0; i < in_dims; ++i) {
    if (i < slice_dims) {
      auto offset = offset_v->at(i);
      auto end = ends_v->at(i);
      auto step = steps_v->at(i);
      if (end < 0) {
        end += in_shape[i];
      }
      offset = step > 0 ? std::clamp(offset, 0L, in_shape[i])
                        : std::clamp(offset, 0L, in_shape[i] - 1);
      end = step > 0 ? std::clamp(end, 0L, in_shape[i])
                     : std::clamp(end, -1L, in_shape[i] - 1);
      output_shape[i] = abs_ceiling_func(end - offset, step);
    } else {
      output_shape[i] = in_shape[i];
    }
  }
  module::setShape(getOutput(), output_shape);
  std::vector<int64_t> out_shape = module::getShape(getOutput());
  auto out_dims = out_shape.size();
  // just support the dims of input & input is equal.
  while (out_dims < in_dims) {
    out_shape.insert(out_shape.begin(), 1);
    out_dims++;
  }
  if (!(module::isNone(getOffsetT()) && module::isNone(getEndsT()) &&
        module::isNone(getStepsT()))) {
    // slice in only one aixs in such case
    int axis = module::getI64Array(getAxes())->at(0);
    auto ends_v = module::getI64Array(getEnds());
    auto in_ends_v = std::valarray<int64_t>(ends_v->data(), ends_v->size());
    auto in_steps_v = std::valarray<int64_t>(steps_v->data(), steps_v->size());
    for (int i = 0; i < out_dims; i++) {
      out_shape[i] = std::min(out_shape[i], in_shape[i]);
    }
    if (!module::isNone(getOffsetT()))
      offset_v->at(axis) = *p.inputs[1];
    if (!module::isNone(getEndsT()))
      ends_v->at(axis) = *p.inputs[2];
    if (!module::isNone(getStepsT()))
      steps_v->at(axis) = *p.inputs[3];
    if (offset_v->at(axis) < 0)
      offset_v->at(axis) += in_shape[axis];
    if (ends_v->at(axis) < 0)
      ends_v->at(axis) += in_shape[axis];
    offset_v->at(axis) =
        steps_v->at(axis) > 0
            ? std::clamp(offset_v->at(axis), 0L, in_shape[axis])
            : std::clamp(offset_v->at(axis), 0L, in_shape[axis] - 1);
    ends_v->at(axis) =
        steps_v->at(axis) > 0
            ? std::clamp(ends_v->at(axis), 0L, in_shape[axis])
            : std::clamp(ends_v->at(axis), -1L, in_shape[axis] - 1);

    out_shape[axis] =
        (ends_v->at(axis) - offset_v->at(axis)) / steps_v->at(axis);
    module::setShape(getOutput(), out_shape);
    out_num_elem = module::getNumElements(getOutput());
  }
  for (int i = 0; i < slice_dims; ++i) {
    if (offset_v->at(i) < 0) {
      offset_v->at(i) += in_shape[i];
    }
    offset_v->at(i) = steps_v->at(i) > 0
                          ? std::clamp(offset_v->at(i), 0L, in_shape[i])
                          : std::clamp(offset_v->at(i), 0L, in_shape[i] - 1);
  }
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

LogicalResult tpu::SliceOp::BackwardN(int64_t &in_idx, int64_t &in_slice,
                                      int64_t out_idx, int64_t out_slice) {
  const auto offset = module::getI64Array(getOffset());
  const auto steps = module::getI64Array(getSteps());
  std::vector<int64_t> in_shape = module::getShape(getInput());
  auto &p = getSliceParam(*this);
  in_idx = out_idx * steps->at(0);
  in_slice = out_slice * steps->at(0) + offset->at(0);
  in_slice += offset->at(0) >= 0 ? 0 : in_shape[0];
  bool is_last = (out_idx + out_slice == p.os_4[0]);
  LocalGenInterface::fixSlice(in_idx, in_slice, p.is_4[0], is_last);
  return success();
}

LogicalResult tpu::SliceOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                      int64_t out_idx, int64_t out_slice) {
  const auto offset = module::getI64Array(getOffset());
  const auto steps = module::getI64Array(getSteps());
  std::vector<int64_t> in_shape = module::getShape(getInput());
  auto &p = getSliceParam(*this);
  in_idx = out_idx * steps->at(2);
  in_slice = out_slice * steps->at(2) + offset->at(2);
  in_slice += offset->at(2) >= 0 ? 0 : in_shape[2];
  bool is_last = (out_idx + out_slice == p.os_4[2]);
  LocalGenInterface::fixSlice(in_idx, in_slice, p.is_4[2], is_last);
  return success();
}

LogicalResult tpu::SliceOp::BackwardW(int64_t &in_idx, int64_t &in_slice,
                                      int64_t out_idx, int64_t out_slice) {
  const auto offset = module::getI64Array(getOffset());
  const auto steps = module::getI64Array(getSteps());
  std::vector<int64_t> in_shape = module::getShape(getInput());
  auto &p = getSliceParam(*this);
  in_idx = out_idx * steps->at(3);
  in_slice = out_slice * steps->at(3) + offset->at(3);
  in_slice += offset->at(3) >= 0 ? 0 : in_shape[3];
  bool is_last = (out_idx + out_slice == p.os_4[3]);
  LocalGenInterface::fixSlice(in_idx, in_slice, p.is_4[3], is_last);
  return success();
}

LogicalResult tpu::SliceOp::LocalGenSupport() {
  auto shape = module::getShape(getInput());
  int num_dims = shape.size();
  if (module::isCV18xx()) {
    if (num_dims != 3 && num_dims != 4) {
      return failure();
    }
    auto p = parseParam();
    if (!p.no_step || p.fusible == true) {
      return failure();
    }
    return (p.offset_4[1] % CV18xx::NPU_NUM == 0) ? success() : failure();
  } else if (module::isBM1684XFamily() || module::isBM1690Family()) {
    if ((int)getRunMode(getOperation()) == 1) {
      return failure();
    }
    const auto offset = module::getI64Array(getOffset());
    const auto steps = module::getI64Array(getSteps());
    // TODO: force layer group to allow that offset->at(0) != 0
    if (num_dims > 1) {
      // TODO: force layer group to allow that offset->at(2) != 0
      if (steps->at(1) != 1)
        return failure();
      // force (1x76725x4) -> (1x76725x1) step in global
      if (num_dims > 2 && shape[1] > 65535)
        return failure();
    }
    if (num_dims > 4) {
      if (num_dims == 5) {
        int64_t in_shape[5];
        int64_t out_shape[5];
        tpu_mlir::group_type_t group_type = GROUP_3D;
        module::getNCDHW(getInput(), in_shape[0], in_shape[1], in_shape[2],
                         in_shape[3], in_shape[4], group_type);
        module::getNCDHW(getOutput(), out_shape[0], out_shape[1], out_shape[2],
                         out_shape[3], out_shape[4], group_type);
        for (int i = 0; i < 5; ++i) {
          if (in_shape[i] != out_shape[i] && (i != 2)) {
            return failure();
          }
        }
        return success();
      }
      return failure();
    }
  } else if (module::isBM1684Family()) {
    auto p = parseParam();
    auto input_dim = module::getShape(getInput()).size();
    if (input_dim != 4)
      return failure();
    bool neg_stride = false;
    std::for_each(p.step_4.begin(), p.step_4.begin() + input_dim, [&](auto s) {
      if (s < 0)
        neg_stride = true;
    });
    if (neg_stride)
      return failure();

    auto input_shape = module::getShape(getInput());
    int c_size = align_up(input_shape[2] * input_shape[3], BM1684::eu_num(4));
    int c_stride = input_shape[1] == 1 ? 1 : c_size * p.step_4[1];
    int n_stride = input_shape[0] == 1
                       ? 1
                       : ceiling_func(input_shape[1], BM1684::NPU_NUM) *
                             c_size * p.step_4[0];
    if (c_stride >= (1 << 19) || n_stride >= (1 << 19))
      return failure();

    if (p.step_4[1] != 1)
      return failure();
    if (input_dim > 1 && p.offset_4[1] % BM1684::NPU_NUM != 0)
      return failure();
    if (module::isUniformQuantized(getInput())) {
      int begin_mask = 0, end_mask = 0;
      auto output_shape = module::getShape(getOutput());
      auto end_index_n = p.offset_4[0] + output_shape[0] * p.step_4[0];
      int output_n = ceil((((end_mask & 0x1) ? input_shape[0] : end_index_n) -
                           ((begin_mask & 0x1) ? 0 : p.offset_4[0])) *
                          1.0 / p.step_4[0]);
      if (BM1684::getStoreMode(getInput()) != STORE_MODE_4N ||
          BM1684::getStoreMode(getOutput()) != STORE_MODE_4N ||
          output_n != input_shape[0]) {
        return failure();
      }
    }
  }
  return success();
}

void tpu::SliceOp::assign_fw_param(void *param) {
  fw_stride_slice_layer_param_t slice_param;
  memset(&slice_param, 0, sizeof(fw_stride_slice_layer_param_t));
  slice_param.shape_size = module::getShape(getInput()).size();
  // only StrideSliceOp need mask
  slice_param.begin_mask = 0;
  slice_param.end_mask = 0;
  slice_param.shrink_axis_mask = 0;
  slice_param.new_axis_mask = 0;
  slice_param.ellipsis_mask = 0;
  auto output_shape = module::getShape(getOutput());
  auto offset = module::getI64Array(getOffset());
  auto steps = module::getI64Array(getSteps());
  for (int i = 0; i < slice_param.shape_size; ++i) {
    slice_param.begin_index[i] = offset->at(i);
    slice_param.end_index[i] = output_shape[i] * steps->at(i) + offset->at(i);
    slice_param.stride[i] = steps->at(i);
  }
  if (module::isUniformQuantized(getInput())) {
    slice_param.buffer_global_addr = module::getAddress(getBuffer());
    auto input_shape = module::getShape(getInput());
    uint64_t input_size = ceiling_func(input_shape[0], (int64_t)4) * 4;
    for (int i = 1; i < slice_param.shape_size; i++) {
      input_size *= input_shape[i];
    }
    slice_param.imm_global_addr = slice_param.buffer_global_addr + input_size;
  }
  slice_param.is_dynamic = false;
  memcpy(param, &slice_param, sizeof(fw_stride_slice_layer_param_t));
}

mlir::Type tpu::SliceOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  if (opd_idx == 0) {
    return type_verify_case_same(getOperation(), 0, mode);
  }
  return do_nothing(mode);
}

bool tpu::SliceOp::support_multi_core() { return false; }
