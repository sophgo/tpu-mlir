//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
using namespace tpu_mlir::backend;

void tpu::SliceOp::codegen_global_bm1684x() {
  auto p = parseParam();
  if (p.fusible) {
    auto in_addr = module::getAddress(getInput());
    auto in_size = module::getBytes(getInput());
    auto out_addr = module::getAddress(getOutput());
    if (in_addr <= out_addr && (in_addr + in_size) >= out_addr) {
      return;
    }
  }
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::fix_shape(input_spec->at(0), p.is_4);
  BM168x::fix_shape(output_spec->at(0), p.os_4);
  strideslice_common_spec_t param = {0};
  param.begin_mask = 0;
  param.end_mask = 0;
  int num_dims = p.is_4.size();
  for (int i = 0; i < num_dims; i++) {
    p.offset_4[i] = p.offset_4[i] < 0 ? p.offset_4[i] + p.is_4[i] : p.offset_4[i];
    param.begin_index[i] = p.offset_4[i];
    param.end_index[i] = p.os_4[i] * p.step_4[i] + p.offset_4[i];
    param.strides[i] = p.step_4[i];
  }
  BM168x::call_global_func("backend_api_strideslice_global", &param,
                           sizeof(param), input_spec->data(),
                           output_spec->data());
}

// =========================================
// LocalGenInterface
// =========================================

int64_t tpu::SliceOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  strideslice_local_spec_t spec = {0};
  auto &common = spec.common;
  common.begin_mask = 0;
  common.end_mask = 0;
  auto output_shape = SmallVector<int64_t>(module::getShape(getOutput()));
  const int num_dims = output_shape.size();
  output_shape[0] = out_nslice;
  if (num_dims > 2) {
    output_shape[2] = out_hslice;
  }
  if (num_dims > 3) {
    output_shape[3] = out_wslice;
  }
  const auto offset = module::getI64Array(getOffset());
  const auto steps = module::getI64Array(getSteps());
  for (int i = 0; i < num_dims; i++) {
    common.begin_index[i] = offset->at(i);
    common.strides[i] = steps->at(i);
    common.end_index[i] =
        common.begin_index[i] + output_shape[i] * common.strides[i];
  }
  local_sec_info_t sec_info;
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;
  // int64_t n, c, d, h, w, on, oc, od, oh, ow;
  // auto input = op->getOperand(0);
  // auto output = op->getResult(0);
  // module::getNCDHW(input, n, c, d, h, w, group_type);
  // module::getNCDHW(output, on, oc, od, oh, ow, group_type);
  sec_info.n_slice = in_nslice;
  sec_info.c_slice = in_cslice;
  sec_info.d_slice = in_dslice;
  sec_info.h_slice = in_hslice;
  sec_info.w_slice = in_wslice;
  sec_info.out_n_slice = out_nslice;
  sec_info.out_h_slice = out_hslice;
  sec_info.out_w_slice = out_wslice;
  return BM168x::call_local_bfsz_func("backend_api_strideslice_local_bfsz",
                                      &spec, sizeof(spec), &sec_info,
                                      input_spec->data(), output_spec->data());
}

void tpu::SliceOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                         int64_t h_step, int64_t d_step,
                                         int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info) {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op, group_type);
  auto output_spec = BM168x::get_output_spec(op, group_type);
  strideslice_local_spec_t spec = {0};
  const auto &gi = getGroupInfo(0, 0, 0, 0, 0);
  spec.buffer_addr = gi.buffer_addr;
  auto &common = spec.common;
  common.begin_mask = 0;
  common.end_mask = 0;
  auto input_shape = SmallVector<int64_t>(module::getShape(getInput()));
  auto output_shape = SmallVector<int64_t>(module::getShape(getOutput()));
  const int num_dims = output_shape.size();
  output_shape[0] = sec_info.out_n_slice;
  if (num_dims > 2) {
    output_shape[2] = sec_info.out_h_slice;
  }
  if (num_dims > 3) {
    output_shape[3] = sec_info.out_w_slice;
  }
  const auto offset = module::getI64Array(getOffset());
  const auto steps = module::getI64Array(getSteps());
  for (int i = 0; i < num_dims; i++) {
    common.begin_index[i] = offset->at(i) < 0 ? offset->at(i) + input_shape[i]
                                              : offset->at(i);
    common.strides[i] = steps->at(i);
    common.end_index[i] =
        common.begin_index[i] + output_shape[i] * common.strides[i];
  }

  BM168x::call_local_func("backend_api_strideslice_local", &spec, sizeof(spec),
                          &sec_info, input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::SliceOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(strideslice_global_spec_t);
  strideslice_global_spec_t param = {0};
  const std::vector<int64_t> input_shape = module::getShape(getInput());
  const std::vector<int64_t> output_shape = module::getShape(getOutput());
  param.common.begin_mask = 0;
  param.common.end_mask = 0;
  const int num_dims = input_shape.size();
  const auto offset = module::getI64Array(getOffset());
  const auto ends = module::getI64Array(getEnds());
  const auto steps = module::getI64Array(getSteps());
  param.shape_size = offset->size();
  param.ellipsis_mask = 0;
  param.new_axis_mask = 0;
  param.shrink_axis_mask = 0;
  param.is_dynamic = !module::isNone(getOffsetT()) ||
                     !module::isNone(getEndsT()) ||
                     !module::isNone(getStepsT());
  param.begin_as_tensor = !module::isNone(getOffsetT());
  param.end_as_tensor = !module::isNone(getEndsT());
  param.stride_as_tensor = !module::isNone(getStepsT());
  int axis = param.is_dynamic ? module::getI64Array(getAxes())->at(0) : 0;
  for (int i = 0; i < num_dims; i++) {
    param.common.begin_index[i] = offset->at(i);
    param.common.strides[i] = steps->at(i);
    // TODO: fix canonicalizers and reactivate this
    // param.common.end_index[i] = ends->at(i);
    auto offset_tmp = offset->at(i) < 0 ? offset->at(i) + input_shape[i] : offset->at(i);
    if (param.begin_as_tensor) {
      param.common.end_index[i] = ends->at(i);
    } else {
      param.common.end_index[i] = ends->at(i) < 0 ? ends->at(i) : output_shape[i] * steps->at(i) + offset_tmp;
    }
  }

  /* for dynamic input shape, it need the begin_mask/end_mask
    to deduction the actual output shape */
  for (int i = 0; i < num_dims; i++) {
    if (!param.begin_as_tensor && input_shape[i] == output_shape[i] ||
        param.begin_as_tensor && axis != i) {
      param.common.begin_mask |= (1 << i);
    }
    if (!param.end_as_tensor &&
            input_shape[i] == output_shape[i] ||
        param.end_as_tensor && axis != i) {
      param.common.end_mask |= (1 << i);
    }
  }
  return BM168x::dynamic_spec_to_buffer(buffer, param);
}

int64_t tpu::SliceOp::get_fw_type_bm1684x() { return FW_BMNET_STRIDESLICE; }

int64_t tpu::SliceOp::dyn_codegen_local_bm1684x(void *buffer) {
  llvm_unreachable("not implement");
}
