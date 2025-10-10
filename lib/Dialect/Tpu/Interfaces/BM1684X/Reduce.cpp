//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
using namespace tpu_mlir::backend;

int64_t tpu::ReduceOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type,
    bool with_hw_margins) {
  int64_t buffer_size = 0;
  switch (BM168x::get_reduce_type(getMode())) {
  case SG_REDUCE_MEAN:
  case SG_REDUCE_SUM:
  case SG_REDUCE_MAX:
    int64_t npu_num = BM168x::NPU_NUM;
    int64_t dtype_size = BM168x::getFmtBytes(BM168x::getDataType(getInput()));
    int64_t eu_num = Arch::eu_num(dtype_size);
    auto axes = module::getI64Array(getAxes());
    assert(axes->size() == 1);
    auto dims = module::getShape(getInput()).size();
    if (axes->at(0) == 3) {
      // NOTE: NOW,  INT8 uses 3-stage reduce-w optimize, while FP16/FP32 uses
      // 2-stage reduce-w. INT8 reduce ==> in_wslice -> npu_num -> 2 -> 1
      // FP16/FP32 reduce ==> in_wslice -> npu_num -> 1
      buffer_size += 2 * ceiling_func(in_cslice, npu_num) * in_hslice * eu_num *
                     dtype_size;
    } else if (dims == 3 && axes->at(0) == 2) {
      buffer_size += 2 * ceiling_func(in_cslice, npu_num) * eu_num * dtype_size;
    }
    return buffer_size;
  }
  llvm_unreachable("unimplemented local reduceOp.");
}

void tpu::ReduceOp::codegen_local_bm1684x_kernel(
    std::vector<group_info_t> &in_group_infos,
    std::vector<group_info_t> &out_group_infos, local_sec_info_t &sec_info,
    std::shared_ptr<std::vector<tensor_spec_t>> input_spec,
    std::shared_ptr<std::vector<tensor_spec_t>> output_spec) {
  auto gi = out_group_infos[0];
  auto axes = module::getI64Array(getAxes());
  assert(axes->size() == 1);
  assert(axes->at(0) == 2 || axes->at(0) == 3);
  auto shape = module::getShape(getInput());
  assert(shape.size() == 4 || (shape.size() == 3 && axes->at(0) == 2));

  reduce_full_local_param_t param = {0};
  param.spec.common.axis_num = 1;
  param.spec.common.axis[0] = axes->at(0);
  param.spec.common.method = BM168x::get_reduce_type(getMode());
  param.spec.common.input_scale = 1.0f;
  param.spec.common.output_scale = 1.0f;
  param.spec.common.keep_dims = getKeepdims() ? 1 : 0;
  param.spec.buffer_addr = gi.buffer_addr;
  BM168x::call_local_func("backend_api_reduce_full_local", &param,
                          sizeof(param), &sec_info, input_spec->data(),
                          output_spec->data());
}

void tpu::ReduceOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto attr = parseParam();
  assert(attr.simplified);
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::fix_shape(input_spec->at(0), {attr.outer_n, attr.outer_c,
                                        attr.axis_dims, attr.inner_dims});
  BM168x::fix_shape(output_spec->at(0),
                    {attr.outer_n, attr.outer_c, 1, attr.inner_dims});

  reduce_full_global_param_t param = {0};
  param.spec.common.axis_num = 1;
  param.spec.common.axis[0] = 2;
  param.spec.common.method = BM168x::get_reduce_type(getMode());
  param.spec.common.input_scale = 1.0f;
  param.spec.common.output_scale = 1.0f;
  param.spec.common.keep_dims = 1;
  param.spec.buffer_addr = module::getAddress(getBuffer());
  param.if_getting_buffer_size = false;
  BM168x::call_global_func("backend_api_reduce_full_global", &param,
                           sizeof(reduce_full_global_param_t),
                           input_spec->data(), output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::ReduceOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(reduce_full_global_spec_t);
  reduce_full_global_spec_t spec = {0};
  auto &&axes = getAxes();
  spec.common.axis_num = axes.size();
  for (int i = 0; i < axes.size(); i++)
    spec.common.axis[i] = (axes[i].cast<IntegerAttr>().getInt());
  spec.common.method = BM168x::get_reduce_type(getMode());
  spec.common.input_scale = 1.0f;
  spec.common.output_scale = 1.0f;
  spec.common.keep_dims = getKeepdims() ? 1 : 0;
  spec.buffer_addr = module::getAddress(getBuffer());
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::ReduceOp::get_fw_type_bm1684x() { return FW_BMNET_REDUCE_FULL; }
