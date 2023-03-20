//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/BM168x/DynCompileCommon.hpp"
using namespace tpu_mlir::backend;

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
