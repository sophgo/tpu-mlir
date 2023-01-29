//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Module.h"




using namespace tpu_mlir::backend;

void tpu::ReduceOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto attr = parseParam();
  assert(attr.simplified);

  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);

  std::vector<int32_t> in_shape = {(int)attr.outer_n, (int)attr.outer_c,
                                   (int)attr.axis_dims, (int)attr.inner_dims};
  std::vector<int32_t> out_shape = {(int)attr.outer_n, (int)attr.outer_c, 1,
                                    (int)attr.inner_dims};
  BM168x::fix_shape(input_spec->at(0), in_shape);
  BM168x::fix_shape(output_spec->at(0), out_shape);

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
  return 0;
}
