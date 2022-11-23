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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float scale_val;
    int begin_axis;
    int end_axis;
    int log;
    int zero_point;
} softmax_common_param_t;

typedef struct {
    softmax_common_param_t common;
} softmax_global_param_t;

typedef struct {
    softmax_common_param_t common;
    uint32_t buffer_addr;
} softmax_local_param_t;

typedef struct {
    softmax_common_param_t common;
} softmax_tflite_fix8b_param_t;

#ifdef __cplusplus
}
#endif
// =========================================
// GloballGenInterface
// =========================================
void tpu::SoftmaxOp::codegen_global_bm1684x() {
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  bool has_table = !table().getType().isa<NoneType>();
  float in_scale = 1.0;
  if (Quant::isUniformQuantized(input())) {
    auto in_qtype = Quant::getUniformQuantizedType(input());
    in_scale = in_qtype.getScale();
  }
  if (Quant::isUniformQuantized(input(), output())) {
    assert(has_table);
    auto out_qtype = Quant::getUniformQuantizedType(output());
    softmax_tflite_fix8b_param_t param = {0};
    auto &common = param.common;
    common.begin_axis = axis();
    common.end_axis = axis();
    common.zero_point = out_qtype.getZeroPoint();
    common.scale_val = out_qtype.getScale();
    BM168x::call_global_func("backend_api_softmax_tflite_fix8b_global", &param,
                             sizeof(param), input_spec->data(),
                             output_spec->data());
  } else {
    softmax_global_param_t param = {0};
    auto &common = param.common;
    common.begin_axis = axis();
    common.end_axis = axis();
    common.scale_val = in_scale;
    BM168x::call_global_func("backend_api_softmax_global", &param,
                             sizeof(param), input_spec->data(),
                             output_spec->data());
  }
}
