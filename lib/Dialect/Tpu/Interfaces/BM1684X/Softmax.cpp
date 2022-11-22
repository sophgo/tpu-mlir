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
  unsigned long long input_addr;
  unsigned long long output_addr;
  int n;
  int c;
  int h;
  int w;
  bool log;
  float scale_val;
  int dtype;
  int zero_point;
  int begin_axis;
  int end_axis;
} softmax_global_param_t;

typedef struct {
  unsigned int input_addr;
  unsigned int output_addr;
  unsigned int buffer_addr;
  int n;
  int c;
  int h;
  int w;
  bool log;
  float scale_val;
  int begin_axis;
  int end_axis;
  int dtype;
} softmax_local_param_t;

typedef struct {
  unsigned long long input_addr;
  unsigned long long output_addr;
  unsigned long long table_addr;
  int n;
  int c;
  int h;
  int w;
  int zero_point;
  float scale_val;
  int begin_axis;
  int end_axis;
  int dtype;
} softmax_tflite_fix8b_param_t;

#ifdef __cplusplus
}
#endif
// =========================================
// GloballGenInterface
// =========================================
void tpu::SoftmaxOp::codegen_global_bm1684x() {
  int64_t n, c, h, w;
  Module::getNCHW(input(), n, c, h, w);
  int outer_num = 1, softmax_num = 1, inner_num = 1;
  auto in_shape = Module::getShape(input());
  int ax = axis();
  for (uint64_t i = 0; i < ax; i++) {
    outer_num *= in_shape[i];
  }
  softmax_num *= in_shape[ax];
  for (uint64_t i = ax + 1; i < in_shape.size(); i++) {
    inner_num *= in_shape[i];
  }
  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  bool has_table = !table().getType().isa<NoneType>();
  float in_scale = 0;
  if (Quant::isUniformQuantized(input())) {
    auto in_qtype = Quant::getUniformQuantizedType(input());
    in_scale = in_qtype.getScale();
  }
  if (Quant::isUniformQuantized(input(), output())) {
    assert(has_table);
    auto out_qtype = Quant::getUniformQuantizedType(output());
    softmax_tflite_fix8b_param_t param = {0};
    param.input_addr = Module::getAddress(input());
    param.output_addr = Module::getAddress(output());
    param.table_addr = Module::getAddress(table());
    param.n = outer_num;
    param.c = softmax_num;
    param.h = 1;
    param.w = inner_num;
    param.zero_point = out_qtype.getZeroPoint();
    param.scale_val = out_qtype.getScale();
    param.dtype = BM168x::getDataType(input());
    BM168x::call_global_func("backend_api_softmax_tflite_fix8b", &param,
                             sizeof(param), input_spec->data(),
                             output_spec->data());
  } else {
    softmax_global_param_t param = {0};
    param.input_addr = Module::getAddress(input());
    param.output_addr = Module::getAddress(output());
    param.n = outer_num;
    param.c = softmax_num;
    param.h = 1;
    param.w = inner_num;

    param.log = false;
    param.dtype = BM168x::getDataType(input());
    param.scale_val = in_scale;
    BM168x::call_global_func("backend_api_softmax_global", &param,
                             sizeof(param), input_spec->data(),
                             output_spec->data());
  }
}
