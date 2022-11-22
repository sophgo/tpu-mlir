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
#include "tpu_mlir/Support/MathUtils.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int pad[4][2];
  int type;
  float constant;
} pad_param_t;

#ifdef __cplusplus
}
#endif

// =========================================
// GlobalGenInterface
// =========================================
void tpu::PadOp::codegen_global_bm1684x() {
  pad_param_t param = {0};
  auto in_shape = Module::getShape(input());
  int pad_dim = in_shape.size();
  auto pads = Module::getI64Array(paddings());
  for (int i = 0; i < pad_dim; i++) {
    param.pad[i][0] = pads->at(i);
    param.pad[i][1] = pads->at(i + pad_dim);
  }
  param.type = mode();

  param.constant = val().convertToDouble();

  auto op = getOperation();
  auto input_spec = BM168x::get_input_spec(op);
  auto output_spec = BM168x::get_output_spec(op);
  BM168x::call_global_func("backend_api_pad", &param, sizeof(param),
                                       input_spec->data(), output_spec->data());
}
