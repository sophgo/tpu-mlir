//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace mlir;
using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace tpu_mlir::backend;

typedef enum {
  FcPerLayerShift = 0,
  FcPerLayerScale = 1,
  FcPerChannelScale = 2,
} FcQScale;

typedef struct {
  float perlayer_scale;
  int if_asymmetic;
  int weight_offset;
  int output_offset;
  int if_bias_float;
} FcQParams;

void tpu::MatMulOp::codegen_global_bm1684() {
  int64_t batch, M, K, N, right_zp;
  bool with_bias, relu;
  double relu_limit;
  parseParam(batch, M, K, N, with_bias, relu, relu_limit, right_zp);
  int using_bias = with_bias ? 1 : 0;
  int if_relu = relu ? 1 : 0;
  int if_right_active = isa<top::WeightOp>(right().getDefiningOp()) ? 0 : 1;
  auto rshift_v = Module::getI64Array(rshifts(), 1, 0);
  assert(rshift_v->size() == 1);
  FcQParams quant_param{0, 0, 0, 0, 0};
  BM1684::instance().dl_nodechip_fc_fix8b_forward_parallel(
      Module::getAddress(input()), Module::getAddress(right()),
      with_bias ? Module::getAddress(bias()) : 0, Module::getAddress(output()),
      0, M, K, N, 0, using_bias, 1, 1, 1, rshift_v->at(0), 0, if_relu, 1,
      if_right_active, 1, 0, FcPerLayerShift, &quant_param,
      (CMD_ID_NODE *)BM1684::instance().cmdid_node);
}
