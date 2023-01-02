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

#include "tpu_mlir/Support/Module.h"



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
  auto p = parseParam();
  int using_bias = p.with_bias ? 1 : 0;
  int if_relu = p.do_relu ? 1 : 0;
  int if_right_active = isa<top::WeightOp>(getRight().getDefiningOp()) ? 0 : 1;
  auto rshift_v = module::getI64Array(getRshifts(), 1, 0);
  assert(rshift_v->size() == 1);
  FcQParams quant_param{0, 0, 0, 0, 0};
  BM1684::instance().dl_nodechip_fc_fix8b_forward_parallel(
      module::getAddress(getInput()), module::getAddress(getRight()),
      p.with_bias ? module::getAddress(getBias()) : 0,
      module::getAddress(getOutput()), 0, p.M, p.K, p.N, 0, using_bias, 1, 1, 1,
      rshift_v->at(0), 0, if_relu, 1, if_right_active, 1, 0, FcPerLayerShift,
      &quant_param, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
}
