//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynCompileCommon.hpp"
using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================

void tpu::NmsOp::codegen_global_bm1684x() {
  llvm_unreachable("Only support dynamic codegen");
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::NmsOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_nms_global_spec_t);
  dyn_nms_global_spec_t spec = {0};
  spec.common.input_num = 2;
  spec.common.keep_topk_per_class = getMaxOutputSize();
  spec.common.center_point_box = getCenterPointBox();
  spec.common.onnx_nms = 1;
  if (getInputs().size() >= 5) {
    auto iou = dyn_cast<top::WeightOp>(getInputs()[3].getDefiningOp());
    auto data = iou.read_as_float();
    spec.common.iou_threshold = data->data()[0];
    auto score_th = dyn_cast<top::WeightOp>(getInputs()[4].getDefiningOp());
    data = score_th.read_as_float();
    spec.common.score_threshold = data->data()[0];
  } else {
    spec.common.iou_threshold = 0.5;
    spec.common.score_threshold = 0.5;
  }

  spec.buffer_addr = module::isBM1688() ? module::getAddress(getBuffer()) : 0;
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::NmsOp::get_fw_type_bm1684x() { return FW_BMNET_ONNX_NMS; }
