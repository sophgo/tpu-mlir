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
    return sizeof(nms_common_spec_t);
  nms_common_spec_t spec = {0};
  spec.input_num = 2;
  spec.keep_topk_per_class = getMaxOutputSize();
  spec.center_point_box = getCenterPointBox();
  spec.onnx_nms = 1;
  if (getInputs().size() == 5) {
    auto iou = dyn_cast<top::WeightOp>(getInputs()[3].getDefiningOp());
    auto data = iou.read_as_float();
    spec.iou_threshold = data->data()[0];
    auto score_th = dyn_cast<top::WeightOp>(getInputs()[4].getDefiningOp());
    data = score_th.read_as_float();
    spec.score_threshold = data->data()[0];
  } else {
    spec.iou_threshold = 0.5;
    spec.score_threshold = 0.5;
  }
  return BM168x::dynamic_spec_to_buffer(buffer, spec);
}

int64_t tpu::NmsOp::get_fw_type_bm1684x() { return FW_BMNET_ONNX_NMS; }
