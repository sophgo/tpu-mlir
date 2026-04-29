//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::MlpOp::init(InferenceParameter &p) { return success(); }

void tpu::MlpOp::deinit(InferenceParameter &p) { return; }

LogicalResult tpu::MlpOp::inference(InferenceParameter &p) { return success(); }

mlir::Type tpu::MlpOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  auto op = getOperation();
  if (opd_idx == 13) {
    // expert_id
    auto opd = op->getOperand(13);
    auto in_op = opd.getDefiningOp();
    if (in_op != nullptr && isa<top::WeightOp, top::NoneOp>(in_op)) {
      return do_nothing(mode);
    }
    auto stype = module::getStorageType(opd);
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    mode = TypeCastMode::DO_CAST;
    auto bitwidth = stype.getIntOrFloatBitWidth();
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      // indices should be int32 in BM1684x
      bitwidth = 32;
    }
    return Builder(op).getIntegerType(bitwidth);
  }
  return type_verify_case_same(op, opd_idx, mode);
}

bool tpu::MlpOp::support_multi_core() {
  return (module::isSG2380() || module::isBM1690Family()) &&
         !module::isOpInGroupParallel(*this);
}
