//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float8.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::RequantFpOp::init(InferenceParameter &p) {
  return success();
}
void tpu::RequantFpOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RequantFpOp::inference(InferenceParameter &p) {
  int64_t zero_point = 0;
  auto o_sType = module::getStorageType(getOutput());
  if (o_sType.isFloat8E4M3FN() || o_sType.isFloat8E5M2()) {
    ;
  } else {
    auto o_qtype = module::getUniformQuantizedType(getOutput());
    zero_point = o_qtype.getZeroPoint();
  }
  auto mode = getQuantMode();
  auto shape = module::getShape(getOutput());
  int64_t length = 1;
  for (int i = 0; i < shape.size(); ++i) {
    length *= shape[i];
  }

  float scale_v = getScale().convertToDouble();
  float offset_v = getOffset().convertToDouble();
  auto round_mode = round_mode_convert(getRoundMode());

  switch (mode) {
  case RequantMode::TFLite:
  case RequantMode::TFLite_LShift: {
#pragma omp parallel for schedule(static, omp_schedule(length))
    for (int64_t i = 0; i < length; ++i) {
      int32_t v = to_int(p.inputs[0][i] * scale_v, round_mode) + zero_point;
      p.outputs[0][i] = saturate(v, o_sType);
    }
  } break;
  case RequantMode::MultiplierShift: {
#pragma omp parallel for schedule(static, omp_schedule(length))
    for (int64_t i = 0; i < length; ++i) {
      int32_t v =
          to_int((float)(p.inputs[0][i]) * scale_v + offset_v, round_mode);
      p.outputs[0][i] = saturate(v, o_sType);
    }
  } break;
  case RequantMode::OnlyScale:
    if (o_sType.isFloat8E4M3FN())
      F8E4M3(p.inputs[0], p.outputs[0], length, 1 / scale_v, true);
    else if (o_sType.isFloat8E5M2())
      F8E5M2(p.inputs[0], p.outputs[0], length, 1.0, true);
    else
      llvm_unreachable("Unknown requant mode");
    break;
  default:
    llvm_unreachable("Unknown requant mode");
    break;
  }
  return success();
}

mlir::Type tpu::RequantFpOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  if (opd_idx == 0) {
    auto op = getOperation();
    auto stype = module::getStorageType(getInput());
    if (stype.isIntOrIndex()) {
      return do_nothing(mode);
    }
    mode = TypeCastMode::DO_CAST;
    auto bitwith = stype.getIntOrFloatBitWidth();
    return Builder(op).getIntegerType(bitwith);
  }
  return do_nothing(mode);
}

ArrayAttr tpu::RequantFpOp::getIndexingMaps() {
  auto shape = module::getShape(getInput());
  AffineMap identity_map =
      AffineMap::getMultiDimIdentityMap(shape.size(), getContext());
  SmallVector<AffineMap> indexingMaps{identity_map, identity_map};
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::RequantFpOp::support_multi_core() { return false; }
