//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::RequantIntOp::getFLOPs() {
  auto qmode = getQuantModeAttr().str();
  auto num = qmode == "Normal" ? 3 : 5;
  return module::getNumElements(getOutput()) * num;
}
LogicalResult top::RequantIntOp::init(InferenceParameter &p) {
  return success();
}
void top::RequantIntOp::deinit(InferenceParameter &p) {}

LogicalResult top::RequantIntOp::inference(InferenceParameter &p) {

  std::map<std::string, RoundingMode> map_mode = {
      {"HalfAwayFromZero", RoundingMode::ROUNDING_HALF_AWAY_FROM_ZERO},
      {"HalfUp", RoundingMode::ROUNDING_HALF_UP},
      {"HalfDown", RoundingMode::ROUNDING_HALF_DOWN},
      {"HalfToEven", RoundingMode::ROUNDING_HALF_TO_EVEN},
      {"HalfToOdd", RoundingMode::ROUNDING_HALF_TO_ODD},
      {"HalfTowardsZero", RoundingMode::ROUNDING_HALF_TOWARDS_ZERO},
      {"TowardsZero", RoundingMode::ROUNDING_TOWARDS_ZERO},
      {"Up", RoundingMode::ROUNDING_UP},
      {"Down", RoundingMode::ROUNDING_DOWN}};

  auto o_sType = module::getStorageType(getOutput());
  auto o_qtype = module::getUniformQuantizedType(getOutput());
  auto qmode = getQuantModeAttr().str();
  auto iter = map_mode.find(getRoundModeAttr().str());
  RoundingMode rmode;
  if (iter != map_mode.end()) {
    rmode = iter->second;
  }
  auto shape = module::getShape(getOutput());
  int64_t inner = 1;
  for (int i = 2; i < shape.size(); ++i) {
    inner *= shape[i];
  }
  int64_t zp_x = 0;
  if (module::isUniformQuantized(getInput())) {
    auto i_qtype = module::getUniformQuantizedType(getInput());
    zp_x = i_qtype.getZeroPoint();
    ASSERT_THIS(qmode == "MultiplierShift");
  }
  auto shift = module::getI64Array(getRshift());
  auto multi = module::getI64Array(getMultiplier());
  auto zero_point = o_qtype.getZeroPoint();
  auto raw_shift = *shift;
  auto raw_multi = *multi;
  ASSERT_THIS(raw_multi.size() == raw_shift.size() &&
              "zero point & multi & shift size missmatch");

  bool per_channel = raw_multi.size() != 1;
  if (qmode == "TFLite_LShift" || qmode == "TFLite") {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      int multi_val = raw_multi[per_channel ? c : 0];
      int shift_val = -raw_shift[per_channel ? c : 0];
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          auto v = zero_point +
                   MultiplyByQuantizedMultiplier((int32_t)(p.inputs[0][offset]),
                                                 (int32_t)multi_val,
                                                 (int32_t)shift_val, rmode);
          p.outputs[0][offset] = saturate(v, o_sType);
        }
      }
    }
  } else if (qmode == "MultiplierShift") {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      int multi_val = raw_multi[per_channel ? c : 0];
      int shift_val = raw_shift[per_channel ? c : 0];
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          auto v = zero_point + applyMultiplierAndRShift(
                                    (p.inputs[0][offset] - zp_x), multi_val,
                                    shift_val,
                                    tpu::RequantMode::MultiplierShift, rmode);
          p.outputs[0][offset] = saturate(v, o_sType);
        }
      }
    }
  } else {
    llvm_unreachable("Unknown requant mode");
  }
  return success();
}

void top::RequantIntOp::shape_inference() {
  common_shape_inference(getOperation());
  auto axis = getRqAxis();
  if (axis < 0) {
    auto in_shape = module::getShape(getInput());
    axis += in_shape.size();
    setRqAxis(axis);
  }
}
