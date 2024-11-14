//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::DequantIntOp::getFLOPs() {
  auto qmode = getQuantModeAttr().str();
  auto num = qmode == "Normal" ? 3 : 5;
  return module::getNumElements(getOutput()) * num;
}
LogicalResult top::DequantIntOp::init(InferenceParameter &p) {
  return success();
}
void top::DequantIntOp::deinit(InferenceParameter &p) {}

LogicalResult top::DequantIntOp::inference(InferenceParameter &p) {

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

  auto qtype = module::getUniformQuantizedType(getInput());
  int64_t num_elem = module::getNumElements(getInput());
  auto shape = module::getShape(getOutput());
  int64_t inner = 1;
  for (int i = 2; i < shape.size(); ++i) {
    inner *= shape[i];
  }
  // auto shift_val = getShift();
  // auto mul_val = getMultiplier();
  // auto offset = qtype.getZeroPoint();
  auto qmode = getQuantModeAttr().str();
  auto iter = map_mode.find(getRoundModeAttr().str());
  RoundingMode rmode;
  if (iter != map_mode.end()) {
    rmode = iter->second;
  }
  auto shift = module::getI64Array(getShift());
  auto multi = module::getI64Array(getMultiplier());
  auto zero_point = qtype.getZeroPoint();
  auto raw_shift = *shift;
  auto raw_multi = *multi;
  ASSERT_THIS(raw_multi.size() == raw_shift.size() &&
              "zero point & multi & shift size missmatch");

  if (qmode == "Normal") {
    if (raw_shift.size() == 1) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t idx = 0; idx < num_elem; idx++) {
        int32_t tmp = (int32_t)p.inputs[0][idx] - zero_point;
        auto v =
            applyMultiplierAndRShift(tmp, raw_multi[0], -raw_shift[0],
                                     tpu::RequantMode::MultiplierShift, rmode);
        p.outputs[0][idx] = v;
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
      for (int c = 0; c < shape[1]; ++c) {
        int64_t multi_val = raw_multi[c];
        int64_t shift_val = raw_shift[c];
        for (int n = 0; n < shape[0]; ++n) {
          for (int i = 0; i < inner; ++i) {
            int offset = (n * shape[1] + c) * inner + i;
            int32_t tmp = (int32_t)p.inputs[0][offset] - zero_point;
            p.outputs[0][offset] = applyMultiplierAndRShift(
                tmp, multi_val, -shift_val, tpu::RequantMode::MultiplierShift,
                rmode);
          }
        }
      }
    }
  } else if (qmode == "TFLite") {
    int64_t lshift_val = getLshift();
    if (raw_shift.size() == 1) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t idx = 0; idx < num_elem; idx++) {
        int64_t tmp = ((int32_t)p.inputs[0][idx] - zero_point) * raw_multi[0]
                      << lshift_val;
        auto v = RightShiftRound(tmp, 31, ROUNDING_HALF_UP);
        v = RightShiftRound(v, -raw_shift[0], rmode);
        p.outputs[0][idx] = v;
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
      for (int c = 0; c < shape[1]; ++c) {
        int64_t multi_val = raw_multi[c];
        int64_t shift_val = p.inputs[1][c * 3 + 1];
        for (int n = 0; n < shape[0]; ++n) {
          for (int i = 0; i < inner; ++i) {
            int offset = (n * shape[1] + c) * inner + i;
            int64_t tmp =
                ((int32_t)p.inputs[0][offset] - zero_point) * multi_val
                << lshift_val;
            p.outputs[0][offset] =
                MultiplyByQuantizedMultiplier(tmp, 1, -shift_val, rmode);
          }
        }
      }
    }
  } else {
    llvm_unreachable("Unknown dequant mode");
  }
  return success();
}

void top::DequantIntOp::shape_inference() {
  common_shape_inference(getOperation());
}
