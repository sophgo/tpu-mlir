//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::RequantIntAxisOp::init(InferenceParameter &p) {
  return success();
}
void tpu::RequantIntAxisOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RequantIntAxisOp::inference(InferenceParameter &p) {
  auto o_sType = module::getStorageType(getOutput());

  auto shape = module::getShape(getOutput());
  auto mode = getQuantMode();
  auto round_mode = round_mode_convert(getRoundMode());
  bool fuse_rq_axis = getFuseRqAxis();
  int64_t inner = 1;
  for (int i = 2; i < shape.size(); ++i) {
    inner *= shape[i];
  }

  int64_t outner = 1;
  for (int i = 0; i < shape.size() - 1; ++i) {
    outner *= shape[i];
  }

  int64_t zp_x = 0;
  if (module::isUniformQuantized(getInput())) {
    auto i_qtype = module::getUniformQuantizedType(getInput());
    zp_x = i_qtype.getZeroPoint();
    assert(mode == tpu::RequantMode::MultiplierShift);
  }

  if (mode == tpu::RequantMode::TFLite_LShift ||
      mode == tpu::RequantMode::TFLite) {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
    for (int c = 0; c < shape[1]; ++c) {
      int64_t multi, shift_val, zero_point;
      if (module::isBM1684X()) {
        multi = p.inputs[1][c * 3];
        shift_val = p.inputs[1][c * 3 + 1];
        zero_point = p.inputs[1][c * 3 + 2];
      } else {
        multi = p.inputs[1][c * 2];
        uint32_t tmp = p.inputs[1][c * 2 + 1];
        shift_val = (int64_t)((char)(tmp & 0xff));
        zero_point = (int64_t)(short)((tmp & 0xffff0000) >> 16);
      }
      for (int n = 0; n < shape[0]; ++n) {
        for (int i = 0; i < inner; ++i) {
          int offset = (n * shape[1] + c) * inner + i;
          int v = zero_point +
                  MultiplyByQuantizedMultiplier((int32_t)(p.inputs[0][offset]),
                                                (int32_t)multi,
                                                (int32_t)shift_val, round_mode);
          p.outputs[0][offset] = saturate(v, o_sType);
        }
      }
    }
  } else if (mode == tpu::RequantMode::MultiplierShift) {
    if (fuse_rq_axis) {
#pragma omp parallel for schedule(static, omp_schedule(shape[shape.size() - 1]))
      for (int w = 0; w < shape[shape.size() - 1]; ++w) {
        int64_t multi, shift_val, zero_point;
        if (module::isBM1684X()) {
          multi = p.inputs[1][w * 3];
          shift_val = p.inputs[1][w * 3 + 1];
          zero_point = p.inputs[1][w * 3 + 2];
        } else {
          multi = p.inputs[1][w * 2];
          uint32_t tmp = p.inputs[1][w * 2 + 1];
          shift_val = (int64_t)((char)(tmp & 0xff));
          zero_point = (int64_t)(short)((tmp & 0xffff0000) >> 16);
        }
        for (int i = 0; i < outner; i++) {

          int offset = i * shape[shape.size() - 1] + w;
          int v =
              zero_point + applyMultiplierAndRShift(
                               (p.inputs[0][offset] - zp_x), multi, shift_val,
                               tpu::RequantMode::MultiplierShift, round_mode);
          p.outputs[0][offset] = saturate(v, o_sType);
        }
      }

    } else {
#pragma omp parallel for schedule(static, omp_schedule(shape[1]))
      for (int c = 0; c < shape[1]; ++c) {
        int64_t multi, rshift_val, zero_point;
        if (module::isBM1684X()) {
          multi = p.inputs[1][c * 3];
          rshift_val = -p.inputs[1][c * 3 + 1];
          zero_point = p.inputs[1][c * 3 + 2];
        } else {
          multi = p.inputs[1][c * 2];
          uint32_t tmp = p.inputs[1][c * 2 + 1];
          rshift_val = (int64_t)(-(char)(tmp & 0xff));
          zero_point = (int64_t)(short)((tmp & 0xffff0000) >> 16);
        }
        for (int n = 0; n < shape[0]; ++n) {
          for (int i = 0; i < inner; ++i) {
            int offset = (n * shape[1] + c) * inner + i;
            int v = zero_point +
                    applyMultiplierAndRShift(
                        (p.inputs[0][offset] - zp_x), multi, rshift_val,
                        tpu::RequantMode::MultiplierShift, round_mode);
            p.outputs[0][offset] = saturate(v, o_sType);
          }
        }
      }
    }
  }
  return success();
}

mlir::Type tpu::RequantIntAxisOp::type_verify(uint64_t opd_idx,
                                              TypeCastMode &mode) {
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

bool tpu::RequantIntAxisOp::support_multi_core() { return false; }
