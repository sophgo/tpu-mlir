//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <llvm/Support/Debug.h>

LogicalResult tpu::MeanStdScaleOp::init(InferenceParameter &p) {
  return success();
}

void tpu::MeanStdScaleOp::deinit(InferenceParameter &p) {}

mlir::Type tpu::MeanStdScaleOp::type_verify(uint64_t opd_idx,
                                            TypeCastMode &mode) {
  return do_nothing(mode);
}

int32_t RightShiftRound(int32_t src, int shift_num,
                        tpu_mlir::RoundingMode round_mode) {
  if (shift_num == 0)
    return src;
  if (shift_num > 63)
    shift_num = 63;
  int32_t val, res;
  if (shift_num < 0) {
    return src << (-shift_num);
  }
  val = src >> shift_num;
  res = val;
  int32_t lo_mask = (1ull << shift_num) - 1;
  int32_t mant = src & lo_mask;
  int32_t mant_0d5 = 1ull << (shift_num - 1);
  if (round_mode == ROUNDING_HALF_TO_EVEN) {
    if (mant == mant_0d5)
      res = val + (val & 1);
    else if (mant > mant_0d5)
      res = val + 1;
  } else if (round_mode == ROUNDING_HALF_AWAY_FROM_ZERO) {
    if (src >= 0 && mant >= mant_0d5)
      res = val + 1;
    else if (src < 0 && mant > mant_0d5)
      res = val + 1;
  } else if (round_mode == ROUNDING_TOWARDS_ZERO) {
    if (src < 0)
      res = val + (mant != 0);
  } else if (round_mode == ROUNDING_DOWN)
    res = val;
  else if (round_mode == ROUNDING_UP)
    res = val + (mant != 0);
  else if (round_mode == ROUNDING_HALF_UP) {
    if (mant >= mant_0d5)
      res = val + 1;
  } else if (round_mode == ROUNDING_HALF_DOWN) {
    if (mant > mant_0d5)
      res = val + 1;
  }
  return res;
}

int64_t applyMultiplierAndRShift(int64_t v, int64_t multiplier, int64_t rshift,
                                 tpu_mlir::RoundingMode round_mode) {
  return RightShiftRound(v * multiplier, (int)rshift, round_mode);
}

int saturate(int data) {
  int64_t max_val = INT8_MAX;
  int64_t min_val = INT8_MIN;
  int ret_val = 0;
  if ((int64_t)data > max_val) {
    ret_val = max_val;
  } else if ((int64_t)data < min_val) {
    ret_val = min_val;
  } else {
    ret_val = data;
  }
  return ret_val;
}

void do_requant_int(int32_t *inputs, std::vector<int64_t> input_shapes,
                    const int input_dims, int multi, int shift_val, int izp,
                    int ozp, float *outputs,
                    tpu_mlir::RoundingMode round_mode) {
  int64_t inner = 1;
  for (int i = 2; i < input_dims; ++i) {
    inner *= input_shapes[i];
  }

  for (int c = 0; c < input_shapes[1]; ++c) {
    for (int n = 0; n < input_shapes[0]; ++n) {
      for (int i = 0; i < inner; ++i) {
        int offset = (n * input_shapes[1] + c) * inner + i;
        int32_t v =
            ozp + applyMultiplierAndRShift((inputs[offset] - izp), multi,
                                           shift_val, round_mode);
        outputs[offset] = saturate(v);
      }
    }
  }
  return;
}

float round_float_number(float number, tpu_mlir::RoundingMode rounding_mode) {
  switch (rounding_mode) {
  case ROUNDING_HALF_TO_EVEN:
    return nearbyintf(number);
  case ROUNDING_HALF_AWAY_FROM_ZERO:
    return (number > 0.0f) ? floorf(number + 0.5f) : ceilf(number - 0.5f);
  case ROUNDING_TOWARDS_ZERO:
    return (number > 0.0f) ? floorf(number) : ceilf(number);
  case ROUNDING_DOWN:
    return floorf(number);
  case ROUNDING_UP:
    return ceilf(number);
  case ROUNDING_HALF_UP:
    return roundf(number);
  case ROUNDING_HALF_DOWN: {
    float intpart;
    modff(number, &intpart);
    float fracpart = number - intpart;
    if (number > 0.0f) {
      return (fracpart > 0.5f) ? intpart + 1.0f : intpart;
    } else {
      return (fracpart < -0.5f) ? intpart - 1.0f : intpart;
    }
  }
  default:
    return number;
  }
}

LogicalResult tpu::MeanStdScaleOp::inference(InferenceParameter &p) {
  auto std = module::getF64Array(getStd());
  auto scale = module::getF64Array(getScale());
  auto mean = module::getF64Array(getMean());
  auto zero_points = module::getF64Array(getZeroPoints());
  auto round_mode =
      round_mode_convert(symbolizeRoundMode(getRoundingMode()).value());
  auto rshift = module::getI64Array(getRshift());
  auto offset = module::getI64Array(getOffset());
  auto multi = module::getI64Array(getMulti());
  std::vector<int64_t> in_shape = module::getShape(getInput());
  auto in_shape_size = in_shape.size();

  int in_zp = (*zero_points)[0];
  int out_zp = (*zero_points)[1];
  int inner_size = 1;
  for (int i = 2; i < in_shape_size; i++) {
    inner_size *= in_shape[i];
  }
  int batch_size = in_shape[0];
  int channels = in_shape[1];

  int elem_num = 1;
  for (int i = 0; i < in_shape.size(); i++) {
    elem_num *= in_shape[i];
  }

  auto idtype = module::getStorageType(getInput().getType());
  auto odtype = module::getStorageType(getResult().getType());
  if ((idtype.isUnsignedInteger(8) || idtype.isSignedInteger(8)) &&
      odtype.isSignedInteger(8)) {
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < inner_size * channels; i++) {
        int global_idx = b * inner_size * channels + i;
        int chn_idx = (i / inner_size) % channels;
        int _rshift = rshift->at(chn_idx);
        int _offset = offset->at(chn_idx);
        int32_t res = 0;
        float tmp = 0.0;
        if (idtype.isUnsignedInteger(8)) {
          tmp = (float)(((int32_t)((uint8_t)p.inputs[0][global_idx] *
                                   (int32_t)multi->at(chn_idx)) +
                         _offset) /
                        pow(2, _rshift));
        } else {
          tmp = (float)(((int32_t)((int8_t)p.inputs[0][global_idx] *
                                   (int32_t)multi->at(chn_idx)) +
                         _offset) /
                        pow(2, _rshift));
        }

        res = round_float_number(tmp, round_mode);
        res = res < -128 ? -128 : res;
        res = res > 127 ? 127 : res;
        p.outputs[0][global_idx] = res;
      }
    }
  } else if (idtype.isF32() && odtype.isSignedInteger(8)) {
    std::vector<int32_t> res;
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < inner_size * channels; i++) {
        int global_idx = b * inner_size * channels + i;
        int chn_idx = (i / inner_size) % channels;
        float mean_float = mean->at(chn_idx);

        int32_t tmp = (int32_t)round_float_number(
            (int32_t)(p.inputs[0][global_idx] - mean_float) *
                (1 / std->at(chn_idx)),
            round_mode);
        res.push_back(tmp);
      }
    }
    // do requant
    do_requant_int(res.data(), in_shape, in_shape.size(), multi->at(0),
                   rshift->at(0), in_zp, out_zp, p.outputs[0], round_mode);
  } else {
    // default
    for (int b = 0; b < batch_size; b++) {
      for (int i = 0; i < inner_size * channels; i++) {
        int global_idx = b * inner_size * channels + i;
        int chn_idx = i / inner_size;
        p.outputs[0][global_idx] =
            (p.inputs[0][global_idx] - mean->at(chn_idx)) / std->at(chn_idx);
      }
    }
  }

  return success();
}

bool tpu::MeanStdScaleOp::support_multi_core() { return false; }

LogicalResult tpu::MeanStdScaleOp::LocalGenSupport() {
  if (module::isBM1684X()) {
    return success();
  } else {
    return failure();
  }
}
