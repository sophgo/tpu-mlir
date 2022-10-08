//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Int8.h"
#include "bitcasts.h"
#include <math.h>

using namespace tpu_mlir;
using namespace tpu_mlir::helper;

namespace tpu_mlir {
/*
Quantize an Activation tensor into INT8 / UINT8
For cv18xx
*/
static int32_t cvi_to_int(float v, int round_mode = 0) {
  int32_t i32_val;
  if (round_mode == 1) { // round to zero
    i32_val = (int)v;
  } else { // round to nearest even
    float fraction, integer;
    float abs_v = std::abs(v);
    fraction = std::modf(abs_v, &integer);
    i32_val = (int)integer;
    if (fraction > 0.5) {
      i32_val = i32_val + 1;
    } else if (fraction == 0.5) {
      if (i32_val & 0x01) {
        i32_val = i32_val + 1;
      }
    }
    if (v < 0)
      i32_val = -i32_val;
  }
  return i32_val;
}

int8_t cvi_f32_to_int8(float v, int round_mode) {
  int32_t i32_val = cvi_to_int(v, round_mode);
  // already rounded just need saturate
  return Quant::to_int8(i32_val);
}

uint8_t cvi_f32_to_uint8(float v, int round_mode) {
  int32_t i32_val = cvi_to_int(v, round_mode);
  // already rounded just need saturate
  return Quant::to_uint8(i32_val);
}

void cvi_f32_to_int8(float *p_src, float *p_dst,
                    float scale, int zero_point,
                    int num, bool is_unsigned,
                    bool is_tpu) {
  assert(is_tpu);
  if (is_tpu) {
    scale = cvi_f32_to_bf16(scale);
    zero_point = cvi_f32_to_bf16(zero_point);
#pragma omp parallel for schedule(static, omp_schedule(num))
    for (int i = 0; i < num; i++) {
      float val = cvi_f32_to_bf16(
                    cvi_f32_to_bf16(
                      cvi_f32_to_bf16(p_src[i], false) * scale)
                  + zero_point, (zero_point != 0));
      if (is_unsigned) {
        p_dst[i] = (float)cvi_f32_to_uint8(val, 0);
      } else {
        p_dst[i] = (float)cvi_f32_to_int8(val, 0);
      }
    }
  } else {
#pragma omp parallel for schedule(static, omp_schedule(num))
    for (int i = 0; i < num; i++) {
      int val = std::round(p_src[i] * scale) + zero_point;
      if (is_unsigned) {
        p_dst[i] = (float)cvi_f32_to_uint8(val, 1);
      } else {
        p_dst[i] = (float)cvi_f32_to_int8(val, 1);
      }
    }
  }
}
// Todo uint16 int16

} // namespace tpu_mlir
