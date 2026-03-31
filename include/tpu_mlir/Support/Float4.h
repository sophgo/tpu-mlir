//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <stdint.h>

namespace tpu_mlir {

float get_f4e2m1_max();

/*
convert f32 to f4e2m1 by uint8
*/
uint8_t f32_to_f4e2m1(float src);
float f4e2m1_to_f32(uint8_t src);

/*
  convert f32 to f8e4m3 and back to f32
*/
float F4E2M1(float src, float step);
void F4E2M1(const float *p_src, float *p_dst, int num, float step);

} // namespace tpu_mlir
