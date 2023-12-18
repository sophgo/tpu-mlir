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

float get_f8e4m3_max();
float get_f8e4m3_min();

float get_f8e5m2_max();
float get_f8e5m2_min();

/*
convert f32 to f8e4m3 f8e5m2 by uint8
*/
uint8_t f32_to_f8e4m3(float src, bool satu);
uint8_t f32_to_f8e5m2(float src, bool satu);
float f8e4m3_to_f32(uint8_t src);
float f8e5m2_to_f32(uint8_t src);

/*
convert f16 to f8e4m3 f8e5m2 by uint8
*/
float f8e4m3_to_f16(uint8_t src);
float f8e5m2_to_f16(uint8_t src);

/*
  convert f32 to f8e4m3 and back to f32
*/
float F8E4M3(float src, float step, bool satu);
float F8E5M2(float src, float step, bool satu);
void F8E4M3(const float *p_src, float *p_dst, int num, float step, bool satu);
void F8E5M2(const float *p_src, float *p_dst, int num, float step, bool satu);

} // namespace tpu_mlir
