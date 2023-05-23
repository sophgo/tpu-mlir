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

/*
convert f32 to f16/bf16 by uint16
*/
uint16_t f32_to_f16(float src);
uint16_t f32_to_bf16(float src, bool is_tpu = true);
float f16_to_f32(uint16_t src);
float bf16_to_f32(uint16_t src);

/*
convert to f32 float to f16/bf16 float
*/
float F16(float src);
void F16(float *p_src, float *p_dst, int num);
float BF16(float src, bool is_tpu = true);
void BF16(float *p_src, float *p_dst, int num, bool is_tpu = true);

float bf16_mul(float lhs, float rhs);
float bf16_add(float lhs, float rhs);
} // namespace tpu_mlir
