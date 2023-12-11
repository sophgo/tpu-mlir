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

typedef union {
  uint16_t bits;
  struct {
    uint16_t frac : 10; // mantissa
    uint16_t exp : 5;   // exponent
    uint16_t sign : 1;  // sign
  } format;
} fp16;

typedef union {
  uint16_t bits;
  struct {
    uint16_t frac : 7; // mantissa
    uint16_t exp : 8;  // exponent
    uint16_t sign : 1; // sign
  } format;
} bf16;

typedef union {
  float fval;
  uint32_t bits;
  struct {
    uint32_t frac : 23; // mantissa
    uint32_t exp : 8;   // exponent
    uint32_t sign : 1;  // sign
  } format;
} fp32;

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
float F16(float src, bool half_away_from_zero);
void F16(float *p_src, float *p_dst, int num);
float BF16(float src, bool is_tpu = true);
void BF16(float *p_src, float *p_dst, int num, bool is_tpu = true);

float bf16_mul(float lhs, float rhs);
float bf16_add(float lhs, float rhs);
} // namespace tpu_mlir
