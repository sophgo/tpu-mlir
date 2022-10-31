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
convert to f32 float to f16/bf16 float
*/
void f32_to_f16(float *p_src, float *p_dst, int num);
uint16_t f32_to_f16(float src);

// f32 to bf16 common
uint16_t f32_to_bf16(float src, bool is_cv18xx = false, bool is_tpu = true);
void f32_to_bf16(float *p_src, float *p_dst, int num, bool is_cv18xx = false,
                 bool is_tpu = true);
// bm168x
uint16_t bm_f32_to_bf16(float src);
// cv18xx
uint16_t cvi_f32_to_bf16(float src, bool is_tpu);
float cvi_f32_to_fbf16(float src, bool is_tpu = true);

/*
convert to f16/bf16 float to f32 float
*/
float f16_to_f32(uint16_t src);
float bf16_to_f32(uint16_t src);

/*
convert to int8 to bf16 float
*/
void cvi_int8_to_bf16(float *p_src, float *p_dst, float scale, int zero_point,
                      int num, bool is_tpu = true);

} // namespace tpu_mlir
