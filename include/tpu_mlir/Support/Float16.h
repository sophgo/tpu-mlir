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
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
uint32_t fp16_ieee_to_fp32_bits(uint16_t h);

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
float fp16_ieee_to_fp32_value(uint16_t h);

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 16-bit floating-point number in IEEE half-precision format, in bit
 * representation.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
uint16_t fp16_ieee_from_fp32_value(float f);

/*
 * Convert a 16-bit floating-point number in ARM alternative half-precision
 * format, in bit representation, to a 32-bit floating-point number in IEEE
 * single-precision format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
uint32_t fp16_alt_to_fp32_bits(uint16_t h);

/*
 * Convert a 16-bit floating-point number in ARM alternative half-precision
 * format, in bit representation, to a 32-bit floating-point number in IEEE
 * single-precision format.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
float fp16_alt_to_fp32_value(uint16_t h);

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 16-bit floating-point number in ARM alternative half-precision format, in bit
 * representation.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
uint16_t fp16_alt_from_fp32_value(float f);

uint16_t float_to_bf16_uint16_simple(float x);

float bf16_uint16_to_float_simple(uint16_t x);

/*
convert to f32 float to f16/bf16 float
*/
void f32_to_f16(float *p_src, float *p_dst, int num);
void f32_to_bf16(float *p_src, float *p_dst, int num);

} // namespace tpu_mlir
