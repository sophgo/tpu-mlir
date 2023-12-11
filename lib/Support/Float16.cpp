//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/Float16.h"
#include "bitcasts.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <float.h>

namespace tpu_mlir {

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
static inline uint32_t fp16_ieee_to_fp32_bits(uint16_t h) {
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t)h << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the bits 0-30
   * of the 32-bit word:
   *
   *      +---+-----+------------+-------------------+
   *      | 0 |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  30  27-31     17-26            0-16
   */
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  /*
   * Renorm shift is the number of bits to shift mantissa left to make the
   * half-precision number normalized. If the initial number is normalized, some
   * of its high 6 bits (sign == 0 and 5-bit exponent) equals one. In this case
   * renorm_shift == 0. If the number is denormalize, renorm_shift > 0. Note
   * that if we shift denormalized nonsign by renorm_shift, the unit bit of
   * mantissa will shift into exponent, turning the biased exponent into 1, and
   * making mantissa normalized (i.e. without leading 1).
   */
#ifdef _MSC_VER
  unsigned long nonsign_bsr;
  _BitScanReverse(&nonsign_bsr, (unsigned long)nonsign);
  uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31;
#else
  uint32_t renorm_shift = __builtin_clz(nonsign);
#endif
  renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
  /*
   * Iff half-precision number has exponent of 15, the addition overflows it
   * into bit 31, and the subsequent shift turns the high 9 bits into 1. Thus
   *   inf_nan_mask ==
   *                   0x7F800000 if the half-precision number had exponent of
   * 15 (i.e. was NaN or infinity) 0x00000000 otherwise
   */
  const int32_t inf_nan_mask =
      ((int32_t)(nonsign + 0x04000000) >> 8) & INT32_C(0x7F800000);
  /*
   * Iff nonsign is 0, it overflows into 0xFFFFFFFF, turning bit 31 into 1.
   * Otherwise, bit 31 remains 0. The signed shift right by 31 broadcasts bit 31
   * into all bits of the zero_mask. Thus zero_mask == 0xFFFFFFFF if the
   * half-precision number was zero (+0.0h or -0.0h) 0x00000000 otherwise
   */
  const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
  /*
   * 1. Shift nonsign left by renorm_shift to normalize it (if the input was
   * denormal)
   * 2. Shift nonsign right by 3 so the exponent (5 bits originally) becomes an
   * 8-bit field and 10-bit mantissa shifts into the 10 high bits of the 23-bit
   * mantissa of IEEE single-precision number.
   * 3. Add 0x70 to the exponent (starting at bit 23) to compensate the
   * different in exponent bias (0x7F for single-precision number less 0xF for
   * half-precision number).
   * 4. Subtract renorm_shift from the exponent (starting at bit 23) to account
   * for renormalization. As renorm_shift is less than 0x70, this can be
   * combined with step 3.
   * 5. Binary OR with inf_nan_mask to turn the exponent into 0xFF if the input
   * was NaN or infinity.
   * 6. Binary ANDNOT with zero_mask to turn the mantissa and exponent into zero
   * if the input was zero.
   * 7. Combine with the sign of the input number.
   */
  return sign |
         ((((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) |
           inf_nan_mask) &
          ~zero_mask);
}

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
static inline float fp16_ieee_to_fp32_value(uint16_t h) {
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t)h << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the high bits
   * of the 32-bit word:
   *
   *      +-----+------------+---------------------+
   *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
   *      +-----+------------+---------------------+
   * Bits  27-31    17-26            0-16
   */
  const uint32_t two_w = w + w;

  /*
   * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become
   * mantissa and exponent of a single-precision floating-point number:
   *
   *       S|Exponent |          Mantissa
   *      +-+---+-----+------------+----------------+
   *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
   *      +-+---+-----+------------+----------------+
   * Bits   | 23-31   |           0-22
   *
   * Next, there are some adjustments to the exponent:
   * - The exponent needs to be corrected by the difference in exponent bias
   * between single-precision and half-precision formats (0x7F - 0xF = 0x70)
   * - Inf and NaN values in the inputs should become Inf and NaN values after
   * conversion to the single-precision number. Therefore, if the biased
   * exponent of the half-precision input was 0x1F (max possible value), the
   * biased exponent of the single-precision output must be 0xFF (max possible
   * value). We do this correction in two steps:
   *   - First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset
   * below) rather than by 0x70 suggested by the difference in the exponent bias
   * (see above).
   *   - Then we multiply the single-precision result of exponent adjustment by
   * 2**(-112) to reverse the effect of exponent adjustment by 0xE0 less the
   * necessary exponent adjustment by 0x70 due to difference in exponent bias.
   *     The floating-point multiplication hardware would ensure than Inf and
   * NaN would retain their value on at least partially IEEE754-compliant
   * implementations.
   *
   * Note that the above operations do not handle denormal inputs (where biased
   * exponent == 0). However, they also do not operate on denormal inputs, and
   * do not produce denormal results.
   */
  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) ||              \
    defined(__GNUC__) && !defined(__STRICT_ANSI__)
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float normalized_value =
      fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  /*
   * Convert denormalized half-precision inputs into single-precision results
   * (always normalized). Zero inputs are also handled here.
   *
   * In a denormalized number the biased exponent is zero, and mantissa has
   * on-zero bits. First, we shift mantissa into bits 0-9 of the 32-bit word.
   *
   *                  zeros           |  mantissa
   *      +---------------------------+------------+
   *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
   *      +---------------------------+------------+
   * Bits             10-31                0-9
   *
   * Now, remember that denormalized half-precision numbers are represented as:
   *    FP16 = mantissa * 2**(-24).
   * The trick is to construct a normalized single-precision number with the
   * same mantissa and thehalf-precision input and with an exponent which would
   * scale the corresponding mantissa bits to 2**(-24). A normalized
   * single-precision floating-point number is represented as: FP32 = (1 +
   * mantissa * 2**(-23)) * 2**(exponent - 127) Therefore, when the biased
   * exponent is 126, a unit change in the mantissa of the input denormalized
   * half-precision number causes a change of the constructud single-precision
   * number by 2**(-24), i.e. the same ammount.
   *
   * The last step is to adjust the bias of the constructed single-precision
   * number. When the input half-precision number is zero, the constructed
   * single-precision number has the value of FP32 = 1 * 2**(126 - 127) =
   * 2**(-1) = 0.5 Therefore, we need to subtract 0.5 from the constructed
   * single-precision number to get the numerical equivalent of the input
   * half-precision number.
   */
  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
      fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  /*
   * - Choose either results of conversion of input as a normalized number, or
   * as a denormalized number, depending on the input exponent. The variable
   * two_w contains input exponent in bits 27-31, therefore if its smaller than
   * 2**27, the input is either a denormal number, or zero.
   * - Combine the result of conversion of exponent and mantissa with the sign
   * of the input number.
   */
  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                          : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 16-bit floating-point number in IEEE half-precision format, in bit
 * representation.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
static uint16_t fp16_ieee_from_fp32_value(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) ||              \
    defined(__GNUC__) && !defined(__STRICT_ANSI__)
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
#else
  const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
  const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (sign >> 16) |
         (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

/*
 * Convert a 16-bit floating-point number in ARM alternative half-precision
 * format, in bit representation, to a 32-bit floating-point number in IEEE
 * single-precision format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
static inline uint32_t fp16_alt_to_fp32_bits(uint16_t h) {
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t)h << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the bits 0-30
   * of the 32-bit word:
   *
   *      +---+-----+------------+-------------------+
   *      | 0 |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  30  27-31     17-26            0-16
   */
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  /*
   * Renorm shift is the number of bits to shift mantissa left to make the
   * half-precision number normalized. If the initial number is normalized, some
   * of its high 6 bits (sign == 0 and 5-bit exponent) equals one. In this case
   * renorm_shift == 0. If the number is denormalize, renorm_shift > 0. Note
   * that if we shift denormalized nonsign by renorm_shift, the unit bit of
   * mantissa will shift into exponent, turning the biased exponent into 1, and
   * making mantissa normalized (i.e. without leading 1).
   */
#ifdef _MSC_VER
  unsigned long nonsign_bsr;
  _BitScanReverse(&nonsign_bsr, (unsigned long)nonsign);
  uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31;
#else
  uint32_t renorm_shift = __builtin_clz(nonsign);
#endif
  renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
  /*
   * Iff nonsign is 0, it overflows into 0xFFFFFFFF, turning bit 31 into 1.
   * Otherwise, bit 31 remains 0. The signed shift right by 31 broadcasts bit 31
   * into all bits of the zero_mask. Thus zero_mask == 0xFFFFFFFF if the
   * half-precision number was zero (+0.0h or -0.0h) 0x00000000 otherwise
   */
  const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
  /*
   * 1. Shift nonsign left by renorm_shift to normalize it (if the input was
   * denormal)
   * 2. Shift nonsign right by 3 so the exponent (5 bits originally) becomes an
   * 8-bit field and 10-bit mantissa shifts into the 10 high bits of the 23-bit
   * mantissa of IEEE single-precision number.
   * 3. Add 0x70 to the exponent (starting at bit 23) to compensate the
   * different in exponent bias (0x7F for single-precision number less 0xF for
   * half-precision number).
   * 4. Subtract renorm_shift from the exponent (starting at bit 23) to account
   * for renormalization. As renorm_shift is less than 0x70, this can be
   * combined with step 3.
   * 5. Binary ANDNOT with zero_mask to turn the mantissa and exponent into zero
   * if the input was zero.
   * 6. Combine with the sign of the input number.
   */
  return sign |
         (((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) &
          ~zero_mask);
}

/*
 * Convert a 16-bit floating-point number in ARM alternative half-precision
 * format, in bit representation, to a 32-bit floating-point number in IEEE
 * single-precision format.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
static inline float fp16_alt_to_fp32_value(uint16_t h) {
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t)h << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the high bits
   * of the 32-bit word:
   *
   *      +-----+------------+---------------------+
   *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
   *      +-----+------------+---------------------+
   * Bits  27-31    17-26            0-16
   */
  const uint32_t two_w = w + w;

  /*
   * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become
   * mantissa and exponent of a single-precision floating-point number:
   *
   *       S|Exponent |          Mantissa
   *      +-+---+-----+------------+----------------+
   *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
   *      +-+---+-----+------------+----------------+
   * Bits   | 23-31   |           0-22
   *
   * Next, the exponent is adjusted for the difference in exponent bias between
   * single-precision and half-precision formats (0x7F - 0xF = 0x70). This
   * operation never overflows or generates non-finite values, as the largest
   * half-precision exponent is 0x1F and after the adjustment is can not exceed
   * 0x8F < 0xFE (largest single-precision exponent for non-finite values).
   *
   * Note that this operation does not handle denormal inputs (where biased
   * exponent == 0). However, they also do not operate on denormal inputs, and
   * do not produce denormal results.
   */
  const uint32_t exp_offset = UINT32_C(0x70) << 23;
  const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset);

  /*
   * Convert denormalized half-precision inputs into single-precision results
   * (always normalized). Zero inputs are also handled here.
   *
   * In a denormalized number the biased exponent is zero, and mantissa has
   * on-zero bits. First, we shift mantissa into bits 0-9 of the 32-bit word.
   *
   *                  zeros           |  mantissa
   *      +---------------------------+------------+
   *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
   *      +---------------------------+------------+
   * Bits             10-31                0-9
   *
   * Now, remember that denormalized half-precision numbers are represented as:
   *    FP16 = mantissa * 2**(-24).
   * The trick is to construct a normalized single-precision number with the
   * same mantissa and thehalf-precision input and with an exponent which would
   * scale the corresponding mantissa bits to 2**(-24). A normalized
   * single-precision floating-point number is represented as: FP32 = (1 +
   * mantissa * 2**(-23)) * 2**(exponent - 127) Therefore, when the biased
   * exponent is 126, a unit change in the mantissa of the input denormalized
   * half-precision number causes a change of the constructud single-precision
   * number by 2**(-24), i.e. the same ammount.
   *
   * The last step is to adjust the bias of the constructed single-precision
   * number. When the input half-precision number is zero, the constructed
   * single-precision number has the value of FP32 = 1 * 2**(126 - 127) =
   * 2**(-1) = 0.5 Therefore, we need to subtract 0.5 from the constructed
   * single-precision number to get the numerical equivalent of the input
   * half-precision number.
   */
  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
      fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  /*
   * - Choose either results of conversion of input as a normalized number, or
   * as a denormalized number, depending on the input exponent. The variable
   * two_w contains input exponent in bits 27-31, therefore if its smaller than
   * 2**27, the input is either a denormal number, or zero.
   * - Combine the result of conversion of exponent and mantissa with the sign
   * of the input number.
   */
  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                          : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 16-bit floating-point number in ARM alternative half-precision format, in bit
 * representation.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
static inline uint16_t fp16_alt_from_fp32_value(float f) {
  const uint32_t w = fp32_to_bits(f);
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t shl1_w = w + w;

  const uint32_t shl1_max_fp16_fp32 = UINT32_C(0x8FFFC000);
  const uint32_t shl1_base =
      shl1_w > shl1_max_fp16_fp32 ? shl1_max_fp16_fp32 : shl1_w;
  uint32_t shl1_bias = shl1_base & UINT32_C(0xFF000000);
  const uint32_t exp_difference = 23 - 10;
  const uint32_t shl1_bias_min = (127 - 1 - exp_difference) << 24;
  if (shl1_bias < shl1_bias_min) {
    shl1_bias = shl1_bias_min;
  }

  const float bias =
      fp32_from_bits((shl1_bias >> 1) + ((exp_difference + 2) << 23));
  const float base = fp32_from_bits((shl1_base >> 1) + (2 << 23)) + bias;

  const uint32_t exp_f = fp32_to_bits(base) >> 13;
  return (sign >> 16) | ((exp_f & UINT32_C(0x00007C00)) +
                         (fp32_to_bits(base) & UINT32_C(0x00000FFF)));
}

/// Cast fp32 data to bf16 data
/// The round mode is the same with default CPU standard
/// Default round mode: round to nearest with tie to even
/// The round mode can be change through fesetround()
static inline bf16 fp32_to_bf16(fp32 &single) {
  bf16 res;
  if (single.format.exp == 255) {
    if (single.format.frac != 0) {
      // NAN which had been checked with IC
      res.bits = 0x7fff;
    } else {
      // INF
      res.bits = (uint16_t)(single.bits >> 16);
    }
  } else if (single.format.exp == 0) {
    // zero
    res.bits = 0x0;
    res.format.sign = single.format.sign;
  } else {
    const uint16_t sign_exp = (single.bits & UINT32_C(0xFF800000)) >> 16;
    const uint32_t mantissa = single.bits & UINT32_C(0x7FFFFF);
    // Use CPU FP32 add to do mantissa >> 16 and rounding
    float base = fp32_from_bits(UINT32_C(0x48000000));
    base = fp32_from_bits(UINT32_C(0x40000000) | mantissa) + base;
    // Get new mantissa
    uint16_t bf16_mantissa = fp32_to_bits(base) & UINT32_C(0x1FF);
    bf16_mantissa = bf16_mantissa - UINT16_C(0x80);
    // Get bf16 bits
    res.bits = sign_exp + bf16_mantissa;
  }
  return res;
}

static inline bf16 fp32_to_bf16_denorm(fp32 src, RoundingMode round_mode) {
  bf16 dst;
  if (((src.format.frac >> 16) & 0x7f) == 0x7f) {
    if ((src.format.frac & 0xffff) == 0x0) { // 0x007f0000
      dst.bits = 0x0;
      dst.format.sign = src.format.sign;
    } else if ((src.format.frac & 0x8000) != 0x8000) { // 0x007f0001-0x007f7fff
      if ((round_mode == tpu_mlir::ROUNDING_HALF_TO_EVEN) ||
          (round_mode == tpu_mlir::ROUNDING_HALF_AWAY_FROM_ZERO) ||
          (round_mode == tpu_mlir::ROUNDING_TOWARDS_ZERO)) {
        dst.bits = 0x0;
      } else if (round_mode == tpu_mlir::ROUNDING_DOWN) {
        if (src.format.sign == 0) {
          dst.bits = 0x0;
        } else {
          dst.bits = 0x80;
        }
      } else { // ROUND_UP
        if (src.format.sign == 0) {
          dst.bits = 0x80;
        } else {
          dst.bits = 0x0;
        }
      }
      dst.format.sign = src.format.sign;
    } else { // 0x007f8000 - 0x007fffff
      if ((round_mode == tpu_mlir::ROUNDING_HALF_TO_EVEN) ||
          (round_mode == tpu_mlir::ROUNDING_HALF_AWAY_FROM_ZERO)) {
        dst.bits = 0x80;
      } else if (round_mode == tpu_mlir::ROUNDING_TOWARDS_ZERO) {
        dst.bits = 0x0;
      } else if (round_mode == tpu_mlir::ROUNDING_DOWN) {
        if (src.format.sign == 0) {
          dst.bits = 0x0;
        } else {
          dst.bits = 0x80;
        }
      } else { // ROUND_UP
        if (src.format.sign == 0) {
          dst.bits = 0x80;
        } else {
          dst.bits = 0x0;
        }
      }
      dst.format.sign = src.format.sign;
    }
  } else {
    dst.bits = 0x0;
    dst.format.sign = src.format.sign;
  }
  return dst;
}

static inline bf16 fp32_to_bf16_all(fp32 &src, RoundingMode round_mode) {
  bf16 dst;
  fp32 fp32val;
  long long temp_r, temp_l;
  if (src.format.exp > 0 && src.format.exp < 255) {
    uint32_t mant = src.bits & 0xFFFF;
    if (round_mode == tpu_mlir::ROUNDING_DOWN) {
      if (src.format.sign == 0) {
        temp_r = (src.bits >> 16);
      } else {
        temp_r = ((src.bits >> 16) + (mant != 0));
      }
    } else if (round_mode == tpu_mlir::ROUNDING_UP) {
      if (src.format.sign == 0) {
        temp_r = ((src.bits >> 16) + (mant != 0));
      } else {
        temp_r = (src.bits >> 16);
      }
    } else {
      temp_r = RightShiftRound<long long>((long long)src.bits, 16, round_mode);
    }
    temp_l = temp_r << 16;
    fp32val.bits = temp_l & 0xFFFFFFFF;
    dst = fp32_to_bf16(fp32val);
  } else if (src.format.exp == 0xff && src.format.frac != 0) {
    dst.bits = 0x7fff;
  } else if (src.format.sign == 0 && src.format.exp == 0xff &&
             src.format.frac == 0) {
    dst.bits = 0x7f80;
  } else if (src.format.sign == 1 && src.format.exp == 0xff &&
             src.format.frac == 0) {
    dst.bits = 0xff80;
  } else if (src.format.sign == 0 && src.format.exp == 0 &&
             src.format.frac == 0) {
    dst.bits = 0x0000;
  } else if (src.format.sign == 1 && src.format.exp == 0 &&
             src.format.frac == 0) {
    dst.bits = 0x8000;
  } else {
    // Denorm fp32, use fp32_to_bf16_denorm
    dst = fp32_to_bf16_denorm(src, round_mode);
  }
  return dst;
}

/// Cast fp32 data to fp16 data
/// The round mode is the same with default CPU standard
/// Default round mode: round to nearest with tie to even
/// The round mode can be change through fesetround()
static inline fp16 fp32_to_fp16(fp32 single) {
  fp16 res;
  if (single.format.exp == 255 && single.format.frac != 0) {
    // NAN which had been checked with IC
    res.bits = UINT16_C(0x7FFF);
    return res;
  }
  res.bits = fp16_ieee_from_fp32_value(single.fval);
  return res;
}

static inline fp16 fp32_to_fp16_all(fp32 &src, RoundingMode round_mode) {
  fp16 dst;
  fp32 fp32val;
  long long temp_r, temp_l;
  if (src.format.exp > 112 && src.format.exp < 255) {
    uint32_t mant = src.bits & 0x1FFF;
    if (round_mode == tpu_mlir::ROUNDING_DOWN) {
      if (src.format.sign == 0) {
        temp_r = (src.bits >> 13);
      } else {
        temp_r = ((src.bits >> 13) + (mant != 0));
      }
    } else if (round_mode == tpu_mlir::ROUNDING_UP) {
      if (src.format.sign == 0) {
        temp_r = ((src.bits >> 13) + (mant != 0));
      } else {
        temp_r = (src.bits >> 13);
      }
    } else {
      temp_r = RightShiftRound<long long>(src.bits, 13, round_mode);
    }
    temp_l = temp_r << 13;
    fp32val.bits = temp_l & 0xFFFFFFFF;
    dst = fp32_to_fp16(fp32val);
  } else if (src.format.exp > 0 && src.format.exp <= 112) {
    int mant = (src.bits & 0x7FFFFF) + (1 << 23);
    mant = src.format.sign ? (0 - mant) : mant;
    int rshift_num = (113 - src.format.exp) + 13;
    mant = RightShiftRound<long long>(mant, rshift_num, round_mode);
    mant = src.format.sign ? (0 - mant) : mant;
    dst.bits = (mant & 0xFFFF);
    dst.format.sign = src.format.sign;
  } else if (src.format.exp == 0xff && src.format.frac != 0) {
    dst.bits = 0x7fff;
  } else if (src.format.sign == 0 && src.format.exp == 0xff &&
             src.format.frac == 0) {
    dst.bits = 0x7c00;
  } else if (src.format.sign == 1 && src.format.exp == 0xff &&
             src.format.frac == 0) {
    dst.bits = 0xfc00;
  } else if (src.format.sign == 0 && src.format.exp == 0 &&
             src.format.frac == 0) {
    dst.bits = 0x0000;
  } else if (src.format.sign == 1 && src.format.exp == 0 &&
             src.format.frac == 0) {
    dst.bits = 0x8000;
  } else {
    // Denorm fp32, use fp32_to_fp16 directly
    dst = fp32_to_fp16(src);
  }
  return dst;
}

static uint16_t bm_f32_to_bf16(float src) {
  fp32 tmp = {.fval = src};
  bf16 ret = fp32_to_bf16_all(tmp, tpu_mlir::ROUNDING_HALF_TO_EVEN);
  return ret.bits;
}

/*
for cv18xx
*/
static uint16_t cvi_f32_to_bf16(float src, bool is_tpu) {
  // To convert a float 32 to bfloat16, a float 32 can be viewed as 32 bits
  // with the following tags:
  //
  // Sign |  Exp (8 bits) | Frac (23 bits)
  //  S     EEEEEEEE         FFFFFFLRTTTTTTTTTTTTTTT
  //
  //  S: Sign bit.
  //  E: Exponent bits.
  //  F: First 6 bits of fraction.
  //  L: Least significant bit of resulting bfloat16 if we truncate away the
  //  rest of the float32. This is also the 7th bit of fraction
  //  R: Rounding bit, 8th bit of fraction.
  //  T: Sticky bits, rest of fraction, 15 bits.

  // At this point, src must be either a normal float, or +/-infinity or
  // zero.
  uint16_t u16_val;
  if (is_tpu) {
    // Fast rounding algorithm that rounds a half value to nearest even. This
    // reduces expected error when we convert a large number of floats.
    //
    // The fast converting algorithm simply adds lsb (L) to 0x7fff (15 bits of
    // 1s) as the rounding bias, adds the rounding bias to the input, then
    // truncates the last 16 bits away.
    uint32_t u32_val = *((uint32_t *)(&src));
    uint32_t lsb = (u32_val >> 16) & 1;
    u32_val += (0x7fff + lsb);
    u16_val = ((uint16_t *)(&u32_val))[1];
    /* HW behavior */
    // infinity set to max finite positive value
    u16_val = ((u16_val & 0x7f80) == 0x7f80) ? 0x7f7f : u16_val;
  } else {
    u16_val = ((uint16_t *)(&src))[1];
  }
  return u16_val;
}

uint16_t f32_to_f16(float src) {
  fp32 tmp = {.fval = src};
  fp16 ret = fp32_to_fp16_all(tmp, tpu_mlir::ROUNDING_HALF_TO_EVEN);
  return ret.bits;
}

uint16_t f32_to_bf16(float src, bool is_tpu) {
  if (module::isCV18xx()) {
    return cvi_f32_to_bf16(src, is_tpu);
  }
  return bm_f32_to_bf16(src);
}

float f16_to_f32(uint16_t src) {
  fp16 half = {.bits = src};
  if (half.format.exp == 31 && half.format.frac != 0) {
    fp32 res = {0};
    // NAN which had beed checked with IC
    res.bits = UINT32_C(0xFFC00000);
    return res.fval;
  }

  return fp16_ieee_to_fp32_value(src);
}

float bf16_to_f32(uint16_t src) {
  unsigned int tmp = src;
  tmp = tmp << 16;
  return *((float *)&tmp);
}

void BF16(float *p_src, float *p_dst, int num, bool is_tpu) {
#pragma omp parallel for schedule(static, omp_schedule(num))
  for (int i = 0; i < num; i++) {
    p_dst[i] = BF16(p_src[i], is_tpu);
  }
}

void F16(float *p_src, float *p_dst, int num) {
#pragma omp parallel for schedule(static, omp_schedule(num))
  for (int i = 0; i < num; i++) {
    p_dst[i] = F16(p_src[i]);
  }
}

float F16(float src) {
  uint16_t tmp = f32_to_f16(src);
  return f16_to_f32(tmp);
}

float F16(float src, bool half_away_from_zero) {
  fp32 tmp = {.fval = src};
  if (half_away_from_zero) {
    fp16 tmp16 = fp32_to_fp16_all(tmp, tpu_mlir::ROUNDING_HALF_AWAY_FROM_ZERO);
    return f16_to_f32(tmp16.bits);
  } else {
    fp16 tmp16 = fp32_to_fp16_all(tmp, tpu_mlir::ROUNDING_HALF_TO_EVEN);
    return f16_to_f32(tmp16.bits);
  }
}

float BF16(float src, bool is_tpu) {
  auto u16_val = f32_to_bf16(src, is_tpu);
  return bf16_to_f32(u16_val);
}

#define BF16_POSITIVE_MAX_VAL 0x7F7F
#define BF16_NEGATIVE_MAX_VAL 0xFF7F
#define BF16_POSITIVE_INF_EXP 0x7F80
#define BF16_NEGATIVE_INF_EXP 0xFF80

#define FP32_POSITIVE_MAX_VAL 0x7F7FFFFF // FLT_MAX (fp32)
#define FP32_NEGATIVE_MAX_VAL 0xFF7FFFFF
#define FP32_POSITIVE_INF_EXP 0x7F800000
#define FP32_NEGATIVE_INF_EXP 0xFF800000

union convert_type_float {
  float fval;
  uint16_t bf16[2];
  uint32_t ival;
};

typedef union convert_type_float convert_int_float;

/* convert float to hex directly */
static inline uint32_t convert_fp32_hex(float val) {
  convert_int_float convert_val;
  convert_val.fval = val;
  return convert_val.ival;
}

static inline int bf16_unnormal_value_fp32(float *fval, int trans_pos) {
  int unnormal = 0;
  if ((convert_fp32_hex(*fval) & FP32_POSITIVE_INF_EXP) == 0) {
    *fval = 0;
  } else if ((convert_fp32_hex(*fval) & FP32_POSITIVE_INF_EXP) ==
             FP32_POSITIVE_INF_EXP) {
    *fval = FLT_MAX; // FP32_NEGATIVE_MAX_VAL;
    unnormal = 1;
  } else if ((convert_fp32_hex(*fval) & FP32_NEGATIVE_INF_EXP) ==
             FP32_NEGATIVE_INF_EXP) {
    /* HW keeps -FLT_MAX to FLT_MAX*/
    *fval = trans_pos ? FLT_MAX : -FLT_MAX; // FP32_NEGATIVE_MAX_VAL;
    unnormal = 1;
  } else if (*fval == FLT_MAX || *fval == -FLT_MAX) {
    /* HW keeps -FLT_MAX to FLT_MAX*/
    if (trans_pos)
      *fval = FLT_MAX;
    unnormal = 1;
  }
  return unnormal;
}

static inline void bf16_cal_add(float *res, float a, float b, int *overflow) {
  float tmp0 = a;
  float tmp1 = b;
  *overflow |= bf16_unnormal_value_fp32(&tmp0, 0);
  *overflow |= bf16_unnormal_value_fp32(&tmp1, 0);
  if (*overflow) {
    *res = ((convert_fp32_hex(a) >> 31) ^ (convert_fp32_hex(b) >> 31))
               ? FLT_MAX
               : -FLT_MAX;
  } else
    *res = tmp0 + tmp1;
}

float bf16_add(float lhs, float rhs) {
  int overflow = 0;
  float tmp = 0.0f;
  bf16_cal_add(&tmp, lhs, rhs, &overflow);
  bf16_unnormal_value_fp32(&tmp, 1);
  return BF16(tmp);
}

static inline int check_max_inf_value(float a) {
  if ((convert_fp32_hex(a) & FP32_POSITIVE_INF_EXP) == FP32_POSITIVE_INF_EXP ||
      (convert_fp32_hex(a) & FP32_NEGATIVE_INF_EXP) == FP32_NEGATIVE_INF_EXP ||
      (convert_fp32_hex(a) & FP32_POSITIVE_MAX_VAL) == FP32_POSITIVE_MAX_VAL ||
      (convert_fp32_hex(a) & FP32_NEGATIVE_MAX_VAL) == FP32_NEGATIVE_MAX_VAL) {
    return 1;
  } else
    return 0;
}

static inline void bf16_cal_mac(float *res, float a, float b, int *overflow) {
  int inf_a = check_max_inf_value(a);
  int inf_b = check_max_inf_value(b);
  float mac;
  if (!inf_a && ((convert_fp32_hex(a) & FP32_POSITIVE_INF_EXP) == 0))
    a = 0;
  if (!inf_b && ((convert_fp32_hex(b) & FP32_POSITIVE_INF_EXP) == 0))
    b = 0;
  if (inf_a || inf_b) {
    *res = ((convert_fp32_hex(a) >> 31) ^ (convert_fp32_hex(b) >> 31))
               ? FLT_MAX
               : -FLT_MAX;
    *overflow = 1;
  } else {
    mac = a * b;
    if ((convert_fp32_hex(mac) & FP32_POSITIVE_INF_EXP) == 0)
      mac = 0;
    else if (check_max_inf_value(mac) ||
             ((convert_fp32_hex(mac) & FP32_POSITIVE_MAX_VAL) ==
              FP32_POSITIVE_MAX_VAL)) {
      *res = FLT_MAX; // convert_hex_fp32(FP32_POSITIVE_MAX_VAL);
      *overflow = 1;
    } else if (check_max_inf_value(mac) ||
               ((convert_fp32_hex(mac) & FP32_NEGATIVE_MAX_VAL) ==
                FP32_NEGATIVE_MAX_VAL)) {
      *res = -FLT_MAX; // convert_hex_fp32(FP32_NEGATIVE_MAX_VAL);
      *overflow = 1;
    }
  }
  if (*overflow == 0) {
    *res += mac;
    if ((convert_fp32_hex(*res) & FP32_POSITIVE_INF_EXP) == 0)
      *res = 0;
    else if (check_max_inf_value(*res) ||
             ((convert_fp32_hex(*res) & FP32_POSITIVE_MAX_VAL) ==
              FP32_POSITIVE_MAX_VAL)) {
      *res = FLT_MAX; // convert_hex_fp32(FP32_POSITIVE_MAX_VAL);
      *overflow = 1;
    } else if (check_max_inf_value(*res) ||
               ((convert_fp32_hex(*res) & FP32_NEGATIVE_MAX_VAL) ==
                FP32_NEGATIVE_MAX_VAL)) {
      *res = -FLT_MAX; // convert_hex_fp32(FP32_NEGATIVE_MAX_VAL);
      *overflow = 1;
    }
  }
}

float bf16_mul(float lhs, float rhs) {
  int overflow = 0;
  float tmp = 0.0f;
  bf16_cal_mac(&tmp, lhs, rhs, &overflow);

  bf16_unnormal_value_fp32(&tmp, 1);

  return BF16(tmp);
}
} // namespace tpu_mlir
