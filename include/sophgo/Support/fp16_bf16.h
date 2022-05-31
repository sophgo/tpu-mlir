//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef FP16_BF16_H_
#define FP16_BF16_H_
#include "fp16_convert.h"
#include <iostream>

//using  std::ostream;

typedef union {
  float    fval;
  uint32_t bits;
  struct {
    uint32_t frac : 23; // mantissa
    uint32_t exp  : 8;  // exponent
    uint32_t sign : 1;  // sign
  } format;
} fp32;

struct fp16{
  union {
    uint16_t bits;
    struct {
      uint16_t frac : 10; // mantissa
      uint16_t exp  : 5;  // exponent
      uint16_t sign : 1;  // sign
    } format;
  }data;

  fp16() {
    data.bits = 0;
  }
  template<typename T>
  fp16(const T&);

  template<typename T>
  operator T() const;

  template<typename T>
  fp16& operator=(const T&);

  template<typename T>
  fp16& operator+=(const T&);

  template<typename T>
  fp16& operator-=(const T&);

  template<typename T>
  fp16& operator*=(const T&);

  template<typename T>
  fp16& operator/=(const T&);

  fp16 operator++ ();
  fp16 operator++ (int);

  friend std::ostream &operator<< (std::ostream &out, const fp16&);
};

struct bf16{
  union {
    uint16_t bits;
    struct {
      uint16_t frac : 7; // mantissa
      uint16_t exp  : 8;  // exponent
      uint16_t sign : 1;  // sign
    } format;
  }data;

  bf16() {
    data.bits = 0;
  }
  template<typename T>
  bf16(const T&);

  template<typename T>
  operator T() const;

  template<typename T>
  bf16& operator=(const T&);

  template<typename T>
  bf16& operator+=(const T&);

  template<typename T>
  bf16& operator-=(const T&);

  template<typename T>
  bf16& operator*=(const T&);

  template<typename T>
  bf16& operator/=(const T&);

  bf16 operator++ ();
  bf16 operator++ (int);

  friend std::ostream &operator<< (std::ostream &out, const bf16&);
};

// Data cast
fp32 fp16_to_fp32(const fp16& half);

fp16 fp32_to_fp16(const fp32& single);

fp32 bf16_to_fp32(const bf16& half);

bf16 fp32_to_bf16(const fp32& single);

// Constructor
template<typename T>
fp16::fp16(const T& val) {
  fp32 val_fp32;
  val_fp32.fval = static_cast<float>(val);
  *this = fp32_to_fp16(val_fp32);
}

template<typename T>
bf16::bf16(const T& val) {
  fp32 val_fp32;
  val_fp32.fval = static_cast<float>(val);
  *this = fp32_to_bf16(val_fp32);
}

// Operator override
template<typename T>
fp16::operator T() const {
    fp32 res_fp32 = fp16_to_fp32(*this);
    return static_cast<T>(res_fp32.fval);
}

template<typename T>
fp16& fp16::operator=(const T& val) {
  fp32 val_fp32;
  val_fp32.fval = static_cast<float>(val);
  *this = fp32_to_fp16(val_fp32);
  return *this;
}

template<typename T>
fp16& fp16::operator+=(const T& val) {
  fp32 val_fp32;
  val_fp32.fval = static_cast<float>(val);
  fp32 self_fp32 = fp16_to_fp32(*this);
  self_fp32.fval += val_fp32.fval;
  *this = fp32_to_fp16(self_fp32);
  return *this;
}

template<typename T>
fp16& fp16::operator-=(const T& val) {
  fp32 val_fp32;
  val_fp32.fval = static_cast<float>(val);
  fp32 self_fp32 = fp16_to_fp32(*this);
  self_fp32.fval -= val_fp32.fval;
  *this = fp32_to_fp16(self_fp32);
  return *this;
}

template<typename T>
fp16& fp16::operator*=(const T& val) {
  fp32 val_fp32;
  val_fp32.fval = static_cast<float>(val);
  fp32 self_fp32 = fp16_to_fp32(*this);
  self_fp32.fval *= val_fp32.fval;
  *this = fp32_to_fp16(self_fp32);
  return *this;
}

template<typename T>
fp16& fp16::operator/=(const T& val) {
  fp32 val_fp32;
  val_fp32.fval = static_cast<float>(val);
  fp32 self_fp32 = fp16_to_fp32(*this);
  self_fp32.fval /= val_fp32.fval;
  *this = fp32_to_fp16(self_fp32);
  return *this;
}

/*
fp16 fp16::operator++ () {
  fp32 self_fp32 = fp16_to_fp32(*this);
  self_fp32.fval += 1;
  *this = fp32_to_fp16(self_fp32);
  return *this;
}

fp16 fp16::operator++ (int dummy) {
  fp16 ret = *this;
  ++(*this);
  return ret;
}
*/

template<typename T>
bf16::operator T() const {
    fp32 res_fp32 = bf16_to_fp32(*this);
    return static_cast<T>(res_fp32.fval);
}

template<typename T>
bf16& bf16::operator=(const T& val) {
  fp32 val_fp32;
  val_fp32.fval = static_cast<float>(val);
  *this = fp32_to_bf16(val_fp32);
  return *this;
}

template<typename T>
bf16& bf16::operator+=(const T& val) {
  fp32 val_fp32;
  val_fp32.fval = static_cast<float>(val);
  fp32 self_fp32 = bf16_to_fp32(*this);
  self_fp32.fval += val_fp32.fval;
  *this = fp32_to_bf16(self_fp32);
  return *this;
}

template<typename T>
bf16& bf16::operator-=(const T& val) {
  fp32 val_fp32;
  val_fp32.fval = static_cast<float>(val);
  fp32 self_fp32 = bf16_to_fp32(*this);
  self_fp32.fval -= val_fp32.fval;
  *this = fp32_to_bf16(self_fp32);
  return *this;
}

template<typename T>
bf16& bf16::operator*=(const T& val) {
  fp32 val_fp32;
  val_fp32.fval = static_cast<float>(val);
  fp32 self_fp32 = bf16_to_fp32(*this);
  self_fp32.fval *= val_fp32.fval;
  *this = fp32_to_bf16(self_fp32);
  return *this;
}

template<typename T>
bf16& bf16::operator/=(const T& val) {
  fp32 val_fp32;
  val_fp32.fval = static_cast<float>(val);
  fp32 self_fp32 = bf16_to_fp32(*this);
  self_fp32.fval /= val_fp32.fval;
  *this = fp32_to_bf16(self_fp32);
  return *this;
}

//  fp16 operator
bool operator== (const fp16& a, const fp16& b);
bool operator!= (const fp16& a, const fp16& b);
bool operator> (const fp16& a, const fp16& b);
bool operator< (const fp16& a, const fp16& b);
bool operator>= (const fp16& a, const fp16& b);
bool operator<= (const fp16& a, const fp16& b);
fp16 operator+(const fp16& a, const fp16& b);
fp16 operator-(const fp16& a, const fp16& b);
fp16 operator*(const fp16& a, const fp16& b);
fp16 operator/(const fp16& a, const fp16& b);

fp16 operator-(const fp16& a);

//  bf16 operator
bool operator== (const bf16& a, const bf16& b);
bool operator!= (const bf16& a, const bf16& b);
bool operator> (const bf16& a, const bf16& b);
bool operator< (const bf16& a, const bf16& b);
bool operator>= (const bf16& a, const bf16& b);
bool operator<= (const bf16& a, const bf16& b);
bf16 operator+(const bf16& a, const bf16& b);
bf16 operator-(const bf16& a, const bf16& b);
bf16 operator*(const bf16& a, const bf16& b);
bf16 operator/(const bf16& a, const bf16& b);

bf16 operator-(const bf16& a);

static uint16_t float_to_fp16_uint16(float x)
{
    fp32 value = {.fval = x};
    return fp32_to_fp16(value).data.bits;
}

static float fp16_uint16_to_float(uint16_t x)
{
    fp16 value = {.bits = x};
    return fp16_to_fp32(value).fval;
}

static uint16_t  float_to_fp16_uint16_nvidia(float m)
{
    unsigned long m2 = *(unsigned long*)(&m);
    // 强制把float转为unsigned long
    // 截取后23位尾数，右移13位，剩余10位；符号位直接右移16位；
    // 指数位麻烦一些，截取指数的8位先右移13位(左边多出3位不管了)
    // 之前是0~255表示-127~128, 调整之后变成0~31表示-15~16
    // 因此要减去127-15=112(在左移10位的位置).
    uint16_t t = ((m2 & 0x007fffff) >> 13) | ((m2 & 0x80000000) >> 16)
        | (((m2 & 0x7f800000) >> 13) - (112 << 10));
    if(m2 & 0x1000)
        t++;  // 四舍五入(尾数被截掉部分的最高位为1, 则尾数剩余部分+1)
    return t ;
}

static float fp16_uint16_to_float_nvidia(uint16_t n)
{
    uint16_t frac = (n & 0x3ff) | 0x400;
    int exp = ((n & 0x7c00) >> 10) - 25;
    float m;
    if(frac == 0 && exp == 0x1f)
        m = INFINITY;
    else if (frac || exp)
        m = frac * pow(2, exp);
    else
        m = 0;
    return (n & 0x8000) ? -m : m;
}

namespace std {

#define DECLARE_F16_ELE_FUN_TO_STD(fun)          \
fp16 fun(const fp16&);                           \
bf16 fun(const bf16&);

DECLARE_F16_ELE_FUN_TO_STD(floor)
DECLARE_F16_ELE_FUN_TO_STD(ceil)
DECLARE_F16_ELE_FUN_TO_STD(log)
DECLARE_F16_ELE_FUN_TO_STD(exp)
DECLARE_F16_ELE_FUN_TO_STD(tanh)
DECLARE_F16_ELE_FUN_TO_STD(fabs)
DECLARE_F16_ELE_FUN_TO_STD(sqrt)
DECLARE_F16_ELE_FUN_TO_STD(sin)
DECLARE_F16_ELE_FUN_TO_STD(cos)
DECLARE_F16_ELE_FUN_TO_STD(abs)

#undef DECLARE_F16_ELE_FUN_TO_STD

#define DECLARE_F16_BINARY_TO_STD(fun)           \
fp16 fun(const fp16&, const fp16&);              \
bf16 fun(const bf16&, const bf16&);

DECLARE_F16_BINARY_TO_STD(min)
DECLARE_F16_BINARY_TO_STD(max)
DECLARE_F16_BINARY_TO_STD(pow)

#undef DECLARE_F16_BINARY_TO_STD

  bool isinf(const fp16&);
  bool isnan(const fp16&);
  bool isfinite(const fp16&);

  bool isinf(const bf16&);
  bool isnan(const bf16&);
  bool isfinite(const bf16&);

}

#endif
