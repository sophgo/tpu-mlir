//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <inttypes.h>
#include <type_traits>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#define MULTI_CORE template <int CoreNum = 1>
#define GET_CoreNum() CoreNum
#define __KERNEL__ __attribute__((deprecated("kernel")))
#define __TEST__ __attribute__((deprecated("test")))
#define __HOST__ __attribute__((deprecated("host")))
#define __AUTOTUNE__ __attribute__((deprecated("autotune")))
#define IS_UNSIGNED(dtype)                                                     \
  (std::is_same_v<dtype, uint8> || std::is_same_v<dtype, uint8 *> ||           \
   std::is_same_v<dtype, uint32> || std::is_same_v<dtype, uint32 *> ||         \
   std::is_same_v<dtype, uint4> || std::is_same_v<dtype, uint4 *> ||           \
   std::is_same_v<dtype, uint16> || std::is_same_v<dtype, uint16 *>)

#define DEFAULT_SDMA_PORT -1
#define None (int)0

using bf16 = __bf16;
using fp16 = __fp16;
using fp32 = float;
using int32 = int;
using uint32 = unsigned int;
using int8 = int8_t;
using uint8 = uint8_t;
using int16 = int16_t;
using uint16 = uint16_t;
using int64 = int64_t;
using uint64 = uint64_t;

using uint = uint32;
using BF16 = bf16;
using FP16 = fp16;
using FP32 = fp32;
using INT8 = int8;
using UINT8 = uint8;

struct fp8e5m2 {
  char data;
};
struct fp8e4m3 {
  char data;
};
struct int4 {};
struct uint4 {};
struct fp20 {
  int data : 20;
};
