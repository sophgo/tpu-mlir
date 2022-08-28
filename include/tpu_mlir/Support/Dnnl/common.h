//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "oneapi/dnnl/dnnl.hpp"

using memory = dnnl::memory;

template <typename data_t>
struct data_traits {};

template <>
struct data_traits<float> {
  static const auto data_type = memory::data_type::f32;

  using uint_type = uint32_t;
};
template <>
struct data_traits<uint8_t> {
  static const auto data_type = memory::data_type::u8;

  using uint_type = uint8_t;
};
template <>
struct data_traits<int8_t> {
  static const auto data_type = memory::data_type::s8;

  using uint_type = uint8_t;
};
template <>
struct data_traits<int32_t> {
  static const auto data_type = memory::data_type::s32;

  using uint_type = uint32_t;
};

inline memory::format_tag get_tag(const ::memory::dims &x) {
  switch (x.size()) {
  case 1:
    return memory::format_tag::a;
  case 2:
    return memory::format_tag::ab;
  case 3:
    return memory::format_tag::abc;
  case 4:
    return memory::format_tag::abcd;
  case 5:
    return memory::format_tag::abcde;
  case 6:
    return memory::format_tag::abcdef;
  default:
    dnnl::error::wrap_c_api(
        dnnl_invalid_arguments,
        ("unsupported dimension size :" + std::to_string(x.size())).c_str());
    return memory::format_tag();
  }
  return memory::format_tag();
}
