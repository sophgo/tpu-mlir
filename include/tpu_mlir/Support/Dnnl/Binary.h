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
#include "tpu_mlir/Support/Dnnl/common.h"
#include "llvm/ADT/ArrayRef.h"

using namespace dnnl;
namespace tpu_mlir {

class Binary {
  using tag = memory::format_tag;
  using dt = memory::data_type;

public:
  Binary();

  template <typename T>
  inline Binary &lhs(T *lhs, llvm::ArrayRef<int64_t> lhs_shape) {
    auto mds =
        memory::desc(lhs_shape, data_traits<T>::data_type, get_tag(lhs_shape));
    lhs_mem = memory(mds, eng, lhs);
    return *this;
  };

  template <typename T>
  inline Binary &rhs(T *rhs, llvm::ArrayRef<int64_t> rhs_shape) {
    auto mds =
        memory::desc(rhs_shape, data_traits<T>::data_type, get_tag(rhs_shape));
    rhs_mem = memory(mds, eng, rhs);
    return *this;
  };

  template <typename T>
  inline Binary &dst(T *dst, llvm::ArrayRef<int64_t> ret_shape) {
    auto mds =
        memory::desc(ret_shape, data_traits<T>::data_type, get_tag(ret_shape));
    dst_mem = memory(mds, eng, dst);
    return *this;
  };

  inline Binary &do_relu(bool do_relu) {
    do_relu_ = do_relu;
    return *this;
  };

  inline Binary &relu_limit(float relu_limit) {
    relu_limit_ = relu_limit;
    return *this;
  };

  inline Binary &algorithem(algorithm algorithm) {
    algorithm_ = algorithm;
    return *this;
  }

  void setup();
  void run();

private:
  engine eng;
  bool do_relu_ = false;
  float relu_limit_ = -1;
  algorithm algorithm_;
  stream engine_stream;
  primitive binary_prim;
  memory lhs_mem;
  memory rhs_mem;
  memory dst_mem;
};
} // namespace tpu_mlir
