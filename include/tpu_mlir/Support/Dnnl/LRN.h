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

class LRN {
  using tag = memory::format_tag;
  using dt = memory::data_type;

public:
  LRN();

  template <typename T>
  inline LRN &src(T *src, llvm::ArrayRef<int64_t> lhs_shape) {
    auto mds =
        memory::desc(lhs_shape, data_traits<T>::data_type, get_tag(lhs_shape));
    src_mem = memory(mds, eng, src);
    return *this;
  };

  template <typename T>
  inline LRN &dst(T *dst, llvm::ArrayRef<int64_t> ret_shape) {
    auto mds =
        memory::desc(ret_shape, data_traits<T>::data_type, get_tag(ret_shape));
    dst_mem = memory(mds, eng, dst);
    return *this;
  };

  inline LRN &algorithem(algorithm algorithm) {
    algorithm_ = algorithm;
    return *this;
  }
  inline LRN &size(int64_t size) {
    size_ = size;
    return *this;
  }

  inline LRN &param(float alpha, float beta, float bias) {
    alpha_ = alpha;
    beta_ = beta;
    bias_ = bias;
    return *this;
  }

  void setup();
  void run();

private:
  engine eng;
  float alpha_, beta_, bias_;
  algorithm algorithm_;
  int64_t size_;
  stream engine_stream;
  primitive lrn_prim;
  memory src_mem;
  memory dst_mem;
};
} // namespace tpu_mlir
