//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common.h"
#include "oneapi/dnnl/dnnl.hpp"
#include "llvm/ADT/ArrayRef.h"
using namespace dnnl;

namespace tpu_mlir {

class PRelu {
public:
  PRelu();
  template <typename T>
  inline PRelu &src(T *src, llvm::ArrayRef<int64_t> src_shape) {
    auto mds =
        memory::desc(src_shape, data_traits<T>::data_type, get_tag(src_shape));
    src_mem = memory(mds, eng, src);
    return *this;
  };

  template <typename T>
  inline PRelu &weights(T *weights, llvm::ArrayRef<int64_t> weights_shape) {
    auto mds =
        memory::desc(weights_shape, data_traits<T>::data_type, get_tag(weights_shape));
    weights_mem = memory(mds, eng, weights);
    return *this;
  };

  template <typename T>
  inline PRelu &dst(T *dst, llvm::ArrayRef<int64_t> ret_shape) {
    auto mds =
        memory::desc(ret_shape, data_traits<T>::data_type, get_tag(ret_shape));
    dst_mem = memory(mds, eng, dst);
    return *this;
  };
  void setup();
  void run();
private:
  engine eng;
  stream eng_stream;
  memory::dims src_shape;
  memory::dims dst_shape;
  primitive prelu_prim;
  memory src_mem;
  memory weights_mem;
  memory dst_mem;
};
} // namespace tpu_mlir
