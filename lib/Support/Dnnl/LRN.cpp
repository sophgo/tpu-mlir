//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/LRN.h"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

namespace tpu_mlir {
LRN::LRN() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  engine_stream = dnnl::stream(eng);
}

void LRN::setup() {
  // memory description with primitive description
  auto op_desc =
      lrn_forward::desc(prop_kind::forward_inference, algorithm_,
                        src_mem.get_desc(), size_, alpha_, beta_, bias_);
  // define a primitive
  auto pd = lrn_forward::primitive_desc(op_desc, eng);
  lrn_prim = lrn_forward(pd);
}

void LRN::run() {
  lrn_prim.execute(engine_stream,
                   {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}});
  engine_stream.wait();
}

} // namespace tpu_mlir
