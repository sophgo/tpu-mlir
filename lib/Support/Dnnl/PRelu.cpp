//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/PRelu.h"
using namespace dnnl;

namespace tpu_mlir {
PRelu::PRelu() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  eng_stream = dnnl::stream(eng);
}

void PRelu::setup(/*float *input, float *output, prelu_attr_t &attr*/) {
  // auto src_md = memory::desc(src_shape, memory::data_type::f32,
  // memory::format_tag::nchw); auto weights_md = memory::desc(weights_shape,
  // memory::data_type::f32, memory::format_tag::nchw);
  auto prelu_pd = prelu_forward::primitive_desc(
      eng, prop_kind::forward_inference, src_mem.get_desc(),
      weights_mem.get_desc(), dst_mem.get_desc());
  prelu_prim = prelu_forward(prelu_pd);
}
void PRelu::run() {
  prelu_prim.execute(eng_stream, {{DNNL_ARG_SRC, src_mem},
                                  {DNNL_ARG_WEIGHTS, weights_mem},
                                  {DNNL_ARG_DST, dst_mem}});
  eng_stream.wait();
}
/*
Binary::Binary() {
eng = dnnl::engine(engine::kind::cpu, 0);
engine_stream = dnnl::stream(eng);
}

void Binary::setup() {
// memory description with primitive description
auto op_desc = binary::desc(algorithm_, lhs_mem.get_desc(),
                            rhs_mem.get_desc(), dst_mem.get_desc());
// define a primitive
using pd_t = binary::primitive_desc;

auto pd = pd_t();
if (do_relu_) {
  dnnl::post_ops ops_eltwise;
  // https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html
  if (relu_limit_ > 0)
    ops_eltwise.append_eltwise(1.0f, dnnl::algorithm::eltwise_clip, 0.f,
                               relu_limit_);
  else
    ops_eltwise.append_eltwise(1.0f, dnnl::algorithm::eltwise_relu, 0.f, 0.f);
  dnnl::primitive_attr attr_po_eltwise;
  attr_po_eltwise.set_post_ops(ops_eltwise);
  pd = pd_t(op_desc, attr_po_eltwise, eng);
} else {
  pd = pd_t(op_desc, eng);
}
binary_prim = binary(pd);
}

void Binary::run() {
binary_prim.execute(engine_stream, {{DNNL_ARG_SRC_0, lhs_mem},
                                    {DNNL_ARG_SRC_1, rhs_mem},
                                    {DNNL_ARG_DST, dst_mem}});
engine_stream.wait();
}
*/
} // namespace tpu_mlir
