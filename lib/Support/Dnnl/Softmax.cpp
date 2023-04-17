
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Softmax.h"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace tpu_mlir {

Softmax::Softmax() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  eng_stream = dnnl::stream(eng);
}

void Softmax::setup(float *input, float *output, softmax_attr_t &attr) {
  this->attr_ = std::move(attr);
  p_input = input;
  p_output = output;

  auto src_md = memory::desc(attr_.src_shape, dt::f32, tag::ncw);
  auto src_mem = memory(src_md, eng, p_input);

  auto dst_md = memory::desc(attr_.dst_shape, dt::f32, tag::ncw);
  auto dst_mem = memory(dst_md, eng, p_output);
  auto alg = attr.log ? algorithm::softmax_log : algorithm::softmax_accurate;

  auto softmax_pd = softmax_forward::primitive_desc(
      eng, prop_kind::forward_inference, alg, src_md, dst_md, attr_.axis);
  softmax_prim = softmax_forward(softmax_pd);
  softmax_args.insert({DNNL_ARG_SRC, src_mem});
  softmax_args.insert({DNNL_ARG_DST, dst_mem});
}

void Softmax::run() {
  softmax_prim.execute(eng_stream, softmax_args);
  eng_stream.wait();
}

} // namespace tpu_mlir
