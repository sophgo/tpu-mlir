
//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Concat.h"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace tpu_mlir {

Concat::Concat() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  eng_stream = dnnl::stream(eng);
}

void Concat::setup(std::vector<float *> inputs, float *output,
                   concat_attr_t &attr) {
  this->attr_ = std::move(attr);
  p_inputs = inputs;
  p_output = output;

  std::vector<memory::desc> src_mds;
  std::vector<memory> src_mems;

  for (uint64_t i = 0; i < attr_.num_src; i++) {
    auto src_shape = attr_.src_shapes[i];
    auto src_md = memory::desc(src_shape, dt::f32, tag::ncw);
    auto src_mem = memory(src_md, eng, p_inputs[i]);

    src_mds.push_back(src_md);
    src_mems.push_back(src_mem);
  }

  auto concat_pd = concat::primitive_desc(eng, attr_.axis, src_mds);
  auto dst_mem = memory(concat_pd.dst_desc(), eng, p_output);
  concat_prim = concat(concat_pd);

  for (int n = 0; n < attr_.num_src; ++n)
    concat_args.insert({DNNL_ARG_MULTIPLE_SRC + n, src_mems[n]});
  concat_args.insert({DNNL_ARG_DST, dst_mem});

  // // Primitive execution: concatenation.
  // concat_prim.execute(eng_stream, concat_args);

  // // Wait for the computation to finalize.
  // eng_stream.wait();
}

void Concat::run() {
  concat_prim.execute(eng_stream, concat_args);
  eng_stream.wait();
}

} // namespace tpu_mlir
