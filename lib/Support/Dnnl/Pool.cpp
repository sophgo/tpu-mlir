//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "sophgo/Support/Dnnl/Pool.h"
#include "sophgo/Support/MathUtils.h"

using namespace dnnl;
using namespace sophgo;

Pooling::Pooling() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  eng_stream = dnnl::stream(eng);
  _pt = _pb = _pr = _pl = 0;
  _izp = 0;
}

Pooling::~Pooling() {}

void Pooling::pad_init(float *input, int n, int ic, int ih, int iw, int& pt, int& pb, int& pl, int& pr, int izp) {
  origin_input = input;
  _pt = pt;
  _pb = pb;
  _pr = pr;
  _pl = pl;
  _izp = izp;
  _n = n;
  _c = ic;
  _h = ih;
  _w = iw;
  if (izp && (_pt > 0 || _pb > 0 || _pr > 0 || _pl > 0)) {
    int input_paded_size = n * ic * (ih + pt + pb) * (iw + pr + pl);
    input_after_pad = std::make_shared<std::vector<float>>(input_paded_size);
    src_shape = {n, ic, ih + pt + pb, iw + pr + pl};
    pt = pb = pr = pl = 0;
    p_input = input_after_pad->data();
  } else {
    src_shape = {n, ic, ih, iw};
    p_input = input;
  }
}




void Pooling::setup(float *input, float *output, int n, int c, int ih, int iw,
                    int oh, int ow, int kh, int kw, int sh, int sw, int pt,
                    int pb, int pl, int pr, bool is_avg, bool count_include_pad,
                    int izp, int pad_value, memory::data_type dt) {
  pad_init(input, n, c, ih, iw, pt, pb, pl, pr, izp);
  memory::dims dst_shape = {n, c, oh, ow};
  memory::dims strides = {sh, sw};
  memory::dims kernel = {kh, kw};
  memory::dims padding_tl = {pt, pl};
  memory::dims padding_br = {pb, pr};
  auto src_md = memory::desc({src_shape}, dt,
                             memory::format_tag::nchw);
  auto dst_md = memory::desc({dst_shape}, dt,
                             memory::format_tag::nchw);
  auto pool_avg_algo = count_include_pad
                           ? algorithm::pooling_avg_include_padding
                           : algorithm::pooling_avg_exclude_padding;
  // pool desc
  auto pool_desc = pooling_forward::desc(
      prop_kind::forward_inference,
      is_avg ? pool_avg_algo : algorithm::pooling_max, src_md, dst_md, strides,
      kernel, padding_tl, padding_br);

  prim_desc = pooling_forward::primitive_desc(pool_desc, eng);
  memory src_memory =
      memory({{src_shape}, memory::data_type::f32, memory::format_tag::nchw},
             eng, p_input);
  memory dst_memory =
      memory({{dst_shape}, memory::data_type::f32, memory::format_tag::nchw},
             eng, output);
  net.clear();
  net_args.clear();
  auto prim_src_memory = src_memory;
  if (prim_desc.src_desc() != src_memory.get_desc()) {
    prim_src_memory = memory(prim_desc.src_desc(), eng);
    net.push_back(reorder(src_memory, prim_src_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, src_memory}, {DNNL_ARG_TO, prim_src_memory}});
  }
  auto prim_dst_memory = memory(prim_desc.dst_desc(), eng);
  net.push_back(pooling_forward(prim_desc));
  net_args.push_back(
      {{DNNL_ARG_SRC, prim_src_memory}, {DNNL_ARG_DST, prim_dst_memory}});
  if (prim_dst_memory != dst_memory) {
    net.push_back(reorder(prim_dst_memory, dst_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, prim_dst_memory}, {DNNL_ARG_TO, dst_memory}});
  }
}

void Pooling::run() {
  if (input_after_pad) {
    pad_tensor(input_after_pad->data(), origin_input, _n, _c, _h, _w, _pt, _pb, _pl, _pr, _izp);
  }
  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(eng_stream, net_args.at(i));
  eng_stream.wait();
}
