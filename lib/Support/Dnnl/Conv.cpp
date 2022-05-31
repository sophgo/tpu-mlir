//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include <string.h>
#include "sophgo/Support/Dnnl/Conv.h"
#include "sophgo/Support/MathUtils.h"

using namespace dnnl;
using namespace sophgo;
Conv::Conv() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  eng_stream = dnnl::stream(eng);
  _pt = _pb = _pr = _pl = 0;
  _izp = 0;
}

Conv::~Conv() {}

void Conv::pad_init(float *input, int n, int ic, int ih, int iw, int &pt,
                    int &pb, int &pl, int &pr, int izp) {
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

void Conv::setup(float *input, float *weight, float *bias, float *output, int n,
                 int ic, int ih, int iw, int oc, int oh, int ow, int kh, int kw,
                 int sh, int sw, int dh, int dw, int pt, int pb, int pl, int pr,
                 int g, bool do_relu, int izp) {
  // printf("Conv para:%d,%d,%d,%d,%d,%d,%d,%d\n", idt, wdt, bdt, odt,
  // per_channel, izp, ozp, do_relu);
  pad_init(input, n, ic, ih, iw, pt, pb, pl, pr, izp);
  dst_shape = {n, oc, oh, ow};
  memory::dims filter_shape = (g != 1) ? memory::dims{g, oc / g, ic / g, kh, kw}
                                       : memory::dims{oc, ic, kh, kw};
  memory::dims bias_shape = {oc};
  memory::dims strides = {sh, sw};

  memory::dims padding_l = {pt, pl};
  memory::dims padding_r = {pb, pr};
  memory::dims dilation = {dh - 1, dw - 1};

  net.clear();
  net_args.clear();
  auto src_md = memory::desc({src_shape}, memory::data_type::f32,
                             memory::format_tag::any);
  auto filter_md = memory::desc({filter_shape}, memory::data_type::f32,
                                memory::format_tag::any);
  auto bias_md = memory::desc({bias_shape}, memory::data_type::f32,
                              memory::format_tag::any);
  auto dst_md = memory::desc({dst_shape}, memory::data_type::f32,
                             memory::format_tag::any);

  auto conv_desc = convolution_forward::desc(
      prop_kind::forward_inference, algorithm::convolution_direct, src_md,
      filter_md, bias_md, dst_md, strides, dilation, padding_l, padding_r);

  post_ops ops;
  primitive_attr conv_attr;

  if (do_relu) {
    const float ops_scale = 1.f;
    const float ops_alpha = 0.f; // relu negative slope
    const float ops_beta = 0.f;
    ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
    conv_attr.set_post_ops(ops);
  }

  conv_prim_desc =
      convolution_forward::primitive_desc(conv_desc, conv_attr, eng);

  // set mkldnn memory
  auto filter_tag =
      (g != 1) ? memory::format_tag::goihw : memory::format_tag::oihw;
  auto filter_memory =
      memory({{filter_shape}, memory::data_type::f32, filter_tag}, eng, weight);
  prim_filter_memory = filter_memory;
  if (conv_prim_desc.weights_desc() != filter_memory.get_desc()) {
    prim_filter_memory = memory(conv_prim_desc.weights_desc(), eng);
    reorder(filter_memory, prim_filter_memory)
        .execute(eng_stream, filter_memory, prim_filter_memory);
  }

  auto prim_bias_memory = memory();
  if (bias != nullptr) {
    auto bias_memory =
        memory({{bias_shape}, memory::data_type::f32, memory::format_tag::x},
               eng, bias);
    prim_bias_memory = bias_memory;
    if (conv_prim_desc.bias_desc() != bias_memory.get_desc()) {
      prim_bias_memory = memory(conv_prim_desc.bias_desc(), eng);
      net.push_back(reorder(bias_memory, prim_bias_memory));
      net_args.push_back(
          {{DNNL_ARG_FROM, bias_memory}, {DNNL_ARG_TO, prim_bias_memory}});
    }
  }

  auto src_memory =
      memory({{src_shape}, memory::data_type::f32, memory::format_tag::nchw},
             eng, p_input);
  auto prim_src_memory = src_memory;
  if (conv_prim_desc.src_desc() != src_memory.get_desc()) {
    prim_src_memory = memory(conv_prim_desc.src_desc(), eng);
    net.push_back(reorder(src_memory, prim_src_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, src_memory}, {DNNL_ARG_TO, prim_src_memory}});
  }

  auto prim_dst_memory = memory(conv_prim_desc.dst_desc(), eng);
  net.push_back(convolution_forward(conv_prim_desc));
  if (bias != nullptr) {
    net_args.push_back({{DNNL_ARG_SRC, prim_src_memory},
                        {DNNL_ARG_WEIGHTS, prim_filter_memory},
                        {DNNL_ARG_BIAS, prim_bias_memory},
                        {DNNL_ARG_DST, prim_dst_memory}});
  } else {
    net_args.push_back({{DNNL_ARG_SRC, prim_src_memory},
                        {DNNL_ARG_WEIGHTS, prim_filter_memory},
                        {DNNL_ARG_DST, prim_dst_memory}});
  }
  // reorder or copy the output
  auto dst_memory =
      memory({{dst_shape}, memory::data_type::f32, memory::format_tag::nchw},
             eng, output);
  if (prim_dst_memory != dst_memory) {
    net.push_back(reorder(prim_dst_memory, dst_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, prim_dst_memory}, {DNNL_ARG_TO, dst_memory}});
  }
}

void Conv::run() {
  if (input_after_pad) {
    pad_tensor(input_after_pad->data(), origin_input, _n, _c, _h, _w, _pt, _pb,
               _pl, _pr, _izp);
  }
  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(eng_stream, net_args.at(i));
  eng_stream.wait();
}
