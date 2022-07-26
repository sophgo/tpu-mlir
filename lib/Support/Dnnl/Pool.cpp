//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Pool.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace dnnl;
using namespace tpu_mlir;

Pooling::Pooling() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  eng_stream = dnnl::stream(eng);
  memset(&_attrs, 0, sizeof(pool_attr_t));
  _izp = 0;
}

Pooling::~Pooling() {}

void Pooling::pad_init(float *input, pool_attr_t &attr, int izp) {
  origin_input = input;
  _izp = izp;
  memcpy(&_attrs, &attr, sizeof(pool_attr_t));
  if (izp && (attr.pad_d > 0 || attr.pad_d_after > 0 || attr.pad_h > 0 ||
              attr.pad_h_after > 0 || attr.pad_w > 0 || attr.pad_w_after > 0)) {
    src_shape = {attr.n, attr.c, attr.id + attr.pad_d + attr.pad_d_after,
                 attr.ih + attr.pad_h + attr.pad_h_after,
                 attr.iw + attr.pad_w + attr.pad_w_after};
    int input_padded_size = src_shape[0] * src_shape[1] * src_shape[2] *
                            src_shape[3] * src_shape[4];
    input_after_pad = std::make_shared<std::vector<float>>(input_padded_size);
    attr.pad_d = attr.pad_d_after = attr.pad_h = attr.pad_h_after = attr.pad_w =
        attr.pad_w_after = 0;
    p_input = input_after_pad->data();
  } else {
    src_shape = {attr.n, attr.c, attr.id, attr.ih, attr.iw};
    p_input = input;
  }
}

void Pooling::setup(float *input, float *output, pool_attr_t attr, bool is_avg,
                    int izp) {
  this->kd = attr.kd;
  this->kh = attr.kh;
  this->kw = attr.kw;
  pad_init(input, attr, izp);
  memory::dims dst_shape = {attr.n, attr.c, attr.od, attr.oh, attr.ow};
  memory::dims strides = {attr.sd, attr.sh, attr.sw};
  memory::dims kernel = {attr.kd, attr.kh, attr.kw};
  memory::dims padding_tl = {attr.pad_d, attr.pad_h, attr.pad_w};
  memory::dims padding_br = {attr.pad_d_after, attr.pad_h_after,
                             attr.pad_w_after};
  auto src_md = memory::desc({src_shape}, memory::data_type::f32,
                             memory::format_tag::ncdhw);
  auto dst_md = memory::desc({dst_shape}, memory::data_type::f32,
                             memory::format_tag::ncdhw);
  auto pool_avg_algo = attr.count_include_pad
                           ? algorithm::pooling_avg_include_padding
                           : algorithm::pooling_avg_exclude_padding;
  // pool desc
  auto pool_desc = pooling_forward::desc(
      prop_kind::forward_inference,
      is_avg ? pool_avg_algo : algorithm::pooling_max, src_md, dst_md, strides,
      kernel, padding_tl, padding_br);

  prim_desc = pooling_forward::primitive_desc(pool_desc, eng);
  memory src_memory =
      memory({{src_shape}, memory::data_type::f32, memory::format_tag::ncdhw},
             eng, p_input);
  memory dst_memory =
      memory({{dst_shape}, memory::data_type::f32, memory::format_tag::ncdhw},
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
    pad_tensor(input_after_pad->data(), origin_input, _attrs.n, _attrs.c,
               _attrs.id, _attrs.ih, _attrs.iw, _attrs.pad_d,
               _attrs.pad_d_after, _attrs.pad_h, _attrs.pad_h_after,
               _attrs.pad_w, _attrs.pad_w_after, _izp);
  }
  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(eng_stream, net_args.at(i));
  eng_stream.wait();
}
