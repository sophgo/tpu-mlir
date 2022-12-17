//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Conv.h"
#include "tpu_mlir/Support/Dnnl/DnnlUtils.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <string.h>

using namespace dnnl;
using namespace tpu_mlir;
Conv::Conv() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  eng_stream = dnnl::stream(eng);
  memset(&_attr, 0, sizeof(conv_attr_t));
}

Conv::~Conv() {}

void Conv::pad_init(float *input, conv_attr_t &attr) {
  origin_input = input;
  memcpy(&_attr, &attr, sizeof(conv_attr_t));
  if (attr.pad_value != 0 && (attr.pdf > 0 || attr.pdb > 0 || attr.pht > 0 ||
                              attr.phb > 0 || attr.pwl > 0 || attr.pwr > 0)) {
    src_shape = {attr.n, attr.ic, attr.id + attr.pdf + attr.pdb,
                 attr.ih + attr.pht + attr.phb, attr.iw + attr.pwl + attr.pwr};
    int input_padded_size = src_shape[0] * src_shape[1] * src_shape[2] *
                            src_shape[3] * src_shape[4];
    input_after_pad = std::make_shared<std::vector<float>>(input_padded_size);
    attr.pdf = attr.pdb = attr.pht = attr.phb = attr.pwl = attr.pwr = 0;
    p_input = input_after_pad->data();
  } else {
    src_shape = {attr.n, attr.ic, attr.id, attr.ih, attr.iw};
    p_input = input;
  }
}

void Conv::filter_init(float *weight, conv_attr_t &attr) {
  origin_weight = weight;
  if (attr.kernel_zp != 0) {
    int weight_size =
        attr.ic * attr.oc * attr.kd * attr.kh * attr.kw / attr.groups;
    weight_after_zp = std::make_shared<std::vector<float>>(weight_size);
    p_weight = weight_after_zp->data();
    tensor_sub_zp(weight_after_zp->data(), origin_weight, weight_size,
                  attr.kernel_zp);
  } else {
    p_weight = weight;
  }
}

void Conv::setup(float *input, float *weight, float *bias, float *output,
                 conv_attr_t attr) {
  pad_init(input, attr);
  filter_init(weight, attr);
  dst_shape = {attr.n, attr.oc, attr.od, attr.oh, attr.ow};
  memory::dims filter_shape =
      (attr.groups != 1)
          ? memory::dims{attr.groups,
                         attr.oc / attr.groups,
                         attr.ic / attr.groups,
                         attr.kd,
                         attr.kh,
                         attr.kw}
          : memory::dims{attr.oc, attr.ic, attr.kd, attr.kh, attr.kw};
  memory::dims bias_shape = {attr.oc};
  memory::dims strides = {attr.sd, attr.sh, attr.sw};

  memory::dims padding_l = {attr.pdf, attr.pht, attr.pwl};
  memory::dims padding_r = {attr.pdb, attr.phb, attr.pwr};
  memory::dims dilation = {attr.dd - 1, attr.dh - 1, attr.dw - 1};

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

  if (bias == nullptr)
    conv_desc = convolution_forward::desc(
        prop_kind::forward_inference, algorithm::convolution_direct, src_md,
        filter_md, dst_md, strides, dilation, padding_l, padding_r);

  // post_ops ops;
  primitive_attr conv_attr;
  post_relu(conv_attr, attr.do_relu, attr.relu_limit);

  conv_prim_desc =
      convolution_forward::primitive_desc(conv_desc, conv_attr, eng);

  // set mkldnn memory
  auto filter_tag = (attr.groups != 1) ? memory::format_tag::goidhw
                                       : memory::format_tag::oidhw;
  auto filter_memory = memory(
      {{filter_shape}, memory::data_type::f32, filter_tag}, eng, p_weight);
  prim_filter_memory = filter_memory;
  if (conv_prim_desc.weights_desc() != filter_memory.get_desc()) {
    prim_filter_memory = memory(conv_prim_desc.weights_desc(), eng);
    net.push_back(reorder(filter_memory, prim_filter_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, filter_memory}, {DNNL_ARG_TO, prim_filter_memory}});
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
      memory({{src_shape}, memory::data_type::f32, memory::format_tag::ncdhw},
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
      memory({{dst_shape}, memory::data_type::f32, memory::format_tag::ncdhw},
             eng, output);
  if (prim_dst_memory != dst_memory) {
    net.push_back(reorder(prim_dst_memory, dst_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, prim_dst_memory}, {DNNL_ARG_TO, dst_memory}});
  }
}

void Conv::run() {
  if (input_after_pad) {
    pad_tensor(input_after_pad->data(), origin_input, _attr.n, _attr.ic,
               _attr.id, _attr.ih, _attr.iw, _attr.pdf, _attr.pdb, _attr.pht,
               _attr.phb, _attr.pwl, _attr.pwr, _attr.pad_value);
  }

  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(eng_stream, net_args.at(i));
  eng_stream.wait();
}
