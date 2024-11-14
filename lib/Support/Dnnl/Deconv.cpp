//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Deconv.h"
#include "tpu_mlir/Support/Dnnl/DnnlUtils.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace dnnl;
using namespace tpu_mlir;
Deconv::Deconv() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  eng_stream = dnnl::stream(eng);
  memset(&_attrs, 0, sizeof(deconv_attr_t));
  _izp = 0;
}

Deconv::~Deconv() {}

void Deconv::pad_init(float *input, deconv_attr_t &attr, int izp) {
  origin_input = input;
  _izp = izp;
  memcpy(&_attrs, &attr, sizeof(deconv_attr_t));
  if (izp) {
    src_shape = {
        attr.n, attr.ic,
        (attr.id - 1) * attr.sd + 1 +
            attr.dd * (2 * attr.kd - 2 - attr.pad_d - attr.pad_d_after) +
            attr.output_pad_d,
        (attr.ih - 1) * attr.sh + 1 +
            attr.dh * (2 * attr.kh - 2 - attr.pad_h - attr.pad_h_after) +
            attr.output_pad_h,
        (attr.iw - 1) * attr.sw + 1 +
            attr.dw * (2 * attr.kw - 2 - attr.pad_w - attr.pad_w_after) +
            attr.output_pad_w};
    int input_padded_size = src_shape[0] * src_shape[1] * src_shape[2] *
                            src_shape[3] * src_shape[4];
    input_after_pad = std::make_shared<std::vector<float>>(input_padded_size);
    attr.pad_d = attr.pad_d_after = attr.pad_h = attr.pad_h_after = attr.pad_w =
        attr.pad_w_after = 0;
    attr.sd = attr.sh = attr.sw = 1;
    attr.dd = attr.dh = attr.dw = 1;
    p_input = input_after_pad->data();
  } else {
    src_shape = {attr.n, attr.ic, attr.id, attr.ih, attr.iw};
    p_input = input;
  }
}

void Deconv::setup(float *input, float *weight, float *bias, float *output,
                   const deconv_attr_t &attr_, int izp) {
  // printf("Conv para:%d,%d,%d,%d,%d,%d,%d,%d\n", idt, wdt, bdt, odt,
  // per_channel, izp, ozp, do_relu);
  auto attr = attr_;
  this->kd = attr.kd;
  this->kh = attr.kh;
  this->kw = attr.kw;
  pad_init(input, attr, izp);
  dst_shape = {attr.n, attr.oc, attr.od, attr.oh, attr.ow};
  memory::dims filter_shape =
      (attr.g != 1) ? memory::dims{attr.g,  attr.oc / attr.g, attr.ic / attr.g,
                                   attr.kd, attr.kh,          attr.kw}
                    : memory::dims{attr.oc, attr.ic, attr.kd, attr.kh, attr.kw};
  memory::dims bias_shape = {attr.oc};
  memory::dims strides = {attr.sd, attr.sh, attr.sw};

  memory::dims padding_l = {attr.pad_d, attr.pad_h, attr.pad_w};
  memory::dims padding_r = {attr.pad_d_after, attr.pad_h_after,
                            attr.pad_w_after};

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
  primitive_attr conv_attr;
  post_relu(conv_attr, attr.do_relu, attr.relu_limit);
  if (_izp != 0) {
    if (bias != nullptr) {
      conv_prim_desc = convolution_forward::primitive_desc(
          eng, prop_kind::forward_inference, algorithm::convolution_direct,
          src_md, filter_md, bias_md, dst_md, strides, dilation, padding_l,
          padding_r, conv_attr);
    } else {
      conv_prim_desc = convolution_forward::primitive_desc(
          eng, prop_kind::forward_inference, algorithm::convolution_direct,
          src_md, filter_md, dst_md, strides, dilation, padding_l, padding_r,
          conv_attr);
    }

    // set mkldnn memory
    auto filter_tag =
        (attr.g != 1) ? memory::format_tag::goidhw : memory::format_tag::oidhw;
    weight_rotated = std::make_shared<std::vector<float>>(
        attr.oc * attr.ic * attr.kh * attr.kw / attr.g);
    for (int i = 0; i < attr.oc * attr.ic / attr.g; ++i) {
      for (int j = 0; j < attr.kh; ++j) {
        for (int k = 0; k < attr.kw; ++k) {
          weight_rotated->data()[i * kh * kw + (attr.kh - j - 1) * attr.kw +
                                 (attr.kw - k - 1)] =
              weight[i * kh * kw + j * attr.kw + k];
        }
      }
    }
    auto filter_memory =
        memory({{filter_shape}, memory::data_type::f32, filter_tag}, eng,
               weight_rotated->data());
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
  } else {
    if (bias != nullptr) {
      deconv_prim_desc = deconvolution_forward::primitive_desc(
          eng, prop_kind::forward_inference, algorithm::deconvolution_direct,
          src_md, filter_md, bias_md, dst_md, strides, dilation, padding_l,
          padding_r, conv_attr);
    } else {
      deconv_prim_desc = deconvolution_forward::primitive_desc(
          eng, prop_kind::forward_inference, algorithm::deconvolution_direct,
          src_md, filter_md, dst_md, strides, dilation, padding_l, padding_r,
          conv_attr);
    }

    // set mkldnn memory
    auto filter_tag =
        (attr.g != 1) ? memory::format_tag::goidhw : memory::format_tag::oidhw;
    auto filter_memory = memory(
        {{filter_shape}, memory::data_type::f32, filter_tag}, eng, weight);
    prim_filter_memory = filter_memory;
    if (deconv_prim_desc.weights_desc() != filter_memory.get_desc()) {
      prim_filter_memory = memory(deconv_prim_desc.weights_desc(), eng);
      reorder(filter_memory, prim_filter_memory)
          .execute(eng_stream, filter_memory, prim_filter_memory);
    }

    auto prim_bias_memory = memory();
    if (bias != nullptr) {
      auto bias_memory =
          memory({{bias_shape}, memory::data_type::f32, memory::format_tag::x},
                 eng, bias);
      prim_bias_memory = bias_memory;
      if (deconv_prim_desc.bias_desc() != bias_memory.get_desc()) {
        prim_bias_memory = memory(deconv_prim_desc.bias_desc(), eng);
        net.push_back(reorder(bias_memory, prim_bias_memory));
        net_args.push_back(
            {{DNNL_ARG_FROM, bias_memory}, {DNNL_ARG_TO, prim_bias_memory}});
      }
    }

    auto src_memory =
        memory({{src_shape}, memory::data_type::f32, memory::format_tag::ncdhw},
               eng, p_input);
    auto prim_src_memory = src_memory;
    if (deconv_prim_desc.src_desc() != src_memory.get_desc()) {
      prim_src_memory = memory(deconv_prim_desc.src_desc(), eng);
      net.push_back(reorder(src_memory, prim_src_memory));
      net_args.push_back(
          {{DNNL_ARG_FROM, src_memory}, {DNNL_ARG_TO, prim_src_memory}});
    }

    auto prim_dst_memory = memory(deconv_prim_desc.dst_desc(), eng);
    net.push_back(deconvolution_forward(deconv_prim_desc));
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
}

void Deconv::run() {
  if (input_after_pad) {
    pad_tensor_for_deconv(input_after_pad->data(), origin_input, _attrs.n,
                          _attrs.ic, _attrs.id, _attrs.ih, _attrs.iw, _attrs.kd,
                          _attrs.kh, _attrs.kw, _attrs.dd, _attrs.dh, _attrs.dw,
                          _attrs.sd, _attrs.sh, _attrs.sw, _attrs.pad_d,
                          _attrs.pad_d_after, _attrs.pad_h, _attrs.pad_h_after,
                          _attrs.pad_w, _attrs.pad_w_after, _attrs.output_pad_d,
                          _attrs.output_pad_h, _attrs.output_pad_w, _izp);
  }
  for (size_t i = 0; i < net.size(); ++i) {
    net.at(i).execute(eng_stream, net_args.at(i));
  }
  eng_stream.wait();
}
