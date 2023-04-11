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

void Conv::activation_init(float *input, conv_attr_t &attr) {
  origin_input = input;
  memcpy(&_attr, &attr, sizeof(conv_attr_t));
  src_shape = {attr.n, attr.ic, attr.id, attr.ih, attr.iw};
  assert(!attr.ins_d);
  bool need_dilate = false;
  if (attr.ins_h || attr.ins_w) {
    int64_t ih_after = (attr.ih - 1) * (attr.ins_h + 1) + 1;
    int64_t iw_after = (attr.iw - 1) * (attr.ins_w + 1) + 1;
    src_shape[3] = ih_after;
    src_shape[4] = iw_after;
    need_dilate = true;
    attr.ins_h = 0;
    attr.ins_w = 0;
  }
  if (attr.pad_value != 0 && (attr.pdf > 0 || attr.pdb > 0 || attr.pht > 0 ||
                              attr.phb > 0 || attr.pwl > 0 || attr.pwr > 0)) {
    src_shape[2] += attr.pdf + attr.pdb;
    src_shape[3] += attr.pht + attr.phb;
    src_shape[4] += attr.pwl + attr.pwr;

    need_dilate = true;
    attr.pdf = attr.pdb = attr.pht = attr.phb = attr.pwl = attr.pwr = 0;
  }
  if (need_dilate) {
    int input_padded_size = src_shape[0] * src_shape[1] * src_shape[2] *
                            src_shape[3] * src_shape[4];
    input_after_pad = std::make_shared<std::vector<float>>(input_padded_size);
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
  activation_init(input, attr);
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

  src_mem =
      memory({{src_shape}, memory::data_type::f32, memory::format_tag::ncdhw},
             eng, p_input);
  auto filter_tag = (attr.groups != 1) ? memory::format_tag::goidhw
                                       : memory::format_tag::oidhw;
  filter_mem = memory({{filter_shape}, memory::data_type::f32, filter_tag}, eng,
                      p_weight);
  if (bias == nullptr) {
    bias0 = std::make_shared<std::vector<float>>(attr.oc, 0);
    bias = bias0->data();
  }
  bias_mem = memory(
      {{bias_shape}, memory::data_type::f32, memory::format_tag::x}, eng, bias);
  dst_mem =
      memory({{dst_shape}, memory::data_type::f32, memory::format_tag::ncdhw},
             eng, output);
  // post_ops ops;
  primitive_attr conv_attr;
  post_relu(conv_attr, attr.do_relu, attr.relu_limit);

  auto conv_prim_desc = convolution_forward::primitive_desc(
      eng, prop_kind::forward_inference, algorithm::convolution_direct,
      src_mem.get_desc(), filter_mem.get_desc(), bias_mem.get_desc(),
      dst_mem.get_desc(), strides, dilation, padding_l, padding_r, conv_attr);
  prim = convolution_forward(conv_prim_desc);
}

void Conv::run() {
  if (input_after_pad) {
    if (_attr.pad_value) {
      dilate_tensor(input_after_pad->data(), origin_input, _attr.n, _attr.ic,
                    _attr.id, _attr.ih, _attr.iw, _attr.pdf, _attr.pdb,
                    _attr.pht, _attr.phb, _attr.pwl, _attr.pwr, _attr.pad_value,
                    _attr.ins_h, _attr.ins_w, 0);
    } else {
      dilate_tensor(input_after_pad->data(), origin_input, _attr.n, _attr.ic,
                    _attr.id, _attr.ih, _attr.iw, 0, 0, 0, 0, 0, 0, 0,
                    _attr.ins_h, _attr.ins_w, 0);
    }
  }

  prim.execute(eng_stream, {{DNNL_ARG_SRC, src_mem},
                            {DNNL_ARG_WEIGHTS, filter_mem},
                            {DNNL_ARG_BIAS, bias_mem},
                            {DNNL_ARG_DST, dst_mem}});
  eng_stream.wait();
}
