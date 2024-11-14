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

using namespace dnnl;
using namespace tpu_mlir;
Conv::Conv() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  eng_stream = dnnl::stream(eng);
  memset(&_attr, 0, sizeof(conv_attr_t));
  backw_init = false;
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

void Conv::diff_filter_init(memory::dims &filter_shape) {
  int64_t size_ = 1;
  for (auto dim : filter_shape)
    size_ *= dim;
  diff_filter = std::make_shared<std::vector<float>>(size_);
  p_gweight = diff_filter->data();
}

void Conv::diff_bias_init(memory::dims &bias_shape) {
  int64_t size_ = 1;
  for (auto dim : bias_shape)
    size_ *= dim;
  diff_bias = std::make_shared<std::vector<float>>(size_);
  p_gbias = diff_bias->data();
}

void Conv::diff_dst_init(memory::dims &dst_shape) {
  int64_t size_ = 1;
  for (auto dim : dst_shape)
    size_ *= dim;
  diff_dst = std::make_shared<std::vector<float>>(size_);
  p_goutput = diff_dst->data();
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

  conv_prim_desc = convolution_forward::primitive_desc(
      eng, prop_kind::forward_inference, algorithm::convolution_direct,
      src_mem.get_desc(), filter_mem.get_desc(), bias_mem.get_desc(),
      dst_mem.get_desc(), strides, dilation, padding_l, padding_r, conv_attr);
  prim = convolution_forward(conv_prim_desc);
}

void Conv::backward_weights_setup() {
  // now to set backward path
  net_bw.clear();
  net_bw_args.clear();

  memory::dims filter_shape =
      (_attr.groups != 1)
          ? memory::dims{_attr.groups,
                         _attr.oc / _attr.groups,
                         _attr.ic / _attr.groups,
                         _attr.kd,
                         _attr.kh,
                         _attr.kw}
          : memory::dims{_attr.oc, _attr.ic, _attr.kd, _attr.kh, _attr.kw};
  memory::dims bias_shape = {_attr.oc};
  memory::dims strides = {_attr.sd, _attr.sh, _attr.sw};

  memory::dims padding_l = {_attr.pdf, _attr.pht, _attr.pwl};
  memory::dims padding_r = {_attr.pdb, _attr.phb, _attr.pwr};
  auto filter_tag = (_attr.groups != 1) ? memory::format_tag::goidhw
                                        : memory::format_tag::oidhw;
  if (_attr.pad_value != 0 &&
      (_attr.pdf > 0 || _attr.pdb > 0 || _attr.pht > 0 || _attr.phb > 0 ||
       _attr.pwl > 0 || _attr.pwr > 0)) {
    padding_l = {0, 0, 0};
    padding_r = {0, 0, 0};
  }
  diff_filter_init(filter_shape);
  diff_bias_init(bias_shape);
  diff_dst_init(dst_shape);

  auto user_diff_filter_memory = memory(
      {{filter_shape}, memory::data_type::f32, filter_tag}, eng, p_gweight);
  auto user_diff_bias_memory =
      memory({{bias_shape}, memory::data_type::f32, memory::format_tag::x}, eng,
             p_gbias);
  auto user_diff_dst_memory =
      memory({{dst_shape}, memory::data_type::f32, memory::format_tag::ncdhw},
             eng, p_goutput);

  // create memory
  auto bw_src_md = memory::desc({src_shape}, memory::data_type::f32,
                                memory::format_tag::any);
  auto bw_diff_filter_md = memory::desc({filter_shape}, memory::data_type::f32,
                                        memory::format_tag::any);
  auto bw_diff_bias_md = memory::desc({bias_shape}, memory::data_type::f32,
                                      memory::format_tag::any);
  auto bw_diff_dst_md = memory::desc({dst_shape}, memory::data_type::f32,
                                     memory::format_tag::any);
  // create backward convolution primitive descriptor
  auto bw_weights_pd = convolution_backward_weights::primitive_desc(
      eng, algorithm::convolution_direct, bw_src_md, bw_diff_filter_md,
      bw_diff_bias_md, bw_diff_dst_md, strides, padding_l, padding_r,
      conv_prim_desc);
  auto bw_src_memory = src_mem;
  if (bw_weights_pd.src_desc() != bw_src_memory.get_desc()) {
    bw_src_memory = memory(bw_weights_pd.src_desc(), eng);
    net_bw.push_back(reorder(src_mem, bw_src_memory));
    net_bw_args.push_back(
        {{DNNL_ARG_FROM, src_mem}, {DNNL_ARG_TO, bw_src_memory}});
  }
  auto diff_dst_memory = user_diff_dst_memory;
  if (bw_weights_pd.diff_dst_desc() != user_diff_dst_memory.get_desc()) {
    diff_dst_memory = memory(bw_weights_pd.diff_dst_desc(), eng);
    net_bw.push_back(reorder(user_diff_dst_memory, diff_dst_memory));
    net_bw_args.push_back({{DNNL_ARG_FROM, user_diff_dst_memory},
                           {DNNL_ARG_TO, diff_dst_memory}});
  }
  // create backward convolution primitive
  net_bw.push_back(convolution_backward_weights(bw_weights_pd));
  net_bw_args.push_back({{DNNL_ARG_SRC, bw_src_memory},
                         {DNNL_ARG_DIFF_DST, diff_dst_memory},
                         {DNNL_ARG_DIFF_BIAS, user_diff_bias_memory}});
  // create reorder primitives between conv diff weights and user diff weights
  // if needed
  auto diff_weights_memory = user_diff_filter_memory;
  if (bw_weights_pd.diff_weights_desc() != user_diff_filter_memory.get_desc()) {
    diff_weights_memory = memory(bw_weights_pd.diff_weights_desc(), eng);
    net_bw_args.back().insert({DNNL_ARG_DIFF_WEIGHTS, diff_weights_memory});
    net_bw.push_back(reorder(diff_weights_memory, user_diff_filter_memory));
    net_bw_args.push_back({{DNNL_ARG_FROM, diff_weights_memory},
                           {DNNL_ARG_TO, user_diff_filter_memory}});
  } else {
    net_bw_args.back().insert({DNNL_ARG_DIFF_WEIGHTS, diff_weights_memory});
  }
  return;
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

void Conv::run_backw(void *dst_grd_input, void *weight_grd_output) {
  if (!backw_init) {
    backward_weights_setup();
    backw_init = true;
  }
  memcpy(p_goutput, dst_grd_input, diff_dst.get()->size() * sizeof(float));
  memset(p_gweight, 0, diff_filter.get()->size() * sizeof(float));
  memset(p_gbias, 0, diff_bias.get()->size() * sizeof(float));
  for (size_t i = 0; i < net_bw.size(); ++i)
    net_bw.at(i).execute(eng_stream, net_bw_args.at(i));
  eng_stream.wait();
  memcpy(weight_grd_output, p_gweight,
         diff_filter.get()->size() * sizeof(float));
}
