#include "sophgo/Support/DnnlConv.h"

using namespace dnnl;
Conv::Conv() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  eng_stream = dnnl::stream(eng);
}

void Conv::setup(float *input, float *weight, float *bias, float *output, int n,
                 int ic, int ih, int iw, int oc, int oh, int ow, int kh, int kw,
                 int sh, int sw, int dh, int dw, int pt, int pb, int pl, int pr,
                 int g) {
  src_shape = {n, ic, ih, iw};
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

  conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);

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
  prim_bias_memory = memory(
      {{bias_shape}, memory::data_type::f32, memory::format_tag::x}, eng, bias);
  auto src_memory =
      memory({{src_shape}, memory::data_type::f32, memory::format_tag::nchw},
             eng, input);
  auto prim_src_memory = src_memory;
  if (conv_prim_desc.src_desc() != src_memory.get_desc()) {
    prim_src_memory = memory(conv_prim_desc.src_desc(), eng);
    net.push_back(reorder(src_memory, prim_src_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, src_memory}, {DNNL_ARG_TO, prim_src_memory}});
  }

  auto prim_dst_memory = memory(conv_prim_desc.dst_desc(), eng);
  net.push_back(convolution_forward(conv_prim_desc));
  net_args.push_back({{DNNL_ARG_SRC, prim_src_memory},
                      {DNNL_ARG_WEIGHTS, prim_filter_memory},
                      {DNNL_ARG_BIAS, prim_bias_memory},
                      {DNNL_ARG_DST, prim_dst_memory}});
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
  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(eng_stream, net_args.at(i));
  eng_stream.wait();
}
