#include<string.h>
#include "sophgo/Support/Dnnl/Conv.h"
#include "sophgo/Support/MathUtils.h"

using namespace dnnl;
using namespace sophgo;
Conv::Conv() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  eng_stream = dnnl::stream(eng);
  _input_paded1 = nullptr;
  _input_paded2 = nullptr;
  _pt = _pb = _pr = _pl = 0;
  _izp = 0;
}

Conv::~Conv() {
  if (_izp && (_pt > 0 || _pb > 0 || _pr > 0 || _pl > 0)) {
    if (_input_paded1) {
      delete []_input_paded1;
      _input_paded1 = nullptr;
    }

    if (_input_paded2) {
      delete []_input_paded2;
      _input_paded2 = nullptr;
    }
  }
}

void Conv::pad_init(float *input, int n, int ic, int ih, int iw, int& pt, int& pb, int& pl, int& pr, int izp) {
  _input = input;
  _pt = pt;
  _pb = pb;
  _pr = pr;
  _pl = pl;
  _izp = izp;
  if (izp && (_pt > 0 || _pb > 0 || _pr > 0 || _pl > 0)) {
    int input_paded_size = n*ic*(ih+pt+pb)*(iw+pr+pl);
    _input_paded1 = new float[input_paded_size];
    _input_paded2 = new float[input_paded_size];
    for (int i = 0; i < input_paded_size; i++) {
      _input_paded1[i] = izp;
      _input_paded2[i] = izp;
    }
    src_shape = {n, ic, ih+pt+pb, iw+pr+pl};
    pt = pb = pr = pl = 0;
  } else {
    src_shape = {n, ic, ih, iw};
    _input_paded2 = input;
  }
}

void Conv::setup(float *input, float *weight, float *bias, float *output, int n,
                 int ic, int ih, int iw, int oc, int oh, int ow, int kh, int kw,
                 int sh, int sw, int dh, int dw, int pt, int pb, int pl, int pr,
                 int g, bool do_relu, int izp, int ozp, int* rshift, int* multiplier, memory::data_type idt,
                 memory::data_type wdt, memory::data_type bdt, memory::data_type odt, bool per_channel, int chip) {
  //printf("Conv para:%d,%d,%d,%d,%d,%d,%d,%d\n", idt, wdt, bdt, odt, per_channel, izp, ozp, do_relu);
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
  auto src_md = memory::desc({src_shape}, idt, memory::format_tag::any);
  auto filter_md = memory::desc({filter_shape}, wdt, memory::format_tag::any);
  auto bias_md = memory::desc({bias_shape}, bdt, memory::format_tag::any);
  auto dst_md = memory::desc({dst_shape}, odt,memory::format_tag::any);

  auto conv_desc = convolution_forward::desc(
      prop_kind::forward_inference, algorithm::convolution_direct, src_md,
      filter_md, bias_md, dst_md, strides, dilation, padding_l, padding_r);

  post_ops ops;
  primitive_attr conv_attr;
  if (memory::data_type::s32 == odt) {
    std::vector<float> conv_scales(oc);
    if (per_channel) {
      for (int o = 0; o < oc; o++) {
          float scale = multiplier[o];
          for (int i = 0; i < abs(rshift[o]); i++) {
            if (rshift > 0) {
              scale /= 2;
            } else if (rshift < 0) {
              scale *= 2;
            }
          }
          conv_scales[o] = scale;
      }
    } else {
      float scale = 1;
      if (multiplier) {
        scale = multiplier[0];
      }

      for (int i = 0; i < abs(rshift[0]); i++) {
        if (rshift > 0) {
          scale /= 2;
        } else if (rshift < 0) {
          scale *= 2;
        }
      }
      std::fill(conv_scales.begin(), conv_scales.end(), scale);
    }

    if (izp) {
      //conv_attr.set_zero_points(DNNL_ARG_SRC, 0, {izp}); //增加此句会导致conv输出cos降低很多，但非0零点的pad需要寻找解决方案 todo
    }
    const float ops_scale = 1.f;
    float ops_alpha = 0.f; // relu negative slope
    float ops_beta = 0.f;
    if (do_relu) {
      ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
      if (chip) {
        ops_alpha = -128;
        ops_beta = 127;
      } else {
        ops_alpha = 0;
        ops_beta = 255;
      }
    } else {
        //1686时，这里需要限制在0到255，后续考虑更好处理方式 wxctodo
      ops_alpha = -128;
      ops_beta = 127;
    }
    if (ozp)
      ops.append_eltwise(1,dnnl::algorithm::eltwise_linear,1,ozp);
    ops.append_eltwise(ops_scale, algorithm::eltwise_clip, ops_alpha, ops_beta);
    ops.append_eltwise(ops_scale, algorithm::eltwise_round, 0, 0);
    conv_attr.set_output_scales(2, conv_scales);
    conv_attr.set_post_ops(ops);
  } else {
    if (do_relu) {
      const float ops_scale = 1.f;
      const float ops_alpha = 0.f; // relu negative slope
      const float ops_beta = 0.f;
      ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
      conv_attr.set_post_ops(ops);
    }
  }
  conv_prim_desc = convolution_forward::primitive_desc(conv_desc, conv_attr, eng);

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
    auto bias_memory = memory(
        {{bias_shape}, memory::data_type::f32, memory::format_tag::x}, eng, bias);
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
             eng, _input_paded2);
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
  if (_izp)
    pad_tensor(_input, _input_paded1, _input_paded2, src_shape[0], src_shape[1], src_shape[2], src_shape[3], _pt, _pb, _pl, _pr);
  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(eng_stream, net_args.at(i));
  eng_stream.wait();
}
