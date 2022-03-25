#include "sophgo/Support/Dnnl/MatMul.h"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace sophgo {
MatMul::MatMul() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  engine_stream = dnnl::stream(eng);
}

void MatMul::setup(float *left, float *right, float *bias, float *output,
                   int64_t batch, int64_t M, int64_t K, int64_t N,
                   bool do_relu, int64_t rshift, memory::data_type ldt,
                   memory::data_type rdt,
                   memory::data_type bdt,
                   memory::data_type odt) {
  //printf("MatMul ldt:%ld, rdt:%ld, bdt:%ld, odt:%ld, rshift:%ld\n", ldt, rdt, bdt, odt, rshift);
  memory::dims src_dims = {batch, M, K};
  memory::dims weights_dims = {batch, K, N};
  memory::dims bias_dims = {1, 1, N};
  memory::dims dst_dims = {batch, M, N};

  net.clear();
  net_args.clear();
  auto src_md = memory::desc(src_dims, ldt, tag::abc);
  auto weights_md = memory::desc(weights_dims, rdt, tag::abc);
  auto bias_md = memory::desc(bias_dims, bdt, tag::abc);
  auto dst_md = memory::desc(dst_dims, odt, tag::abc);
  auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);
  post_ops ops;
  matmul::primitive_desc matmul_pd;
  primitive_attr matmul_attr;
  if (memory::data_type::s32 == odt) {
    float scale = 1;
    for (int i = 0; i < abs(rshift); i++) {
      if (rshift > 0) {
        scale /= 2;
      } else if (rshift < 0) {
        scale *= 2;
      }
    }
    std::vector<float> fc_scales = {scale};
    matmul_attr.set_output_scales(0, fc_scales);

    const float ops_scale = 1.f;
    if (do_relu) {
      const float ops_alpha = 0.f; // relu negative slope
      const float ops_beta = 0.f;
      ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
    }
    const float ops_alpha = -128;
    const float ops_beta = 127;
    ops.append_eltwise(ops_scale, algorithm::eltwise_clip, ops_alpha, ops_beta);
    ops.append_eltwise(ops_scale, algorithm::eltwise_round, 0, 0);
    matmul_attr.set_post_ops(ops);
    matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);
  } else {
    if (do_relu) {
      const float ops_scale = 1.f;
      const float ops_alpha = 0.f; // relu negative slope
      const float ops_beta = 0.f;
      ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha, ops_beta);
      matmul_attr.set_post_ops(ops);
      matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);
    } else {
      matmul_pd = matmul::primitive_desc(matmul_d, eng);
    }
  }

  auto src_float_memory = memory({{src_dims}, memory::data_type::f32, memory::format_tag::abc}, eng, left);
  auto prim_src_memory = src_float_memory;
  if (matmul_pd.src_desc() != src_float_memory.get_desc()) {
    prim_src_memory = memory(matmul_pd.src_desc(), eng);
    net.push_back(reorder(src_float_memory, prim_src_memory));
    net_args.push_back({{DNNL_ARG_FROM, src_float_memory}, {DNNL_ARG_TO, prim_src_memory}});
  }

  auto weights_float_memory = memory({{weights_dims}, memory::data_type::f32, memory::format_tag::abc}, eng, right);
  auto prim_weights_memory = weights_float_memory;
  if (matmul_pd.weights_desc() != weights_float_memory.get_desc()) {
    prim_weights_memory = memory(matmul_pd.weights_desc(), eng);
    net.push_back(reorder(weights_float_memory, prim_weights_memory));
    net_args.push_back({{DNNL_ARG_FROM, weights_float_memory}, {DNNL_ARG_TO, prim_weights_memory}});
  }

  auto prim_bias_memory = memory();
  if (bias != nullptr) {
    auto bias_float_memory = memory({{bias_dims}, memory::data_type::f32, memory::format_tag::abc}, eng, bias);
    prim_bias_memory = bias_float_memory;
    if (matmul_pd.bias_desc() != bias_float_memory.get_desc()) {
      prim_bias_memory = memory(matmul_pd.bias_desc(), eng);
      net.push_back(reorder(bias_float_memory, prim_bias_memory));
      net_args.push_back({{DNNL_ARG_FROM, bias_float_memory}, {DNNL_ARG_TO, prim_bias_memory}});
    }
  }

  auto prim_dst_memory = memory(matmul_pd.dst_desc(), eng);
  net.push_back(matmul(matmul_pd));
  if (bias != nullptr) {
    net_args.push_back({{DNNL_ARG_SRC, prim_src_memory},
                        {DNNL_ARG_WEIGHTS, prim_weights_memory},
                        {DNNL_ARG_BIAS, prim_bias_memory},
                        {DNNL_ARG_DST, prim_dst_memory}});
  } else {
    net_args.push_back({{DNNL_ARG_SRC, prim_src_memory},
                        {DNNL_ARG_WEIGHTS, prim_weights_memory},
                        {DNNL_ARG_DST, prim_dst_memory}});
  }

  // reorder or copy the output
  auto dst_memory =
      memory({{dst_dims}, memory::data_type::f32, memory::format_tag::abc}, eng, output);
  if (prim_dst_memory != dst_memory) {
    net.push_back(reorder(prim_dst_memory, dst_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, prim_dst_memory}, {DNNL_ARG_TO, dst_memory}});
  }
}

void MatMul::run() {
  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(engine_stream, net_args.at(i));
  engine_stream.wait();
}

}
