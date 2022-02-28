#include "sophgo/Support/DnnlMatMul.h"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

MatMul::MatMul() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  engine_stream = dnnl::stream(eng);
}

void MatMul::setup(float *left, float *right, float *bias, float *output,
                   int64_t batch, int64_t M, int64_t K, int64_t N,
                   bool do_relu) {
  memory::dims src_dims = {batch, M, K};
  memory::dims weights_dims = {batch, K, N};
  memory::dims bias_dims = {1, 1, N};
  memory::dims dst_dims = {batch, M, N};
  auto src_md = memory::desc(src_dims, dt::f32, tag::abc);
  auto weights_md = memory::desc(weights_dims, dt::f32, tag::abc);
  auto bias_md = memory::desc(bias_dims, dt::f32, tag::abc);
  auto dst_md = memory::desc(dst_dims, dt::f32, tag::abc);

  auto src_mem = memory(src_md, eng, left);
  auto weights_mem = memory(weights_md, eng, right);
  auto bias_mem = memory(bias_md, eng, bias);
  auto dst_mem = memory(dst_md, eng, output);
  auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);

  matmul::primitive_desc matmul_pd;
  if (do_relu) {
    const float scale = 1.0f;
    const float alpha = 0.f;
    const float beta = 0.f;
    post_ops matmul_ops;
    matmul_ops.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
    primitive_attr matmul_attr;
    matmul_attr.set_post_ops(matmul_ops);
    matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);
  } else {
    matmul_pd = matmul::primitive_desc(matmul_d, eng);
  }
  primitive = matmul(matmul_pd);
  matmul_args.clear();
  matmul_args.insert({DNNL_ARG_SRC, src_mem});
  matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
  if (bias != nullptr) {
    matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
  }
  matmul_args.insert({DNNL_ARG_DST, dst_mem});
}

void MatMul::run() {
  primitive.execute(engine_stream, matmul_args);
  engine_stream.wait();
}
