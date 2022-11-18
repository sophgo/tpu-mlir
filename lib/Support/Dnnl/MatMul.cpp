//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/MatMul.h"
#include "tpu_mlir/Support/Dnnl/DnnlUtils.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace tpu_mlir {
MatMul::MatMul() {
  eng = dnnl::engine(engine::kind::cpu, 0);
  engine_stream = dnnl::stream(eng);
}

void MatMul::right_init(float *right, int64_t right_zp, int64_t batch,
                        int64_t K, int64_t N, bool right_transpose) {
  int64_t weight_len = batch * K * N;
  p_right = right;
  if (right_zp != 0 ) {
    right_after_zp = std::make_shared<std::vector<float>>(weight_len);
    p_right = right_after_zp->data();
    tensor_sub_zp(right_after_zp->data(), right, weight_len, right_zp);
  }
  if (right_transpose) {
    tensor_hw_transpose(p_right, p_right, 1, batch, N, K);
  }
}

void MatMul::setup(float *left, float *right, float *bias, float *output,
                   int64_t batch, int64_t M, int64_t K, int64_t N,
                   bool do_relu, double relu_limit, int64_t right_zp,
                   bool right_transpose) {
  // printf("MatMul ldt:%ld, rdt:%ld, bdt:%ld, odt:%ld, rshift:%ld\n", ldt, rdt,
  // bdt, odt, rshift);
  memory::dims src_dims = {batch, M, K};
  memory::dims weights_dims = {batch, K, N};
  memory::dims bias_dims = {1, 1, N};
  memory::dims dst_dims = {batch, M, N};
  int64_t weight_len = batch * K * N;
  right_init(right, right_zp, batch, K, N, right_transpose);
  net.clear();
  net_args.clear();
  auto src_md = memory::desc(src_dims, memory::data_type::f32, tag::abc);
  auto weights_md =
      memory::desc(weights_dims, memory::data_type::f32, tag::abc);
  auto bias_md = memory::desc(bias_dims, memory::data_type::f32, tag::abc);
  auto dst_md = memory::desc(dst_dims, memory::data_type::f32, tag::abc);
  auto matmul_d = matmul::desc(src_md, weights_md, bias_md, dst_md);
  post_ops ops;
  matmul::primitive_desc matmul_pd;
  primitive_attr matmul_attr;

  post_relu(matmul_attr, do_relu, relu_limit);

  matmul_pd = matmul::primitive_desc(matmul_d, matmul_attr, eng);

  auto src_float_memory = memory(
      {{src_dims}, memory::data_type::f32, memory::format_tag::abc}, eng, left);
  auto prim_src_memory = src_float_memory;
  if (matmul_pd.src_desc() != src_float_memory.get_desc()) {
    prim_src_memory = memory(matmul_pd.src_desc(), eng);
    net.push_back(reorder(src_float_memory, prim_src_memory));
    net_args.push_back(
        {{DNNL_ARG_FROM, src_float_memory}, {DNNL_ARG_TO, prim_src_memory}});
  }

  auto weights_float_memory =
      memory({{weights_dims}, memory::data_type::f32, memory::format_tag::abc},
             eng, p_right);
  auto prim_weights_memory = weights_float_memory;
  if (matmul_pd.weights_desc() != weights_float_memory.get_desc()) {
    prim_weights_memory = memory(matmul_pd.weights_desc(), eng);
    net.push_back(reorder(weights_float_memory, prim_weights_memory));
    net_args.push_back({{DNNL_ARG_FROM, weights_float_memory},
                        {DNNL_ARG_TO, prim_weights_memory}});
  }

  auto prim_bias_memory = memory();
  bias0 = std::make_shared<std::vector<float>>(dst_dims[2], 0);
  auto bias_float_memory =
      memory({{bias_dims}, memory::data_type::f32, memory::format_tag::abc},
             eng, bias != nullptr ? bias : bias0->data());
  prim_bias_memory = bias_float_memory;
  if (matmul_pd.bias_desc() != bias_float_memory.get_desc()) {
    prim_bias_memory = memory(matmul_pd.bias_desc(), eng);
    net.push_back(reorder(bias_float_memory, prim_bias_memory));
    net_args.push_back({{DNNL_ARG_FROM, bias_float_memory},
                        {DNNL_ARG_TO, prim_bias_memory}});
  }

  auto prim_dst_memory = memory(matmul_pd.dst_desc(), eng);
  net.push_back(matmul(matmul_pd));
  net_args.push_back({{DNNL_ARG_SRC, prim_src_memory},
                      {DNNL_ARG_WEIGHTS, prim_weights_memory},
                      {DNNL_ARG_BIAS, prim_bias_memory},
                      {DNNL_ARG_DST, prim_dst_memory}});

  // reorder or copy the output
  auto dst_memory =
      memory({{dst_dims}, memory::data_type::f32, memory::format_tag::abc}, eng,
             output);
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

} // namespace tpu_mlir
