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
                        int64_t batch_low, int64_t K, int64_t N,
                        bool right_transpose) {
  int64_t weight_len = batch * batch_low * K * N;
  p_right = right;
  origin_right = right;
  if (right_zp != 0 || right_transpose) {
    right_after_init = std::make_shared<std::vector<float>>(weight_len);
    p_right = right_after_init->data();
    // tensor_sub_zp(right_after_init->data(), right, weight_len, right_zp);
    right_has_zp_ = right_zp != 0;
    right_transpose_ = right_transpose;
  }
}

void MatMul::input_init(float *input, int64_t input_zp, int64_t batch,
                        int64_t batch_low, int64_t M, int64_t K,
                        bool input_transpose) {
  int64_t input_len = batch * batch_low * K * M;
  p_input = input;
  origin_input = input;
  if (input_zp != 0 || input_transpose) {
    input_after_init = std::make_shared<std::vector<float>>(input_len);
    p_input = input_after_init->data();
    input_has_zp_ = input_zp != 0;
    input_transpose_ = input_transpose;
  }
}

void MatMul::output_init(float *output, int64_t batch, int64_t batch_low,
                         int64_t M, int64_t N, bool output_transpose) {
  int64_t output_len = batch * batch_low * M * N;
  origin_output = output;
  if (output_transpose) {
    output_after_trans = std::make_shared<std::vector<float>>(output_len);
    output_transpose_ = output_transpose;
  }
}

void MatMul::setup(float *left, float *right, float *bias, float *output,
                   int64_t batch, int64_t batch_low, int64_t M, int64_t K,
                   int64_t N, bool do_relu, double relu_limit, int64_t right_zp,
                   int64_t input_zp, bool right_transpose, bool input_transpose,
                   bool output_transpose, bool hdim_is_batch) {
  // printf("MatMul ldt:%ld, rdt:%ld, bdt:%ld, odt:%ld, rshift:%ld\n", ldt, rdt,
  // bdt, odt, rshift);
  memory::dims src_dims = {batch * batch_low, M, K};
  memory::dims weights_dims = {batch * batch_low, K, N};
  memory::dims bias_dims = {1, 1, N};
  memory::dims dst_dims = {batch * batch_low, M, N};
  right_init(right, right_zp, batch, batch_low, K, N, right_transpose);
  input_init(left, input_zp, batch, batch_low, M, K, input_transpose);
  output_init(output, batch, batch_low, M, N, output_transpose);
  batch_ = batch;
  batch_low_ = batch_low;
  M_ = M;
  N_ = N;
  K_ = K;
  right_zp_ = right_zp;
  input_zp_ = input_zp;
  hdim_is_batch_ = hdim_is_batch;
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

  auto src_float_memory =
      memory({{src_dims}, memory::data_type::f32, memory::format_tag::abc}, eng,
             p_input);
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
    net_args.push_back(
        {{DNNL_ARG_FROM, bias_float_memory}, {DNNL_ARG_TO, prim_bias_memory}});
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
  float *p_input_after = origin_input;
  float *p_right_after = origin_right;
  if (right_transpose_) {
    if (hdim_is_batch_) {
      tensor_hc_transpose(right_after_init->data(), origin_right, batch_, K_,
                          batch_low_, N_);
    } else {
      tensor_hw_transpose(right_after_init->data(), origin_right, 1, batch_, N_,
                          K_);
    }
    p_right_after = right_after_init->data();
  }
  if (input_transpose_) {
    if (hdim_is_batch_) {
      tensor_hc_transpose(input_after_init->data(), origin_input, batch_, M_,
                          batch_low_, K_);
    } else {
      tensor_hw_transpose(input_after_init->data(), origin_input, 1, batch_, K_,
                          M_);
    }
    p_input_after = input_after_init->data();
  }
  if (right_has_zp_) {
    int64_t weight_len = batch_ * batch_low_ * K_ * N_;
    tensor_sub_zp(right_after_init->data(), p_right_after, weight_len,
                  right_zp_);
  }
  if (input_has_zp_) {
    int64_t input_len = batch_ * batch_low_ * K_ * M_;
    tensor_sub_zp(input_after_init->data(), p_input_after, input_len,
                  input_zp_);
  }
  for (size_t i = 0; i < net.size(); ++i)
    net.at(i).execute(engine_stream, net_args.at(i));
  engine_stream.wait();
  if (output_transpose_) {
    if (hdim_is_batch_) {
      tensor_hc_transpose(output_after_trans->data(), origin_output, batch_,
                          batch_low_, M_, N_);
    } else {
      tensor_hw_transpose(output_after_trans->data(), origin_output, batch_,
                          batch_low_, M_, N_);
    }
    memcpy(origin_output, output_after_trans->data(), output_after_trans->size() * sizeof(float));
  }
}

} // namespace tpu_mlir
