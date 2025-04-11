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

void MatMul::dequant_weight(float *dq_weight, float *weight, float *scale,
                            float *zp, int weight_len, int q_group_size,
                            int weight_bits) {
  int compress_ratio = 8 / weight_bits;
  uint8_t mask = (weight_bits == 4) ? 0xF : 0xFF;
  for (int i = 0; i < weight_len; i++) {
    int quant_idx = i / q_group_size;
    auto zp_i = zp[quant_idx];
    auto scale_i = scale[quant_idx];
    for (int j = 0; j < compress_ratio; j++) {
      dq_weight[i + j] =
          (((((int)(weight[i / compress_ratio]) >> (weight_bits * j)) & mask) -
            (int)(zp_i)) *
           scale_i);
    }
    i += compress_ratio - 1;
  }
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
  if (need_broadcast_) {
    right_broadcasted = std::make_shared<std::vector<float>>(weight_len);
    p_right = right_broadcasted->data();
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
  if (need_broadcast_) {
    input_broadcasted = std::make_shared<std::vector<float>>(input_len);
    p_input = input_broadcasted->data();
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

/**[M, K] @ [K, N] => [M, N]*/
void MatMul::setup(float *left, float *right, float *bias, float *output,
                   int64_t batch, int64_t batch_low, int64_t M, int64_t K,
                   int64_t N, bool do_relu, double relu_limit, int64_t right_zp,
                   int64_t input_zp, bool right_transpose, bool input_transpose,
                   bool output_transpose, bool hdim_is_batch,
                   const std::vector<int64_t> &L_shape,
                   const std::vector<int64_t> &R_shape, int dims_merge_2_M) {
  // printf("MatMul ldt:%ld, rdt:%ld, bdt:%ld, odt:%ld, rshift:%ld\n", ldt, rdt,
  // bdt, odt, rshift);
  memory::dims src_dims = {batch * batch_low, M, K};
  memory::dims weights_dims = {batch * batch_low, K, N};
  if (L_shape.size() && R_shape.size()) {
    L_shape_ = L_shape;
    R_shape_ = R_shape;
    while (L_shape_.size() - 2 >= 0 && L_shape_[L_shape_.size() - 1] != K) {
      L_shape_[L_shape_.size() - 2] *= L_shape_[L_shape_.size() - 1];
      L_shape_.pop_back();
    }
    while (L_shape_.size() < 4) {
      L_shape_.insert(L_shape_.begin(), 1);
    }
    while (R_shape_.size() < 4) {
      R_shape_.insert(R_shape_.begin(), 1);
    }
    l_b = 1;
    r_b = 1;
    for (int i = 0; i < L_shape_.size() - 2; i++) {
      l_b *= L_shape_[i];
    }
    for (int i = 0; i < R_shape_.size() - 2; i++) {
      r_b *= R_shape_[i];
    }
    if (L_shape_.size() == R_shape_.size()) {
      for (int i = 0; i < R_shape_.size() - 2; i++) {
        if (L_shape_[i] != R_shape_[i] && R_shape.size() > 2 &&
            (L_shape_[i] == 1 || (R_shape_[i] == 1 && (R_shape_.size() - 2 -
                                                       dims_merge_2_M) > i))) {
          need_broadcast_ = true;
          if (L_shape_.size() > 4)
            llvm_unreachable(
                "Not support broadcast in MatMul of dims larger than 4");
          break;
        }
      }
    }
  }
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
  src_mem = memory({src_dims, memory::data_type::f32, tag::abc}, eng, p_input);
  weight_mem =
      memory({weights_dims, memory::data_type::f32, tag::abc}, eng, p_right);
  if (bias == nullptr) {
    bias0 = std::make_shared<std::vector<float>>(N_, 0);
    bias = bias0->data();
  }
  bias_mem = memory({bias_dims, memory::data_type::f32, tag::abc}, eng, bias);
  dst_mem = memory({dst_dims, memory::data_type::f32, tag::abc}, eng, output);
  primitive_attr relu_attr;
  post_relu(relu_attr, do_relu, relu_limit);
  auto matmul_pd = matmul::primitive_desc(
      eng, src_mem.get_desc(), weight_mem.get_desc(), bias_mem.get_desc(),
      dst_mem.get_desc(), relu_attr);
  prim = matmul(matmul_pd);
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
    p_right_after = right_after_init->data();
  }
  if (input_has_zp_) {
    int64_t input_len = batch_ * batch_low_ * K_ * M_;
    tensor_sub_zp(input_after_init->data(), p_input_after, input_len,
                  input_zp_);
    p_input_after = input_after_init->data();
  }

  if (need_broadcast_) {
    int l_length = K_ * M_;
    int r_length = K_ * N_;
    int max_n = std::max(L_shape_[0], R_shape_[0]);
    int max_c = std::max(L_shape_[1], R_shape_[1]);
    for (int i = 0; i < max_n; i++) {
      for (int j = 0; j < max_c; j++) {
        memcpy(input_broadcasted->data() + (i * max_c + j) * l_length,
               p_input_after +
                   ((i % L_shape_[0]) * L_shape_[1] + j % L_shape_[1]) *
                       l_length,
               l_length * sizeof(float));
        memcpy(right_broadcasted->data() + (i * max_c + j) * r_length,
               p_right_after +
                   ((i % R_shape_[0]) * R_shape_[1] + j % R_shape_[1]) *
                       r_length,
               r_length * sizeof(float));
      }
    }
  }
  prim.execute(engine_stream, {{DNNL_ARG_SRC, src_mem},
                               {DNNL_ARG_WEIGHTS, weight_mem},
                               {DNNL_ARG_BIAS, bias_mem},
                               {DNNL_ARG_DST, dst_mem}});
  engine_stream.wait();
  if (output_transpose_) {
    if (hdim_is_batch_) {
      tensor_hc_transpose(output_after_trans->data(), origin_output, batch_,
                          batch_low_, M_, N_);
    } else {
      tensor_hw_transpose(output_after_trans->data(), origin_output, batch_,
                          batch_low_, M_, N_);
    }
    memcpy(origin_output, output_after_trans->data(),
           output_after_trans->size() * sizeof(float));
  }
}

} // namespace tpu_mlir
