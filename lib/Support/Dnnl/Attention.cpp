//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Attention.h"
#include "tpu_mlir/Support/Dnnl/DnnlUtils.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Dnnl/MatMul.h"
#include "tpu_mlir/Support/Dnnl/Softmax.h"
#include "tpu_mlir/Support/Dnnl/Binary.h"
#include "tpu_mlir/Support/Float16.h"

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace tpu_mlir {

Attention::Attention() {
}

void Attention::setup(float *input, float *keys, float *values,
                      float *queries_weight, float *queries_bias, float *keys_weight,
                      float *keys_bias, float *values_weight, float *values_bias,
                      float *out_weight, float *out_bias, float *musk, float *table,
                      float *output, int64_t *quant_param,
                      int64_t batch, int64_t M_q, int64_t M_k, int64_t N_q, int64_t N_k,
                      int64_t d, float scale, bool add_result, int dtype) {
  // int64_t d = K / head;
  scale_ = scale;
  M_k_ = M_k;
  M_q_ = M_q;
  batch_ = batch;
  add_result_ = add_result;
  dtype_ = dtype;
  p_table = table;
  p_musk = musk;
  if (keys == nullptr) {
    keys = input;
  }
  if (values == nullptr) {
    values = keys;
  }
  if (quant_param != nullptr) {
    q_mul = quant_param[0];
    q_sft = quant_param[1];
    q_zp = quant_param[2];
    k_mul = quant_param[3];
    k_sft = quant_param[4];
    k_zp = quant_param[5];
    v_mul = quant_param[6];
    v_sft = quant_param[7];
    v_zp = quant_param[8];
    m0_mul = quant_param[9];
    m0_sft = quant_param[10];
    m0_zp = quant_param[11];
    m1_mul = quant_param[12];
    m1_sft = quant_param[13];
    m1_zp = quant_param[14];
    s_zp = quant_param[15];
  }
  // queries
  matmulq = new MatMul();
  q_data = std::make_shared<std::vector<float>>(batch * M_q * d);
  p_queries = q_data->data();
  ((MatMul *)matmulq)->setup(input, queries_weight, queries_bias, p_queries, 1, 1,
                             batch * M_q, N_q, d, 0, -1, 0, 0, 0, 0, 0, 0);
  // keys
  matmulk = new MatMul();
  k_data = std::make_shared<std::vector<float>>(batch * M_k * d);
  p_keys = k_data->data();
  ((MatMul *)matmulk)->setup(keys, keys_weight, keys_bias, p_keys, 1, 1,
                             batch * M_k, N_k, d, 0, -1, 0, 0, 0, 0, 0, 0);
  // values
  matmulv = new MatMul();
  v_data = std::make_shared<std::vector<float>>(batch * M_k * d);
  p_values = v_data->data();
  ((MatMul *)matmulv)->setup(values, values_weight, values_bias, p_values, 1, 1,
                             batch * M_k, N_k, d, 0, -1, 0, 0, 0, 0, 0, 0);
  // matmul0
  num_elem = batch * M_q * M_k;
  matmul0 = new MatMul();
  data_0 = std::make_shared<std::vector<float>>(num_elem);
  p_mat0 = data_0->data();
  ((MatMul *)matmul0)->setup(p_queries, p_keys, nullptr, p_mat0, batch, 1,
                             M_q, d, M_k, 0, -1, 0, 0, 1, 0, 0, 0);
  std::vector<int64_t> lshape = {batch, M_q, M_k};
  // binary
  if (musk != nullptr) {
    data_binary = std::make_shared<std::vector<float>>(num_elem);
    p_binary = data_binary->data();
    binary = new Binary();
    // std::vector<int64_t> lshape = {batch, M_q, M_k};
    std::vector<int64_t> rshape = {batch, 1, M_k};
    (*(Binary *)binary)
        .hs(p_mat0, musk, lshape, rshape)
        .dst(p_binary, lshape)
        .algorithem(algorithm::binary_add)
        .setup();
  } else {
    p_binary = p_mat0;
  }
  // // softmax
  softmax = new Softmax();
  softmax_attr_t attr;
  attr.src_shape = lshape;
  attr.dst_shape = lshape;
  attr.axis = 2;
  attr.log = 0;
  data_softmax = std::make_shared<std::vector<float>>(num_elem);
  p_softmax = data_softmax->data();
  ((Softmax *)softmax)->setup(p_binary, p_softmax, attr);
  // matmul1
  matmul1 = new MatMul();
  data_1 = std::make_shared<std::vector<float>>(batch * M_q * d);
  p_mat1 = data_1->data();
  ((MatMul *)matmul1)->setup(p_softmax, p_values, nullptr, p_mat1, batch, 1,
                             M_q, M_k, d, 0, -1, 0, 0, 0, 0, 0, 0);
  if (add_result) {
    data_out = std::make_shared<std::vector<float>>(batch * M_q * d);
    p_mat1_out = data_out->data();
    num_elem_out = batch * M_q * N_q;
  } else {
    p_mat1_out = output;
  }
  p_output = output;
  // matmul out
  matmul_out = new MatMul();
  ((MatMul *)matmul_out)->setup(p_mat1, out_weight, out_bias, p_mat1_out, 1, 1,
                                batch * M_q, d, N_q, 0, -1, 0, 0, 0, 0, 0, 0);
}

// type = {0:fp32, 1:fp16, 2:bf16, 3:int8}
void type_cast(float* data, int64_t num, int type) {
  if (type == 1) {
    F16(data, data, num);
  } else if (type == 2) {
    BF16(data, data, num);
  }
  return;
}

// type = {0:fp32, 1:fp16, 2:bf16, 3:int8}
void requant(float* data, int64_t num,
             int64_t mul=1, int rshift=0, int64_t zp=0, bool out_i16=false) {
#pragma omp parallel for schedule(static, omp_schedule(num))
  for (int64_t i = 0; i < num; ++i) {
    int64_t v = 0;
    v = RightShiftRound((int64_t)data[i] * mul, rshift, ROUNDING_HALF_AWAY_FROM_ZERO);
    if (out_i16) {
      data[i] = (v + zp);
    } else {
      data[i] = to_int8(v + zp);
    }
  }
  return;
}

const float MAX_VAL = 127;
void softmax_with_musk(float* output, float* input, float* exp_table, float* musk,
                       float scale, int zp, int n, int c, int w) {
  if (musk != nullptr) {
#pragma omp parallel for schedule(static, omp_schedule(w))
    for (int i = 0; i < w; ++i) {
      musk[i] = (float)(musk[i] != 0) * 255.f;
    }
#pragma omp parallel for schedule(static, omp_schedule(w))
    for (int i = 0; i < n*c; ++i) {
      for (int j = 0; j < w; ++j) {
        input[i * w + j] = to_int8(input[i * w + j] - musk[j]);
      }
    }
  }
#pragma omp parallel for schedule(static, omp_schedule(n*c))
  for (int i = 0; i < n*c; ++i) {
    int64_t out_offset = i * w;
    float sum = 0.f;
    for (int j = 0; j < w; ++j) {
      auto offset = to_uint8(MAX_VAL - input[out_offset + j]);
      sum += exp_table[offset];
    }
    for (int j = 0; j < w; ++j) {
      auto offset = to_uint8(MAX_VAL - input[out_offset + j]);
      float prob_rescaled = exp_table[offset];
      prob_rescaled = prob_rescaled / (sum * scale);
      int prob_rnd = static_cast<int32_t>(std::round(prob_rescaled));
      output[out_offset + j] = to_uint8(prob_rnd + zp);
    }
  }
}

void Attention::run() {
  int mode = dtype_;
  if (dtype_ < 3) {
    ((MatMul *)matmulq)->run();
    type_cast(p_queries, q_data->size(), mode);
    ((MatMul *)matmulk)->run();
    type_cast(p_keys, k_data->size(), mode);
    ((MatMul *)matmulv)->run();
    type_cast(p_values, v_data->size(), mode);
    ((MatMul *)matmul0)->run();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; i++) {
      p_mat0[i] *= scale_;
    }
    if (binary != nullptr) {
      ((Binary *)binary)->run();
    }
    ((Softmax *)softmax)->run();
    type_cast(p_softmax, data_softmax->size(), mode);
    ((MatMul *)matmul1)->run();
    type_cast(p_mat1, data_1->size(), mode);
    ((MatMul *)matmul_out)->run();
    type_cast(p_mat1_out, data_1->size(), mode);
    if (add_result_) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem_out))
      for (int64_t i = 0; i < num_elem_out; i++) {
        p_output[i] += p_mat1_out[i];
      }
      type_cast(p_output, data_1->size(), mode);
    }
  } else {
    ((MatMul *)matmulq)->run();
    requant(p_queries, q_data->size(), q_mul, q_sft, q_zp);
    ((MatMul *)matmulk)->run();
    requant(p_keys, k_data->size(), k_mul, k_sft, k_zp);
    ((MatMul *)matmulv)->run();
    requant(p_values, v_data->size(), v_mul, v_sft, v_zp);
    ((MatMul *)matmul0)->run();
    requant(p_mat0, data_0->size(), m0_mul, m0_sft, m0_zp);
    softmax_with_musk(p_softmax, p_mat0, p_table, p_musk, scale_, s_zp, batch_, M_q_, M_k_);
    ((MatMul *)matmul1)->run();
    requant(p_mat1, data_1->size(), m1_mul, m1_sft, m1_zp);
    ((MatMul *)matmul_out)->run();
  }
}

void Attention::deinit() {
  delete ((MatMul *)matmulq);
  delete ((MatMul *)matmulk);
  delete ((MatMul *)matmulv);
  delete ((MatMul *)matmul0);
  if (binary != nullptr)
    delete ((Binary *)binary);
  delete ((Softmax *)softmax);
  delete ((MatMul *)matmul1);
  delete ((MatMul *)matmul_out);
}

} // namespace tpu_mlir
