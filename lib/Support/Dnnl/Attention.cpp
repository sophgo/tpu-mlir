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

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

namespace tpu_mlir {

Attention::Attention() {
}

void Attention::setup(float *input, float *keys, float *values,
                      float *queries_weight, float *queries_bias, float *keys_weight,
                      float *keys_bias, float *values_weight, float *values_bias,
                      float *out_weight, float *out_bias, float *musk, float *output,
                      int64_t batch, int64_t M_q, int64_t M_k, int64_t K,
                      int64_t d, float scale, bool add_result) {
  // int64_t d = K / head;
  scale_ = scale;
  add_result_ = add_result;
  // queries
  matmulq = new MatMul();
  q_data = std::make_shared<std::vector<float>>(batch * M_q * d);
  p_queries = q_data->data();
  ((MatMul *)matmulq)->setup(input, queries_weight, queries_bias, p_queries, batch, 1,
                             M_q, K, d, 0, -1, 0, 0, 0, 0, 0, 0);
  // keys
  matmulk = new MatMul();
  k_data = std::make_shared<std::vector<float>>(batch * M_k * d);
  p_keys = k_data->data();
  ((MatMul *)matmulk)->setup(keys, keys_weight, keys_bias, p_keys, batch, 1,
                             M_k, K, d, 0, -1, 0, 0, 0, 0, 0, 0);
  // values
  matmulv = new MatMul();
  v_data = std::make_shared<std::vector<float>>(batch * M_k * d);
  p_values = v_data->data();
  ((MatMul *)matmulv)->setup(values, values_weight, values_bias, p_values, batch, 1,
                             M_k, K, d, 0, -1, 0, 0, 0, 0, 0, 0);
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
    std::vector<int64_t> rshape = {1, 1, M_k};
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
    num_elem_out = batch * M_q * K;
  } else {
    p_mat1_out = output;
  }
  p_output = output;
  // matmul out
  matmul_out = new MatMul();
  ((MatMul *)matmul_out)->setup(p_mat1, out_weight, out_bias, p_mat1_out, batch, 1,
                                M_q, d, K, 0, -1, 0, 0, 0, 0, 0, 0);
}

void Attention::run() {
  ((MatMul *)matmulq)->run();
  ((MatMul *)matmulk)->run();
  ((MatMul *)matmulv)->run();
  ((MatMul *)matmul0)->run();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int64_t i = 0; i < num_elem; i++) {
    p_mat0[i] *= scale_;
  }
  if (binary != nullptr)
    ((Binary *)binary)->run();
  ((Softmax *)softmax)->run();
  ((MatMul *)matmul1)->run();
  ((MatMul *)matmul_out)->run();
  if (add_result_) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem_out; i++) {
      p_output[i] += p_mat1_out[i];
    }
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
