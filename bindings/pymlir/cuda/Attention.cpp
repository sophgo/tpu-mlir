//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../pycuda.h"
#include "cuda_helper.h"

void py_cuda::cudaAttentionOp(top::AttentionOp op) {
  auto in_shape = module::getShape(op.getInput());
  auto key_shape = module::getShape(op.getKeys());
  int B = in_shape[0], M_q = in_shape[1], N_q = in_shape[2];
  int M_k = key_shape[1], N_k = key_shape[2];
  int head = op.getHead();
  int d = N_q / head;
  float scale = (float)(1.0 / std::sqrt((double)d));

  bool has_q_bias = !module::isNone(op.getQueriesBias());
  bool has_k_bias = !module::isNone(op.getKeysBias());
  bool has_v_bias = !module::isNone(op.getValuesBias());
  bool has_o_bias = !module::isNone(op.getOutBias());
  int Hd = head * d;

  // Q/K/V projections: [B*M, N] @ [N, Hd] → [B*M, Hd]
  auto Q_out = cuda_malloc(B * M_q * Hd * sizeof(float));
  cuda::mmF32(getCudaData(op.getInput()), getCudaData(op.getQueriesWeight()),
              Q_out.get(), false, B * M_q, N_q, Hd);
  if (has_q_bias)
    cuda::addAxis(Q_out.get(), getCudaData(op.getQueriesBias()),
                  Q_out.get(), B * M_q, Hd, 1, cuda::DT_F32);

  auto K_out = cuda_malloc(B * M_k * Hd * sizeof(float));
  cuda::mmF32(getCudaData(op.getKeys()), getCudaData(op.getKeysWeight()),
              K_out.get(), false, B * M_k, N_k, Hd);
  if (has_k_bias)
    cuda::addAxis(K_out.get(), getCudaData(op.getKeysBias()),
                  K_out.get(), B * M_k, Hd, 1, cuda::DT_F32);

  auto V_out = cuda_malloc(B * M_k * Hd * sizeof(float));
  cuda::mmF32(getCudaData(op.getValues()), getCudaData(op.getValuesWeight()),
              V_out.get(), false, B * M_k, N_k, Hd);
  if (has_v_bias)
    cuda::addAxis(V_out.get(), getCudaData(op.getValuesBias()),
                  V_out.get(), B * M_k, Hd, 1, cuda::DT_F32);

  // Permute [B, M, H, d] → [B, H, M, d]
  auto Q_perm = cuda_malloc(B * head * M_q * d * sizeof(float));
  auto K_perm = cuda_malloc(B * head * M_k * d * sizeof(float));
  auto V_perm = cuda_malloc(B * head * M_k * d * sizeof(float));
  cuda::bmPermuteBMHD(Q_out.get(), Q_perm.get(), B, M_q, head, d);
  cuda::bmPermuteBMHD(K_out.get(), K_perm.get(), B, M_k, head, d);
  cuda::bmPermuteBMHD(V_out.get(), V_perm.get(), B, M_k, head, d);
  Q_out.reset(); K_out.reset(); V_out.reset();

  // scores = Q @ K^T / scale → [B, H, M_q, M_k]
  auto scores = cuda_malloc(B * head * M_q * M_k * sizeof(float));
  cuda::bmAttentionQK(Q_perm.get(), K_perm.get(), scores.get(),
                       B, head, M_q, M_k, d, scale);

  // softmax
  cuda::bmSoftmax(scores.get(), nullptr, scores.get(),
                   B * head * M_q, M_k, 1, false);

  // context = scores @ V → [B, H, M_q, d]
  auto context = cuda_malloc(B * head * M_q * d * sizeof(float));
  cuda::bmAttentionPV(scores.get(), V_perm.get(), context.get(),
                       B, head, M_q, M_k, d);
  scores.reset(); K_perm.reset(); V_perm.reset();

  // Reverse permute [B, H, M_q, d] → [B, M_q, Hd]
  auto ctx_2d = cuda_malloc(B * M_q * Hd * sizeof(float));
  cuda::bmPermuteBMHD(context.get(), ctx_2d.get(), B, head, M_q, d);
  context.reset();

  // output = ctx_2d @ O_weight + O_bias
  auto out_shape = module::getShape(op.getOutput());
  int out_dim = out_shape[out_shape.size() - 1];
  auto out_tmp = cuda_malloc(B * M_q * out_dim * sizeof(float));
  cuda::mmF32(ctx_2d.get(), getCudaData(op.getOutWeight()),
              out_tmp.get(), false, B * M_q, Hd, out_dim);
  if (has_o_bias)
    cuda::addAxis(out_tmp.get(), getCudaData(op.getOutBias()),
                  out_tmp.get(), B * M_q, out_dim, 1, cuda::DT_F32);

  CHECK_CUDA(cudaMemcpy(getCudaData(op.getOutput()), out_tmp.get(),
                        B * M_q * out_dim * sizeof(float),
                        cudaMemcpyDeviceToDevice));
}
