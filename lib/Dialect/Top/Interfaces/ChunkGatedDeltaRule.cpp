//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

// Helper: simple matrix multiply C[M,N] = A[M,K] * B[K,N]
static void matmul(const float *A, const float *B, float *C, int M, int K,
                   int N) {
  std::memset(C, 0, M * N * sizeof(float));
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      float a_val = A[i * K + k];
      for (int j = 0; j < N; j++) {
        C[i * N + j] += a_val * B[k * N + j];
      }
    }
  }
}

// Helper: matrix multiply C[M,N] = A[M,K] * B^T[N,K]
static void matmul_nt(const float *A, const float *B, float *C, int M, int K,
                      int N) {
  std::memset(C, 0, M * N * sizeof(float));
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += A[i * K + k] * B[j * K + k];
      }
      C[i * N + j] = sum;
    }
  }
}

int64_t top::ChunkGatedDeltaRuleOp::getFLOPs() {
  // query/key/value shape: [B, S, H, D] (before internal transpose to [B, H, S,
  // D]) recurrent_state shape: [B, H, D, D]
  auto v_shape = module::getShape(getValue());
  int64_t B = v_shape[0];
  int64_t S = v_shape[1];
  int64_t H = v_shape[2];
  int64_t D = v_shape[3];
  int64_t cs = getChunkSize();
  // pad S to be divisible by chunk_size
  int64_t S_pad = ((S + cs - 1) / cs) * cs;
  int64_t nc = S_pad / cs;

  // MatMul FLOPs (each matmul of [M,K]*[K,N] = 2*M*K*N):
  //
  // Non-loop matmuls (over all chunks):
  // 1. k_beta @ key^T:          (B,H,nc,cs,D) x (B,H,nc,D,cs) =>
  // B*H*nc*cs^2*D*2
  // 2. attn @ v_beta:            (B,H,nc,cs,cs) x (B,H,nc,cs,D) =>
  // B*H*nc*cs^2*D*2
  // 3. attn @ (k_beta*g_exp):    (B,H,nc,cs,cs) x (B,H,nc,cs,D) =>
  // B*H*nc*cs^2*D*2
  // 4. query @ key^T:            (B,H,nc,cs,D) x (B,H,nc,D,cs) =>
  // B*H*nc*cs^2*D*2
  //
  // Loop matmuls (nc iterations, summed):
  // 5. k_cumdecay_i @ state:     (B,H,cs,D) x (B,H,D,D) => B*H*nc*cs*D^2*2
  // 6. q_g_i @ state:            (B,H,cs,D) x (B,H,D,D) => B*H*nc*cs*D^2*2
  // 7. intra_chunk_attn_i @ v_new:(B,H,cs,cs) x (B,H,cs,D) => B*H*nc*cs^2*D*2
  // 8. k_g_diff_t_i @ v_new:     (B,H,D,cs) x (B,H,cs,D) => B*H*nc*cs*D^2*2

  int64_t common = 2 * B * H * nc;
  // #1 + #2 + #3 + #4 + #7: 5 * cs^2 * D
  int64_t chunk_matmul_flops = common * cs * cs * D * 5;
  // #5 + #6 + #8: 3 * cs * D^2
  int64_t recurrent_matmul_flops = common * cs * D * D * 3;

  return chunk_matmul_flops + recurrent_matmul_flops;
}

LogicalResult top::ChunkGatedDeltaRuleOp::init(InferenceParameter &p) {
  return success();
}
void top::ChunkGatedDeltaRuleOp::deinit(InferenceParameter &p) {}

LogicalResult top::ChunkGatedDeltaRuleOp::inference(InferenceParameter &p) {
  // Input order from .td:
  // 0:query, 1:key, 2:value, 3:g, 4:beta, 5:recurrent_state,
  // 6:triu_mask, 7:strict_triu_mask, 8:tril_mask, 9:eye
  // Output: 0:attn_out

  // --- Extract shapes and attributes ---
  // value shape: [B, S, H, D]
  auto v_shape = module::getShape(getValue());
  auto q_shape = module::getShape(getQuery());

  const int64_t B = v_shape[0];
  const int64_t S = v_shape[1];
  const int64_t num_v_head = v_shape[2];
  const int64_t v_head_dim = v_shape[3];
  const int64_t num_qk_head = q_shape[2];
  const int64_t k_head_dim = q_shape[3];
  const int64_t num_qk_group = num_v_head / num_qk_head;
  const int64_t cs = getChunkSize();
  const bool use_l2norm = getUseQkL2norm();
  const double scale_val = getScale().convertToDouble();

  const int64_t pad_size = (cs - S % cs) % cs;
  const int64_t S_pad = S + pad_size;
  const int64_t nc = S_pad / cs;

  float *in_query = p.inputs[0];
  float *in_key = p.inputs[1];
  float *in_value = p.inputs[2];
  float *in_g = p.inputs[3];
  float *in_beta = p.inputs[4];
  float *in_recurrent_state = p.inputs[5];
  float *out_attn = p.outputs[0];

  const int64_t H = num_v_head;
  const float scale = static_cast<float>(scale_val);

  // --- L2 norm (optional) ---
  std::vector<float> query_raw(in_query,
                               in_query + B * S * num_qk_head * k_head_dim);
  std::vector<float> key_raw(in_key, in_key + B * S * num_qk_head * k_head_dim);
  if (use_l2norm) {
    auto l2norm_inplace = [](float *data, int64_t n, int64_t dim) {
      for (int64_t i = 0; i < n; i++) {
        float sum_sq = 0;
        for (int64_t d = 0; d < dim; d++)
          sum_sq += data[i * dim + d] * data[i * dim + d];
        float inv_norm = 1.0f / std::sqrt(sum_sq + 1e-6f);
        for (int64_t d = 0; d < dim; d++)
          data[i * dim + d] *= inv_norm;
      }
    };
    l2norm_inplace(query_raw.data(), B * S * num_qk_head, k_head_dim);
    l2norm_inplace(key_raw.data(), B * S * num_qk_head, k_head_dim);
  }

  // --- Transpose [B,S,H,D] -> [B,H,S_pad,D] with GQA repeat and padding ---
  std::vector<float> query_t(B * H * S_pad * k_head_dim, 0.0f);
  std::vector<float> key_t(B * H * S_pad * k_head_dim, 0.0f);
  std::vector<float> value_t(B * H * S_pad * v_head_dim, 0.0f);
  std::vector<float> g_t(B * H * S_pad, 0.0f);
  std::vector<float> beta_t(B * H * S_pad, 0.0f);

  for (int64_t b = 0; b < B; b++) {
    for (int64_t s = 0; s < S; s++) {
      // Transpose query/key with GQA repeat
      for (int64_t qh = 0; qh < num_qk_head; qh++) {
        int64_t src = ((b * S + s) * num_qk_head + qh) * k_head_dim;
        for (int64_t grp = 0; grp < num_qk_group; grp++) {
          int64_t vh = qh * num_qk_group + grp;
          int64_t dst = ((b * H + vh) * S_pad + s) * k_head_dim;
          std::memcpy(&query_t[dst], &query_raw[src],
                      k_head_dim * sizeof(float));
          std::memcpy(&key_t[dst], &key_raw[src], k_head_dim * sizeof(float));
        }
      }
      // Transpose value, g, beta
      for (int64_t h = 0; h < H; h++) {
        std::memcpy(&value_t[((b * H + h) * S_pad + s) * v_head_dim],
                    &in_value[((b * S + s) * H + h) * v_head_dim],
                    v_head_dim * sizeof(float));
        g_t[(b * H + h) * S_pad + s] = in_g[(b * S + s) * H + h];
        beta_t[(b * H + h) * S_pad + s] = in_beta[(b * S + s) * H + h];
      }
    }
  }

  // --- Scale query ---
  for (int64_t i = 0; i < B * H * S_pad * k_head_dim; i++)
    query_t[i] *= scale;

  // --- Output buffer [B, H, S_pad, v_head_dim] ---
  std::vector<float> output_t(B * H * S_pad * v_head_dim, 0.0f);

  // --- Process per (b, h) ---
  for (int64_t b = 0; b < B; b++) {
    for (int64_t h = 0; h < H; h++) {
      int64_t bh = b * H + h;
      float *q = &query_t[bh * S_pad * k_head_dim];
      float *k = &key_t[bh * S_pad * k_head_dim];
      float *v = &value_t[bh * S_pad * v_head_dim];
      float *gp = &g_t[bh * S_pad];
      float *bp = &beta_t[bh * S_pad];

      // k_beta = key * beta, v_beta = value * beta
      std::vector<float> k_beta(S_pad * k_head_dim);
      std::vector<float> v_beta(S_pad * v_head_dim);
      for (int64_t s = 0; s < S_pad; s++) {
        float bv = bp[s];
        for (int64_t d = 0; d < k_head_dim; d++)
          k_beta[s * k_head_dim + d] = k[s * k_head_dim + d] * bv;
        for (int64_t d = 0; d < v_head_dim; d++)
          v_beta[s * v_head_dim + d] = v[s * v_head_dim + d] * bv;
      }

      // g cumsum per chunk
      std::vector<float> gc(S_pad);
      for (int64_t c = 0; c < nc; c++) {
        float cumsum = 0;
        for (int64_t i = 0; i < cs; i++) {
          cumsum += gp[c * cs + i];
          gc[c * cs + i] = cumsum;
        }
      }

      // Pre-allocate per-chunk result buffers
      std::vector<float> val_chunks(nc * cs * v_head_dim);
      std::vector<float> kcd_chunks(nc * cs * k_head_dim);
      std::vector<float> ia_chunks(nc * cs * cs);
      std::vector<float> qg_chunks(nc * cs * k_head_dim);
      std::vector<float> kgdt_chunks(nc * k_head_dim * cs);
      std::vector<float> gle_chunks(nc);

      // Temp buffers reused across chunks
      std::vector<float> decay(cs * cs);
      std::vector<float> kb_kt(cs * cs);
      std::vector<float> attn_mat(cs * cs);
      std::vector<float> kb_g(cs * k_head_dim);
      std::vector<float> qkt(cs * cs);
      std::vector<float> row_buf(cs);

      for (int64_t c = 0; c < nc; c++) {
        int64_t co = c * cs;
        float *q_c = q + co * k_head_dim;
        float *k_c = k + co * k_head_dim;
        float *kb_c = k_beta.data() + co * k_head_dim;
        float *vb_c = v_beta.data() + co * v_head_dim;
        float *gc_c = gc.data() + co;

        // decay_mask[i][j] = exp(gc[i] - gc[j]) for i >= j, else 0
        std::memset(decay.data(), 0, cs * cs * sizeof(float));
        for (int64_t i = 0; i < cs; i++)
          for (int64_t j = 0; j <= i; j++)
            decay[i * cs + j] = std::exp(gc_c[i] - gc_c[j]);

        // attn = -(k_beta @ key^T * decay), zero on upper triangle + diagonal
        matmul_nt(kb_c, k_c, kb_kt.data(), cs, k_head_dim, cs);
        std::memset(attn_mat.data(), 0, cs * cs * sizeof(float));
        for (int64_t i = 1; i < cs; i++)
          for (int64_t j = 0; j < i; j++)
            attn_mat[i * cs + j] = -(kb_kt[i * cs + j] * decay[i * cs + j]);

        // Forward substitution: (I - L)^{-1}
        for (int64_t i = 1; i < cs; i++) {
          for (int64_t j = 0; j < i; j++)
            row_buf[j] = attn_mat[i * cs + j];
          for (int64_t j = 0; j < i; j++) {
            float corr = 0;
            for (int64_t kk = 0; kk < i; kk++)
              corr += row_buf[kk] * attn_mat[kk * cs + j];
            attn_mat[i * cs + j] = row_buf[j] + corr;
          }
        }

        // + I
        for (int64_t i = 0; i < cs; i++)
          attn_mat[i * cs + i] += 1.0f;

        // val_chunks[c] = attn @ v_beta
        matmul(attn_mat.data(), vb_c, val_chunks.data() + c * cs * v_head_dim,
               cs, cs, v_head_dim);

        // kb_g = k_beta * exp(gc)
        for (int64_t i = 0; i < cs; i++) {
          float eg = std::exp(gc_c[i]);
          for (int64_t d = 0; d < k_head_dim; d++)
            kb_g[i * k_head_dim + d] = kb_c[i * k_head_dim + d] * eg;
        }

        // kcd_chunks[c] = attn @ kb_g
        matmul(attn_mat.data(), kb_g.data(),
               kcd_chunks.data() + c * cs * k_head_dim, cs, cs, k_head_dim);

        // intra_chunk_attn = (q @ k^T * decay), zero strict upper triangle
        matmul_nt(q_c, k_c, qkt.data(), cs, k_head_dim, cs);
        float *ia = ia_chunks.data() + c * cs * cs;
        for (int64_t i = 0; i < cs; i++) {
          for (int64_t j = 0; j <= i; j++)
            ia[i * cs + j] = qkt[i * cs + j] * decay[i * cs + j];
          for (int64_t j = i + 1; j < cs; j++)
            ia[i * cs + j] = 0.0f;
        }

        // q_g = query * exp(gc)
        float *qg = qg_chunks.data() + c * cs * k_head_dim;
        for (int64_t i = 0; i < cs; i++) {
          float eg = std::exp(gc_c[i]);
          for (int64_t d = 0; d < k_head_dim; d++)
            qg[i * k_head_dim + d] = q_c[i * k_head_dim + d] * eg;
        }

        // g_last_exp = exp(gc[cs-1])
        gle_chunks[c] = std::exp(gc_c[cs - 1]);

        // k_g_diff_t = (key * exp(g_last - gc))^T : [k_head_dim, cs]
        float g_last = gc_c[cs - 1];
        float *kgdt = kgdt_chunks.data() + c * k_head_dim * cs;
        for (int64_t i = 0; i < cs; i++) {
          float eg = std::exp(g_last - gc_c[i]);
          for (int64_t d = 0; d < k_head_dim; d++)
            kgdt[d * cs + i] = k_c[i * k_head_dim + d] * eg;
        }
      }

      // --- Recurrent loop over chunks ---
      std::vector<float> state(k_head_dim * v_head_dim);
      std::memcpy(state.data(),
                  in_recurrent_state + bh * k_head_dim * v_head_dim,
                  k_head_dim * v_head_dim * sizeof(float));

      float *out_bh = output_t.data() + bh * S_pad * v_head_dim;
      std::vector<float> v_prime(cs * v_head_dim);
      std::vector<float> v_new(cs * v_head_dim);
      std::vector<float> attn_inter(cs * v_head_dim);
      std::vector<float> intra_v(cs * v_head_dim);
      std::vector<float> state_upd(k_head_dim * v_head_dim);

      for (int64_t c = 0; c < nc; c++) {
        float *vc = val_chunks.data() + c * cs * v_head_dim;
        float *kcd = kcd_chunks.data() + c * cs * k_head_dim;
        float *qg = qg_chunks.data() + c * cs * k_head_dim;
        float *ia = ia_chunks.data() + c * cs * cs;
        float *kgdt = kgdt_chunks.data() + c * k_head_dim * cs;

        // v_prime = k_cumdecay[c] @ state
        matmul(kcd, state.data(), v_prime.data(), cs, k_head_dim, v_head_dim);

        // v_new = value[c] - v_prime
        for (int64_t i = 0; i < cs * v_head_dim; i++)
          v_new[i] = vc[i] - v_prime[i];

        // attn_inter = q_g[c] @ state
        matmul(qg, state.data(), attn_inter.data(), cs, k_head_dim, v_head_dim);

        // intra_v = intra_chunk_attn[c] @ v_new
        matmul(ia, v_new.data(), intra_v.data(), cs, cs, v_head_dim);

        // output[c] = attn_inter + intra_v
        float *out_c = out_bh + c * cs * v_head_dim;
        for (int64_t i = 0; i < cs * v_head_dim; i++)
          out_c[i] = attn_inter[i] + intra_v[i];

        // state = state * g_last_exp + k_g_diff_t[c] @ v_new
        matmul(kgdt, v_new.data(), state_upd.data(), k_head_dim, cs,
               v_head_dim);
        float gle = gle_chunks[c];
        for (int64_t i = 0; i < k_head_dim * v_head_dim; i++)
          state[i] = state[i] * gle + state_upd[i];
      }
    }
  }

  // --- Transpose output [B,H,S_pad,D] -> [B,S,H,D] and remove padding ---
  for (int64_t b = 0; b < B; b++)
    for (int64_t s = 0; s < S; s++)
      for (int64_t h = 0; h < H; h++)
        std::memcpy(&out_attn[((b * S + s) * H + h) * v_head_dim],
                    &output_t[((b * H + h) * S_pad + s) * v_head_dim],
                    v_head_dim * sizeof(float));

  return success();
}

// if keep_dims, output shape = input shape
// else input = [1, M_q, q_head, d], output = [1, M_q, q_head*d]
void top::ChunkGatedDeltaRuleOp::shape_inference() {
  auto value = getValue();
  auto v_shape = module::getShape(value);
  module::setShapeOrVerify(getAttnOut(), v_shape);
}
