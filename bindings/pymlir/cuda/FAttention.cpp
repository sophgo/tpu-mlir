#include "../pycuda.h"
#include "cuda_helper.h"


void py_cuda::cudaFAttentionOp(top::FAttentionOp op) {
  int batch = op.getBatch();
  int M_q = op.getMq();
  int M_k = op.getMk();
  int q_head = op.getQHead();
  int kv_head = op.getKvHead();
  int d = op.getDim();
  (void)kv_head; // GQA support pending for top FAttention.
  float scale = (float)op.getScale().convertToDouble();
  int Hd = q_head * d;

  auto scores = cuda_malloc(batch * q_head * M_q * M_k * sizeof(float));
  cuda::bmAttentionQK(getCudaData(op.getQueries()), getCudaData(op.getKeys()),
                      scores.get(), batch, q_head, M_q, M_k, d, scale);

  cuda::bmSoftmax(scores.get(), nullptr, scores.get(), batch * q_head * M_q,
                  M_k, 1, false);

  auto context = cuda_malloc(batch * q_head * M_q * d * sizeof(float));
  cuda::bmAttentionPV(scores.get(), getCudaData(op.getValues()), context.get(),
                      batch, q_head, M_q, M_k, d);
  scores.reset();

  auto ctx_perm = cuda_malloc(batch * q_head * M_q * d * sizeof(float));
  cuda::bmPermuteBMHD(context.get(), ctx_perm.get(), batch, q_head, M_q, d);
  context.reset();

  cudaMemcpy(getCudaData(op.getOutput()), ctx_perm.get(),
             batch * M_q * Hd * sizeof(float), cudaMemcpyDeviceToDevice);
}

void py_cuda::cudaFAttentionOp(tpu::FAttentionOp op) {
  auto out_type = module::getStorageType(op.getOutput());
  bool is_bf16 = out_type.isBF16();
  int batch = op.getBatch();
  int M_q = op.getMq();
  int M_k = op.getMk();
  uint64_t d = op.getDim();
  uint64_t q_head = op.getQHead();
  auto kv_head = op.getKvHead();
  float scale = op.getScale().convertToDouble();
  scale = is_bf16 ? scale : F16(scale);
  bool has_mask = !module::isNone(op.getMask());
  // Q * K
  if (out_type.isF32()) {
    void *Q = getCudaData(op.getQueries()); // batch * M_q * q_head * dim
    void *K = getCudaData(op.getKeys()); // batch * M_k * kv_head * dim
    void *V = getCudaData(op.getValues()); // batch * M_k * kv_head * dim
    void *mask = has_mask ? getCudaData(op.getMask()) : nullptr; // M_q * M_k
    void *output = getCudaData(op.getOutput()); // batch * M_q * (q_head * dim)
    cuda::GQA(Q, K, V, has_mask ? mask : nullptr, output, batch, M_q, M_k, q_head, kv_head, d, scale, is_bf16);
  } else {
    auto Q = newCudaData(op.getQueries(), cuda::DT_F32); // batch * M_q * q_head * dim
    auto K = newCudaData(op.getKeys(), cuda::DT_F32); // batch * M_k * kv_head * dim
    auto V = newCudaData(op.getValues(), cuda::DT_F32); // batch * M_k * kv_head * dim
    auto mask = has_mask ? newCudaData(op.getMask(), cuda::DT_F32) : nullptr; // M_q * M_k
    auto output = newCudaData(op.getOutput(), cuda::DT_F32); // batch * M_q * (q_head * dim)
    cuda::GQA(Q.get(), K.get(), V.get(), has_mask ? mask.get() : nullptr, output.get(),
                     batch, M_q, M_k, q_head, kv_head, d, scale, is_bf16);
    if (is_bf16) {
      cuda::convertType(output.get(), getCudaData(op.getOutput()),
                        batch * M_q * q_head * d, cuda::DT_F32, cuda::DT_BF16);
    } else {
      cuda::convertType(output.get(), getCudaData(op.getOutput()),
                        batch * M_q * q_head * d, cuda::DT_F32, cuda::DT_F16);
    }
  }
}