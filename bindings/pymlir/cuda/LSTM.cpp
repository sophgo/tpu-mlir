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
#include <cublas_v2.h>

static cublasHandle_t getCublasHandle() {
  static cublasHandle_t handle = nullptr;
  if (!handle) cublasCreate(&handle);
  return handle;
}

void py_cuda::cudaLSTMOp(top::LSTMOp op) {
  auto in_shape = module::getShape(op.getInput());
  int seq_len = op.getBatchFirst() ? in_shape[1] : in_shape[0];
  int batch_size = op.getBatchFirst() ? in_shape[0] : in_shape[1];
  int input_size = in_shape[2];
  int hidden_size = op.getHiddenSize();
  int num_dir = op.getBidirectional() ? 2 : 1;
  bool batch_first = op.getBatchFirst();
  float *input_data = (float *)getCudaData(op.getInput());
  float *filter_data = (float *)getCudaData(op.getFilter());
  float *rec_data = (float *)getCudaData(op.getRecurrence());
  bool have_h0 = !module::isNone(op.getInitialH());
  bool have_c0 = !module::isNone(op.getInitialC());
  bool have_bias = !module::isNone(op.getBias());
  bool has_Y = !module::isNone(op.getY());
  bool has_Yh = !module::isNone(op.getYH());
  bool has_Yc = !module::isNone(op.getYC());
  float *Y_data = has_Y ? (float *)getCudaData(op.getY()) : nullptr;
  float *Yh_data = has_Yh ? (float *)getCudaData(op.getYH()) : nullptr;
  float *Yc_data = has_Yc ? (float *)getCudaData(op.getYC()) : nullptr;
  float *bias_data = have_bias ? (float *)getCudaData(op.getBias()) : nullptr;

  int total = batch_size * hidden_size;
  size_t gate_bytes = total * sizeof(float);
  size_t state_bytes = gate_bytes;
  auto all_buf = cuda_malloc(gate_bytes * 8 + state_bytes * 2);
  float *x_i = (float *)all_buf.get();
  float *x_o = x_i + total;
  float *x_f = x_o + total;
  float *x_c = x_f + total;
  float *h_i = x_c + total;
  float *h_o = h_i + total;
  float *h_f = h_o + total;
  float *h_c = h_f + total;
  float *cell_buf = h_c + total;
  float *hidden_buf = cell_buf + total;

  cublasHandle_t handle = getCublasHandle();
  float alpha = 1.0f, beta0 = 0.0f;

  for (int d = 0; d < num_dir; d++) {
    if (have_h0) {
      cudaMemcpy(hidden_buf,
                 (float *)getCudaData(op.getInitialH()) + d * total,
                 state_bytes, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemset(hidden_buf, 0, state_bytes);
    }
    if (have_c0) {
      cudaMemcpy(cell_buf,
                 (float *)getCudaData(op.getInitialC()) + d * total,
                 state_bytes, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemset(cell_buf, 0, state_bytes);
    }

    float *W_i = filter_data + d * 4 * hidden_size * input_size;
    float *W_o = W_i + hidden_size * input_size;
    float *W_f = W_o + hidden_size * input_size;
    float *W_c = W_f + hidden_size * input_size;
    float *R_i = rec_data + d * 4 * hidden_size * hidden_size;
    float *R_o = R_i + hidden_size * hidden_size;
    float *R_f = R_o + hidden_size * hidden_size;
    float *R_c = R_f + hidden_size * hidden_size;

    for (int t = 0; t < seq_len; t++) {
      int t_idx = (d == 1) ? (seq_len - 1 - t) : t;
      float *x_t;
      if (batch_first) {
        x_t = input_data + t_idx * batch_size * input_size;
      } else {
        x_t = input_data + t_idx * batch_size * input_size;
      }

      // input matmuls: x_i = x_t * W_i^T  (4 gates)
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, input_size,
                  &alpha, W_i, input_size, x_t, input_size,
                  &beta0, x_i, hidden_size);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, input_size,
                  &alpha, W_o, input_size, x_t, input_size,
                  &beta0, x_o, hidden_size);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, input_size,
                  &alpha, W_f, input_size, x_t, input_size,
                  &beta0, x_f, hidden_size);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, input_size,
                  &alpha, W_c, input_size, x_t, input_size,
                  &beta0, x_c, hidden_size);

      // hidden matmuls: h_i = hidden * R_i^T  (4 gates)
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, hidden_size,
                  &alpha, R_i, hidden_size, hidden_buf, hidden_size,
                  &beta0, h_i, hidden_size);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, hidden_size,
                  &alpha, R_o, hidden_size, hidden_buf, hidden_size,
                  &beta0, h_o, hidden_size);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, hidden_size,
                  &alpha, R_f, hidden_size, hidden_buf, hidden_size,
                  &beta0, h_f, hidden_size);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, hidden_size,
                  &alpha, R_c, hidden_size, hidden_buf, hidden_size,
                  &beta0, h_c, hidden_size);

      if (bias_data) {
        float *b = bias_data + d * 8 * hidden_size;
        cuda::bmLSTMAddBias(x_i, b, batch_size, hidden_size);
        cuda::bmLSTMAddBias(x_o, b + hidden_size, batch_size, hidden_size);
        cuda::bmLSTMAddBias(x_f, b + 2 * hidden_size, batch_size, hidden_size);
        cuda::bmLSTMAddBias(x_c, b + 3 * hidden_size, batch_size, hidden_size);
        cuda::bmLSTMAddBias(h_i, b + 4 * hidden_size, batch_size, hidden_size);
        cuda::bmLSTMAddBias(h_o, b + 5 * hidden_size, batch_size, hidden_size);
        cuda::bmLSTMAddBias(h_f, b + 6 * hidden_size, batch_size, hidden_size);
        cuda::bmLSTMAddBias(h_c, b + 7 * hidden_size, batch_size, hidden_size);
      }

      cuda::bmLSTMCell(x_i, x_o, x_f, x_c, h_i, h_o, h_f, h_c,
                         cell_buf, hidden_buf, total, 1.0f);

      if (has_Y && Y_data) {
        int y_offset;
        if (batch_first) {
          y_offset = (t_idx * num_dir + d) * hidden_size;
        } else {
          y_offset = (t_idx * num_dir + d) * batch_size * hidden_size;
        }
        cudaMemcpy(Y_data + y_offset, hidden_buf, state_bytes,
                   cudaMemcpyDeviceToDevice);
      }
    }

    if (has_Yh && Yh_data)
      cudaMemcpy(Yh_data + d * total, hidden_buf, state_bytes,
                 cudaMemcpyDeviceToDevice);
    if (has_Yc && Yc_data)
      cudaMemcpy(Yc_data + d * total, cell_buf, state_bytes,
                 cudaMemcpyDeviceToDevice);
  }
}

void py_cuda::cudaLSTMOp(tpu::LSTMOp op) {
  auto in_shape = module::getShape(op.getInput());
  int seq_len = op.getBatchFirst() ? in_shape[1] : in_shape[0];
  int batch_size = op.getBatchFirst() ? in_shape[0] : in_shape[1];
  int input_size = in_shape[2];
  int hidden_size = op.getHiddenSize();
  int num_dir = op.getBidirectional() ? 2 : 1;
  bool batch_first = op.getBatchFirst();
  float *input_data = (float *)getCudaData(op.getInput());
  float *filter_data = (float *)getCudaData(op.getFilter());
  float *rec_data = (float *)getCudaData(op.getRecurrence());
  bool have_h0 = !module::isNone(op.getInitialH());
  bool have_c0 = !module::isNone(op.getInitialC());
  bool have_bias = !module::isNone(op.getBias());
  bool has_Y = !module::isNone(op.getY());
  bool has_Yh = !module::isNone(op.getYH());
  bool has_Yc = !module::isNone(op.getYC());
  float *Y_data = has_Y ? (float *)getCudaData(op.getY()) : nullptr;
  float *Yh_data = has_Yh ? (float *)getCudaData(op.getYH()) : nullptr;
  float *Yc_data = has_Yc ? (float *)getCudaData(op.getYC()) : nullptr;
  float *bias_data = have_bias ? (float *)getCudaData(op.getBias()) : nullptr;

  int total = batch_size * hidden_size;
  size_t gate_bytes = total * sizeof(float);
  size_t state_bytes = gate_bytes;
  auto all_buf = cuda_malloc(gate_bytes * 8 + state_bytes * 2);
  float *x_i = (float *)all_buf.get();
  float *x_o = x_i + total;
  float *x_f = x_o + total;
  float *x_c = x_f + total;
  float *h_i = x_c + total;
  float *h_o = h_i + total;
  float *h_f = h_o + total;
  float *h_c = h_f + total;
  float *cell_buf = h_c + total;
  float *hidden_buf = cell_buf + total;

  cublasHandle_t handle = getCublasHandle();
  float alpha = 1.0f, beta0 = 0.0f;

  for (int d = 0; d < num_dir; d++) {
    if (have_h0) {
      cudaMemcpy(hidden_buf,
                 (float *)getCudaData(op.getInitialH()) + d * total,
                 state_bytes, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemset(hidden_buf, 0, state_bytes);
    }
    if (have_c0) {
      cudaMemcpy(cell_buf,
                 (float *)getCudaData(op.getInitialC()) + d * total,
                 state_bytes, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemset(cell_buf, 0, state_bytes);
    }

    float *W_i = filter_data + d * 4 * hidden_size * input_size;
    float *W_o = W_i + hidden_size * input_size;
    float *W_f = W_o + hidden_size * input_size;
    float *W_c = W_f + hidden_size * input_size;
    float *R_i = rec_data + d * 4 * hidden_size * hidden_size;
    float *R_o = R_i + hidden_size * hidden_size;
    float *R_f = R_o + hidden_size * hidden_size;
    float *R_c = R_f + hidden_size * hidden_size;

    for (int t = 0; t < seq_len; t++) {
      int t_idx = (d == 1) ? (seq_len - 1 - t) : t;
      float *x_t;
      if (batch_first) {
        x_t = input_data + t_idx * batch_size * input_size;
      } else {
        x_t = input_data + t_idx * batch_size * input_size;
      }

      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, input_size,
                  &alpha, W_i, input_size, x_t, input_size,
                  &beta0, x_i, hidden_size);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, input_size,
                  &alpha, W_o, input_size, x_t, input_size,
                  &beta0, x_o, hidden_size);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, input_size,
                  &alpha, W_f, input_size, x_t, input_size,
                  &beta0, x_f, hidden_size);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, input_size,
                  &alpha, W_c, input_size, x_t, input_size,
                  &beta0, x_c, hidden_size);

      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, hidden_size,
                  &alpha, R_i, hidden_size, hidden_buf, hidden_size,
                  &beta0, h_i, hidden_size);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, hidden_size,
                  &alpha, R_o, hidden_size, hidden_buf, hidden_size,
                  &beta0, h_o, hidden_size);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, hidden_size,
                  &alpha, R_f, hidden_size, hidden_buf, hidden_size,
                  &beta0, h_f, hidden_size);
      cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                  hidden_size, batch_size, hidden_size,
                  &alpha, R_c, hidden_size, hidden_buf, hidden_size,
                  &beta0, h_c, hidden_size);

      if (bias_data) {
        float *b = bias_data + d * 8 * hidden_size;
        cuda::bmLSTMAddBias(x_i, b, batch_size, hidden_size);
        cuda::bmLSTMAddBias(x_o, b + hidden_size, batch_size, hidden_size);
        cuda::bmLSTMAddBias(x_f, b + 2 * hidden_size, batch_size, hidden_size);
        cuda::bmLSTMAddBias(x_c, b + 3 * hidden_size, batch_size, hidden_size);
        cuda::bmLSTMAddBias(h_i, b + 4 * hidden_size, batch_size, hidden_size);
        cuda::bmLSTMAddBias(h_o, b + 5 * hidden_size, batch_size, hidden_size);
        cuda::bmLSTMAddBias(h_f, b + 6 * hidden_size, batch_size, hidden_size);
        cuda::bmLSTMAddBias(h_c, b + 7 * hidden_size, batch_size, hidden_size);
      }

      cuda::bmLSTMCell(x_i, x_o, x_f, x_c, h_i, h_o, h_f, h_c,
                         cell_buf, hidden_buf, total, 1.0f);

      if (has_Y && Y_data) {
        int y_offset;
        if (batch_first) {
          y_offset = (t_idx * num_dir + d) * hidden_size;
        } else {
          y_offset = (t_idx * num_dir + d) * batch_size * hidden_size;
        }
        cudaMemcpy(Y_data + y_offset, hidden_buf, state_bytes,
                   cudaMemcpyDeviceToDevice);
      }
    }

    if (has_Yh && Yh_data)
      cudaMemcpy(Yh_data + d * total, hidden_buf, state_bytes,
                 cudaMemcpyDeviceToDevice);
    if (has_Yc && Yc_data)
      cudaMemcpy(Yc_data + d * total, cell_buf, state_bytes,
                 cudaMemcpyDeviceToDevice);
  }
}
