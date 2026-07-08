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

void py_cuda::cudaGRUOp(top::GRUOp op) {
  auto in_shape = module::getShape(op.getInput());
  int seq_len = op.getBatchFirst() ? in_shape[1] : in_shape[0];
  int batch_size = op.getBatchFirst() ? in_shape[0] : in_shape[1];
  int input_size = in_shape[2];
  int hidden_size = op.getHiddenSize();
  int num_dir = op.getBidirectional() ? 2 : 1;
  bool batch_first = op.getBatchFirst();
  bool linear_before_reset = op.getLinearBeforeReset();

  float *input_data = (float *)getCudaData(op.getInput());
  float *filter_data = (float *)getCudaData(op.getFilter());
  float *rec_data = (float *)getCudaData(op.getRecurrence());
  float *bias_data = module::isNone(op.getBias()) ? nullptr
                     : (float *)getCudaData(op.getBias());
  float *h_init = module::isNone(op.getInitialH()) ? nullptr
                     : (float *)getCudaData(op.getInitialH());
  bool has_Y = !module::isNone(op.getY());
  bool has_Yh = !module::isNone(op.getYH());
  float *Y_data = has_Y ? (float *)getCudaData(op.getY()) : nullptr;
  float *Yh_data = has_Yh ? (float *)getCudaData(op.getYH()) : nullptr;

  int total = batch_size * hidden_size;
  size_t gate_bytes = total * sizeof(float);
  size_t h_bytes = gate_bytes;
  auto all_buf = cuda_malloc(gate_bytes * 6);
  float *x_gi = (float *)all_buf.get();
  float *x_gr = x_gi + total;
  float *x_gh = x_gr + total;
  float *h_gi = x_gh + total;
  float *h_gr = h_gi + total;
  float *h_gh = h_gr + total;

  for (int d = 0; d < num_dir; d++) {
    auto h_prev_buf = cuda_malloc(h_bytes);
    float *h_prev = (float *)h_prev_buf.get();

    if (h_init != nullptr) {
      cudaMemcpy(h_prev, h_init + d * total, h_bytes, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemset(h_prev, 0, h_bytes);
    }

    float *W_z = filter_data + d * 3 * hidden_size * input_size;
    float *W_r = W_z + hidden_size * input_size;
    float *W_h = W_r + hidden_size * input_size;
    float *R_z = rec_data + d * 3 * hidden_size * hidden_size;
    float *R_r = R_z + hidden_size * hidden_size;
    float *R_h = R_r + hidden_size * hidden_size;

    float *b_iz = nullptr;
    float *b_ir = nullptr;
    float *b_ih = nullptr;
    float *b_hz = nullptr;
    float *b_hr = nullptr;
    float *b_hh = nullptr;
    if (bias_data != nullptr) {
      float *bias = bias_data + d * 6 * hidden_size;
      b_iz = bias;
      b_ir = b_iz + hidden_size;
      b_ih = b_ir + hidden_size;
      b_hz = b_ih + hidden_size;
      b_hr = b_hz + hidden_size;
      b_hh = b_hr + hidden_size;
    }

    for (int t = 0; t < seq_len; t++) {
      int t_idx = (d == 1) ? (seq_len - 1 - t) : t;

      int stride = batch_size * input_size;
      float *x_t = input_data + t_idx * stride;

      cuda::mmF32(x_t, W_z, x_gi, batch_size, input_size, hidden_size, false, true);
      cuda::mmF32(x_t, W_r, x_gr, batch_size, input_size, hidden_size, false, true);
      cuda::mmF32(x_t, W_h, x_gh, batch_size, input_size, hidden_size, false, true);

      cuda::mmF32(h_prev, R_z, h_gi, batch_size, hidden_size, hidden_size, false, true);
      cuda::mmF32(h_prev, R_r, h_gr, batch_size, hidden_size, hidden_size, false, true);
      cuda::mmF32(h_prev, R_h, h_gh, batch_size, hidden_size, hidden_size, false, true);

      if (bias_data != nullptr) {
        cuda::addAxis(x_gi, b_iz, x_gi, 1, batch_size, hidden_size, cuda::DT_F32);
        cuda::addAxis(x_gr, b_ir, x_gr, 1, batch_size, hidden_size, cuda::DT_F32);
        cuda::addAxis(x_gh, b_ih, x_gh, 1, batch_size, hidden_size, cuda::DT_F32);
        cuda::addAxis(h_gi, b_hz, h_gi, 1, batch_size, hidden_size, cuda::DT_F32);
        cuda::addAxis(h_gr, b_hr, h_gr, 1, batch_size, hidden_size, cuda::DT_F32);
        cuda::addAxis(h_gh, b_hh, h_gh, 1, batch_size, hidden_size, cuda::DT_F32);
      }

      cuda::bmGruCell(x_gi, x_gr, x_gh, h_gi, h_gr, h_gh,
                       h_prev, h_prev, total, linear_before_reset);

      if (has_Y && Y_data != nullptr) {
        int y_offset;
        if (batch_first) {
          y_offset = (t_idx * batch_size + d * batch_size) * hidden_size;
        } else {
          y_offset = (t_idx * num_dir + d) * batch_size * hidden_size;
        }
        cudaMemcpy(Y_data + y_offset, h_prev, h_bytes, cudaMemcpyDeviceToDevice);
      }
    }

    if (has_Yh && Yh_data != nullptr) {
      cudaMemcpy(Yh_data + d * total, h_prev, h_bytes, cudaMemcpyDeviceToDevice);
    }
  }
}

void py_cuda::cudaGRUOp(tpu::GRUOp op) {
  auto in_shape = module::getShape(op.getInput());
  int seq_len = op.getBatchFirst() ? in_shape[1] : in_shape[0];
  int batch_size = op.getBatchFirst() ? in_shape[0] : in_shape[1];
  int input_size = in_shape[2];
  int hidden_size = op.getHiddenSize();
  int num_dir = op.getBidirectional() ? 2 : 1;
  bool batch_first = op.getBatchFirst();
  bool linear_before_reset = op.getLinearBeforeReset();

  float *input_data = (float *)getCudaData(op.getInput());
  float *filter_data = (float *)getCudaData(op.getFilter());
  float *rec_data = (float *)getCudaData(op.getRecurrence());
  float *bias_data = module::isNone(op.getBias()) ? nullptr
                     : (float *)getCudaData(op.getBias());
  float *h_init = module::isNone(op.getInitialH()) ? nullptr
                     : (float *)getCudaData(op.getInitialH());
  bool has_Y = !module::isNone(op.getY());
  bool has_Yh = !module::isNone(op.getYH());
  float *Y_data = has_Y ? (float *)getCudaData(op.getY()) : nullptr;
  float *Yh_data = has_Yh ? (float *)getCudaData(op.getYH()) : nullptr;

  int total = batch_size * hidden_size;
  size_t gate_bytes = total * sizeof(float);
  size_t h_bytes = gate_bytes;
  auto all_buf = cuda_malloc(gate_bytes * 6);
  float *x_gi = (float *)all_buf.get();
  float *x_gr = x_gi + total;
  float *x_gh = x_gr + total;
  float *h_gi = x_gh + total;
  float *h_gr = h_gi + total;
  float *h_gh = h_gr + total;

  for (int d = 0; d < num_dir; d++) {
    auto h_prev_buf = cuda_malloc(h_bytes);
    float *h_prev = (float *)h_prev_buf.get();

    if (h_init != nullptr) {
      cudaMemcpy(h_prev, h_init + d * total, h_bytes, cudaMemcpyDeviceToDevice);
    } else {
      cudaMemset(h_prev, 0, h_bytes);
    }

    float *W_z = filter_data + d * 3 * hidden_size * input_size;
    float *W_r = W_z + hidden_size * input_size;
    float *W_h = W_r + hidden_size * input_size;
    float *R_z = rec_data + d * 3 * hidden_size * hidden_size;
    float *R_r = R_z + hidden_size * hidden_size;
    float *R_h = R_r + hidden_size * hidden_size;

    float *b_iz = nullptr;
    float *b_ir = nullptr;
    float *b_ih = nullptr;
    float *b_hz = nullptr;
    float *b_hr = nullptr;
    float *b_hh = nullptr;
    if (bias_data != nullptr) {
      float *bias = bias_data + d * 6 * hidden_size;
      b_iz = bias;
      b_ir = b_iz + hidden_size;
      b_ih = b_ir + hidden_size;
      b_hz = b_ih + hidden_size;
      b_hr = b_hz + hidden_size;
      b_hh = b_hr + hidden_size;
    }

    for (int t = 0; t < seq_len; t++) {
      int t_idx = (d == 1) ? (seq_len - 1 - t) : t;
      int stride = batch_size * input_size;
      float *x_t = input_data + t_idx * stride;

      cuda::mmF32(x_t, W_z, x_gi, batch_size, input_size, hidden_size, false, true);
      cuda::mmF32(x_t, W_r, x_gr, batch_size, input_size, hidden_size, false, true);
      cuda::mmF32(x_t, W_h, x_gh, batch_size, input_size, hidden_size, false, true);

      cuda::mmF32(h_prev, R_z, h_gi, batch_size, hidden_size, hidden_size, false, true);
      cuda::mmF32(h_prev, R_r, h_gr, batch_size, hidden_size, hidden_size, false, true);
      cuda::mmF32(h_prev, R_h, h_gh, batch_size, hidden_size, hidden_size, false, true);

      if (bias_data != nullptr) {
        cuda::addAxis(x_gi, b_iz, x_gi, 1, batch_size, hidden_size, cuda::DT_F32);
        cuda::addAxis(x_gr, b_ir, x_gr, 1, batch_size, hidden_size, cuda::DT_F32);
        cuda::addAxis(x_gh, b_ih, x_gh, 1, batch_size, hidden_size, cuda::DT_F32);
        cuda::addAxis(h_gi, b_hz, h_gi, 1, batch_size, hidden_size, cuda::DT_F32);
        cuda::addAxis(h_gr, b_hr, h_gr, 1, batch_size, hidden_size, cuda::DT_F32);
        cuda::addAxis(h_gh, b_hh, h_gh, 1, batch_size, hidden_size, cuda::DT_F32);
      }

      cuda::bmGruCell(x_gi, x_gr, x_gh, h_gi, h_gr, h_gh,
                       h_prev, h_prev, total, linear_before_reset);

      if (has_Y && Y_data != nullptr) {
        int y_offset;
        if (batch_first) {
          y_offset = (t_idx * batch_size + d * batch_size) * hidden_size;
        } else {
          y_offset = (t_idx * num_dir + d) * batch_size * hidden_size;
        }
        cudaMemcpy(Y_data + y_offset, h_prev, h_bytes, cudaMemcpyDeviceToDevice);
      }
    }

    if (has_Yh && Yh_data != nullptr) {
      cudaMemcpy(Yh_data + d * total, h_prev, h_bytes, cudaMemcpyDeviceToDevice);
    }
  }
}
