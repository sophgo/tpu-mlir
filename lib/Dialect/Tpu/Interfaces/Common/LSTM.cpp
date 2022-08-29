//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

LogicalResult tpu::LSTMOp::init(InferenceParameter &p) { return success(); }
void tpu::LSTMOp::deinit(InferenceParameter &p) {}

static inline float sigmoid_(float x) {
  //return static_cast<float>(1.f / (1.f + std::exp(-x)));
  return 0.5 * tanh(0.5 * x) + 0.5;
}

static inline float tanh_(float x) {
  return tanh(x);
}

static void lstm_compute(InferenceParameter &p,
                         float *x_wi,
                         float *h_wi,
                         float *x_bi,
                         float *h,
                         float *c,
                         int num_layer,
                         int num_dir,
                         int seq_length,
                         int batch_size,
                         int input_size,
                         int hidden_size,
                         bool forward) {
  //(TODO) num_layers > 1
  //input += seq_length * batch * input_size * num_layer; //(TODO check!)
  float *input = p.inputs[0];
  float *output = p.outputs[0];
  if (!forward) {
    x_wi += 4 * input_size * hidden_size;
    x_bi += 2 * 4 * hidden_size;
    h_wi += 4 * hidden_size * hidden_size;
    //h_bi += 2 * 4 * hidden_size;
    h += batch_size * hidden_size;
    c += batch_size * hidden_size;
    output += batch_size * hidden_size;
  }
  float* x_wo = x_wi + input_size * hidden_size;
  float* x_wf = x_wo + input_size * hidden_size;
  float* x_wc = x_wf + input_size * hidden_size;
  float* x_bo = x_bi + hidden_size;
  float* x_bf = x_bo + hidden_size;
  float* x_bc = x_bf + hidden_size;
  float* h_wo = h_wi + hidden_size * hidden_size;
  float* h_wf = h_wo + hidden_size * hidden_size;
  float* h_wc = h_wf + hidden_size * hidden_size;
  float* h_bi = x_bi + 4 * hidden_size;
  //if (!forward) h_bi += 2 * 4 * hidden_size;
  float* h_bo = h_bi + hidden_size;
  float* h_bf = h_bo + hidden_size;
  float* h_bc = h_bf + hidden_size;

  std::vector<float> x_i(batch_size * hidden_size);
  std::vector<float> x_o(batch_size * hidden_size);
  std::vector<float> x_f(batch_size * hidden_size);
  std::vector<float> x_c(batch_size * hidden_size);
  std::vector<float> h_i(batch_size * hidden_size);
  std::vector<float> h_o(batch_size * hidden_size);
  std::vector<float> h_f(batch_size * hidden_size);
  std::vector<float> h_c(batch_size * hidden_size);
  std::vector<float> gi(batch_size * hidden_size);
  std::vector<float> go(batch_size * hidden_size);
  std::vector<float> gf(batch_size * hidden_size);
  std::vector<float> gc(batch_size * hidden_size);

  for (int s = 0; s < seq_length; s++) {
    // matrixmul
    // x op w_x : [batch, x_size] op [x_size, hidden_size] => [batch, hidden_size]
    // h op w_h : [batch, h_size] op [h_size, hidden_size] => [batch, hidden_size]
    // note: h_size = hidden_size?
    int seq_idx = forward ? s : (seq_length - s - 1);
    float *x = input + seq_idx * batch_size * input_size;

    dnnl_mm(x, x_wi, x_bi, x_i.data(), batch_size, input_size, hidden_size, false);
    dnnl_mm(x, x_wo, x_bo, x_o.data(), batch_size, input_size, hidden_size, false);
    dnnl_mm(x, x_wf, x_bf, x_f.data(), batch_size, input_size, hidden_size, false);
    dnnl_mm(x, x_wc, x_bc, x_c.data(), batch_size, input_size, hidden_size, false);

    dnnl_mm(h, h_wi, h_bi, h_i.data(), batch_size, hidden_size, hidden_size, false);
    dnnl_mm(h, h_wo, h_bo, h_o.data(), batch_size, hidden_size, hidden_size, false);
    dnnl_mm(h, h_wf, h_bf, h_f.data(), batch_size, hidden_size, hidden_size, false);
    dnnl_mm(h, h_wc, h_bc, h_c.data(), batch_size, hidden_size, hidden_size, false);

    for (int batch = 0; batch < batch_size; batch++) {
      float *xi = x_i.data() + batch * hidden_size;
      float *xo = x_o.data() + batch * hidden_size;
      float *xf = x_f.data() + batch * hidden_size;
      float *xc = x_c.data() + batch * hidden_size;
      float *hi = h_i.data() + batch * hidden_size;
      float *ho = h_o.data() + batch * hidden_size;
      float *hf = h_f.data() + batch * hidden_size;
      float *hc = h_c.data() + batch * hidden_size;
      float *cell_state = c + batch * hidden_size;
      float *hidden_state = output + (seq_idx * num_dir * batch_size + batch) * hidden_size;
      for (int i = 0; i < hidden_size; i++) {
        gi[i] = sigmoid_(xi[i] + hi[i]);
        go[i] = sigmoid_(xo[i] + ho[i]);
        gf[i] = sigmoid_(xf[i] + hf[i]);
        gc[i] = tanh_(xc[i] + hc[i]);
        cell_state[i] = gf[i] * cell_state[i] + gi[i] * gc[i];
        hidden_state[i] = go[i] * tanh_(cell_state[i]);
      }
    }
    h = output + seq_idx * num_dir * batch_size * hidden_size;
  }
}

LogicalResult tpu::LSTMOp::inference(InferenceParameter &p) {

  auto module = Module::getModuleOp(getOperation());
  int nInputs = p.inputs.size();//{input, W, R, (bias), (initial_h), (initial_c)}

  auto input_shape = Module::getShape(input());
  int64_t seq_length = batch_first() ? input_shape[1] : input_shape[0];
  int64_t batch_size = batch_first() ? input_shape[0] : input_shape[1];
  int64_t input_size = input_shape[2];
  auto w_shape = Module::getShape(filter());
  auto r_shape = Module::getShape(recurrence());
  int64_t hidden_size = r_shape[2];
  int64_t num_dir = w_shape[0];

  bool have_initial_h = !initial_h().getType().isa<NoneType>();
  bool have_initial_c = !initial_c().getType().isa<NoneType>();

  auto B = p.inputs[3];
  if (!have_bias())
    B = std::vector<float>(num_dir * 8 * hidden_size, 0.0f).data();
  auto initial_h = p.inputs[3];
  auto initial_c = p.inputs[4];
  if (have_bias()) {
    initial_h = p.inputs[4];
    initial_c = p.inputs[5];
  }
  if (!have_initial_h) {
    initial_h = std::vector<float>(num_dir * batch_size * hidden_size, 0.0f).data();
  }
  if (!have_initial_c) {
    initial_c = std::vector<float>(num_dir * batch_size * hidden_size, 0.0f).data();
  }

  for (int i = 0; i < num_layers(); i++) {
    lstm_compute(p, p.inputs[1], p.inputs[2], B, initial_h, initial_c, i, num_dir, seq_length, batch_size, input_size, hidden_size, true);
    if (bidirectional()) {
      lstm_compute(p, p.inputs[1], p.inputs[2], B, initial_h, initial_c, i, num_dir, seq_length, batch_size, input_size, hidden_size, false);
    }
  }

  return success();
}
