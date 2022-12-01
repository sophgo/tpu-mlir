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

tpu::LSTMOp::lstm_attr_t tpu::LSTMOp::parseParam() {
  lstm_attr_t attr = {0};
  auto in_shape = Module::getShape(input());
  assert(in_shape.size() == 3);
  if (batch_first()) {
    attr.batch_size = in_shape[0];
    attr.seq_len = in_shape[1];
    attr.batch_first = true;
  } else {
    attr.batch_size = in_shape[1];
    attr.seq_len = in_shape[0];
    attr.batch_first = false;
  }
  attr.num_direction = bidirectional() ? 2 : 1;
  attr.hidden_size = hidden_size();
  attr.input_size = in_shape[2];
  attr.have_bias = !bias().getType().isa<NoneType>();
  attr.have_h0 = !initial_h().getType().isa<NoneType>();
  attr.have_c0 = !initial_c().getType().isa<NoneType>();
  attr.output_y = !Y().getType().isa<NoneType>();
  attr.output_yh = !Y_h().getType().isa<NoneType>();
  attr.output_yc = !Y_c().getType().isa<NoneType>();
  return attr;
}

LogicalResult tpu::LSTMOp::init(InferenceParameter &p) { return success(); }
void tpu::LSTMOp::deinit(InferenceParameter &p) {}

static inline float sigmoid_(float x) {
  // return static_cast<float>(1.f / (1.f + std::exp(-x)));
  return 0.5 * tanh(0.5 * x) + 0.5;
}

static inline float tanh_(float x) { return tanh(x); }

static void lstm_compute(InferenceParameter &p,
                         const tpu::LSTMOp::lstm_attr_t &attr, float *bias,
                         float *h, float *c, bool forward) {
  //(TODO) num_layers > 1
  // input += seq_length * batch * input_size * num_layer; //(TODO check!)
  float *input = p.inputs[0];
  float *output = p.outputs[0];
  float *x_wi = p.inputs[1];
  float *h_wi = p.inputs[2];
  float *x_bi = bias;
  float *last_h = p.outputs[1]; // Y_h
  float *last_c = p.outputs[2]; // Y_c
  if (!forward) {
    x_wi += 4 * attr.input_size * attr.hidden_size;
    x_bi += 2 * 4 * attr.hidden_size;
    h_wi += 4 * attr.hidden_size * attr.hidden_size;
    // h_bi += 2 * 4 * attr.hidden_size;
    h += attr.batch_size * attr.hidden_size;
    c += attr.batch_size * attr.hidden_size;
    output += attr.batch_size * attr.hidden_size;
    last_h += attr.batch_size * attr.hidden_size;
    last_c += attr.batch_size * attr.hidden_size;
  }
  float *x_wo = x_wi + attr.input_size * attr.hidden_size;
  float *x_wf = x_wo + attr.input_size * attr.hidden_size;
  float *x_wc = x_wf + attr.input_size * attr.hidden_size;
  float *x_bo = x_bi + attr.hidden_size;
  float *x_bf = x_bo + attr.hidden_size;
  float *x_bc = x_bf + attr.hidden_size;
  float *h_wo = h_wi + attr.hidden_size * attr.hidden_size;
  float *h_wf = h_wo + attr.hidden_size * attr.hidden_size;
  float *h_wc = h_wf + attr.hidden_size * attr.hidden_size;
  float *h_bi = x_bi + 4 * attr.hidden_size;
  // if (!forward) h_bi += 2 * 4 * attr.hidden_size;
  float *h_bo = h_bi + attr.hidden_size;
  float *h_bf = h_bo + attr.hidden_size;
  float *h_bc = h_bf + attr.hidden_size;

  std::vector<float> x_i(attr.batch_size * attr.hidden_size);
  std::vector<float> x_o(attr.batch_size * attr.hidden_size);
  std::vector<float> x_f(attr.batch_size * attr.hidden_size);
  std::vector<float> x_c(attr.batch_size * attr.hidden_size);
  std::vector<float> h_i(attr.batch_size * attr.hidden_size);
  std::vector<float> h_o(attr.batch_size * attr.hidden_size);
  std::vector<float> h_f(attr.batch_size * attr.hidden_size);
  std::vector<float> h_c(attr.batch_size * attr.hidden_size);
  std::vector<float> gi(attr.batch_size * attr.hidden_size);
  std::vector<float> go(attr.batch_size * attr.hidden_size);
  std::vector<float> gf(attr.batch_size * attr.hidden_size);
  std::vector<float> gc(attr.batch_size * attr.hidden_size);

  for (int s = 0; s < attr.seq_len; s++) {
    // matrixmul
    // x op w_x : [batch, x_size] op [x_size, attr.hidden_size] => [batch,
    // attr.hidden_size] h op w_h : [batch, h_size] op [h_size,
    // attr.hidden_size] => [batch, attr.hidden_size] note: h_size =
    // attr.hidden_size?
    int seq_idx = forward ? s : (attr.seq_len - s - 1);
    float *x = input + seq_idx * attr.batch_size * attr.input_size;

    dnnl_mm(x, x_wi, x_bi, x_i.data(), attr.batch_size, attr.input_size,
            attr.hidden_size, false);
    dnnl_mm(x, x_wo, x_bo, x_o.data(), attr.batch_size, attr.input_size,
            attr.hidden_size, false);
    dnnl_mm(x, x_wf, x_bf, x_f.data(), attr.batch_size, attr.input_size,
            attr.hidden_size, false);
    dnnl_mm(x, x_wc, x_bc, x_c.data(), attr.batch_size, attr.input_size,
            attr.hidden_size, false);

    dnnl_mm(h, h_wi, h_bi, h_i.data(), attr.batch_size, attr.hidden_size,
            attr.hidden_size, false);
    dnnl_mm(h, h_wo, h_bo, h_o.data(), attr.batch_size, attr.hidden_size,
            attr.hidden_size, false);
    dnnl_mm(h, h_wf, h_bf, h_f.data(), attr.batch_size, attr.hidden_size,
            attr.hidden_size, false);
    dnnl_mm(h, h_wc, h_bc, h_c.data(), attr.batch_size, attr.hidden_size,
            attr.hidden_size, false);

    for (int batch = 0; batch < attr.batch_size; batch++) {
      float *xi = x_i.data() + batch * attr.hidden_size;
      float *xo = x_o.data() + batch * attr.hidden_size;
      float *xf = x_f.data() + batch * attr.hidden_size;
      float *xc = x_c.data() + batch * attr.hidden_size;
      float *hi = h_i.data() + batch * attr.hidden_size;
      float *ho = h_o.data() + batch * attr.hidden_size;
      float *hf = h_f.data() + batch * attr.hidden_size;
      float *hc = h_c.data() + batch * attr.hidden_size;
      float *cell_state = c + batch * attr.hidden_size;
      float *hidden_state = h + batch * attr.hidden_size;
      if (attr.output_y) {
        hidden_state =
            output + (seq_idx * attr.num_direction * attr.batch_size + batch) *
                         attr.hidden_size;
      }
      for (int i = 0; i < attr.hidden_size; i++) {
        gi[i] = sigmoid_(xi[i] + hi[i]);
        go[i] = sigmoid_(xo[i] + ho[i]);
        gf[i] = sigmoid_(xf[i] + hf[i]);
        gc[i] = tanh_(xc[i] + hc[i]);
        cell_state[i] = gf[i] * cell_state[i] + gi[i] * gc[i];
        hidden_state[i] = go[i] * tanh_(cell_state[i]);
      }
    }
    if (attr.output_y) {
      h = output +
          seq_idx * attr.num_direction * attr.batch_size * attr.hidden_size;
    }
  }
  if (attr.output_yh) {
    memcpy(last_h, h, attr.batch_size * attr.hidden_size * sizeof(float));
  }
  if (attr.output_yc) {
    memcpy(last_c, c, attr.batch_size * attr.hidden_size * sizeof(float));
  }
}

LogicalResult tpu::LSTMOp::inference(InferenceParameter &p) {

  auto module = Module::getModuleOp(getOperation());
  auto attr = parseParam();

  auto h0_buffer = std::make_shared<std::vector<float>>(
      attr.num_direction * attr.batch_size * attr.hidden_size, 0.0f);
  auto initial_h = h0_buffer->data();
  if (attr.have_h0) {
    memcpy(initial_h, p.inputs[4], h0_buffer->size() * sizeof(float));
  }

  auto c0_buffer = std::make_shared<std::vector<float>>(
      attr.num_direction * attr.batch_size * attr.hidden_size, 0.0f);
  auto initial_c = c0_buffer->data();
  if (attr.have_c0) {
    memcpy(initial_c, p.inputs[5], c0_buffer->size() * sizeof(float));
  }

  auto bias_buffer = std::make_shared<std::vector<float>>(
      attr.num_direction * 8 * attr.hidden_size, 0.0f);
  auto B = bias_buffer->data();
  if (attr.have_bias) {
    memcpy(B, p.inputs[3], bias_buffer->size() * sizeof(float));
  }

  lstm_compute(p, attr, B, initial_h, initial_c, true);
  if (attr.num_direction == 2) {
    lstm_compute(p, attr, B, initial_h, initial_c, false);
  }

  return success();
}
