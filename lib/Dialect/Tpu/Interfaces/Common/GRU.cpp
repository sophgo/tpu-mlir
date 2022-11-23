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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

tpu::GRUOp::gru_attr_t tpu::GRUOp::parseParam() {
  gru_attr_t attr = {0};
  auto in_shape = Module::getShape(input());
  auto out_shape = Module::getShape(Y());
  assert(in_shape.size() == 3);
  assert(out_shape.size() == 4);
  if (batch_first()) {
    attr.batch_size = in_shape[0];
    attr.seq_len = in_shape[1];
    attr.num_direction = out_shape[2];
    attr.batch_first = true;
  } else {
    attr.batch_size = in_shape[1];
    attr.seq_len = in_shape[0];
    attr.num_direction = out_shape[1];
    attr.batch_first = false;
  }
  attr.input_size = in_shape[2];
  attr.hidden_size = out_shape[3];
  attr.have_bias = !bias().getType().isa<NoneType>();
  attr.have_h0 = !initial_h().getType().isa<NoneType>();
  attr.output_h = !Y_h().getType().isa<NoneType>();
  return attr;
}

LogicalResult tpu::GRUOp::init(InferenceParameter &p) { return success(); }
void tpu::GRUOp::deinit(InferenceParameter &p) {}

static inline float sigmoid_(float x) {
  // return static_cast<float>(1.f / (1.f + std::exp(-x)));
  return 0.5 * tanh(0.5 * x) + 0.5;
}

static inline float tanh_(float x) { return tanh(x); }

static void gru_compute(InferenceParameter &p,
                        const tpu::GRUOp::gru_attr_t &attr, float *bias,
                        float *h, bool forward) {
  //(TODO) num_layers > 1
  // input += seq_length * batch * input_size * num_layer; //(TODO check!)
  float *input = p.inputs[0];
  float *output = p.outputs[0];
  float *x_wz = p.inputs[1];
  float *h_wz = p.inputs[2];
  float *x_bz = bias;
  float *last_h = p.outputs[1]; // Y_h
  if (!forward) {
    x_wz += 3 * attr.input_size * attr.hidden_size;
    x_bz += 2 * 3 * attr.hidden_size;
    h_wz += 3 * attr.hidden_size * attr.hidden_size;
    h += attr.batch_size * attr.hidden_size;
    output += attr.batch_size * attr.hidden_size;
    last_h += attr.batch_size * attr.hidden_size;
  }
  float *prev_hidden_state = h;
  float *x_wr = x_wz + attr.input_size * attr.hidden_size;
  float *x_wh = x_wr + attr.input_size * attr.hidden_size;

  float *x_br = x_bz + attr.hidden_size;
  float *x_bh = x_br + attr.hidden_size;

  float *h_wr = h_wz + attr.hidden_size * attr.hidden_size;
  float *h_wh = h_wr + attr.hidden_size * attr.hidden_size;

  float *h_bz = x_bz + 3 * attr.hidden_size;
  float *h_br = h_bz + attr.hidden_size;
  float *h_bh = h_br + attr.hidden_size;

  std::vector<float> x_z(attr.batch_size * attr.hidden_size);
  std::vector<float> x_r(attr.batch_size * attr.hidden_size);
  std::vector<float> x_h(attr.batch_size * attr.hidden_size);

  std::vector<float> h_z(attr.batch_size * attr.hidden_size);
  std::vector<float> h_r(attr.batch_size * attr.hidden_size);
  std::vector<float> h_h(attr.batch_size * attr.hidden_size);

  std::vector<float> g_z(attr.batch_size * attr.hidden_size);
  std::vector<float> g_r(attr.batch_size * attr.hidden_size);
  std::vector<float> g_h(attr.batch_size * attr.hidden_size);

  for (int s = 0; s < attr.seq_len; s++) {
    // matrixmul
    // x op w_x : [batch, x_size] op [x_size, attr.hidden_size] => [batch,
    // attr.hidden_size] h op w_h : [batch, h_size] op [h_size,
    // attr.hidden_size] => [batch, attr.hidden_size] note: h_size =
    // attr.hidden_size?
    int seq_idx = forward ? s : (attr.seq_len - s - 1);
    float *x = input + seq_idx * attr.batch_size * attr.input_size;

    dnnl_mm(x, x_wz, x_bz, x_z.data(), attr.batch_size, attr.input_size,
            attr.hidden_size, false);
    dnnl_mm(x, x_wr, x_br, x_r.data(), attr.batch_size, attr.input_size,
            attr.hidden_size, false);
    dnnl_mm(x, x_wh, x_bh, x_h.data(), attr.batch_size, attr.input_size,
            attr.hidden_size, false);

    dnnl_mm(prev_hidden_state, h_wz, h_bz, h_z.data(), attr.batch_size,
            attr.hidden_size, attr.hidden_size, false);
    dnnl_mm(prev_hidden_state, h_wr, h_br, h_r.data(), attr.batch_size,
            attr.hidden_size, attr.hidden_size, false);
    dnnl_mm(prev_hidden_state, h_wh, h_bh, h_h.data(), attr.batch_size,
            attr.hidden_size, attr.hidden_size, false);

    for (int batch = 0; batch < attr.batch_size; batch++) {
      float *xz = x_z.data() + batch * attr.hidden_size;
      float *xr = x_r.data() + batch * attr.hidden_size;
      float *xh = x_h.data() + batch * attr.hidden_size;

      float *hz = h_z.data() + batch * attr.hidden_size;
      float *hr = h_r.data() + batch * attr.hidden_size;
      float *hh = h_h.data() + batch * attr.hidden_size;
      float *pre_state = prev_hidden_state + batch * attr.hidden_size;
      float *hidden_state =
          output + (seq_idx * attr.num_direction * attr.batch_size + batch) *
                       attr.hidden_size;
      for (int i = 0; i < attr.hidden_size; i++) {
        hz[i] = sigmoid_(hz[i] + xz[i]);
        hr[i] = sigmoid_(hr[i] + xr[i]);
        hh[i] = tanh_(hr[i] * hh[i] + xh[i]);
        hidden_state[i] = (1 - hz[i]) * hh[i] + hz[i] * pre_state[i];
      }
    }
    prev_hidden_state = output + seq_idx * attr.num_direction *
                                     attr.batch_size * attr.hidden_size;
  }
  if (attr.output_h) {
    memcpy(last_h, prev_hidden_state,
           attr.batch_size * attr.hidden_size * sizeof(float));
  }
}

LogicalResult tpu::GRUOp::inference(InferenceParameter &p) {

  auto module = Module::getModuleOp(getOperation());
  auto attr = parseParam();

  auto h0_buffer = std::make_shared<std::vector<float>>(
      attr.num_direction * attr.batch_size * attr.hidden_size, 0.0f);
  auto initial_h = h0_buffer->data();
  if (attr.have_h0) {
    memcpy(initial_h, p.inputs[4], h0_buffer->size() * sizeof(float));
  }

  auto bias_buffer = std::make_shared<std::vector<float>>(
      attr.num_direction * 6 * attr.hidden_size, 0.0f);
  auto B = bias_buffer->data();
  if (attr.have_bias) {
    memcpy(B, p.inputs[3], bias_buffer->size() * sizeof(float));
  }

  gru_compute(p, attr, B, initial_h, true);
  if (attr.num_direction == 2) {
    gru_compute(p, attr, B, initial_h, false);
  }

  return success();
}
