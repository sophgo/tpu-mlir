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
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/LutFunc.h"
#include "tpu_mlir/Support/MathUtils.h"



static float sigmoid_(float data, InferenceParameter &p) {
  float var = BF16(data);
  bf16_lut_slope(&var, &var, 1, p.inputs[5], p.inputs[6], -12, 12);
  return var;
}

static float tanh_(float data, InferenceParameter &p) {
  float var = BF16(data);
  bf16_lut_slope(&var, &var, 1, p.inputs[7], p.inputs[8], -15, 15);
  return var;
}

lstm_attr_t tpu::LSTMCVIOp::parseParam() {
  lstm_attr_t attr = {0};
  auto r_shape = module::getShape(recurrence());
  auto in_shape = module::getShape(input());
  assert(in_shape.size() == 3);
  attr.batch_size = in_shape[1];
  attr.seq_len = in_shape[0];
  attr.input_size = in_shape[2];
  attr.batch_first = false;
  attr.num_direction = bidirectional() ? 2 : 1;
  attr.hidden_size = r_shape[2];
  attr.have_bias = !bias().getType().isa<mlir::NoneType>();
  attr.have_h0 = !initial_h().getType().isa<mlir::NoneType>();
  attr.have_c0 = !initial_c().getType().isa<mlir::NoneType>();
  attr.output_y = !Y().getType().isa<mlir::NoneType>();
  attr.output_yh = !Y_h().getType().isa<mlir::NoneType>();
  attr.output_yc = !Y_c().getType().isa<mlir::NoneType>();
  return attr;
}

LogicalResult tpu::LSTMCVIOp::init(InferenceParameter &p) { return success(); }
void tpu::LSTMCVIOp::deinit(InferenceParameter &p) {}

static void lstm_compute(InferenceParameter &p, const lstm_attr_t &attr,
                         float *bias, float *h, float *c, bool forward) {
  float *output = p.outputs[0];
  float *last_h = p.outputs[1]; // Y_h
  float *last_c = p.outputs[2]; // Y_c

  float *input = p.inputs[0];
  float *r_wi = p.inputs[1];
  float *r_bi = bias;

  if (!forward) {
    r_wi += 4 * attr.hidden_size * attr.hidden_size;
    r_bi += 4 * attr.hidden_size;
    h += attr.batch_size * attr.hidden_size;
    c += attr.batch_size * attr.hidden_size;
    input = input + 4 * attr.hidden_size;
    output += attr.batch_size * attr.hidden_size;
    last_h += attr.batch_size * attr.hidden_size;
    last_c += attr.batch_size * attr.hidden_size;
  }

  float *r_wo = r_wi + attr.hidden_size * attr.hidden_size;
  float *r_wf = r_wo + attr.hidden_size * attr.hidden_size;
  float *r_wc = r_wf + attr.hidden_size * attr.hidden_size;

  float *r_bo = r_bi + attr.hidden_size;
  float *r_bf = r_bo + attr.hidden_size;
  float *r_bc = r_bf + attr.hidden_size;

  std::vector<float> gate_i(attr.batch_size * attr.hidden_size);
  std::vector<float> gate_o(attr.batch_size * attr.hidden_size);
  std::vector<float> gate_f(attr.batch_size * attr.hidden_size);
  std::vector<float> gate_c(attr.batch_size * attr.hidden_size);

  for (int s = 0; s < attr.seq_len; s++) {
    int seq_idx = forward ? s : (attr.seq_len - s - 1);
    float *x = input + seq_idx * attr.batch_size * attr.input_size;

    dnnl_mm(h, r_wi, r_bi, gate_i.data(), attr.batch_size, attr.hidden_size,
            attr.hidden_size, false);
    dnnl_mm(h, r_wo, r_bo, gate_o.data(), attr.batch_size, attr.hidden_size,
            attr.hidden_size, false);
    dnnl_mm(h, r_wf, r_bf, gate_f.data(), attr.batch_size, attr.hidden_size,
            attr.hidden_size, false);
    dnnl_mm(h, r_wc, r_bc, gate_c.data(), attr.batch_size, attr.hidden_size,
            attr.hidden_size, false);

    BF16(gate_i.data(), gate_i.data(), gate_i.size());
    BF16(gate_o.data(), gate_o.data(), gate_o.size());
    BF16(gate_f.data(), gate_f.data(), gate_f.size());
    BF16(gate_c.data(), gate_c.data(), gate_c.size());

    for (int batch = 0; batch < attr.batch_size; batch++) {
      float *xi = x + batch * attr.input_size;
      float *xo = xi + attr.hidden_size;
      float *xf = xo + attr.hidden_size;
      float *xc = xf + attr.hidden_size;
      float *gi = gate_i.data() + batch * attr.hidden_size;
      float *go = gate_o.data() + batch * attr.hidden_size;
      float *gf = gate_f.data() + batch * attr.hidden_size;
      float *gc = gate_c.data() + batch * attr.hidden_size;
      float *cell_state = c + batch * attr.hidden_size;
      float *hidden_state = h + batch * attr.hidden_size;
      if (attr.output_y) {
        hidden_state =
            output + (seq_idx * attr.num_direction * attr.batch_size + batch) *
                         attr.hidden_size;
      }
#pragma omp parallel for schedule(static, omp_schedule(attr.hidden_size))
      for (int i = 0; i < attr.hidden_size; i++) {
        gi[i] = sigmoid_(xi[i] + gi[i], p);
        go[i] = sigmoid_(xo[i] + go[i], p);
        gf[i] = sigmoid_(xf[i] + gf[i], p);
        gc[i] = tanh_(xc[i] + gc[i], p);
        cell_state[i] = BF16(BF16(gf[i] * cell_state[i]) + BF16(gi[i] * gc[i]));
        hidden_state[i] = BF16(go[i] * tanh_(cell_state[i], p));
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

LogicalResult tpu::LSTMCVIOp::inference(InferenceParameter &p) {
  auto attr = parseParam();

  auto bias_buffer = std::make_shared<std::vector<float>>(
      attr.num_direction * 4 * attr.hidden_size, 0.0f);
  auto B = bias_buffer->data();
  if (attr.have_bias) {
    memcpy(B, p.inputs[2], bias_buffer->size() * sizeof(float));
  }

  auto h0_buffer = std::make_shared<std::vector<float>>(
      attr.num_direction * attr.batch_size * attr.hidden_size, 0.0f);
  auto initial_h = h0_buffer->data();
  if (attr.have_h0) {
    memcpy(initial_h, p.inputs[3], h0_buffer->size() * sizeof(float));
  }

  auto c0_buffer = std::make_shared<std::vector<float>>(
      attr.num_direction * attr.batch_size * attr.hidden_size, 0.0f);
  auto initial_c = c0_buffer->data();
  if (attr.have_c0) {
    memcpy(initial_c, p.inputs[4], c0_buffer->size() * sizeof(float));
  }

  lstm_compute(p, attr, B, initial_h, initial_c, true);
  if (attr.num_direction == 2) {
    lstm_compute(p, attr, B, initial_h, initial_c, false);
  }

  return success();
}
