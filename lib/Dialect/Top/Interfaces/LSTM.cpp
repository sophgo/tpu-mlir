//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::LSTMOp::getFLOPs() {
  // flops: 4 * (4 * hidden_size * input_size + 4 * hidden_size * hidden_size)
  auto in_shape = module::getShape(getInput());
  ASSERT_THIS(in_shape.size() == 3);

  int64_t batch_size;
  int64_t seq_len;

  if (getBatchFirst()) {
    batch_size = in_shape[0];
    seq_len = in_shape[1];
  } else {
    batch_size = in_shape[1];
    seq_len = in_shape[0];
  }

  int64_t input_size = in_shape[2];
  int64_t hidden_size = getHiddenSize();
  int64_t num_direction = getBidirectional() ? 2 : 1;
  int64_t flops =
      4 * (4 * hidden_size * input_size + 4 * hidden_size * hidden_size);

  return flops * batch_size * seq_len * num_direction;
}

lstm_attr_t top::LSTMOp::parseParam() {
  lstm_attr_t attr = {0};
  auto in_shape = module::getShape(getInput());
  ASSERT_THIS(in_shape.size() == 3);
  if (getBatchFirst()) {
    attr.batch_size = in_shape[0];
    attr.seq_len = in_shape[1];
    attr.batch_first = true;
  } else {
    attr.batch_size = in_shape[1];
    attr.seq_len = in_shape[0];
    attr.batch_first = false;
  }
  attr.input_size = in_shape[2];
  attr.num_direction = getBidirectional() ? 2 : 1;
  attr.hidden_size = getHiddenSize();
  attr.have_bias = !getBias().getType().isa<NoneType>();
  attr.have_h0 = !getInitialH().getType().isa<NoneType>();
  attr.have_c0 = !getInitialC().getType().isa<NoneType>();
  attr.have_cont = !getCont().getType().isa<NoneType>();
  attr.output_y = !getY().getType().isa<NoneType>();
  attr.output_yh = !getYH().getType().isa<NoneType>();
  attr.output_yc = !getYC().getType().isa<NoneType>();
  return attr;
}

LogicalResult top::LSTMOp::init(InferenceParameter &p) {
  if (!module::isPlatform(module::Platform::TORCH)) {
    return success();
  }
  auto attr = parseParam();
  if (attr.output_y == false || attr.batch_size == 1 ||
      attr.num_direction == 1) {
    return success();
  }
  auto num = module::getNumElements(getY());
  float *buffer = new float[num];
  p.handle = (void *)buffer;
  return success();
}

void top::LSTMOp::deinit(InferenceParameter &p) {
  if (p.handle) {
    float *buffer = (float *)p.handle;
    delete[] buffer;
    p.handle = nullptr;
  }
}

static inline float sigmoid_(float x) {
  // return static_cast<float>(1.f / (1.f + std::exp(-x)));
  return 0.5 * tanh(0.5 * x) + 0.5;
}

static inline float tanh_(float x) { return tanh(x); }

static void lstm_compute(InferenceParameter &p, const lstm_attr_t &attr,
                         float *bias, float *h, float *c, bool forward) {
  //(TODO) num_layers > 1
  // input += seq_length * batch * input_size * num_layer; //(TODO check!)
  float *input = p.inputs[0];
  float *output = p.handle != nullptr ? (float *)p.handle : p.outputs[0];
  float *x_wi = p.inputs[1];
  float *h_wi = p.inputs[2];
  float *x_bi = bias;
  float *last_h = p.outputs[1]; // Y_h
  float *last_c = p.outputs[2]; // Y_c
  float *conts = p.inputs[6];

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
      float cont = 1.0f;
      if (attr.have_cont) {
        cont = conts[s * attr.batch_size + batch];
      }
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
        gi[i] = sigmoid_(xi[i] + cont * hi[i]);
        go[i] = sigmoid_(xo[i] + cont * ho[i]);
        gf[i] = sigmoid_(xf[i] + cont * hf[i]);
        gc[i] = tanh_(xc[i] + cont * hc[i]);
        cell_state[i] = cont * gf[i] * cell_state[i] + gi[i] * gc[i];
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

LogicalResult top::LSTMOp::inference(InferenceParameter &p) {
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
  if (p.handle) {
    float *buffer = (float *)p.handle;
    function_permute(buffer, p.outputs[0],
                     {1, attr.seq_len, attr.num_direction, attr.batch_size,
                      attr.hidden_size},
                     {0, 1, 3, 2, 4});
  }
  return success();
}

void top::LSTMOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  ASSERT_THIS(in_shape.size() == 3);
  int64_t num_dir = 1;
  if (getBidirectional()) {
    num_dir = 2;
  }
  int64_t seq_len, batch_size;
  if (getBatchFirst()) {
    batch_size = in_shape[0];
    seq_len = in_shape[1];
  } else {
    seq_len = in_shape[0];
    batch_size = in_shape[1];
  }
  int64_t hidden_size = getHiddenSize();
  std::vector<int64_t> shape0;
  std::vector<int64_t> shape1;
  std::vector<int64_t> shape2;
  if (getBatchFirst()) {
    shape0 = {batch_size, seq_len, num_dir, hidden_size};
    shape1 = {batch_size, num_dir, hidden_size};
    shape2 = {batch_size, num_dir, hidden_size};
  } else {
    if (module::isPlatform(module::Platform::TORCH)) {
      shape0 = {seq_len, batch_size, num_dir, hidden_size};
    } else {
      shape0 = {seq_len, num_dir, batch_size, hidden_size};
    }
    shape1 = {num_dir, batch_size, hidden_size};
    shape2 = {num_dir, batch_size, hidden_size};
  }
  if (module::isNone(getY()) == false) {
    module::setShapeOrVerify(getY(), shape0);
  }
  if (module::isNone(getYH()) == false) {
    module::setShapeOrVerify(getYH(), shape1);
  }
  if (module::isNone(getYC()) == false) {
    module::setShapeOrVerify(getYC(), shape2);
  }
}
