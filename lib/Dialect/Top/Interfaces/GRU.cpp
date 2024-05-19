//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::GRUOp::getFLOPs() {
  // flops： 2 * (3 * hidden_size * input_size + 3 * hidden_size * hidden_size)
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
      2 * (3 * hidden_size * input_size + 3 * hidden_size * hidden_size);

  return flops * batch_size * seq_len * num_direction;
}

gru_attr_t top::GRUOp::parseParam() {
  gru_attr_t attr = {0};
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
  attr.num_direction = getBidirectional() ? 2 : 1;
  attr.hidden_size = getHiddenSize();
  attr.input_size = in_shape[2];
  attr.have_bias = !getBias().getType().isa<NoneType>();
  attr.have_h0 = !getInitialH().getType().isa<NoneType>();
  attr.output_y = !getY().getType().isa<NoneType>();
  attr.output_yh = !getYH().getType().isa<NoneType>();
  return attr;
}

LogicalResult top::GRUOp::init(InferenceParameter &p) {
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

void top::GRUOp::deinit(InferenceParameter &p) {
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

static void gru_compute(InferenceParameter &p, const gru_attr_t &attr,
                        float *bias, float *h, bool forward) {
  //(TODO) num_layers > 1
  // input += seq_length * batch * input_size * num_layer; //(TODO check!)
  float *input = p.inputs[0];
  float *output = p.handle != nullptr ? (float *)p.handle : p.outputs[0];
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
      float *hidden_state = pre_state;
      if (attr.output_y) {
        hidden_state =
            output + (seq_idx * attr.num_direction * attr.batch_size + batch) *
                         attr.hidden_size;
      }
      for (int i = 0; i < attr.hidden_size; i++) {
        hz[i] = sigmoid_(hz[i] + xz[i]);
        hr[i] = sigmoid_(hr[i] + xr[i]);
        hh[i] = tanh_(hr[i] * hh[i] + xh[i]);
        hidden_state[i] = (1 - hz[i]) * hh[i] + hz[i] * pre_state[i];
      }
    }
    if (attr.output_y) {
      prev_hidden_state = output + seq_idx * attr.num_direction *
                                       attr.batch_size * attr.hidden_size;
    }
  }
  if (attr.output_yh) {
    memcpy(last_h, prev_hidden_state,
           attr.batch_size * attr.hidden_size * sizeof(float));
  }
}

LogicalResult top::GRUOp::inference(InferenceParameter &p) {
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
  if (p.handle) {
    float *buffer = (float *)p.handle;
    function_permute(buffer, p.outputs[0],
                     {1, attr.seq_len, attr.num_direction, attr.batch_size,
                      attr.hidden_size},
                     {0, 1, 3, 2, 4});
  }
  return success();
}

void top::GRUOp::shape_inference() {
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
  if (getBatchFirst()) {
    shape0 = {batch_size, seq_len, num_dir, hidden_size};
    shape1 = {batch_size, num_dir, hidden_size};
  } else {
    if (module::isPlatform(module::Platform::TORCH)) {
      shape0 = {seq_len, batch_size, num_dir, hidden_size};
    } else {
      shape0 = {seq_len, num_dir, batch_size, hidden_size};
    }
    shape1 = {num_dir, batch_size, hidden_size};
  }
  if (module::isNone(getY()) == false) {
    module::setShapeOrVerify(getY(), shape0);
  }
  if (module::isNone(getYH()) == false) {
    module::setShapeOrVerify(getYH(), shape1);
  }
}
