//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

using namespace tpu_mlir::backend;

#ifdef __cplusplus
extern "C" {
#endif

// BATCH_FIRST: x = [batch, seq_len, input_size], y = [batch, seq_len, num_dir,
// hidden_size] BATCH_TORCH: x = [seq_len, batch, input_size], y = [seq_len,
// batch, num_dir, hidden_size] BATCH_ONNX:  x = [seq_len, batch, input_size], y
// = [seq_len, num_dir, batch, hidden_size]
typedef enum {
  BATCH_TORCH = 0,
  BATCH_FIRST = 1,
  BATCH_ONNX = 2,
} lstm_batch_t;

typedef struct {
  uint64_t x_global_addr;
  uint64_t h0_global_addr;
  uint64_t c0_global_addr;
  uint64_t y_global_addr;
  uint64_t hn_global_addr;
  uint64_t cn_global_addr;
  uint64_t w_global_addr;
  uint64_t b_global_addr;
  uint64_t z_global_addr;
  bool bias;
  bool output_y;
  bool output_yh;
  bool output_yc;
  int sequence;
  int batch;
  int x_size;
  int h_size;
  lstm_batch_t batch_mode;
  bool bidirection;
  int num_layers;
  int dtype;
} pytorch_lstm_param_t;

#ifdef __cplusplus
}
#endif
// =========================================
// GlobalGenInterface
// =========================================
void tpu::LSTMOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_spec(
      ValueRange{getInput(), getInitialH(), getInitialC(), getFilter()});
  auto output_spec = BM168x::get_output_spec(op);
  // 1684x pytorch lstm out is [seq_length, batch_size, num_dir * hidden_size]
  pytorch_lstm_param_t p = {0};
  p.x_global_addr = module::getAddress(getInput());
  p.w_global_addr = module::getAddress(getFilter());
  p.b_global_addr = module::getAddress(getBias());
  p.h0_global_addr = module::getAddress(getInitialH());
  p.c0_global_addr = module::getAddress(getInitialC());
  p.y_global_addr = module::getAddress(getY());
  p.hn_global_addr = module::getAddress(getYH());
  p.cn_global_addr = module::getAddress(getYC());
  p.z_global_addr = module::getAddress(getBuffer());

  p.bias = attr.have_bias;
  p.output_y = attr.output_y;
  p.output_yh = attr.output_yh;
  p.output_yc = attr.output_yc;
  p.sequence = attr.seq_len;
  p.batch = attr.batch_size;
  p.x_size = attr.input_size;
  p.h_size = attr.hidden_size;
  p.batch_mode = attr.batch_first ? BATCH_FIRST
                                  : (module::isPlatform(module::Platform::TORCH)
                                         ? BATCH_TORCH
                                         : BATCH_ONNX);
  p.bidirection = (attr.num_direction == 2);
  p.num_layers = 1;
  p.dtype = BM168x::getDataType(getInput());
  BM168x::call_global_func("backend_api_pytorch_lstm_global", &p,
                           sizeof(pytorch_lstm_param_t), input_spec->data(),
                           output_spec->data());
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::LSTMOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(pytorch_lstm_param_t);
  auto attr = parseParam();
  auto op = getOperation();
  auto input_spec = BM168x::get_spec(
      ValueRange{getInput(), getInitialH(), getInitialC(), getFilter()});
  auto output_spec = BM168x::get_output_spec(op);
  // 1684x pytorch lstm out is [seq_length, batch_size, num_dir * hidden_size]
  pytorch_lstm_param_t p = {0};
  p.x_global_addr = module::getAddress(getInput());
  p.w_global_addr = module::getAddress(getFilter());
  p.b_global_addr = module::getAddress(getBias());
  p.h0_global_addr = module::getAddress(getInitialH());
  p.c0_global_addr = module::getAddress(getInitialC());
  p.y_global_addr = module::getAddress(getY());
  p.hn_global_addr = module::getAddress(getYH());
  p.cn_global_addr = module::getAddress(getYC());
  p.z_global_addr = module::getAddress(getBuffer());

  p.bias = attr.have_bias;
  p.output_y = attr.output_y;
  p.output_yh = attr.output_yh;
  p.output_yc = attr.output_yc;
  p.sequence = attr.seq_len;
  p.batch = attr.batch_size;
  p.x_size = attr.input_size;
  p.h_size = attr.hidden_size;
  p.batch_mode = attr.batch_first ? BATCH_FIRST
                                  : (module::isPlatform(module::Platform::TORCH)
                                         ? BATCH_TORCH
                                         : BATCH_ONNX);
  p.bidirection = (attr.num_direction == 2);
  p.num_layers = 1;
  p.dtype = BM168x::getDataType(getInput());
  return BM168x::dynamic_spec_to_buffer(buffer, p);
}

int64_t tpu::LSTMOp::get_fw_type_bm1684x() { return FW_BMNET_PYTORCH_LSTM; }
