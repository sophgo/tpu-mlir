//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
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
} gru_batch_t;

typedef struct {
  uint64_t xGlobalAddr;
  uint64_t h0GlobalAddr;
  uint64_t yGlobalAddr;
  uint64_t hnGlobalAddr;
  uint64_t wGlobalAddr;
  uint64_t bGlobalAddr;
  uint64_t zGlobalAddr;
  bool bias;
  bool outputY;
  bool outputYh;
  int sequence;
  int batch;
  int xSize;
  int hSize;
  int batchMode;
  bool bidirectional;
  int numLayers;
  int dtype;
} gru_param_t;

#ifdef __cplusplus
}
#endif
// =========================================
// GlobalGenInterface
// =========================================
void tpu::GRUOp::codegen_global_bm1684x() {
  auto attr = parseParam();
  gru_param_t p = {0};
  p.xGlobalAddr = module::getAddress(getInput());
  p.wGlobalAddr = module::getAddress(getFilter());
  p.bGlobalAddr = module::getAddress(getBias());
  p.h0GlobalAddr = module::getAddress(getInitialH());
  p.yGlobalAddr = module::getAddress(getY());
  p.hnGlobalAddr = module::getAddress(getYH());
  p.zGlobalAddr = module::getAddress(getBuffer());

  p.bias = attr.have_bias;
  p.outputY = attr.output_y;
  p.outputYh = attr.output_yh;
  p.sequence = attr.seq_len;
  p.batch = attr.batch_size;
  p.xSize = attr.input_size;
  p.hSize = attr.hidden_size;
  p.batchMode = attr.batch_first ? BATCH_FIRST
                                 : (module::isPlatform(module::Platform::TORCH)
                                        ? BATCH_TORCH
                                        : BATCH_ONNX);
  p.bidirectional = (attr.num_direction == 2);
  p.numLayers = 1;
  p.dtype = BM168x::getDataType(getInput());
  BM168x::call_global_func("backend_api_gru_global", &p, sizeof(gru_param_t));
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::GRUOp::dyn_codegen_global_bm1684x(void *buffer) {
  if (!buffer)
    return sizeof(dyn_glu_global_spec_t);
  auto attr = parseParam();
  dyn_glu_global_spec_t p = {0};
  p.xGlobalAddr = module::getAddress(getInput());
  p.wGlobalAddr = module::getAddress(getFilter());
  p.bGlobalAddr = module::getAddress(getBias());
  p.h0GlobalAddr = module::getAddress(getInitialH());
  p.yGlobalAddr = module::getAddress(getY());
  p.hnGlobalAddr = module::getAddress(getYH());
  p.zGlobalAddr = module::getAddress(getBuffer());
  p.common.bias = attr.have_bias;
  p.common.outputY = attr.output_y;
  p.common.outputYh = attr.output_yh;
  p.common.sequence = attr.seq_len;
  p.common.batch = attr.batch_size;
  p.common.xSize = attr.input_size;
  p.common.hSize = attr.hidden_size;
  p.common.batchMode =
      attr.batch_first
          ? BATCH_FIRST
          : (module::isPlatform(module::Platform::TORCH) ? BATCH_TORCH
                                                         : BATCH_ONNX);
  p.common.bidirectional = (attr.num_direction == 2);
  p.common.numLayers = 1;
  p.common.dtype = BM168x::getDataType(getInput());
  return BM168x::dynamic_spec_to_buffer(buffer, p);
}

int64_t tpu::GRUOp::get_fw_type_bm1684x() { return FW_BMNET_GRU; }
