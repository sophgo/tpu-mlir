//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

typedef enum {
  BATCH_TORCH = 0,
  BATCH_FIRST = 1,
  BATCH_ONNX = 2,
} lstm_batch_t;

void tpu::LSTMOp::codegen_global_bm1684() {
  auto attr = parseParam();
  int batch_mode =
      attr.batch_first
          ? BATCH_FIRST
          : (module::isPlatform(module::Platform::TORCH) ? BATCH_TORCH
                                                         : BATCH_ONNX);
  BM1684::instance().dl_nodechip_pytorch_lstm(
      module::getAddress(getInput()),    // uint64_t xGlobalAddr,
      module::getAddress(getInitialH()), // uint64_t h0GlobalAddr,
      module::getAddress(getInitialC()), // uint64_t c0GlobalAddr,
      module::getAddress(getY()),        // uint64_t yGlobalAddr,
      module::getAddress(getYH()),       // uint64_t hnGlobalAddr,
      module::getAddress(getYC()),       // uint64_t cnGlobalAddr,
      module::getAddress(getFilter()),   // uint64_t wGlobalAddr,
      module::getAddress(getBias()),     // uint64_t bGlobalAddr, getBias
      module::getAddress(getBuffer()),   // z
      attr.have_bias,                    // bool bias,
      attr.output_y, attr.output_yh, attr.output_yc,
      attr.seq_len,     // int sequence,
      attr.batch_size,  // int batch,
      attr.input_size,  // module::getShape(getInput())[2],//int xSize,
      attr.hidden_size, // module::getShape(getInitialH())[2],//int hSize,
      batch_mode,       // int batchMode,
      attr.num_direction == 2, // bool bidirectional,
      1,                       // int numLayers,
      (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
}

uint32_t tpu::LSTMOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::LSTMOp::get_fw_type_bm1684() { return -1; }
