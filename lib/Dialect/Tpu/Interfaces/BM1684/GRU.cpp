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

void tpu::GRUOp::codegen_global_bm1684() {
  auto attr = parseParam();
  BM1684::instance().dl_nodechip_gru(
      module::getAddress(getInput()),    // uint64_t xGlobalAddr,
      module::getAddress(getInitialH()), // uint64_t h0GlobalAddr,
      module::getAddress(getY()),        // uint64_t yGlobalAddr,
      module::getAddress(getYH()),       // uint64_t hnGlobalAddr,
      module::getAddress(getFilter()),   // uint64_t wGlobalAddr,
      module::getAddress(getBias()),     // uint64_t bGlobalAddr,
      module::getAddress(getBuffer()),   // uint64_t zGlobalAddr,
      attr.have_bias,                    // bool bias,
      attr.output_y, attr.output_yh,
      attr.seq_len,              // int sequence,
      attr.batch_size,           // int batch,
      attr.input_size,           // int xSize,
      attr.hidden_size,          // int hSize,
      attr.batch_first,          // bool batchfirst,
      (attr.num_direction == 2), // bool bidirectional,
      1,                         // int numLayers
      (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
}

uint32_t tpu::GRUOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::GRUOp::get_fw_type_bm1684() { return -1; }
