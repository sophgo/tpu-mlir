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

void tpu::ReverseOp::codegen_global_bm1684() {
  auto input = getInput();
  auto output = getOutput();
  auto input_addr = module::getAddress(input);
  auto output_addr = module::getAddress(output);
  auto shape = module::getShape(input);
  int *input_shape = new int[MAX_SHAPE_DIMS];
  for (int i = 0; i < shape.size(); ++i) {
    input_shape[i] = shape[i];
  }
  if (!module::isUniformQuantized(getOutput())) {
    BM1684::instance().dl_nodechip_reverse_forward_v2(
        input_addr, output_addr, input_shape, shape.size(), getAxis(),
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    UNREACHABLE_THIS("Not Implemented");
  }
}

uint32_t tpu::ReverseOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::ReverseOp::get_fw_type_bm1684() { return -1; }
