//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"

using namespace tpu_mlir::backend;

void tpu::Space2BatchOp::codegen_global_bm1684() {
  auto input = getInput();
  auto output = getOutput();
  auto buffer = getBuffer();
  int64_t block_h = getBlockH();
  int64_t block_w = getBlockW();
  auto pads_v = module::getI64Array(getPads());
  int64_t pad_top = pads_v->at(0);
  int64_t pad_bottom = pads_v->at(1);
  int64_t pad_left = pads_v->at(2);
  int64_t pad_right = pads_v->at(3);
  auto in_addr = module::getAddress(input);
  auto out_addr = module::getAddress(output);
  auto buffer_addr = module::getAddress(buffer);
  int *input_shape = new int[MAX_SHAPE_DIMS];
  for (auto v : llvm::enumerate(module::getShape(input)))
    input_shape[v.index()] = (int)v.value();
  int *output_shape = new int[MAX_SHAPE_DIMS];
  for (auto v : llvm::enumerate(module::getShape(output)))
    output_shape[v.index()] = (int)v.value();
  int block_size[2] = {(int)block_h, (int)block_w};
  int pad_sizes[4] = {(int)pad_top, (int)pad_bottom, (int)pad_left,
                      (int)pad_right};
  if (module::isUniformQuantized(input)) {
    BM1684::instance().dl_nodechip_space2batch_fix8b(
        in_addr, out_addr, buffer_addr, NULL, input_shape, 4,
        2 /*STORE_MODE_4N*/, 2 /*STORE_MODE_4N*/, block_size, pad_sizes,
        output_shape, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_space2batch(
        in_addr, out_addr, buffer_addr, input_shape, 4, block_size, pad_sizes,
        NULL, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
  delete[] input_shape;
  delete[] output_shape;
}

uint32_t tpu::Space2BatchOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::Space2BatchOp::get_fw_type_bm1684() { return -1; }
