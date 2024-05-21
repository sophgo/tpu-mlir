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

void tpu::Batch2SpaceOp::codegen_global_bm1684() {
  auto input = getInput();
  auto output = getOutput();
  auto buffer = getBuffer();
  int64_t block_h = getBlockH();
  int64_t block_w = getBlockW();
  auto crops_v = module::getI64Array(getCrops());
  int64_t crop_top = crops_v->at(0);
  int64_t crop_bottom = crops_v->at(1);
  int64_t crop_left = crops_v->at(2);
  int64_t crop_right = crops_v->at(3);
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
  int crop_sizes[4] = {(int)crop_top, (int)crop_bottom, (int)crop_left,
                       (int)crop_right};
  if (module::isUniformQuantized(input)) {
    uint64_t imm_global_addr = 0;
    int n = input_shape[0] / (block_h * block_w);
    uint64_t input_size = ceiling_func(input_shape[0], 4) * 4 * input_shape[1] *
                          input_shape[2] * input_shape[3];
    uint64_t output_size = ceiling_func(n, 4) * 4 * input_shape[1] *
                           input_shape[2] * block_h * input_shape[3] * block_w;
    uint64_t buffer_size = 0;
    buffer_size = (input_size > output_size) ? input_size : output_size;
    imm_global_addr = buffer_addr + buffer_size;
    BM1684::instance().dl_nodechip_batch2space_fix8b(
        in_addr, out_addr, buffer_addr, imm_global_addr, NULL, input_shape, 4,
        2 /*STORE_MODE_4N*/, 2 /*STORE_MODE_4N*/, block_size, crop_sizes,
        output_shape, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_batch2space(
        in_addr, out_addr, buffer_addr, input_shape, 4, block_size, crop_sizes,
        NULL, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
  delete[] input_shape;
  delete[] output_shape;
}

uint32_t tpu::Batch2SpaceOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::Batch2SpaceOp::get_fw_type_bm1684() { return -1; }
