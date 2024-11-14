//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

// =========================================
// GlobalGenInterface
// =========================================
void tpu::TileOp::codegen_global_bm1684() {
  auto input = getInput();
  auto output = getOutput();
  auto input_format = BM1684::getStoreMode(input);
  auto output_format = BM1684::getStoreMode(output);
  auto output_shape = module::getShape(getOutput());
  auto input_shape = module::getShape(getInput());
  int input_dim = input_shape.size();
  int in_shape[input_dim];
  module::getGlobalShape(getInput(), in_shape, input_dim);
  auto input_addr = module::getAddress(input);
  auto output_addr = module::getAddress(output);
  uint64_t buffer_global_addr = module::getAddress(getBuffer());
  int tile_coeff[input_dim];
  for (int i = 0; i < input_dim; i++) {
    tile_coeff[i] = output_shape[i] / input_shape[i];
  }
  if (module::isUniformQuantized(getInput())) {
    BM1684::instance().dl_nodechip_tile_full_fix8b(
        input_addr, output_addr, buffer_global_addr, NULL,
        (const uint32_t *)in_shape, (const int *)tile_coeff, input_dim,
        input_format, output_format, 0,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    BM1684::instance().dl_nodechip_tile_full(
        input_addr, buffer_global_addr, output_addr, (const uint32_t *)in_shape,
        (const int *)tile_coeff, input_dim, input_format, output_format, 0,
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

/*
note: local_codegen and dynamic for bm1684 are not supported.
*/

int64_t tpu::TileOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  // auto output_shape = module::getShape(getOutput());
  // auto input_shape = module::getShape(getInput());
  // int input_dim = input_shape.size();
  // int tile_coeff[input_dim];
  // for (int i = 0; i < input_dim; i++) {
  //   tile_coeff[i] = output_shape[i] / input_shape[i];
  // }
  // int64_t local_buffer_size = 0;
  // if (tile_coeff[1] > 1) {
  //   auto NPU_NUM = BM1684::NPU_NUM;
  //   auto EU_NUM = BM1684::eu_num(sizeof(float));
  //   int64_t type_len = module::getDtypeSize(getOutput());
  //   auto top_shape = module::getShape(getOutput());
  //   int64_t buffer_size = ceiling_func(top_shape[1], NPU_NUM) *
  //                         align_up(top_shape[2] * top_shape[3], EU_NUM);
  //   local_buffer_size =
  //       ceiling_func(top_shape[0], 4 / type_len) * buffer_size *
  //       sizeof(float);
  // }
  // return local_buffer_size;
  return 0;
}

void tpu::TileOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                       local_sec_info_t &sec_info) {
  // auto input = getInput();
  // auto output = getOutput();
  // auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step);
  // auto out_gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  // int la_input = in_gi.out_addr;
  // int la_output = out_gi.out_addr;
  // int la_buffer = out_gi.buffer_addr;
  // int64_t n, c, h, w;
  // module::getNCHW(getInput(), n, c, h, w);
  // int input_dim = module::getShape(input).size();
  // int input_shape[input_dim];
  // module::getGlobalShape(getInput(), input_shape, input_dim);
  // int output_dim = module::getShape(output).size();
  // int output_shape[output_dim];
  // module::getGlobalShape(getInput(), output_shape, output_dim);
  // int *tile_coeff = new int[input_dim];
  // for (int i = 0; i < input_dim; i++) {
  //   tile_coeff[i] = output_shape[i] / input_shape[i];
  // }
  // auto input_dtype = BM1684::getDataType(input);
  // auto in_gdma_format = BM168x::getGdmaFormat(input_dtype);
  // auto output_dtype = BM1684::getDataType(output);
  // auto out_gdma_format = BM168x::getGdmaFormat(output_dtype);
  // if (module::isUniformQuantized(getInput())) {
  //   BM1684::instance().dl_nodechip_tile_fix8b_local(
  //       la_input, la_output, la_buffer, (const int *)input_shape,
  //       (const int *)tile_coeff, input_dim, in_gdma_format, out_gdma_format,
  //       0, (CMD_ID_NODE *)BM1684::instance()->bdc_node, (CMD_ID_NODE
  //       *)BM1684::instance()->cmdid_node);
  // } else {
  //   BM1684::instance().dl_nodechip_tile_local(
  //       la_input, la_output, (const int *)input_shape, (const int
  //       *)tile_coeff, input_dim, in_gdma_format, out_gdma_format, 0,
  //       (CMD_ID_NODE *)BM1684::instance()->bdc_node);
  // }
  llvm_unreachable("Not supported now");
}

uint32_t tpu::TileOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  // GLOBAL_IR_COMMON(tile);
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::TileOp::get_fw_type_bm1684() {
  // return FW_BMNET_TILE;
  return -1;
}

int32_t tpu::TileOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
