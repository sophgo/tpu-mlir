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

void tpu::MaskedFillOp::codegen_global_bm1684() {
  auto input = getBrn();
  auto mask = getCond();
  auto mask_addr = module::getAddress(mask);
  auto input_addr = module::getAddress(input);
  auto top_addr = module::getAddress(getOutput());
  float value = getConstVal().convertToDouble();
  int *input_shape = new int[MAX_SHAPE_DIMS];
  int *mask_shape = new int[MAX_SHAPE_DIMS];
  int input_dims = module::getShape(input).size();
  int mask_dims = module::getShape(mask).size();

  for (auto v : llvm::enumerate(module::getShape(input)))
    input_shape[v.index()] = (int)v.value();
  for (auto v : llvm::enumerate(module::getShape(mask)))
    mask_shape[v.index()] = (int)v.value();

  BM1684::instance().dl_nodechip_masked_fill_global(
      input_addr, mask_addr, top_addr, (uint32_t *)input_shape,
      (uint32_t *)mask_shape, input_dims, mask_dims, *((uint32_t *)&value),
      (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
}

int64_t tpu::MaskedFillOp::getBufferSize_bm1684(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_hslice, int64_t out_nslice, int64_t out_hslice) {
  int64_t buffer_size = 0;
  int64_t n, c, h, w;
  module::getNCHW(getOutput(), n, c, h, w);
  auto EU_NUM = BM1684::eu_num(sizeof(int32_t));
  buffer_size = out_nslice * ceiling_func(c, BM1684::NPU_NUM) *
                align_up(out_hslice * w, EU_NUM) * sizeof(float);

  return buffer_size;
}

void tpu::MaskedFillOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                             local_sec_info_t &sec_info) {
  if (module::isUniformQuantized(getOutput())) {
    UNREACHABLE_THIS("Not Implemented");
  }
  auto input_dims = module::getShape(getBrn()).size();
  auto mask_dims = module::getShape(getCond()).size();
  auto output_dims = module::getShape(getCond()).size();
  assert(input_dims == mask_dims);
  assert(input_dims == output_dims);
  assert(output_dims == mask_dims);
  int input_shape[input_dims], mask_shape[mask_dims];
  module::getLocalShape(getBrn(), n_step, h_step, input_shape);
  module::getLocalShape(getCond(), n_step, h_step, mask_shape);
  auto top_ginfo = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto input_ginfo =
      LocalGenInterface::getGroupInfo(getBrn(), n_step, h_step, 0, 0, 0);
  auto mask_ginfo =
      LocalGenInterface::getGroupInfo(getCond(), n_step, h_step, 0, 0, 0);

  float value = getConstVal().convertToDouble();
  BM1684::instance().dl_nodechip_masked_fill_local(
      input_ginfo.out_addr, mask_ginfo.out_addr, top_ginfo.buffer_addr,
      top_ginfo.out_addr, input_shape, mask_shape, 4, 4, *((uint32_t *)&value),
      (CMD_ID_NODE *)BM1684::instance()->bdc_node);
}

uint32_t tpu::MaskedFillOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
int64_t tpu::MaskedFillOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::MaskedFillOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
