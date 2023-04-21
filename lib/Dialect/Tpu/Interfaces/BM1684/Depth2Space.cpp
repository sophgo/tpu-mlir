//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"

#include "tpu_mlir/Support/Module.h"



using namespace tpu_mlir::backend;

void tpu::Depth2SpaceOp::codegen_global_bm1684() {
  auto input = getInput();
  auto output = getOutput();
  auto input_addr = module::getAddress(input);
  auto output_addr = module::getAddress(output);
  int input_dims = module::getShape(input).size();
  auto input_dtype = BM1684::getDataType(input);
  auto gdma_format = BM1684::GDMA_VALUE_FORMAT_FLOAT32;
  // melloc
  int *input_shape = new int[MAX_SHAPE_DIMS];
  int *block_sizes = new int[MAX_SHAPE_DIMS];
  // assign param and call func
  module::getGlobalShape(input, input_shape);
  block_sizes[0] = getBlockH();
  block_sizes[1] = getBlockW();
  if (input_dtype == DTYPE_INT8 || input_dtype == DTYPE_UINT8) {
    input_shape[0] = (input_shape[0] + 3) >> 2;
  }
  BM1684::instance().dl_nodechip_depth2space(
        input_addr, output_addr, input_shape, input_dims, block_sizes,
        getInIs_NCHW(), getOutIs_NCHW(), getIsInversed(), getIs_CRD(),
        gdma_format, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
  // release
  delete[] input_shape;
  delete[] block_sizes;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
uint32_t tpu::Depth2SpaceOp::dyn_codegen_global_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}

int64_t tpu::Depth2SpaceOp::get_fw_type_bm1684() {
  return -1;
}