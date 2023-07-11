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

void tpu::GatherOp::codegen_global_bm1684() {
  if (module::isWeight(getIndices())) {
    auto indices_shape = module::getShape(getIndices());
    auto input_shape = module::getShape(getInput());
    int dims_count = 0;
    for (int i = 0; i < indices_shape.size(); ++i) {
      if (indices_shape[i] > 1)
        dims_count++;
    }
    if (dims_count > 1)
      llvm_unreachable("not support");
    auto indices_op = cast<top::WeightOp>(getIndices().getDefiningOp());
    auto indices_data = indices_op.read<int>();
    int block_size = 1;
    for (int i = 0; i < input_shape.size(); ++i) {
      if (i == getAxis())
        continue;
      block_size *= input_shape[i];
    }
    int DTYPE_SIZE = module::getDtypeSize(getInput());
    int DTYPE = DTYPE_SIZE == 4 ? 0 : 3;
    for (int i = 0; i < indices_data->size(); ++i) {
      int in_offset = indices_data->at(i);
      int out_offset = i;
      auto input_addr =
          module::getAddress(getInput()) + in_offset * block_size * DTYPE_SIZE;
      auto output_addr = module::getAddress(getOutput()) +
                         out_offset * block_size * DTYPE_SIZE;
      BM1684::instance().dl_nodechip_global_memcpy_ex(
          input_addr, output_addr, 1, block_size, block_size, DTYPE, DTYPE,
          block_size, (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
    }
  } else {
    llvm_unreachable("need dynamic");
  }
}

uint32_t tpu::GatherOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  if (module::isWeight(getInput())) {
    llvm_unreachable("not support");
  }
  ir_layer_info_t *layer_info = (ir_layer_info_t *)ir_layer_info;
  dynamic_common_ir_layer_info(layer_info, getInput(), getOutput());
  fw_index_select_layer_param_t fw_param = {0};
  fw_param.buffer_addr = module::getAddress(getBuffer());
  fw_param.dim = getAxis();
  fw_param.index_is_coeff = module::isWeight(getIndices());
  fw_param.index_num = module::getNumElements(getIndices());
  layer_info->fw_layer_param_u.fw_index_select_layer_param = fw_param;
  return sizeof(fw_index_select_layer_param_t);
}

int64_t tpu::GatherOp::get_fw_type_bm1684() { return FW_BMNET_INDEX_SELECT; }
