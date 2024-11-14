//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

// in TPU1684's Arg Definition, Nullptr is not 0.
// which can be found in TPU1684/fireware_core/src/global_layer/nodechip_arg.c:6
#ifndef NULL_BM1684PTR
#define NULL_BM1684PTR 9223372036854775807
#endif

using namespace tpu_mlir::backend;

void tpu::ArgOp::codegen_global_bm1684() {

  uint64_t bottom_global_offset = module::getAddress(getInput());
  uint64_t index_global_offset;
  if (getResult(0).getType().isa<NoneType>())
    index_global_offset = NULL_BM1684PTR;
  else
    index_global_offset = module::getAddress(getResult(0));

  uint64_t value_global_offset;
  if (getResult(1).getType().isa<NoneType>())
    value_global_offset = NULL_BM1684PTR;
  else
    value_global_offset = module::getAddress(getResult(1));

  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w, true);

  auto bottom_dim = module::getShape(getInput());
  int shape_size = bottom_dim.size();
  int axis = getAxis();
  if (axis < 0)
    axis += shape_size;

  int method = llvm::StringSwitch<int>(getMode())
                   .Case("ArgMax", 0)
                   .Case("ArgMin", 1)
                   .Default(-1);
  if (shape_size <= 4 || axis < 3) {
    BM1684::instance().dl_nodechip_arg(
        bottom_global_offset, value_global_offset, index_global_offset, n, c, h,
        w, axis, method, 1, getSelectLastIndex(),
        (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  } else {
    int w = 1;
    for (int i = axis + 1; i < shape_size; ++i)
      w *= bottom_dim[i];
    int n = 1;
    for (int i = 0; i < axis - 1; ++i)
      n *= bottom_dim[i];

    BM1684::instance().dl_nodechip_arg(
        bottom_global_offset, value_global_offset, index_global_offset, n,
        bottom_dim[axis - 1], bottom_dim[axis], w, 2, method, 1,
        getSelectLastIndex(), (CMD_ID_NODE *)BM1684::instance()->cmdid_node);
  }
}

uint32_t tpu::ArgOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  ir_layer_info_t *layer_info = (ir_layer_info_t *)ir_layer_info;
  dynamic_common_ir_layer_info(layer_info, getInput(), getIndices());
  fw_arg_layer_param_t layer_param = {0};
  layer_param.ic = module::getShape(getInput())[1];
  layer_param.axis = getAxis();
  layer_param.input_sign = module::isSign(getInput());
  layer_param.method = llvm::StringSwitch<int>(getMode())
                           .Case("ArgMax", 0)
                           .Case("ArgMin", 1)
                           .Default(-1);
  layer_param.is_index_int32 = 1;
  layer_param.select_last_index = getSelectLastIndex();
  if (DSIZE_8 == layer_info->data_size) {
    llvm_unreachable("not implement");
  }
  layer_info->fw_layer_param_u.fw_arg_layer_param = layer_param;
  return sizeof(fw_arg_layer_param_t);
}

int64_t tpu::ArgOp::get_fw_type_bm1684() { return FW_BMNET_ARG; }
