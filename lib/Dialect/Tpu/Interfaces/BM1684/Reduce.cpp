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

#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::backend;

void tpu::ReduceOp::codegen_global_bm1684() {
  int i_dims = module::getShape(getInput()).size();
  auto in_addr = module::getAddress(getInput());
  auto out_addr = module::getAddress(getOutput());
  // int input_shapes[i_dims];
  // module::getGlobalShape(getInput(), input_shapes);
  uint32_t *input_shape = new uint32_t[MAX_SHAPE_DIMS];
  for (auto v : llvm::enumerate(module::getShape(getInput())))
    input_shape[v.index()] = (uint32_t)v.value();
  int method = BM1684::get_reduce_type(getMode());
  auto &&axes = getAxes();
  int axis_num = axes.size();
  int axis_list[axis_num];
  for (int i = 0; i < axes.size(); i++)
    axis_list[i] = (axes[i].cast<IntegerAttr>().getInt());
  auto buffer_addr = module::getAddress(getBuffer());
  if (method == 7 /*BM_REUDCE_L2*/) {
    llvm_unreachable("ReduceL2 Not Support");
  } else {
    if (false == module::isUniformQuantized(getInput())) {
      BM1684::instance().dl_nodechip_reduce_full_v3(
          in_addr, out_addr, input_shape, i_dims, axis_list, axis_num, method,
          buffer_addr, 0, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    } else {
      int keep_dims = getKeepdims() ? 1 : 0;
      int bottom_sign = module::isSign(getInput()) ? 1 : 0;
      int store_mode = STORE_MODE_4N;
      float bottom_scale = 1.0f;
      float top_scale = 1.0f;
      BM1684::instance().dl_nodechip_reduce_full_fix8b(
          in_addr, out_addr, buffer_addr, input_shape, i_dims, axis_list,
          axis_num, method, keep_dims, bottom_sign, store_mode, bottom_scale,
          top_scale, (CMD_ID_NODE *)BM1684::instance().cmdid_node);
    }
  }
}

uint32_t tpu::ReduceOp::dyn_codegen_global_bm1684(void* ir_layer_info) {
  llvm_unreachable("Not Implemented");
  return 0;
}
int64_t tpu::ReduceOp::get_fw_type_bm1684() {
  return -1;
}