//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1690.h"

using namespace tpu_mlir::backend;

void BM1690::load_functions() {
  BM1684X::load_functions();
  CAST_FUNCTION(tpu_sync_all);
  CAST_FUNCTION(tpu_core_context_setup);
  CAST_FUNCTION(sdma_tensor_general_move_gen_cmd);
  CAST_FUNCTION(tensor_stride_move_reduce_gen_cmd);
}

void BM1690::before_codegen() {
  BM1684X::before_codegen();
  useCode0 = true;
}

void BM1690::after_codegen(int64_t flops) {
  BM168x::after_codegen(flops);
  for (int i = 0, n = multiCode.size(); i < n; i++) {
    useCore(i);
    dl_store_cmd_end();
  }
  // useCore(0); // Reset buffer swapping.
  useCode0 = false;
}

void BM1690::setCoreNum(int core) {
  for (int i = core - multiCode.size(); i > 0; i--) {
    multiCode.push_back(std::make_unique<BM168x::Code>());
    // initialize all of them
    auto _code = multiCode.back();
    _code->cmdid_node = dl_create_cmd_id_node();
    _code->bdc_node = dl_create_cmd_id_node();
    _code->gdma_node = dl_create_cmd_id_node();
  }
}

int BM1690::getCurrentCoreID() {
  for (auto [id, _code] : llvm::enumerate(multiCode)) {
    if (_code == code)
      return id;
  }
  llvm_unreachable("cant not find current codeGen core.");
}

void BM1690::useCore(int coreID) {
  if (code == multiCode[coreID]) {
    return;
  }
  code = multiCode[coreID];
  dl_backend_api_set_core_info(coreID, core_num);
}
