//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/SG2380.h"

using namespace tpu_mlir::backend;

void SG2380::load_functions() {
  BM1684X::load_functions();
  CAST_FUNCTION(tpu_sync_all);
  CAST_FUNCTION(tpu_core_context_setup);
  CAST_FUNCTION(tensor_normal_decompress_gen_cmd);
  CAST_FUNCTION(tensor_racu_decompress_gen_cmd);
  CAST_FUNCTION(tensor_racu_compress_gen_cmd);
  CAST_FUNCTION(a16mm_data_split_trans);
  CAST_FUNCTION(gen_riscv_code_begin);
  CAST_FUNCTION(gen_riscv_code_end);
}

void SG2380::before_codegen() {
  BM1684X::before_codegen();
  useCode0 = true;
  setenv("GEN_RISCV_CODE", "1", 1);
  dl_gen_riscv_code_begin();
}

void SG2380::after_codegen(int64_t flops) {
  BM168x::after_codegen(flops);
  for (int i = 0, n = multiCode.size(); i < n; i++) {
    useCore(i);
    dl_store_cmd_end();
  }
  useCore(0); // Reset buffer swapping.
  useCode0 = false;
  dl_gen_riscv_code_end();
  // system("riscv_code_opt.py -f riscv_code_0.h");
  // system("riscv_code_opt.py -f riscv_code_1.h");
  // system("riscv_code_opt.py -f riscv_code_2.h");
  // system("riscv_code_opt.py -f riscv_code_3.h");
  unsetenv("GEN_RISCV_CODE");
}

void SG2380::setCoreNum(int core) {
  for (int i = core - multiCode.size(); i > 0; i--) {
    multiCode.push_back(std::make_unique<BM168x::Code>());
    // initialize all of them
    auto _code = multiCode.back();
    _code->cmdid_node = dl_create_cmd_id_node();
    _code->bdc_node = dl_create_cmd_id_node();
    _code->gdma_node = dl_create_cmd_id_node();
  }
}

int SG2380::getCurrentCoreID() {
  for (auto [id, _code] : llvm::enumerate(multiCode)) {
    if (_code == code)
      return id;
  }
  llvm_unreachable("can not find current codeGen core.");
}

void SG2380::useCore(int coreID) {
  if (code == multiCode[coreID]) {
    return;
  }
  code = multiCode[coreID];
  dl_backend_api_set_core_info(coreID, core_num);
}
