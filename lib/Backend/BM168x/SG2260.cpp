//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/SG2260.h"

using namespace tpu_mlir::backend;

void SG2260::load_functions() {
  BM1684X::load_functions();
  CAST_FUNCTION(tpu_sync_all);
  CAST_FUNCTION(tpu_core_context_setup);
}

void SG2260::before_codegen() {
  BM168x::before_codegen();
  useCode0 = true;
}

void SG2260::after_codegen(int64_t flops) {
  BM168x::after_codegen(flops);
  for (int i = 0, n = multiCodes.size(); i < n; i++) {
    useCore(i);
    dl_store_cmd_end();
  }
  useCore(0); // Reset buffer swapping.
  useCode0 = false;
}

void SG2260::setCoreNum(int core) {
  for (int i = core - multiCodes.size(); i > 0; i--) {
    multiCodes.push_back(std::make_unique<BM168x::Code>());
    // initialize all of them
    auto _code = multiCodes.back();
    _code->cmdid_node = dl_create_cmd_id_node();
    _code->bdc_node = dl_create_cmd_id_node();
    _code->gdma_node = dl_create_cmd_id_node();
    _code->gdma_buffer.reserve(0x1000000);
    _code->bdc_buffer.reserve(0x1000000);
    _code->gdma_group_id.clear();
    _code->gdma_group_id.push_back(0);
    _code->bdc_group_id.clear();
    _code->bdc_group_id.push_back(0);
    _code->gdma_bytes.clear();
    _code->bdc_bytes.clear();
    _code->gdma_buffer.clear();
    _code->bdc_buffer.clear();
    _code->cmdid_groupnum = 1;
  }
}

int SG2260::getCurrentCoreID() {
  for (auto [id, _code] : llvm::enumerate(multiCodes)) {
    if (_code == code)
      return id;
  }
  llvm_unreachable("cant not find current codeGen core.");
}

void SG2260::useCore(int coreID) {
  if (code == multiCodes[coreID]) {
    return;
  }

  // We can not configure the backend to switch to another command buffer. This
  // is a workaround solution: swapping the buffer and "use codes[0]" only.
  if (useCode0) { // restore buffer
    auto itr = std::find(multiCodes.begin(), multiCodes.end(), code);
    multiCodes[0]->gdma_buffer.swap((*itr)->gdma_buffer);
    multiCodes[0]->bdc_buffer.swap((*itr)->bdc_buffer);
  }

  code = multiCodes[coreID];

  if (useCode0) { // use buffer 0
    multiCodes[0]->gdma_buffer.swap(code->gdma_buffer);
    multiCodes[0]->bdc_buffer.swap(code->bdc_buffer);
  }
  dl_set_cmd_len_ptr((void *)&code->gdma_bytes, (void *)&code->bdc_bytes);
  dl_set_total_id_ptr(&code->gdma_total_id, &code->bdc_total_id,
                      code->cmdid_node, (void *)&code->gdma_group_id,
                      (void *)&code->bdc_group_id, &code->cmdid_groupnum);
}
