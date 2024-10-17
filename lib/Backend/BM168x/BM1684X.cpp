//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace tpu_mlir::backend;

uint32_t BM1684X::get_bdc_len(int bdc_num, int group_id) {
  if (bdc_num == 0) {
    return 0;
  }
  auto group_num = dl_backend_api_get_tpu_inst_group_number();
  assert(group_id < group_num);
  return dl_backend_api_get_tpu_inst_size_per_group("tiu:0:0")[group_id];
}

uint32_t BM1684X::get_gdma_len(int gdma_num, int group_id) {
  if (gdma_num == 0) {
    return 0;
  }
  auto group_num = dl_backend_api_get_tpu_inst_group_number();
  assert(group_id < group_num);
  return dl_backend_api_get_tpu_inst_size_per_group("gdma:0:0")[group_id];
}

void BM1684X::load_functions() {
  BM168x::load_functions();
  CAST_FUNCTION(cmd_id_divide);
  CAST_FUNCTION(cmd_id_merge);
  CAST_FUNCTION(load_lookup_tables);
  CAST_FUNCTION(store_cmd_end);
  CAST_FUNCTION(set_cmd_len_ptr);
  CAST_FUNCTION(sg_set_profile_dump);
  CAST_FUNCTION(sg_set_profile_path);
  CAST_FUNCTION(sg_stas_dump);
  CAST_FUNCTION(sg_flops_dump);
  CAST_FUNCTION(sg_stas_reset);
  CAST_FUNCTION(tensor_broadcast_move_gen_cmd);
  CAST_FUNCTION(tpu_set_id_node);
  CAST_FUNCTION(tpu_get_id_node);

  // multi-core switch interface from backend
  CAST_FUNCTION(backend_api_set_core_info);
  CAST_FUNCTION(backend_api_get_tpu_inst_size);
  CAST_FUNCTION(backend_api_get_tpu_inst_data);
  CAST_FUNCTION(backend_api_get_tpu_inst_group_number);
  CAST_FUNCTION(backend_api_get_tpu_inst_number_per_group);
  CAST_FUNCTION(backend_api_get_tpu_inst_size_per_group);
  CAST_FUNCTION(backend_api_clear_tpu_inst_data);
  CAST_FUNCTION(backend_api_get_total_id);
}

void BM1684X::start_env() {
  BM168x::start_env();
  assert(core_num != 0);
  dl_backend_api_set_core_info(0, core_num);
  dl_allow_store_cmd(); // TODO: remove it!
  dl_forbid_atomic_cmodel();
  dl_load_lookup_tables();
}

void BM1684X::before_codegen() {
  dl_allow_store_cmd();
  BM168x::before_codegen();
  dl_backend_api_clear_tpu_inst_data();
}

void BM1684X::after_codegen(int64_t flops) {
  BM168x::after_codegen(flops);
  dl_store_cmd_end();
  dl_forbid_store_cmd();
}

void BM1684X::load_custom_functions() {
  CAST_FUNCTION(set_id_node);
  CAST_FUNCTION(enable_active_mode);
  CAST_FUNCTION(set_ts_fe_cmd_id_ptr);
}
