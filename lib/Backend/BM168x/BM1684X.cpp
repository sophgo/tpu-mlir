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
  assert(group_id < code->bdc_bytes.size());
  return code->bdc_bytes[group_id];
}

uint32_t BM1684X::get_gdma_len(int gdma_num, int group_id) {
  if (gdma_num == 0) {
    return 0;
  }
  assert(group_id < code->gdma_bytes.size());
  return code->gdma_bytes[group_id];
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
}

void BM1684X::start_env() {
  BM168x::start_env();
  dl_load_lookup_tables();
  dl_set_cmd_len_ptr((void *)&code->gdma_bytes, (void *)&code->bdc_bytes);
}

void BM1684X::after_codegen(int64_t flops) {
  BM168x::after_codegen(flops);
  if (module::getChip() == module::Chip::BM1686 || module::getChip() == module::Chip::CV186X) {
    int cmd_type = (gdma_buffer[1] & 0x0f);
    if(cmd_type != 6) {
      uint64_t src_addr = ((uint64_t)(gdma_buffer[17] & 0xff) << 32) | ((uint64_t)gdma_buffer[16]);
      uint64_t dst_addr = ((uint64_t)(gdma_buffer[19] & 0xff) << 32) | ((uint64_t)gdma_buffer[18]);
      bool src_in_global = (src_addr >> 39) & 0x1;
      bool dst_in_global = (dst_addr >> 39) & 0x1;
      if (src_in_global) {
        uint64_t origin_addr = src_addr & ((1ull << 35) - 1);
        if (origin_addr >= module::getCoeffAddr() && origin_addr < module::getNeuronAddr()) {
          gdma_buffer[17] = (gdma_buffer[17] & 0x9f) | 0x10;
        } else if (origin_addr >= module::getNeuronAddr() && origin_addr < module::getNeuronAddr() + module::getNeuronSize()) {
          gdma_buffer[17] = (gdma_buffer[17] & 0xaf) | 0x20;
        }
      }
      if (dst_in_global) {
        uint64_t origin_addr =  dst_addr & ((1ull << 35) - 1);
        if (origin_addr >= module::getCoeffAddr() && origin_addr < module::getNeuronAddr()) {
          gdma_buffer[19] = (gdma_buffer[19] & 0x9f) | 0x10;
        } else if (origin_addr >= module::getNeuronAddr() && origin_addr < module::getNeuronAddr() + module::getNeuronSize()) {
          gdma_buffer[19] = (gdma_buffer[19] & 0xaf) | 0x20;
        }
      }
      // cmd type: 0:DMA_tensor, 1:DMA_matrix, 2:DMA_masked_select, 3:DMA_general
      // 4:DMA_cw_trans, 5:DMA_nonzero, 6:DMA_sys, 7:DMA_gather, 8:DMA_scatter
      // 9:DMA_reverse 10:DMA_compress 11: DMA_decompress
      if (cmd_type == 2 || cmd_type == 7 || cmd_type == 8 || cmd_type == 0xa || cmd_type == 0xb) {
          uint64_t index_addr = ((uint64_t)(gdma_buffer[21] & 0xff) << 32) | ((uint64_t)gdma_buffer[20]);
          if ((index_addr >> 39) & 0x1) {
            uint64_t origin_addr = index_addr & ((1ull << 35) - 1);
            if (origin_addr >= module::getCoeffAddr() && origin_addr < module::getNeuronAddr()) {
              gdma_buffer[21] = (gdma_buffer[21] & 0x9f) | 0x10;
            } else if (origin_addr >= module::getNeuronAddr() && origin_addr < module::getNeuronAddr() + module::getNeuronSize()) {
              gdma_buffer[21] = (gdma_buffer[21] & 0xaf) | 0x20;
            }
          }
        }
    }
  }
  dl_store_cmd_end();
}
