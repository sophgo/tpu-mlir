//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1690.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/TPUNnvlcUtil.h"

using namespace tpu_mlir::backend;

void tpu::LoadToL2MOp::codegen_global_bm1684x() {
  llvm_unreachable("global not support");
}

int64_t tpu::LoadToL2MOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::LoadToL2MOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                             int64_t h_step, int64_t d_step,
                                             int64_t w_step,
                                             group_type_t group_type,
                                             local_sec_info_t &sec_info) {
  return;
}

void tpu::LoadToL2MOp::codegen_only_for_LoadToL2MOp(
    std::pair<int, int> &core_num_idx) {
  if (module::getChip() != module::Chip::BM1690) {
    return;
  }
  auto op = getOperation();
  auto dst_addr = module::getAddress(op->getOperand(1));
  auto src_addr = module::getAddress(op->getOperand(0));
  llvm::errs() << "LoadToL2MOp src_addr:" << src_addr
               << " dst_addr:" << dst_addr << " for "
               << module::getName(op).str() << "\n";
  assert((BM1690::COEFF_START_ADDR && src_addr >= BM1690::COEFF_START_ADDR) ||
         (BM1690::CTX_START_ADDR && src_addr >= BM1690::CTX_START_ADDR));
  int total_size = module::getNumElements(getOutput());
  int num_per_core = ceiling_func(total_size, core_num_idx.first);
  auto move_size = std::min(
      num_per_core, (int)(total_size - num_per_core * core_num_idx.second));
  if (move_size < 1) {
    return;
  }
  auto data_type = BM1690::getDataType(getOutput());
  auto gdma_format = BM1690::getGdmaFormat(data_type);
  auto fmt_bytes = BM1690::getFmtBytes(data_type);
  auto pid_node = (CMD_ID_NODE *)BM1690::instance()->cmdid_node;
  int slice_c = move_size, slice_h = 1;
  if (slice_c > 65535) {
    slice_c = 65535;
    slice_h = ceiling_func(move_size, 65535);
  }
  BM1690::instance().dl_sdma_tensor_general_move_gen_cmd(
      src_addr + num_per_core * core_num_idx.second * fmt_bytes, 1, slice_c,
      slice_h, 1, slice_c * slice_h, slice_h, 1, 1, gdma_format,
      dst_addr + num_per_core * core_num_idx.second * fmt_bytes, 1, slice_c,
      slice_h, 1, slice_c * slice_h, slice_h, 1, 1,
      0,  // transpose
      -1, // port
      pid_node);

  return;
}

// dynamic codegen
int64_t tpu::LoadToL2MOp::dyn_codegen_local_bm1684x(void *buffer) {
  // no need to implement it
  return 0;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::LoadToL2MOp::dyn_codegen_global_bm1684x(void *buffer) {
  // no need to implement it
  return 0;
}

int64_t tpu::LoadToL2MOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
