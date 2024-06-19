//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/TPUNnvlcUtil.h"
#include "tpu_mlir/Backend/BM168x/BM1690.h"

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
                                        int64_t w_step, group_type_t group_type,
                                        local_sec_info_t &sec_info) {
  return;
}

void tpu::LoadToL2MOp::codegen_only_for_LoadToL2MOp() {
  auto op = getOperation();
  auto dst_addr = BM1690::L2_SRAM_START_ADDR + getL2mAddr();

  auto opd = op->getOperand(0).getDefiningOp();
  auto src_addr = module::getAddress(opd->getResult(0));
  llvm::errs() <<"LoadToL2MOp dst_addr:"<<dst_addr<<" src_addr:"<<src_addr<<"\n";

  assert((BM1690::COEFF_START_ADDR && src_addr >= BM1690::COEFF_START_ADDR) ||
      (BM1690::CTX_START_ADDR && src_addr >= BM1690::CTX_START_ADDR));

  // assert(BM1690::COEFF_START_ADDR && BM1690::CTX_START_ADDR &&
  //     BM1690::L2_SRAM_START_ADDR && dst_addr > BM1690::COEFF_START_ADDR &&
  //     dst_addr > BM1690::CTX_START_ADDR &&
  //     dst_addr >= BM1690::L2_SRAM_START_ADDR);

  auto shape = op->getOperand(0).getType().cast<RankedTensorType>().getShape();
  int N = 1, C = 1, H = 1, W = 1;
  if (shape.size() == 4) {
    N = shape[0];
    C = shape[1];
    H = shape[2];
    W = shape[3];
  } else if (shape.size() == 2) {
    H = shape[0];
    W = shape[1];
  }

  auto data_type = BM1690::getDataType(getOutput());
  auto gdma_format = BM1690::getGdmaFormat(data_type);
  // gdma_format = BM1690::GDMA_VALUE_FORMAT_FLOAT16;

  auto pid_node = (CMD_ID_NODE *)BM1690::instance()->cmdid_node;
  BM1690::instance().dl_sdma_tensor_general_move_gen_cmd(
      src_addr,
      N, C, H, W,
      C * H * W, H * W, W, 1,
      gdma_format,
      dst_addr,
      N, C, H, W,
      C * H * W, H * W, W, 1,
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
