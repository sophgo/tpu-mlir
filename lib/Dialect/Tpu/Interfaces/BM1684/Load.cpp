//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/BM168x/BM1684.h"

#include "tpu_mlir/Support/MathUtils.h"

using namespace tpu_mlir::backend;

void tpu::LoadOp::codegen_global_bm1684() {
  UNREACHABLE_THIS("Not Implemented");
}

int64_t tpu::LoadOp::getBufferSize_bm1684(int64_t in_lmem_bytes,
                                          int64_t out_lmem_bytes,
                                          int64_t in_nslice, int64_t in_hslice,
                                          int64_t out_nslice,
                                          int64_t out_hslice) {
  return 0;
}

void tpu::LoadOp::codegen_local_bm1684(int64_t n_step, int64_t h_step,
                                       local_sec_info_t &sec_info) {
  auto pid_node = (CMD_ID_NODE *)(*BM168x::instance())->gdma_node;
  auto gi = getGroupInfo(n_step, h_step, 0, 0, 0);
  auto local_offset = gi.buffer_addr;
  assert(false == gi.overstepped);
  auto data_type = BM168x::getDataType(getOutput());
  // FP32 INT8 INT16 are all move as FLOAT32 because of 2N 4N store mode
  auto gdma_format = BM168x::GDMA_VALUE_FORMAT_FLOAT32;
  int64_t fmt_bytes = (int64_t)BM168x::getFmtBytes(DTYPE_FP32);
  int64_t N, C, D, H, W;
  module::getNCDHW(getOutput(), N, C, D, H, W, (group_type_t)gi.type);
  if (getDoBcast()) {
    C = BM168x::NPU_NUM;
  }
  int64_t c_num_local = ceiling_func(C, BM168x::NPU_NUM);
  int64_t n_idx = gi.n_idx;
  int64_t local_N = gi.n_slice, local_C = C, local_H = gi.h_slice, local_W = W;
  if (fmt_bytes != 4) {
    auto castOp = dyn_cast_or_null<top::WeightOp>(getInput().getDefiningOp());
    if (!module::isWeight(getInput()) ||
        (castOp.getStoreMode().has_value() && castOp.getStoreMode() == "4N")) {
      int64_t N_align = 4 / fmt_bytes;
      fmt_bytes = 4;
      gdma_format = BM168x::GDMA_VALUE_FORMAT_FLOAT32;
      N = ceiling_func(N, N_align);
      local_N = ceiling_func(gi.n_slice, N_align);
      n_idx = gi.n_idx / 4;
      assert(gi.n_idx % 4 == 0);
    } else if (castOp.getStoreMode().has_value() &&
               castOp.getStoreMode() == "2N") {
      int64_t N_align = 4 / fmt_bytes;
      fmt_bytes = 4;
      gdma_format = BM168x::GDMA_VALUE_FORMAT_FLOAT32;
      N = ceiling_func(N, N_align);
      local_N = ceiling_func(gi.n_slice, N_align);
      n_idx = gi.n_idx / 2;
      assert(gi.n_idx % 2 == 0);
    } else {
      // do nothing
    }
  }
  auto g_stride = BM168x::getGlobalStride(N, C, H, W);
  g_stride.N *= D;
  g_stride.C *= D;
  if (getDoBcast() == true) {
    g_stride.N = 0;
    g_stride.C = 0;
    g_stride.H = 0;
  }
  int64_t c_local_stride =
      gi.eu_align ? align_up(local_H * local_W, Arch::eu_num(fmt_bytes))
                  : local_H * local_W;
  auto g_addr = module::getAddress(getInput());

  int64_t use_3ic = getUse_3icOptimize();
  if (use_3ic) {
    auto use_op = *getOutput().getUsers().begin();
    auto conv_op = dyn_cast<tpu::Conv2DOp>(use_op);
    int64_t Nin, Cin, Hin, Win;
    module::getNCHW(conv_op.getInput(), Nin, Cin, Hin, Win);
    auto conv_param = conv_op.parseParam();
    auto kernel = module::getI64Array(conv_op.getKernelShape());
    int data_len = 4;
    int n_idx_trans = n_idx / 4;
    int n_slice_trans = ceiling_func(gi.n_slice, 4);
    for (int i = 0; i < n_slice_trans; i++) {
      int src_N = Cin; // ic
      int src_C = conv_param.kh;
      int src_H = gi.h_slice;
      int src_W = gi.w_slice; // w

      int src_N_stride = (Hin + conv_param.pht + conv_param.pwl) * Win;
      int src_C_stride = Win;
      int src_H_stride = Win * conv_param.sh;
      int src_W_stride = 1;

      int dst_N = 1;
      int dst_C = Cin * conv_param.kh;
      int dst_H = gi.h_slice;
      int dst_W = gi.w_slice;

      int dst_W_stride = 1;
      int dst_H_stride = dst_W;

      auto NPU_NUM = BM1684::NPU_NUM;
      auto EU_NUM = BM1684::eu_num(sizeof(float));
      int dst_C_stride = (gi.h_slice * dst_W + EU_NUM - 1) / EU_NUM * EU_NUM;
      int dst_N_stride =
          (Cin * conv_param.kh + NPU_NUM - 1) / NPU_NUM * dst_C_stride;
      BM1684::instance().dl_tensor_general_move_gen_cmd(
          g_addr +
              (uint64_t)(n_idx_trans + i) * Cin *
                  (Hin + conv_param.pht + conv_param.phb) * Win * data_len +
              (uint64_t)(gi.h_idx > 0 ? gi.h_idx + conv_param.pht : gi.h_idx) *
                  Win * data_len +
              gi.w_idx * data_len,
          0, // no use
          src_N, src_C, src_H, src_W, src_N_stride, src_C_stride, src_H_stride,
          src_W_stride, 0, local_offset + i * dst_N_stride * data_len, 0, dst_N,
          dst_C, dst_H, dst_W, dst_N_stride, dst_C_stride, dst_H_stride,
          dst_W_stride, 0,
          GDMA_VALUE_DIR_S2L, // 0
          0, pid_node);
      return;
    }
  }

  // table in lut have to be store in L2 SRAM
  if (BM1684::instance().isL2Load(getOperation())) {
    BM1684::instance().dl_tensor_general_move_gen_cmd(
        g_addr,                         /*local_addr or global_addr*/
        0, 1, 1, 1, 256, 1, 1, 1, 1, 0, /*GDMA_VALUE_FORMAT_FLOAT32*/
        gi.out_addr + 0x10000000,       /*0x10000000 is L2SRAM start addr*/
        0, 1, 1, 1, 256, 1, 1, 1, 1, 0, /*GDMA_VALUE_FORMAT_FLOAT32*/
        GDMA_VALUE_DIR_S2S, 0, pid_node);
  } else {
    auto store_mode = BM168x::getStoreMode(getOutput());
    int64_t n_idx_trans = n_idx, local_N_trans = local_N;
    if (module::isWeight(getInput())) {
      // bm1684_tensor_gdma_ctrl::coeff_load_cmdgen
      gdma_format = BM168x::getGdmaFormat(data_type);
      fmt_bytes = (int64_t)BM168x::getFmtBytes(data_type);
      // bm1684_tensor_gdma_ctrl::coeff_neuron_load_cmdgen
      auto castOp = dyn_cast_or_null<top::WeightOp>(getInput().getDefiningOp());
      if (castOp.getStoreMode().has_value() && castOp.getStoreMode() == "4N") {
        int64_t N_align = 4 / fmt_bytes;
        fmt_bytes = 4;
        gdma_format = BM168x::GDMA_VALUE_FORMAT_FLOAT32;
        N = ceiling_func(N, N_align);
        local_N = ceiling_func(gi.n_slice, N_align);
        n_idx_trans = gi.n_idx / 4;
        assert(gi.n_idx % 4 == 0);
      } else if (castOp.getStoreMode().has_value() &&
                 castOp.getStoreMode() == "2N") {
        int64_t N_align = 4 / fmt_bytes;
        fmt_bytes = 4;
        gdma_format = BM168x::GDMA_VALUE_FORMAT_FLOAT32;
        N = ceiling_func(N, N_align);
        local_N = ceiling_func(gi.n_slice, N_align);
        n_idx_trans = gi.n_idx / 2;
        assert(gi.n_idx % 2 == 0);
      } else {
        // do nothing
      }
    } else {
      // bm1684_tensor_gdma_ctrl::neuron_load_cmdgen
      if (STORE_MODE_1N == store_mode) {
        n_idx_trans = n_idx;
        local_N_trans = local_N;
        gdma_format = BM168x::getGdmaFormat(data_type);
        if (BM168x::GDMA_VALUE_FORMAT_INT8 == gdma_format) {
          c_local_stride =
              align_up(local_H * local_W, Arch::eu_num(fmt_bytes) * 4);
        } else if (BM168x::GDMA_VALUE_FORMAT_INT16 == gdma_format) {
          c_local_stride =
              align_up(local_H * local_W, Arch::eu_num(fmt_bytes) * 2);
        }
        fmt_bytes = (int64_t)BM168x::getFmtBytes(data_type);
      } else {
        switch (data_type) {
        case DTYPE_INT8:
        case DTYPE_UINT8:
          n_idx_trans = n_idx / 4;
          local_N_trans = ceiling_func(local_N, 4);
          break;
        case DTYPE_INT16:
        case DTYPE_UINT16:
        case DTYPE_FP16:
          n_idx_trans = n_idx / 2;
          local_N_trans = ceiling_func(local_N, 2);
          break;
        default:
          n_idx_trans = n_idx;
          local_N_trans = local_N;
          break;
        }
      }
    }
    for (int64_t d_idx = 0; d_idx < D; d_idx++) {
      int64_t cur_local_offset =
          d_idx * local_N_trans * c_num_local * c_local_stride * fmt_bytes;
      int64_t npu_idx = 0;
      int64_t c_idx_in = 0;
      if (fmt_bytes != 4) {
        c_num_local = ceiling_func(C * D, BM168x::NPU_NUM);
        npu_idx = (c_idx_in + d_idx * C) % BM168x::NPU_NUM;
        cur_local_offset = (c_idx_in + d_idx * C) / BM168x::NPU_NUM *
                           c_local_stride * fmt_bytes;
      }
      int64_t cur_global_offset =
          fmt_bytes * (n_idx_trans * C * D * H * W + d_idx * H * W +
                       gi.h_idx * W + gi.w_idx);
      BM168x::instance()->dl_tensor_stride_move_gen_cmd(
          gi.out_addr + cur_local_offset, npu_idx, g_addr + cur_global_offset,
          local_N_trans, local_C, local_H, local_W, g_stride.N, g_stride.C,
          g_stride.H, g_stride.W, c_num_local * c_local_stride, c_local_stride,
          W, 1, gdma_format, GDMA_VALUE_DIR_S2L, 0, pid_node);
    } // depth loop
  }
}

uint32_t tpu::LoadOp::dyn_codegen_global_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}

int64_t tpu::LoadOp::get_fw_type_bm1684() { return -1; }

int32_t tpu::LoadOp::dyn_codegen_local_bm1684(void *ir_layer_info) {
  UNREACHABLE_THIS("Not Implemented");
  return 0;
}
