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

void tpu::LoadOp::codegen_global_bm1684x() {
  llvm_unreachable("global not support");
}

int64_t tpu::LoadOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type) {
  return 0;
}

void tpu::LoadOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                        int64_t h_step, int64_t d_step,
                                        int64_t w_step, group_type_t group_type,
                                        local_sec_info_t &sec_info) {
  auto pid_node = (CMD_ID_NODE *)(*BM168x::instance())->gdma_node;
  // for matmul second right matrix

  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  if (group_type == GROUP_MM_OPT3 && module::IsRightMat(getOutput())) {
    if (module::IsSecondMatInMlp(getOutput())) {
      gi = getGroupInfo(n_step, 0, d_step, w_step, h_step);
      llvm::errs() << "IsSecondMatInMlp\n";
    }
  }

  int64_t N, C, D, H, W;
  int64_t real_cslice, real_hslice, real_wslice, real_dslice;
  int64_t gdma_format;
  module::getNCDHW(getOutput(), N, C, D, H, W, group_type);
  auto data_type = BM168x::getDataType(getOutput());

  auto in = this->getOperand();
  bool do_nnvlc_decompress = false;
  bool do_nnvlc2_decompress = false;
  uint8_t bias0, bias1;
  int32_t is_signed, zero_guard;
  if (module::isWeight(in)) {
    do_nnvlc_decompress =
        in.getDefiningOp<top::WeightOp>().getDoCompress().has_value() &&
        in.getDefiningOp<top::WeightOp>().getDoCompress().value();
    if (do_nnvlc_decompress) {
      bias0 = (uint8_t)in.getDefiningOp<top::WeightOp>().getBias0().value();
      bias1 = (uint8_t)in.getDefiningOp<top::WeightOp>().getBias1().value();
      is_signed = in.getDefiningOp<top::WeightOp>().getIsSigned().value();
      zero_guard = in.getDefiningOp<top::WeightOp>().getZeroGuard().value();
    }
  } else {
    do_nnvlc2_decompress = this->getCompressInfo().has_value() &&
                           this->getCompressInfo()->getDoDecompress();
    if (do_nnvlc2_decompress) {
      auto cinfo = this->getCompressInfo();
      bias0 = (uint8_t)cinfo->getBias0();
      bias1 = (uint8_t)cinfo->getBias1();
      is_signed = cinfo->getIsSigned();
      zero_guard = cinfo->getZeroGuard();
    }
  }

  real_cslice = gi.c_slice;
  real_hslice = gi.h_slice;
  real_wslice = gi.w_slice;
  real_dslice = gi.d_slice;
  if (data_type == DTYPE_UINT4 || data_type == DTYPE_INT4) {
    gdma_format = BM168x::GDMA_VALUE_FORMAT_INT8;
    data_type = DTYPE_INT8;
    if (gi.h_slice == H) { //  H*W is odd
      if ((W * H & 0x1) == 1) {
        W = align_up(W * H, (int64_t)2) / 2;
        real_wslice = align_up(real_wslice * real_hslice, (int64_t)2) / 2;
        H = 1;
        real_hslice = 1;
      } else {                //  H*W is even
        if ((W & 0x1) == 1) { // W is odd and H is even
          real_hslice >>= 1;  // real_hslice is even ?  to do
          H = H >> 1;
        } else {
          W >>= 1;
          real_wslice >>= 1;
        }
      }
    } else {
      real_hslice = gi.h_slice; // to do for int4
      W >>= 1;
      real_wslice >>= 1;
    }
  }
  gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  int64_t g_addr = -1;
  auto input = getInput();
  auto inputOp = input.getDefiningOp();
  // if (inputOp != nullptr && !input.isa<BlockArgument>()) {
  if (inputOp != nullptr && module::isOpInGroup(inputOp)) {
    // In the case of tensor store followed by load, the input to load is the
    // output of the previous store
    if (isa<tpu::StoreOp>(inputOp)) {
      auto buffer = inputOp->getOperand(1);
      assert(!isa<top::NoneOp>(buffer.getDefiningOp()));
      g_addr = module::getAddress(buffer);
    }
    if (isa<tpu::LoadToL2MOp>(inputOp)) {
      g_addr = module::getAddress(inputOp->getOperand(1));
    }

    for (auto user : inputOp->getUsers()) {
      if (isa<tpu::StoreOp>(user)) {
        auto user2 = *user->getUsers().begin();
        if (isa<tpu::SliceMergeOp>(user2)) {
          g_addr = module::getAddress(user2->getResult(0));
        } else {
          g_addr = module::getAddress(user->getResult(0));
          // for (auto [i, opd]: llvm::enumerate(user2->getOperands())) {
          // }
        }
        break;
      }
    }
  } else {
    g_addr = module::getAddress(getInput());
  }
  // llvm::errs() <<"loadOp g_addr:"<<g_addr<<"\n";
#if 0
  //note: trick for imgToCol pattern
  const auto defBySliceOp = [&] (Value v) {
      std::function<std::pair<bool, Operation *>(Value)>
         f = [&f] (Value v) -> std::pair<bool, Operation *> {
      if (isa<BlockArgument>(v)) {
        auto argument = cast<BlockArgument>(v);
        int index = argument.getArgNumber();
        auto op = argument.getOwner()->getParentOp();
        if (isa<FunctionOpInterface>(op))
          return std::make_pair(false, nullptr);
        else {
          auto in = op->getOpOperand(index).get();
          return  f(in);
        }
      } else if (isa<tpu::ReshapeOp>(v.getDefiningOp())) {
        auto in = v.getDefiningOp()->getOpOperand(0).get();
        return f(in);
      } else if (isa<tpu::SliceOp>(v.getDefiningOp())) {
        auto sliceOp = cast<tpu::SliceOp>(v.getDefiningOp());
        if (sliceOp.getDiscard())
          return std::make_pair(true, v.getDefiningOp());
        else
          return std::make_pair(false, nullptr);
      } else
        return std::make_pair(false, nullptr);};
    return f(v);};
  auto [fromSlice, sliceOp] = defBySliceOp(getInput());
#endif
  // int64_t dhw = D * H * W;
  // int64_t eu_num = BM168x::eu_num(fmt_bytes);
  int64_t use_3ic = getUse_3icOptimize();
  if (use_3ic < 4 && use_3ic > 0) {
    auto g_stride = BM168x::getGlobalStride(N, C, H, W);
    if (getDoBcast() == true) {
      real_cslice = BM168x::NPU_NUM;
      g_stride.N = 0;
      g_stride.C = 0;
      g_stride.H = 0;
    }
    auto s_stride = BM168x::getLocalStride(gi.n_slice, real_cslice, real_hslice,
                                           real_wslice, fmt_bytes, gi.eu_align);
    int64_t g_offset = (gi.n_idx * g_stride.N + gi.c_idx * g_stride.C +
                        gi.h_idx * g_stride.H + gi.w_idx) *
                       fmt_bytes;
    auto use_op = *getOutput().user_begin();
    auto conv_op = dyn_cast<tpu::Conv2DOp>(use_op);
    auto kernel = module::getI64Array(conv_op.getKernelShape());
    int64_t to_ic =
        use_3ic == 1
            ? kernel->at(0)
            : (use_3ic == 2 ? kernel->at(1) : kernel->at(0) * kernel->at(1));
    assert(real_cslice * to_ic <= Arch::NPU_NUM);
    for (int64_t i = 0; i < real_cslice; ++i) {
      BM168x::instance()->dl_tensor_broadcast_move_gen_cmd(
          g_addr + g_offset + i * W * H * fmt_bytes, 0, gi.out_addr, i * to_ic,
          gi.n_slice, real_hslice, real_wslice, to_ic, g_stride.N, g_stride.H,
          s_stride.N, s_stride.H, gdma_format, true, GDMA_VALUE_DIR_S2L,
          pid_node);
    }
  } else if (do_nnvlc_decompress) {
    if (!module::isBM1688()) {
      llvm_unreachable("compress only support bm1688");
    }
    // nnvlc1.0
    auto ishape = in.getDefiningOp<top::WeightOp>().getType().getShape();
    N = ishape[0];
    C = ishape[1];
    H = ishape[2];
    W = (ishape.size() == 3) ? 1 : ishape[3];
    int stride_h = W;
    int stride_c = W * H;
    if (gi.eu_align) {
      stride_c = align_up(W * H, Arch::eu_num(fmt_bytes));
    }
    int stride_n = ceiling_func(C, Arch::NPU_NUM) * stride_c;
    BM168x::instance()->dl_tensor_normal_decompress_gen_cmd(
        gi.out_addr, g_addr, N, C, H, W, stride_n, stride_c, stride_h, bias0,
        bias1, is_signed, zero_guard, gdma_format, pid_node);
  } else {
    int64_t c_num_local = ceiling_func(real_cslice, Arch::NPU_NUM);
    int64_t c_stride = gi.eu_align ? align_up(real_hslice * real_wslice,
                                              Arch::eu_num(fmt_bytes))
                                   : real_hslice * real_wslice;
    int64_t channel_num = real_cslice;
    const int64_t csecs = ceiling_func(channel_num, (int64_t)MAX_TPU_DIM);
    if (real_dslice <= gi.n_slice) {
      for (int64_t d = 0; d < real_dslice; d++) {
        int64_t channel_index = 0;
        while (channel_index < csecs) {
          int64_t cur_cslice =
              std::min(channel_num - channel_index * (int64_t)MAX_TPU_DIM,
                       (int64_t)MAX_TPU_DIM);
          int64_t real_c_num_local =
              (channel_index * (int64_t)MAX_TPU_DIM) / Arch::NPU_NUM;
          int64_t dst_offset_c = real_c_num_local * c_stride * fmt_bytes;
          int64_t real_npu_idx =
              (channel_index * (int64_t)MAX_TPU_DIM) % Arch::NPU_NUM;
          int64_t cur_local_offset =
              d * gi.n_slice * c_num_local * c_stride * fmt_bytes +
              dst_offset_c;
          if (do_nnvlc2_decompress) {
            shape_t nnvlc_shape;
            nnvlc_shape.n = N;
            nnvlc_shape.c = C;
            nnvlc_shape.h = H;
            nnvlc_shape.w = W;
            auto nnvlc_dtype = module::getStorageType(getOutput());
            int max_meta_bytes = tpu_compress_RACU_max_meta_bytes(nnvlc_shape);
            shape_t meta_stride = tpu_compress_RACU_meta_stride(nnvlc_shape);
            shape_t racu_stride =
                tpu_compress_RACU_racu_stride(nnvlc_shape, nnvlc_dtype);
            int64_t meta_gaddr =
                g_addr + (gi.n_idx * meta_stride.n +
                          div_up(gi.c_idx, Arch::NPU_NUM) * meta_stride.c +
                          gi.h_idx * meta_stride.h + gi.w_idx * meta_stride.w) *
                             4;
            int64_t racu_gaddr =
                g_addr + align_up(max_meta_bytes, Arch::EU_BYTES) +
                (gi.n_idx * racu_stride.n +
                 div_up(gi.c_idx, Arch::NPU_NUM) * racu_stride.c +
                 gi.h_idx * racu_stride.h + gi.w_idx * racu_stride.w);

            BM168x::instance()->dl_tensor_racu_decompress_gen_cmd(
                gi.out_addr + cur_local_offset, racu_gaddr, meta_gaddr,
                gi.n_slice, cur_cslice, real_hslice, real_wslice,
                c_num_local * c_stride, c_stride, real_wslice, racu_stride.n,
                racu_stride.c, racu_stride.h, meta_stride.n, meta_stride.c,
                bias0, bias1, is_signed, zero_guard, gdma_format, pid_node);
          } else {
            int64_t src_offset_c =
                (channel_index * (int64_t)MAX_TPU_DIM + gi.c_idx) * H * W *
                fmt_bytes;
            int64_t cur_global_offset = gi.n_idx * C * D * H * W * fmt_bytes +
                                        (gi.d_idx + d) * H * W * fmt_bytes +
                                        gi.h_idx * W * fmt_bytes +
                                        gi.w_idx * fmt_bytes + src_offset_c;
            if (module::isDebugCmdEnable("codegen_debug")) {
              llvm::errs() << "loadOp, gi.n_idx:" << gi.n_idx
                           << ", gi.c_idx:" << gi.c_idx
                           << ", gi.d_idx:" << gi.d_idx
                           << ", gi.h_idx:" << gi.h_idx
                           << ", gi.w_idx:" << gi.w_idx << ", d:" << d
                           << ", C:" << C << ", D:" << D << ", H:" << H
                           << ", W:" << W << ", gi.out_addr:" << gi.out_addr
                           << ", cur_local_offset:" << cur_local_offset
                           << ", g_addr:" << g_addr
                           << ", cur_global_offset:" << cur_global_offset
                           << ", gi.n_slice:" << gi.n_slice
                           << ", cur_cslice:" << cur_cslice
                           << ", real_hslice:" << real_hslice
                           << ", real_wslice:" << real_wslice
                           << ", c_num_local:" << c_num_local
                           << ", c_stride:" << c_stride << "\n";
            }
            BM168x::instance()->dl_tensor_stride_move_gen_cmd(
                gi.out_addr + cur_local_offset, real_npu_idx,
                g_addr + cur_global_offset, gi.n_slice, cur_cslice, real_hslice,
                real_wslice, C * D * H * W, D * H * W, W, 1,
                c_num_local * c_stride, c_stride, real_wslice, 1, gdma_format,
                GDMA_VALUE_DIR_S2L, 0, pid_node);
          }
          channel_index++;
        }
      }      // depth loop
    } else { // HAVE DEPTH,3D [N,C,D,H,W]->[d,n_slice,c,h_slice,w]
      for (int64_t i = 0; i < gi.n_slice; i++) {
        int64_t cur_local_offset = i * c_num_local * c_stride * fmt_bytes;
        int64_t cur_global_offset =
            (gi.n_idx + i) * C * D * H * W * fmt_bytes +
            gi.c_idx * D * H * W * fmt_bytes + gi.d_idx * H * W * fmt_bytes +
            gi.h_idx * W * fmt_bytes + gi.w_idx * fmt_bytes;
        BM168x::instance()->dl_tensor_stride_move_gen_cmd(
            gi.out_addr + cur_local_offset, 0, g_addr + cur_global_offset,
            real_dslice, real_cslice, real_hslice, real_wslice, H * W,
            D * H * W, W, 1, gi.n_slice * c_num_local * c_stride, c_stride,
            real_wslice, 1, gdma_format, GDMA_VALUE_DIR_S2L, 0, pid_node);
      } // nslice loop
    }
  }
}

// dynamic codegen
int64_t tpu::LoadOp::dyn_codegen_local_bm1684x(void *buffer) {
  // no need to implement it
  return 0;
}

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::LoadOp::dyn_codegen_global_bm1684x(void *buffer) {
  // no need to implement it
  return 0;
}

int64_t tpu::LoadOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
