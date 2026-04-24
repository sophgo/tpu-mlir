//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Common/AffineUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "tpu_mlir/Backend/BM168x/BM1690.h"
#include "tpu_mlir/Backend/BM168x/BM1690E.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/TPUNnvlcUtil.h"
#include <algorithm>
#include <cstring>
#include <mutex>
#include <thread>

using namespace tpu_mlir::backend;
using namespace tpu_mlir;

static AffineSharedCache g_store_affine_shared_cache;
static AffineAnchorCache g_store_affine_anchor_cache;
static std::mutex g_store_affine_cache_mutex;

void tpu::StoreOp::codegen_global_bm1684x() {
  UNREACHABLE_THIS("Not Implemented");
}

int64_t tpu::StoreOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type,
    bool with_hw_margins) {
  return 0;
}

void tpu::StoreOp::codegen_local_bm1684x(int64_t n_step, int64_t c_step,
                                         int64_t h_step, int64_t d_step,
                                         int64_t w_step,
                                         group_type_t group_type,
                                         local_sec_info_t &sec_info) {
  auto op = getOperation();
  CMD_ID_NODE *pid_node = (CMD_ID_NODE *)(*BM168x::instance())->gdma_node;
  // llvm::errs() <<"StoreOp codegen, n_step:"<<n_step<<", c_step:"<<c_step<<",
  // h_step:"<<h_step<<"\n";
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto v = op->getOperand(0);
  auto pre_op = v.getDefiningOp();
  if (isa<tpu::MoveOp>(pre_op)) {
    auto moveOp = dyn_cast<tpu::MoveOp>(pre_op);
    auto vec_move_dest_addr = *module::getI64Array(moveOp.getMoveDestAdd());
    int idx = v.cast<OpResult>().getResultNumber();
    gi.out_addr = vec_move_dest_addr[idx];
    // llvm::errs() <<"StoreOp codegen, idx:"<<idx<<",
    // vec_move_dest_addr:"<<gi.out_addr<<"\n";
  }
  int64_t N, C, D, H, W, real_hslice, real_wslice, real_dslice, real_cslice;

  // set nnvlc param
  bool do_compress = this->getCompressInfo().has_value() &&
                     this->getCompressInfo()->getDoCompress();
  uint8_t bias0, bias1;
  int32_t is_signed, zero_guard;
  if (do_compress) {
    auto cinfo = this->getCompressInfo();
    bias0 = (uint8_t)cinfo->getBias0();
    bias1 = (uint8_t)cinfo->getBias1();
    is_signed = cinfo->getIsSigned();
    zero_guard = cinfo->getZeroGuard();
  }

  int64_t gdma_format;
  auto shape = module::getShape(getInput());
  auto data_type = BM168x::getDataType(getOutput());
  real_hslice = gi.h_slice;
  real_wslice = gi.w_slice;
  real_dslice = gi.d_slice;
  real_cslice = gi.c_slice;
  if (data_type == DTYPE_UINT4 || data_type == DTYPE_INT4) {
    gdma_format = BM168x::GDMA_VALUE_FORMAT_INT8;
    data_type = DTYPE_INT8;
    if (shape.size() == 2) {
      N = 1;
      C = shape[0];
      D = 1;
      H = 1;
      W = align_up(shape[1], (int64_t)2) / 2;
      real_wslice = align_up(shape[1], (int64_t)2) / 2;
    } else {
      module::getNCDHW(getOutput(), N, C, D, H, W, group_type);
      if (gi.h_slice == H) {
        if ((W * H & 0x1) == 1) {
          W = align_up(W * H, (int64_t)2) / 2;
          real_wslice = align_up(real_wslice * real_hslice, (int64_t)2) / 2;
          H = 1;
          real_hslice = 1;
        } else {
          if ((W & 0x1) == 1) {
            real_hslice >>= 1;
            H >>= 1;
          } else {
            W >>= 1;
            real_wslice >>= 1;
          }
        }
      } else {
        real_hslice = gi.h_slice; // to do for int4
      }
    }
  } else {
    module::getNCDHW(getOutput(), N, C, D, H, W, group_type);
  }

  gdma_format = BM168x::getGdmaFormat(data_type);
  auto fmt_bytes = BM168x::getFmtBytes(data_type);
  auto out_value = getOutput();
  int64_t g_addr = -1;

  auto parent = op->getParentOp();
  assert(isa_and_nonnull<tpu::GroupOp>(parent));
  auto users = parent->getResult(0).getUsers();
  auto group_next_op = users.empty() ? nullptr : *(users.begin());
  bool have_more_groupOp = group_next_op && isa<SliceMergeOp>(group_next_op);
  auto buffer = op->getOperand(1);
  if (!isa<top::NoneOp>(buffer.getDefiningOp())) {
    g_addr = module::getAddress(buffer);
    // llvm::errs() <<"get buffer addr: " << g_addr<<"\n";
  } else {
    for (auto user : out_value.getUsers()) {
      if (isa<SliceMergeOp>(user)) {
        auto res = user->getResult(0);
        if (have_more_groupOp) {
          auto yieldOp = *(res.getUsers().begin());
          assert(isa<tpu::YieldOp>(yieldOp));
          int opd_idx = -1;
          for (OpOperand &opd : yieldOp->getOpOperands()) {
            if (res == opd.get()) {
              opd_idx = opd.getOperandNumber();
              break;
            }
          }
          assert(opd_idx >= 0);
          llvm::errs() << "opd_idx:" << opd_idx << "\n";
          group_next_op = *(parent->getResult(opd_idx).getUsers().begin());
          res = group_next_op->getResult(0);
        }
        g_addr = module::getAddress(res);
        break;
      } else if (isa<tpu::YieldOp>(user)) {
        if (have_more_groupOp) {
          int opd_idx = -1;
          for (OpOperand &opd : user->getOpOperands()) {
            if (out_value == opd.get()) {
              opd_idx = opd.getOperandNumber();
              break;
            }
          }
          assert(opd_idx >= 0);
          group_next_op = *(parent->getResult(opd_idx).getUsers().begin());
          g_addr = module::getAddress(group_next_op->getResult(0));
          llvm::errs() << "have_more_groupOp, opd_idx:" << opd_idx
                       << " g_addr:" << g_addr << "\n";
          break;
        }
      }
    }
    if (g_addr == -1) {
      g_addr = module::getAddress(out_value);
    }
  }

  if (getOperation()->hasAttr("indexing_map_l2s")) {
    auto indexingMapAttr =
        getOperation()->getAttr("indexing_map_l2s").dyn_cast<AffineMapAttr>();
    auto indexingMap = indexingMapAttr.getValue();
    int pad_const_val = 0;
    bool has_pad = false;

    if (getOperation()->hasAttr("pad_const_val")) {
      auto const_val_attr =
          getOperation()->getAttr("pad_const_val").dyn_cast<FloatAttr>();
      if (const_val_attr) {
        float float_val = const_val_attr.getValueAsDouble();

        // Convert float to appropriate format based on data_type
        if (data_type == DTYPE_FP16) {
          uint16_t f16_bits = f32_to_f16(float_val);
          std::memcpy(&pad_const_val, &f16_bits, sizeof(uint16_t));
        } else if (data_type == DTYPE_BFP16) {
          uint16_t bf16_bits = f32_to_bf16(float_val);
          std::memcpy(&pad_const_val, &bf16_bits, sizeof(uint16_t));
        } else if (data_type == DTYPE_INT8 || data_type == DTYPE_UINT8) {
          uint8_t uint8_val = (uint8_t)float_val;
          std::memcpy(&pad_const_val, &uint8_val, sizeof(uint8_t));
        } else if (data_type == DTYPE_INT16 || data_type == DTYPE_UINT16) {
          int16_t int16_val = (int16_t)float_val;
          std::memcpy(&pad_const_val, &int16_val, sizeof(int16_t));
        } else {
          // For FP32, INT32, and other types, bit-cast/copy as-is
          std::memcpy(&pad_const_val, &float_val, sizeof(float));
        }
        has_pad = true;
      }
    }
    int64_t iN, iC, iD, iH, iW;
    module::getNCDHW(getInput(), iN, iC, iD, iH, iW, group_type);
    assert(iD == 1);
    std::vector<int64_t> iShape = module::getShape(getInput());
    std::vector<int64_t> iShape_ext = {iN, iC, iH, iW};
    std::vector<int64_t> oShape = module::getShape(getOutput());

    // When Pad is fused, track the valid (non-padding) output region
    std::vector<int64_t> pad_before(oShape.size(), 0);
    std::vector<int64_t> pad_after(oShape.size(), 0);
    if (has_pad) {
      auto paddingsArr =
          module::getI64Array(getPaddings(), oShape.size() * 2, 0);
      if (paddingsArr && paddingsArr->size() >= oShape.size() * 2) {
        for (int i = 0; i < (int)oShape.size(); ++i) {
          pad_before[i] = (*paddingsArr)[i];
          pad_after[i] = (*paddingsArr)[i + oShape.size()];
        }
      }
    }

    std::vector<int64_t> iStride = getCompactStrideFromShape(iShape);
    std::vector<int64_t> iStride_ext = getCompactStrideFromShape(iShape_ext);
    std::vector<int64_t> oStride = getCompactStrideFromShape(oShape);

    // Row-scan approach: directly produce FreeTensorDmaInfo
    std::vector<int64_t> lShape_ext = {gi.n_slice, real_cslice,
                                       real_hslice * real_wslice};
    if (real_cslice % Arch::NPU_NUM == 0) {
      lShape_ext[0] *= real_cslice / Arch::NPU_NUM;
      lShape_ext[1] = Arch::NPU_NUM;
    }
    const std::vector<int64_t> lStride_ext =
        getCompactStrideFromShape(lShape_ext);

    const int64_t lStride_ext_0 = lStride_ext[0];
    const int64_t lStride_ext_1 = lStride_ext[1];
    const int64_t lStride_HW = real_hslice * real_wslice;
    const int64_t real_cslice_lStride = real_cslice * lStride_HW;
    const bool same_i_shape = (iShape == iShape_ext);
    const int64_t map_num_inputs = indexingMap.getNumInputs();

    const int64_t iStride_ext_0 = iStride_ext[0];
    const int64_t iStride_ext_1 = iStride_ext[1];
    const int64_t iStride_ext_2 = iStride_ext[2];
    const int64_t iStride_ext_3 = iStride_ext[3];

    const int64_t n_idx = gi.n_idx;
    const int64_t c_idx = gi.c_idx;
    const int64_t h_idx = gi.h_idx;
    const int64_t w_idx = gi.w_idx;

    // Extract FloorDiv info for step alignment checks
    struct FloorDivInfo {
      int dim = -1;
      int64_t offset = 0;
      int64_t divisor = 1;
    };

    auto isConst = [&](AffineExpr expr, int64_t &val) -> bool {
      if (auto c = expr.dyn_cast<AffineConstantExpr>()) {
        val = c.getValue();
        return true;
      }
      return false;
    };

    std::function<bool(AffineExpr, int &, int64_t &)> matchDimPlusConst;
    matchDimPlusConst = [&](AffineExpr expr, int &dim,
                            int64_t &offset) -> bool {
      if (auto d = expr.dyn_cast<AffineDimExpr>()) {
        dim = d.getPosition();
        offset = 0;
        return true;
      }
      if (auto bin = expr.dyn_cast<AffineBinaryOpExpr>()) {
        int64_t cval = 0;
        int tmpDim = -1;
        int64_t tmpOff = 0;
        if (bin.getKind() == AffineExprKind::Add) {
          if (isConst(bin.getRHS(), cval) &&
              matchDimPlusConst(bin.getLHS(), tmpDim, tmpOff)) {
            dim = tmpDim;
            offset = tmpOff + cval;
            return true;
          }
          if (isConst(bin.getLHS(), cval) &&
              matchDimPlusConst(bin.getRHS(), tmpDim, tmpOff)) {
            dim = tmpDim;
            offset = tmpOff + cval;
            return true;
          }
        }
      }
      return false;
    };

    std::function<bool(AffineExpr, FloorDivInfo &)> extractFloorDiv;
    extractFloorDiv = [&](AffineExpr expr, FloorDivInfo &info) -> bool {
      if (auto bin = expr.dyn_cast<AffineBinaryOpExpr>()) {
        if (bin.getKind() == AffineExprKind::FloorDiv) {
          int64_t divisor = 1;
          if (!isConst(bin.getRHS(), divisor) || divisor == 0)
            return false;
          int dim = -1;
          int64_t offset = 0;
          if (!matchDimPlusConst(bin.getLHS(), dim, offset))
            return false;
          info.dim = dim;
          info.offset = offset;
          info.divisor = divisor;
          return true;
        }
        if (bin.getKind() == AffineExprKind::Add) {
          FloorDivInfo tmp;
          if (extractFloorDiv(bin.getLHS(), tmp)) {
            info = tmp;
            return true;
          }
          if (extractFloorDiv(bin.getRHS(), tmp)) {
            info = tmp;
            return true;
          }
        }
      }
      return false;
    };

    SmallVector<FloorDivInfo, 4> floorDivInfos;
    for (auto expr : indexingMap.getResults()) {
      FloorDivInfo info;
      if (extractFloorDiv(expr, info)) {
        floorDivInfos.push_back(info);
      }
    }

    std::vector<FreeTensorDmaInfo> normal_dma_infos;

    auto is_coords_in_src = [&](const std::vector<int64_t> &coords) {
      if (!is_coords_valid(coords, oShape))
        return false;
      if (!has_pad)
        return true;
      for (int i = 0; i < (int)coords.size(); ++i) {
        if (coords[i] < pad_before[i] ||
            coords[i] >= (oShape[i] - pad_after[i])) {
          return false;
        }
      }
      return true;
    };

    // When real_wslice == 1, H elements are contiguous in local memory
    // (stride=1), so scan along H instead of W to find larger contiguous
    // blocks.
    const bool hslice_as_wslice = (real_wslice == 1 && real_hslice > 1);
    const int64_t scan_size = hslice_as_wslice ? real_hslice : real_wslice;
    const int64_t rows_per_n =
        hslice_as_wslice ? real_cslice : real_cslice * real_hslice;
    const int64_t total_rows = gi.n_slice * rows_per_n;

    // Build cache key
    const std::string shared_cache_key =
        build_affine_shared_cache_key(indexingMap, iShape, oShape, gi.n_slice,
                                      real_cslice, real_hslice, real_wslice);
    const AffineOffsetKey offset_key = {n_idx, c_idx, h_idx, w_idx};

    // Compile affine map for fast evaluation
    const auto compiled_map = compileAffineMap(indexingMap);
    const auto safeDeltaDims = computeSafeDeltaDims(indexingMap);

    // Analyze contiguous block structure for fast scanning
    int scan_dim_idx = -1;
    if (hslice_as_wslice) {
      if (map_num_inputs == (int64_t)iShape.size() && same_i_shape) {
        scan_dim_idx = map_num_inputs >= 3 ? 2 : -1;
      } else if (map_num_inputs >= 3) {
        scan_dim_idx = map_num_inputs - 2;
      }
    } else {
      if (map_num_inputs == (int64_t)iShape.size() && same_i_shape) {
        scan_dim_idx = map_num_inputs - 1;
      } else {
        scan_dim_idx = map_num_inputs - 1;
      }
    }
    AffineBlockAnalysis block_analysis;
    if (scan_dim_idx >= 0) {
      block_analysis =
          analyzeContiguousBlocks(indexingMap, scan_dim_idx, oStride);
    }

    // Check cache: exact match + safe delta reuse
    bool full_cache_hit = false;
    {
      std::lock_guard<std::mutex> lock(g_store_affine_cache_mutex);
      full_cache_hit = affine_cache_lookup_with_delta(
          g_store_affine_shared_cache, g_store_affine_anchor_cache,
          shared_cache_key, offset_key, iStride, oStride, iShape, oShape,
          compiled_map, safeDeltaDims, /*is_load=*/false, same_i_shape,
          normal_dma_infos);
    }

    if (!full_cache_hit) {

      // Process a range of rows [row_begin, row_end) and append results to
      // out_infos
      auto process_row_range = [&](int64_t row_begin, int64_t row_end,
                                   std::vector<FreeTensorDmaInfo> &out_infos) {
        std::vector<int64_t> map_input_coords(map_num_inputs, 0);
        std::vector<int64_t> oCoords(compiled_map.results.size());
        RowPatternCache row_cache;

        for (int64_t row = row_begin; row < row_end; ++row) {
          int64_t n_local, c_local, h_local_base;
          if (hslice_as_wslice) {
            n_local = row / real_cslice;
            c_local = row % real_cslice;
            h_local_base = 0;
          } else {
            n_local = row / rows_per_n;
            int64_t rem = row % rows_per_n;
            c_local = rem / real_hslice;
            h_local_base = rem % real_hslice;
          }

          // Try row pattern cache: reuse block pattern from a previous row
          // with the same (c_local, h_local_base) but different n_local
          RowPatternKey rp_key{c_local, h_local_base};
          auto rp_it = row_cache.find(rp_key);
          if (rp_it != row_cache.end() && !rp_it->second.blocks.empty()) {
            const auto &pattern = rp_it->second;
            auto get_mapped_gOffset_quick = [&](int64_t cur_pos,
                                                int64_t &gOff) -> bool {
              int64_t iCoord_ext_0 = n_local + n_idx;
              int64_t iCoord_ext_1 = c_local + c_idx;
              int64_t iCoord_ext_2 =
                  (hslice_as_wslice ? cur_pos : h_local_base) + h_idx;
              int64_t iCoord_ext_3 = (hslice_as_wslice ? 0 : cur_pos) + w_idx;
              if (map_num_inputs == (int64_t)iShape.size() && same_i_shape) {
                map_input_coords[0] = iCoord_ext_0;
                map_input_coords[1] = iCoord_ext_1;
                map_input_coords[2] = iCoord_ext_2;
                map_input_coords[3] = iCoord_ext_3;
              } else {
                int64_t iOffset_ext =
                    iCoord_ext_0 * iStride_ext_0 +
                    iCoord_ext_1 * iStride_ext_1 +
                    iCoord_ext_2 * iStride_ext_2 +
                    iCoord_ext_3 * iStride_ext_3;
                map_input_coords = offset_2_coords(iOffset_ext, iStride);
                if ((int64_t)map_input_coords.size() != map_num_inputs)
                  return false;
              }
              for (auto &info : floorDivInfos) {
                if (info.dim >= 0 && info.dim < (int)map_input_coords.size()) {
                  int64_t val = map_input_coords[info.dim] + info.offset;
                  if (val < 0 || val % info.divisor != 0)
                    return false;
                }
              }
              evalCompiledMapInto(compiled_map, map_input_coords, oCoords);
              if (!is_coords_in_src(oCoords))
                return false;
              gOff = coords_2_offset(oCoords, oStride);
              return true;
            };

            const auto &first_block = pattern.blocks.front();
            int64_t cur_first_gOffset = 0;
            bool delta_valid = get_mapped_gOffset_quick(
                first_block.scan_pos, cur_first_gOffset);

            if (delta_valid) {
              int64_t g_delta = cur_first_gOffset - pattern.ref_first_gOffset;
              bool delta_ok = true;
              if (pattern.blocks.size() > 1) {
                const auto &last_block = pattern.blocks.back();
                int64_t cur_last_gOffset = 0;
                if (get_mapped_gOffset_quick(last_block.scan_pos,
                                             cur_last_gOffset)) {
                  int64_t expected = last_block.gOffset + g_delta;
                  if (cur_last_gOffset != expected)
                    delta_ok = false;
                } else {
                  delta_ok = false;
                }
              }

              if (delta_ok) {
                int64_t n_l_delta =
                    (n_local - pattern.ref_n_local) * real_cslice_lStride;
                for (const auto &blk : pattern.blocks) {
                  int64_t new_lOffset = blk.lOffset + n_l_delta;
                  int64_t new_l_nidx = new_lOffset / lStride_ext_0;
                  int64_t new_l_cidx =
                      (new_lOffset % lStride_ext_0) / lStride_ext_1;
                  out_infos.push_back(FreeTensorDmaInfo{
                      blk.gOffset + g_delta, new_lOffset, 1, 1, 1,
                      blk.entry_W, blk.entry_W, blk.entry_W, blk.entry_W,
                      new_l_nidx, new_l_cidx});
                }
                continue;
              }
            }
          }

          // Full scan for this row (cache miss or delta invalid)
          auto get_mapped_gOffset = [&](int64_t cur_pos,
                                        int64_t &gOff) -> bool {
            int64_t iCoord_ext_0 = n_local + n_idx;
            int64_t iCoord_ext_1 = c_local + c_idx;
            int64_t iCoord_ext_2 =
                (hslice_as_wslice ? cur_pos : h_local_base) + h_idx;
            int64_t iCoord_ext_3 = (hslice_as_wslice ? 0 : cur_pos) + w_idx;

            if (map_num_inputs == (int64_t)iShape.size() && same_i_shape) {
              map_input_coords[0] = iCoord_ext_0;
              map_input_coords[1] = iCoord_ext_1;
              map_input_coords[2] = iCoord_ext_2;
              map_input_coords[3] = iCoord_ext_3;
            } else {
              int64_t iOffset_ext =
                  iCoord_ext_0 * iStride_ext_0 + iCoord_ext_1 * iStride_ext_1 +
                  iCoord_ext_2 * iStride_ext_2 + iCoord_ext_3 * iStride_ext_3;
              map_input_coords = offset_2_coords(iOffset_ext, iStride);
              if ((int64_t)map_input_coords.size() != map_num_inputs) {
                return false;
              }
            }

            for (auto &info : floorDivInfos) {
              if (info.dim >= 0 && info.dim < (int)map_input_coords.size()) {
                int64_t val = map_input_coords[info.dim] + info.offset;
                if (val < 0 || val % info.divisor != 0)
                  return false;
              }
            }

            evalCompiledMapInto(compiled_map, map_input_coords, oCoords);
            if (!is_coords_in_src(oCoords)) {
              return false;
            }
            gOff = coords_2_offset(oCoords, oStride);
            return true;
          };

          std::vector<RowBlockEntry> row_blocks;
          int64_t scan_pos = 0;
          while (scan_pos < scan_size) {
            int64_t gOffset = 0;
            if (!get_mapped_gOffset(scan_pos, gOffset)) {
              scan_pos++;
              continue;
            }

            int64_t entry_W = findContiguousLength(
                block_analysis, scan_pos, scan_size, gOffset,
                get_mapped_gOffset);

            int64_t h_local = hslice_as_wslice ? scan_pos : h_local_base;
            int64_t w_local = hslice_as_wslice ? 0 : scan_pos;
            int64_t lOffset = n_local * real_cslice_lStride +
                              c_local * lStride_HW + h_local * real_wslice +
                              w_local;
            int64_t l_nidx = lOffset / lStride_ext_0;
            int64_t l_cidx = (lOffset % lStride_ext_0) / lStride_ext_1;

            out_infos.push_back(FreeTensorDmaInfo{gOffset, lOffset, 1, 1, 1,
                                                  entry_W, entry_W, entry_W,
                                                  entry_W, l_nidx, l_cidx});
            row_blocks.push_back(
                RowBlockEntry{scan_pos, entry_W, gOffset, lOffset,
                              l_nidx, l_cidx});

            scan_pos += entry_W;
          }

          // Store row pattern for future reuse (only first occurrence)
          if (!row_blocks.empty() &&
              row_cache.find(rp_key) == row_cache.end()) {
            RowPattern pattern;
            pattern.ref_n_local = n_local;
            pattern.ref_first_gOffset = row_blocks.front().gOffset;
            pattern.blocks = std::move(row_blocks);
            row_cache[rp_key] = std::move(pattern);
          }
        }
      };

      constexpr int kMaxThreads = 16;
      int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
      if (hw_threads <= 0)
        hw_threads = 1;
      int num_threads = std::min<int64_t>(kMaxThreads, total_rows);
      num_threads = std::min(num_threads, hw_threads);

      if (num_threads <= 1 || total_rows < 64) {
        process_row_range(0, total_rows, normal_dma_infos);
      } else {
        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        std::vector<std::vector<FreeTensorDmaInfo>> thread_infos(num_threads);
        int64_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;

        for (int t = 0; t < num_threads; ++t) {
          int64_t begin = t * rows_per_thread;
          int64_t end = std::min<int64_t>(total_rows, begin + rows_per_thread);
          workers.emplace_back([&, t, begin, end]() {
            process_row_range(begin, end, thread_infos[t]);
          });
        }
        for (auto &th : workers) {
          th.join();
        }
        for (int t = 0; t < num_threads; ++t) {
          normal_dma_infos.insert(normal_dma_infos.end(),
                                  thread_infos[t].begin(),
                                  thread_infos[t].end());
        }
      }

      // Step 3: Stride fusion
      bool allow_h_fuse = (W == real_wslice);
      if (allow_h_fuse) {
        normal_dma_infos =
            fuse_dma_info_with_stride(normal_dma_infos, DIM_H, lStride_ext);
      }
      normal_dma_infos =
          fuse_dma_info_with_stride(normal_dma_infos, DIM_C, lStride_ext);
      // Very conservative pre-DIM_N reordering:
      // only for uniform sub-patterns in wslice==1 path, and
      // only when there is no padding. Keep other Store flows unchanged.
      if (!has_pad && hslice_as_wslice &&
          real_cslice <= (int64_t)Arch::NPU_NUM && !normal_dma_infos.empty()) {
        const auto &base = normal_dma_infos.front();
        bool can_sort_for_dim_n =
            (base.entry_N == 1 && base.entry_C > 0 && base.entry_C <= 32 &&
             (base.entry_C % 2 == 0) && base.entry_H == 1);
        for (const auto &e : normal_dma_infos) {
          if (e.entry_N != 1 || e.entry_C != base.entry_C ||
              e.entry_H != base.entry_H || e.entry_W != base.entry_W ||
              e.g_stride_c != base.g_stride_c ||
              e.g_stride_h != base.g_stride_h) {
            can_sort_for_dim_n = false;
            break;
          }
        }
        if (can_sort_for_dim_n) {
          const int64_t lStride_ext_0 = lStride_ext[0];
          std::sort(normal_dma_infos.begin(), normal_dma_infos.end(),
                    [lStride_ext_0](const FreeTensorDmaInfo &a,
                                    const FreeTensorDmaInfo &b) {
                      int64_t a_cpos = a.lOffset % lStride_ext_0;
                      int64_t b_cpos = b.lOffset % lStride_ext_0;
                      if (a_cpos != b_cpos)
                        return a_cpos < b_cpos;
                      return a.l_nidx < b.l_nidx;
                    });
        }
      }
      normal_dma_infos =
          fuse_dma_info_with_stride(normal_dma_infos, DIM_N, lStride_ext);

      // Store result in cache + anchor cache
      {
        std::lock_guard<std::mutex> lock(g_store_affine_cache_mutex);
        affine_cache_store(g_store_affine_shared_cache, shared_cache_key,
                           offset_key, normal_dma_infos);
        affine_anchor_store(g_store_affine_anchor_cache, shared_cache_key,
                            offset_key, normal_dma_infos);
      }
    } // end if (!full_cache_hit)

    auto l_aligned_stride = BM168x::getLocalStride(
        lShape_ext[0], lShape_ext[1], 1, lShape_ext[2], fmt_bytes, gi.eu_align);

    // step5. Fill padding regions with pad constant (if PadOp exists)
    // Only fill once at the first slice; fill only actual padding areas, not
    // the entire padded region
    if (has_pad) {
      // Check if this is the first slice in all dimensions
      bool is_first_slice = (gi.n_idx == 0) && (gi.c_idx == 0) &&
                            (gi.h_idx == 0) && (gi.w_idx == 0);

      if (is_first_slice) {
        // Get paddings attribute (set by prior Pass)
        auto paddingsArr = module::getI64Array(getPaddings(), 8, 0);

        std::vector<int64_t> pad_before(4, 0);
        std::vector<int64_t> pad_after(4, 0);
        if (paddingsArr && paddingsArr->size() >= 8) {
          // Layout: [left_0, left_1, left_2, left_3, right_0, right_1, right_2,
          // right_3]
          for (int i = 0; i < 4; ++i) {
            pad_before[i] = (*paddingsArr)[i];
            pad_after[i] = (*paddingsArr)[i + 4];
          }
        }

        // Derive src_dim (Pad's input shape) from oShape by subtracting padding
        // When there's a chain like Permute->Pad, oShape is the padded output
        // and src_dim is the shape Pad operates on (after preceding transforms)
        std::vector<int64_t> oShape_4d = module::getShape(getOutput());
        while (oShape_4d.size() < 4) {
          oShape_4d.insert(oShape_4d.begin(), 1);
        }

        std::vector<int64_t> src_dim(4, 0);
        for (int i = 0; i < 4; ++i) {
          src_dim[i] = oShape_4d[i] - pad_before[i] - pad_after[i];
        }

        std::vector<int64_t> dst_dim =
            oShape_4d; // dst_dim is the padded output shape
        std::vector<int64_t> dst_stride = getCompactStrideFromShape(dst_dim);
        if (dst_dim[0] == 1)
          dst_stride[0] = 1; // Handle N=1 case

        // Initialize pad_dim to src_dim, will be modified per dimension
        std::vector<int64_t> pad_dim = src_dim;

        // Fill padding regions following nodechip_pad_constant logic:
        // Iterate dimensions in reverse (W, H, C, N) and fill padding on each
        // side
        for (int i = 3; i >= 0; --i) {
          // Update pad_dim: set dimension i+1 to its destination size for
          // remaining dims
          if (i < 3) {
            pad_dim[i + 1] = dst_dim[i + 1];
          }

          // Fill before padding for dimension i
          if (pad_before[i] > 0) {
            // Set dimensions: dims < i use src_dim, dim i uses pad_before
            for (int k = 0; k < i; ++k) {
              pad_dim[k] = std::max(src_dim[k], int64_t(1));
            }
            pad_dim[i] = std::max(pad_before[i], int64_t(1));

            // Compute offset for "before" padding region
            int64_t before_offset = 0;
            for (int k = 0; k < i; ++k) {
              before_offset += pad_before[k] * dst_stride[k];
            }

            int64_t fill_addr = g_addr + before_offset * fmt_bytes;
            BM168x::instance()->dl_fill_constant_gen_global_cmd_stride(
                fill_addr, &pad_const_val, gdma_format, pad_dim[0], pad_dim[1],
                pad_dim[2], pad_dim[3], dst_stride[0], dst_stride[1],
                dst_stride[2], 1,
                true, // enable stride
                pid_node);
          }

          // Fill after padding for dimension i
          if (pad_after[i] > 0) {
            // Set dimensions: dims < i use dst_dim, dim i uses pad_after
            for (int k = 0; k < i; ++k) {
              pad_dim[k] = std::max(dst_dim[k], int64_t(1));
            }
            pad_dim[i] = std::max(pad_after[i], int64_t(1));

            // Compute offset for "after" padding region
            // Offset to position after source data in dimension i
            int64_t after_offset = (pad_before[i] + src_dim[i]) * dst_stride[i];

            int64_t fill_addr = g_addr + after_offset * fmt_bytes;
            BM168x::instance()->dl_fill_constant_gen_global_cmd_stride(
                fill_addr, &pad_const_val, gdma_format, pad_dim[0], pad_dim[1],
                pad_dim[2], pad_dim[3], dst_stride[0], dst_stride[1],
                dst_stride[2], 1,
                true, // enable stride
                pid_node);
          }
        }
      }
    }

    // step6. Codegen for Normal data regions (copy from local to global)

    for (auto &normal_info : normal_dma_infos) {
      int64_t global_addr = normal_info.gOffset * fmt_bytes + g_addr;
      std::vector<int64_t> lOffset_coords =
          offset_2_coords(normal_info.lOffset, lStride_ext);

      int64_t local_addr =
          (lOffset_coords[0] * l_aligned_stride.N +
           lOffset_coords[1] / Arch::NPU_NUM * l_aligned_stride.C +
           lOffset_coords[2]) *
              fmt_bytes +
          gi.out_addr;
      int64_t npu_idx = lOffset_coords[1] % Arch::NPU_NUM;

      auto l_real_stride = BM168x::getLocalStride(
          normal_info.entry_N, normal_info.entry_C, normal_info.entry_H,
          normal_info.entry_W, fmt_bytes, gi.eu_align);

      BM168x::instance()->dl_tensor_stride_move_gen_cmd(
          local_addr, npu_idx, global_addr, normal_info.entry_N,
          normal_info.entry_C, normal_info.entry_H, normal_info.entry_W,
          l_real_stride.N, l_real_stride.C, l_real_stride.H, l_real_stride.W,
          normal_info.g_stride_n, normal_info.g_stride_c,
          normal_info.g_stride_h, 1, gdma_format, GDMA_VALUE_DIR_L2S, 0,
          pid_node);
    }

    return;
  }
  // ==================== Affine Opt END ======================== //

  bool need_all_reduce = false;
  if (!module::isNone(getBuffer())) {
    if (!isa<tpu::OutBufferOp>(getBuffer().getDefiningOp())) {
      need_all_reduce = true;
    }
    g_addr = module::getAddress(buffer);
    llvm::errs() << "  will store to l2m, addr:" << g_addr << "\n";
  }

  int64_t c_num_local = ceiling_func(real_cslice, Arch::NPU_NUM);
  int64_t c_stride =
      gi.eu_align ? align_up(real_hslice * real_wslice, Arch::eu_num(fmt_bytes))
                  : real_hslice * real_wslice;
  int64_t channel_num = real_cslice;
  if (real_dslice <= gi.n_slice) {
    const int64_t csecs = ceiling_func(channel_num, (int64_t)MAX_TPU_DIM);
    for (int64_t d = 0; d < real_dslice; d++) {
      int64_t channel_index = 0;
      while (channel_index < csecs) {
        int64_t cur_cslice =
            std::min(channel_num - channel_index * (int64_t)MAX_TPU_DIM,
                     (int64_t)MAX_TPU_DIM);
        int64_t real_c_num_local =
            (channel_index * (int64_t)MAX_TPU_DIM) / Arch::NPU_NUM;
        int64_t src_offset_c = real_c_num_local * c_stride * fmt_bytes;
        int64_t real_npu_idx =
            (channel_index * (int64_t)MAX_TPU_DIM) % Arch::NPU_NUM;
        int64_t cur_local_offset =
            d * gi.n_slice * c_num_local * c_stride * fmt_bytes + src_offset_c;
        if (do_compress) {
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

          int64_t racu_cur_global_offset =
              gi.n_idx * racu_stride.n +
              div_up(gi.c_idx, Arch::NPU_NUM) * racu_stride.c +
              gi.h_idx * racu_stride.h + gi.w_idx * racu_stride.w;
          int64_t meta_cur_global_offset =
              (gi.n_idx * meta_stride.n +
               div_up(gi.c_idx, Arch::NPU_NUM) * meta_stride.c +
               gi.h_idx * meta_stride.h + gi.w_idx * meta_stride.w) *
              4;
          BM168x::instance()->dl_tensor_racu_compress_gen_cmd(
              gi.out_addr + cur_local_offset,
              g_addr + racu_cur_global_offset +
                  align_up(max_meta_bytes, Arch::EU_BYTES),
              g_addr + meta_cur_global_offset, gi.n_slice, cur_cslice,
              real_hslice, real_wslice, c_num_local * c_stride, c_stride,
              real_wslice, racu_stride.n, racu_stride.c, racu_stride.h,
              meta_stride.n, meta_stride.c, bias0, bias1, is_signed, zero_guard,
              gdma_format, pid_node);
        } else {
          int64_t dst_offset_c =
              (channel_index * (int64_t)MAX_TPU_DIM + gi.c_idx) * H * W *
              fmt_bytes;
          int64_t cur_global_offset = gi.n_idx * C * D * H * W * fmt_bytes +
                                      (gi.d_idx + d) * H * W * fmt_bytes +
                                      gi.h_idx * W * fmt_bytes +
                                      gi.w_idx * fmt_bytes + dst_offset_c;
          if (module::isDebugCmdEnable("codegen_debug")) {
            llvm::errs() << "storeOp, gi.n_idx:" << gi.n_idx
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

          if (need_all_reduce) {
            if (module::getChip() == module::Chip::BM1690) {
              BM1690::instance().dl_tensor_stride_move_reduce_gen_cmd(
                  gi.out_addr + cur_local_offset, real_npu_idx,
                  g_addr + cur_global_offset, gi.n_slice, cur_cslice,
                  real_hslice, real_wslice, c_num_local * c_stride, c_stride,
                  real_wslice, 1, C * D * H * W, D * H * W, W, 1, gdma_format,
                  GDMA_VALUE_DIR_L2S, // 1,
                  0, 1, 4, 0,
                  pid_node); // 1:reduce_psum_op, rw, 4:add, 0:MASTER_THREAD
            } else if (module::getChip() == module::Chip::BM1690E) {
              BM1690E::instance().dl_tensor_stride_move_reduce_gen_cmd(
                  gi.out_addr + cur_local_offset, real_npu_idx,
                  g_addr + cur_global_offset, gi.n_slice, cur_cslice,
                  real_hslice, real_wslice, c_num_local * c_stride, c_stride,
                  real_wslice, 1, C * D * H * W, D * H * W, W, 1, gdma_format,
                  GDMA_VALUE_DIR_L2S, // 1,
                  0, 1, 4, 0,
                  pid_node); // 1:reduce_psum_op, rw, 4:add, 0:MASTER_THREAD
            }
          } else {
            BM168x::instance()->dl_tensor_stride_move_gen_cmd(
                gi.out_addr + cur_local_offset, real_npu_idx,
                g_addr + cur_global_offset, gi.n_slice, cur_cslice, real_hslice,
                real_wslice, c_num_local * c_stride, c_stride, real_wslice, 1,
                C * D * H * W, D * H * W, W, 1, gdma_format,
                GDMA_VALUE_DIR_L2S, // 1,
                0, pid_node);
          }
        }
        channel_index++;
      }
    }
  } else { // HAVE DEPTH,3D [D,n_slice,C,h_slice,W] -> [N,C,D,H,W]
    for (int64_t i = 0; i < gi.n_slice; i++) {
      int64_t cur_local_offset = i * c_num_local * c_stride * fmt_bytes;
      int64_t cur_global_offset =
          (gi.n_idx + i) * C * D * H * W * fmt_bytes +
          gi.c_idx * D * H * W * fmt_bytes + gi.d_idx * H * W * fmt_bytes +
          gi.h_idx * W * fmt_bytes + gi.w_idx * fmt_bytes;
      BM168x::instance()->dl_tensor_stride_move_gen_cmd(
          gi.out_addr + cur_local_offset, 0, g_addr + cur_global_offset,
          real_dslice, real_cslice, real_hslice, real_wslice,
          gi.n_slice * c_num_local * c_stride, c_stride, real_wslice, 1, H * W,
          D * H * W, W, 1, gdma_format,
          GDMA_VALUE_DIR_L2S, // 1,
          0, pid_node);
    }
  }
}

// dynamic codegen
int64_t tpu::StoreOp::dyn_codegen_local_bm1684x(void *buffer) { return 0; }

// ======================================
// Dynamic GlobalGenInterface
// ======================================
int64_t tpu::StoreOp::dyn_codegen_global_bm1684x(void *buffer) { return 0; }

int64_t tpu::StoreOp::get_fw_type_bm1684x() { return FW_LAYER_UNKNOWN; }
