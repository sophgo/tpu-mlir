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
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/TPUNnvlcUtil.h"
#include <algorithm>
#include <mutex>
#include <thread>

using namespace tpu_mlir::backend;

static AffineSharedCache g_load_affine_shared_cache;
static AffineAnchorCache g_load_affine_anchor_cache;
static std::mutex g_load_affine_cache_mutex;

void tpu::LoadOp::codegen_global_bm1684x() {
  llvm_unreachable("global not support");
}

int64_t tpu::LoadOp::getBufferSize_bm1684x(
    int64_t in_lmem_bytes, int64_t out_lmem_bytes, int64_t in_nslice,
    int64_t in_cslice, int64_t in_hslice, int64_t in_dslice, int64_t in_wslice,
    int64_t out_nslice, int64_t out_cslice, int64_t out_hslice,
    int64_t out_dslice, int64_t out_wslice, group_type_t group_type,
    bool with_hw_margins) {
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
  int64_t real_cidx, real_hidx, real_widx, real_didx;
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
  auto is_idx = getIsIdxWeight();
  if (is_idx) {
    auto user = this->getResult().getUsers().begin();
    auto it = *user;
    auto getWeightOpFromValue = [&](mlir::Value value) -> top::WeightOp {
      Value currentValue = value;
      while (auto blockArg = currentValue.dyn_cast<mlir::BlockArgument>()) {
        auto parentOp = this->getOperation()->getParentOp();
        if (auto groupOp = dyn_cast<tpu::GroupOp>(parentOp)) {
          unsigned argIndex = blockArg.getArgNumber();
          if (argIndex < groupOp->getNumOperands()) {
            currentValue = groupOp->getOperand(argIndex);
            continue;
          }
        }
        break;
      }
      if (auto definingOp = currentValue.getDefiningOp()) {
        return dyn_cast<top::WeightOp>(definingOp);
      }
      return nullptr;
    };
    if (isa<tpu::UpsampleOp>(it)) {
      auto g_param = this->getOperation()
                         ->getAttr(LocalGenInterface::kLayerGroupAttrName)
                         .cast<tpu::LayerGroupAttr>();
      int64_t w_slice_size = g_param.getWSlice().size();
      auto indices = getWeightOpFromValue(this->getInput());
      auto indices_idx = module::getI64Array(indices.getIndicesIdxAttr());
      auto indices_slice = module::getI64Array(indices.getIndicesSliceAttr());
      int64_t step = h_step * w_slice_size + w_step;
      real_cidx = 0;
      real_hidx = 0;
      real_widx = indices_idx->at(step);
      real_didx = gi.d_idx;
      real_cslice = Arch::NPU_NUM;
      real_hslice = 1;
      real_wslice = indices_slice->at(step);
      real_dslice = gi.d_slice;
    } else {
      llvm::errs() << "not support this idx reorder!\n";
    }
  } else {
    real_cidx = gi.c_idx;
    real_hidx = gi.h_idx;
    real_widx = gi.w_idx;
    real_didx = gi.d_idx;
    real_cslice = gi.c_slice;
    real_hslice = gi.h_slice;
    real_wslice = gi.w_slice;
    real_dslice = gi.d_slice;
  }
  if (data_type == DTYPE_UINT4 || data_type == DTYPE_INT4) {
    gdma_format = BM168x::GDMA_VALUE_FORMAT_INT8;
    data_type = DTYPE_INT8;
    // note: H & W are assumed to be compactedly stored together,
    if (gi.h_slice == H) {
      if ((W * H & 0x1) == 1) { //  H*W is odd,
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

  // step1： auto vectorize. step2:  auto tensorize.
  // step3:  auto tensorize with STRIDE optimization.
  // step4:  codegen for each tensor

  if (getOperation()->hasAttr("indexing_map_s2l")) {
    auto indexingMapAttr =
        getOperation()->getAttr("indexing_map_s2l").dyn_cast<AffineMapAttr>();
    auto indexingMap = indexingMapAttr.getValue();

    assert(D == 1);
    std::vector<int64_t> oShape = module::getShape(getOutput());
    std::vector<int64_t> oShape_ext = {N, C, H, W};
    std::vector<int64_t> iShape = module::getShape(getInput());

    std::vector<int64_t> oStride = getCompactStrideFromShape(oShape);
    std::vector<int64_t> oStride_ext = getCompactStrideFromShape(oShape_ext);
    std::vector<int64_t> iStride = getCompactStrideFromShape(iShape);

    // Step 1/2 combined: Row-scan approach
    // Instead of greedy vectorize + tensorize, directly scan rows (n,c,h)
    // and find contiguous W-blocks by checking adjacent affine-mapped offsets.
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
    const bool same_o_shape = (oShape == oShape_ext);
    const int64_t map_num_inputs = indexingMap.getNumInputs();

    const int64_t oStride_ext_0 = oStride_ext[0];
    const int64_t oStride_ext_1 = oStride_ext[1];
    const int64_t oStride_ext_2 = oStride_ext[2];
    const int64_t oStride_ext_3 = oStride_ext[3];

    const int64_t n_idx = gi.n_idx;
    const int64_t c_idx = gi.c_idx;
    const int64_t h_idx = gi.h_idx;
    const int64_t w_idx = gi.w_idx;

    std::vector<FreeTensorDmaInfo> free_tensor_dma_infos;

    // When real_wslice == 1, H elements are contiguous in local memory
    // (stride=1), so scan along H instead of W to find larger contiguous
    // blocks.
    const bool hslice_as_wslice = (real_wslice == 1 && real_hslice > 1);
    const int64_t scan_size = hslice_as_wslice ? real_hslice : real_wslice;
    const int64_t rows_per_n =
        hslice_as_wslice ? real_cslice : real_cslice * real_hslice;
    const int64_t total_rows = gi.n_slice * rows_per_n;

    // Build cache key for this affine pattern
    const std::string shared_cache_key =
        build_affine_shared_cache_key(indexingMap, iShape, oShape, gi.n_slice,
                                      real_cslice, real_hslice, real_wslice);
    const AffineOffsetKey offset_key = {n_idx, c_idx, h_idx, w_idx};

    // Compile affine map for fast evaluation
    const auto compiled_map = compileAffineMap(indexingMap);
    const auto safeDeltaDims = computeSafeDeltaDims(indexingMap);

    // Analyze contiguous block structure for fast scanning
    // Determine which input dimension is the scan dimension
    int scan_dim_idx = -1;
    if (hslice_as_wslice) {
      // scanning along h, which is oCoord_ext_2
      if (map_num_inputs == 4 && same_o_shape) {
        scan_dim_idx = 2;
      } else if (map_num_inputs == (int64_t)oShape.size()) {
        // For non-4D, approximate: scan maps to the second-to-last position
        scan_dim_idx = map_num_inputs - 2;
      }
    } else {
      // scanning along w, which is oCoord_ext_3 (or last dim)
      if (map_num_inputs == 4 && same_o_shape) {
        scan_dim_idx = 3;
      } else {
        scan_dim_idx = map_num_inputs - 1;
      }
    }
    AffineBlockAnalysis block_analysis;
    if (scan_dim_idx >= 0) {
      block_analysis =
          analyzeContiguousBlocks(indexingMap, scan_dim_idx, iStride);
    }

    // Check cache: exact match + safe delta reuse
    bool full_cache_hit = false;
    {
      std::lock_guard<std::mutex> lock(g_load_affine_cache_mutex);
      full_cache_hit = affine_cache_lookup_with_delta(
          g_load_affine_shared_cache, g_load_affine_anchor_cache,
          shared_cache_key, offset_key, iStride, oStride, iShape, oShape,
          compiled_map, safeDeltaDims, /*is_load=*/true, same_o_shape,
          free_tensor_dma_infos);
    }

    if (!full_cache_hit) {

      // Process a range of rows [row_begin, row_end) and append results to
      // out_infos
      auto process_row_range = [&](int64_t row_begin, int64_t row_end,
                                   std::vector<FreeTensorDmaInfo> &out_infos) {
        std::vector<int64_t> map_input_coords(map_num_inputs, 0);
        std::vector<int64_t> iCoords(compiled_map.results.size());
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
            // Compute gOffset delta by evaluating only the first block's
            // position for the current n_local
            auto get_mapped_gOffset_quick = [&](int64_t cur_pos,
                                                int64_t &gOff) -> bool {
              int64_t oCoord_ext_0 = n_local + n_idx;
              int64_t oCoord_ext_1 = c_local + c_idx;
              int64_t oCoord_ext_2 =
                  (hslice_as_wslice ? cur_pos : h_local_base) + h_idx;
              int64_t oCoord_ext_3 = (hslice_as_wslice ? 0 : cur_pos) + w_idx;
              if (map_num_inputs == 4 && same_o_shape) {
                map_input_coords[0] = oCoord_ext_0;
                map_input_coords[1] = oCoord_ext_1;
                map_input_coords[2] = oCoord_ext_2;
                map_input_coords[3] = oCoord_ext_3;
              } else {
                int64_t oOffset =
                    oCoord_ext_0 * oStride_ext_0 +
                    oCoord_ext_1 * oStride_ext_1 +
                    oCoord_ext_2 * oStride_ext_2 +
                    oCoord_ext_3 * oStride_ext_3;
                map_input_coords = offset_2_coords(oOffset, oStride);
                if ((int64_t)map_input_coords.size() != map_num_inputs)
                  return false;
              }
              evalCompiledMapInto(compiled_map, map_input_coords, iCoords);
              if (!is_coords_valid(iCoords, iShape))
                return false;
              gOff = coords_2_offset(iCoords, iStride);
              return true;
            };

            const auto &first_block = pattern.blocks.front();
            int64_t cur_first_gOffset = 0;
            bool delta_valid = get_mapped_gOffset_quick(
                first_block.scan_pos, cur_first_gOffset);

            if (delta_valid) {
              int64_t g_delta = cur_first_gOffset - pattern.ref_first_gOffset;
              // Verify delta is uniform by checking last block too
              bool delta_ok = true;
              if (pattern.blocks.size() > 1) {
                const auto &last_block = pattern.blocks.back();
                int64_t cur_last_gOffset = 0;
                if (get_mapped_gOffset_quick(last_block.scan_pos,
                                             cur_last_gOffset)) {
                  int64_t expected =
                      last_block.gOffset + g_delta;
                  if (cur_last_gOffset != expected)
                    delta_ok = false;
                } else {
                  delta_ok = false;
                }
              }

              if (delta_ok) {
                // Compute lOffset delta for different n_local
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
                continue; // row done via cache
              }
            }
          }

          // Full scan for this row (cache miss or delta invalid)
          auto get_mapped_gOffset = [&](int64_t cur_pos,
                                        int64_t &gOff) -> bool {
            int64_t oCoord_ext_0 = n_local + n_idx;
            int64_t oCoord_ext_1 = c_local + c_idx;
            int64_t oCoord_ext_2 =
                (hslice_as_wslice ? cur_pos : h_local_base) + h_idx;
            int64_t oCoord_ext_3 = (hslice_as_wslice ? 0 : cur_pos) + w_idx;

            if (map_num_inputs == 4 && same_o_shape) {
              map_input_coords[0] = oCoord_ext_0;
              map_input_coords[1] = oCoord_ext_1;
              map_input_coords[2] = oCoord_ext_2;
              map_input_coords[3] = oCoord_ext_3;
            } else {
              int64_t oOffset =
                  oCoord_ext_0 * oStride_ext_0 + oCoord_ext_1 * oStride_ext_1 +
                  oCoord_ext_2 * oStride_ext_2 + oCoord_ext_3 * oStride_ext_3;
              map_input_coords = offset_2_coords(oOffset, oStride);
              if ((int64_t)map_input_coords.size() != map_num_inputs) {
                return false;
              }
            }

            evalCompiledMapInto(compiled_map, map_input_coords, iCoords);
            if (!is_coords_valid(iCoords, iShape)) {
              return false;
            }
            gOff = coords_2_offset(iCoords, iStride);
            return true;
          };

          // Collect blocks for this row (for caching)
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

      constexpr int kMaxThreads =16;
      int hw_threads = static_cast<int>(std::thread::hardware_concurrency());
      if (hw_threads <= 0)
        hw_threads = 1;
      int num_threads = std::min<int64_t>(kMaxThreads, total_rows);
      num_threads = std::min(num_threads, hw_threads);

      if (num_threads <= 1 || total_rows < 64) {
        process_row_range(0, total_rows, free_tensor_dma_infos);
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
          free_tensor_dma_infos.insert(free_tensor_dma_infos.end(),
                                       thread_infos[t].begin(),
                                       thread_infos[t].end());
        }
      }

      // Step 3: Stride fusion
      free_tensor_dma_infos =
          fuse_dma_info_with_stride(free_tensor_dma_infos, DIM_H, lStride_ext);
      free_tensor_dma_infos =
          fuse_dma_info_with_stride(free_tensor_dma_infos, DIM_C, lStride_ext);
      // Sort + relaxed DIM_N: enable only when real_cslice <= NPU_NUM.
      // When C fits in one NPU group, ceiling_func(entry_C, NPU_NUM) == 1
      // for all entries, ensuring l_real_stride.N matches l_aligned_stride.N.
      if (real_cslice <= (int64_t)Arch::NPU_NUM) {
        const int64_t lStride_ext_0 = lStride_ext[0];
        std::sort(free_tensor_dma_infos.begin(), free_tensor_dma_infos.end(),
                  [lStride_ext_0](const FreeTensorDmaInfo &a,
                                  const FreeTensorDmaInfo &b) {
                    if (a.entry_C != b.entry_C)
                      return a.entry_C < b.entry_C;
                    if (a.entry_H != b.entry_H)
                      return a.entry_H < b.entry_H;
                    if (a.entry_W != b.entry_W)
                      return a.entry_W < b.entry_W;
                    if (a.g_stride_c != b.g_stride_c)
                      return a.g_stride_c < b.g_stride_c;
                    if (a.g_stride_h != b.g_stride_h)
                      return a.g_stride_h < b.g_stride_h;
                    int64_t a_cpos = a.lOffset % lStride_ext_0;
                    int64_t b_cpos = b.lOffset % lStride_ext_0;
                    if (a_cpos != b_cpos)
                      return a_cpos < b_cpos;
                    return a.l_nidx < b.l_nidx;
                  });
      }
      free_tensor_dma_infos =
          fuse_dma_info_with_stride(free_tensor_dma_infos, DIM_N, lStride_ext);

      // Store result in cache + anchor cache
      {
        std::lock_guard<std::mutex> lock(g_load_affine_cache_mutex);
        affine_cache_store(g_load_affine_shared_cache, shared_cache_key,
                           offset_key, free_tensor_dma_infos);
        affine_anchor_store(g_load_affine_anchor_cache, shared_cache_key,
                            offset_key, free_tensor_dma_infos);
      }
    } // end if (!full_cache_hit)

    // step last: codegen

    auto l_aligned_stride = BM168x::getLocalStride(
        lShape_ext[0], lShape_ext[1], 1, lShape_ext[2], fmt_bytes, gi.eu_align);

    for (auto &free_tensor_dma_info : free_tensor_dma_infos) {

      int64_t npu_idx, local_addr, global_addr;
      // All blocks are valid data (padding filtered out earlier)
      global_addr = free_tensor_dma_info.gOffset * fmt_bytes + g_addr;

      std::vector<int64_t> lOffset_coords =
          offset_2_coords(free_tensor_dma_info.lOffset, lStride_ext);
      local_addr = (lOffset_coords[0] * l_aligned_stride.N +
                    lOffset_coords[1] / Arch::NPU_NUM * l_aligned_stride.C +
                    lOffset_coords[2]) *
                       fmt_bytes +
                   gi.out_addr;
      npu_idx = lOffset_coords[1] % Arch::NPU_NUM;

      auto l_real_stride = BM168x::getLocalStride(
          free_tensor_dma_info.entry_N, free_tensor_dma_info.entry_C,
          free_tensor_dma_info.entry_H, free_tensor_dma_info.entry_W, fmt_bytes,
          gi.eu_align);

      // Generate DMA command for valid data block
      BM168x::instance()->dl_tensor_stride_move_gen_cmd(
          local_addr, npu_idx, global_addr, free_tensor_dma_info.entry_N,
          free_tensor_dma_info.entry_C, free_tensor_dma_info.entry_H,
          free_tensor_dma_info.entry_W, free_tensor_dma_info.g_stride_n,
          free_tensor_dma_info.g_stride_c, free_tensor_dma_info.g_stride_h, 1,
          l_real_stride.N, l_real_stride.C, l_real_stride.H, l_real_stride.W,
          gdma_format, GDMA_VALUE_DIR_S2L, 0, pid_node);
    }
    return;
  }
  // ==================== Affine Opt END ======================== //

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
                (channel_index * (int64_t)MAX_TPU_DIM + real_cidx) * H * W *
                fmt_bytes;
            int64_t cur_global_offset = gi.n_idx * C * D * H * W * fmt_bytes +
                                        (gi.d_idx + d) * H * W * fmt_bytes +
                                        real_hidx * W * fmt_bytes +
                                        real_widx * fmt_bytes + src_offset_c;
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
