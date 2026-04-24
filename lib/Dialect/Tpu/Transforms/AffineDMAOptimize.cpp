//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "../Interfaces/Common/AffineUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "tpu_mlir/Backend/Arch.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Passes.h"
#include "tpu_mlir/Support/Module.h"
#include <optional>

using namespace llvm;
using namespace mlir;

namespace tpu_mlir {
namespace tpu {

static bool isSupportedGroupType(tpu::GroupOp groupOp) {
  auto groupTypeAttr =
      groupOp->getAttr("group_type").dyn_cast_or_null<IntegerAttr>();
  if (!groupTypeAttr) {
    return false;
  }
  auto groupType = groupTypeAttr.getInt();
  return groupType == GROUP_NORMAL || groupType == GROUP_SMALL_C ||
         groupType == GROUP_MM;
}

// Fuse coordinate mappings of transform ops (Slice/Pad/Permute/Reshape)
// into a single composite affine map on Load/Store, enabling direct
// output-to-input coordinate computation at codegen.
//
// Pipeline:
//  1. addIndexingMapsToOps*()   — annotate per-op indexing_map attrs
//  2. composeIndexingMaps*Op()  — trace & compose into Load/Store attr
//  3. cleanupIndexingMaps()     — remove intermediate attrs

// Build an affine map for a reshape operation.
static std::optional<AffineMap>
buildReshapeAffineMap(ArrayRef<int64_t> inputShape,
                      ArrayRef<int64_t> outputShape, bool isLoadDirection,
                      MLIRContext *ctx) {
  int inRank = inputShape.size();
  int outRank = outputShape.size();

  // Step 1: partition reshape into dimension groups
  SmallVector<std::tuple<int, int, int, int>> groups;
  bool success = true;

  // Pre-validation: check dimension validity and total element count
  int64_t totalInSize = 1, totalOutSize = 1;
  for (int i = 0; i < inRank; ++i) {
    if (inputShape[i] <= 0) {
      success = false;
      break;
    }
    totalInSize *= inputShape[i];
  }
  for (int i = 0; i < outRank && success; ++i) {
    if (outputShape[i] <= 0) {
      success = false;
      break;
    }
    totalOutSize *= outputShape[i];
  }
  if (success && totalInSize != totalOutSize) {
    success = false;
  }

  // Recursive matching to find dimension groups
  std::function<bool(int, int)> backtrack = [&](int inIdx, int outIdx) -> bool {
    if (inIdx == inRank && outIdx == outRank)
      return true;
    if (inIdx >= inRank || outIdx >= outRank)
      return false;

    int inStart = inIdx, outStart = outIdx;

    for (int inLen = 1; inLen <= inRank - inIdx; ++inLen) {
      int64_t inSize = 1;
      for (int i = 0; i < inLen; ++i)
        inSize *= inputShape[inIdx + i];

      for (int outLen = 1; outLen <= outRank - outIdx; ++outLen) {
        int64_t outSize = 1;
        for (int i = 0; i < outLen; ++i)
          outSize *= outputShape[outIdx + i];

        if (inSize == outSize) {
          groups.push_back({inStart, inIdx + inLen, outStart, outIdx + outLen});

          if (backtrack(inIdx + inLen, outIdx + outLen))
            return true;

          groups.pop_back();
        }
      }
    }

    return false;
  };

  success = backtrack(0, 0);
  if (!success)
    return std::nullopt;

  // Step 2: generate affine expressions for each group
  SmallVector<AffineExpr> exprs;

  for (auto [inStart, inEnd, outStart, outEnd] : groups) {
    int inCount = inEnd - inStart;
    int outCount = outEnd - outStart;

    // Determine "from" (linearization source) and "to" (decomposition target)
    int fromStart, fromEnd, toStart, toEnd;
    ArrayRef<int64_t> fromShape, toShape;
    if (isLoadDirection) {
      fromStart = outStart;
      fromEnd = outEnd;
      toStart = inStart;
      toEnd = inEnd;
      fromShape = outputShape;
      toShape = inputShape;
    } else {
      fromStart = inStart;
      fromEnd = inEnd;
      toStart = outStart;
      toEnd = outEnd;
      fromShape = inputShape;
      toShape = outputShape;
    }

    // Special case: 1-to-1 mapping
    if (inCount == 1 && outCount == 1) {
      exprs.push_back(getAffineDimExpr(fromStart, ctx));
      continue;
    }

    // Compute linearIdx from "from" dimensions
    AffineExpr linearIdx = getAffineConstantExpr(0, ctx);
    int64_t stride = 1;
    for (int i = fromEnd - 1; i >= fromStart; --i) {
      linearIdx = linearIdx +
                  getAffineDimExpr(i, ctx) * getAffineConstantExpr(stride, ctx);
      stride *= fromShape[i];
    }

    // Convert linearIdx to "to" coordinates
    int64_t divisor = 1;
    for (int i = toStart; i < toEnd; ++i)
      divisor *= toShape[i];

    for (int i = toStart; i < toEnd; ++i) {
      divisor /= toShape[i];
      if (i == toStart) {
        exprs.push_back(
            linearIdx.floorDiv(getAffineConstantExpr(divisor, ctx)));
      } else if (i == toEnd - 1) {
        exprs.push_back(linearIdx % getAffineConstantExpr(toShape[i], ctx));
      } else {
        exprs.push_back(
            linearIdx.floorDiv(getAffineConstantExpr(divisor, ctx)) %
            getAffineConstantExpr(toShape[i], ctx));
      }
    }
  }

  int numInputs = isLoadDirection ? outRank : inRank;
  return AffineMap::get(numInputs, 0, exprs, ctx);
}

// Add indexing_map_s2l to Permute, Reshape, Slice ops for the Load path.
static void addIndexingMapsToOpsForLoad(ModuleOp mOp) {
  auto ctx = mOp.getContext();

  mOp.walk([&](Operation *op) {
    if (auto permuteOp = dyn_cast<tpu::PermuteOp>(op)) {
      if (!permuteOp->hasAttr("indexing_map_s2l")) {
        auto order = permuteOp.getOrder();
        auto outputShape = module::getShape(permuteOp.getOutput());
        auto rank = outputShape.size();

        // Check if the last dimension participates in the permute
        int lastOrderVal = order[rank - 1].cast<IntegerAttr>().getInt();
        if (lastOrderVal != rank - 1) {
          return;
        }

        // Build inverse order mapping
        SmallVector<int> order_inv(rank);
        for (int i = 0; i < rank; ++i) {
          int orderVal = order[i].cast<IntegerAttr>().getInt();
          order_inv[orderVal] = i;
        }

        SmallVector<AffineExpr> exprs;
        for (int i = 0; i < rank; ++i) {
          exprs.push_back(getAffineDimExpr(order_inv[i], ctx));
        }
        permuteOp->setAttr(
            "indexing_map_s2l",
            AffineMapAttr::get(AffineMap::get(rank, 0, exprs, ctx)));
      }
    }

    if (auto reshapeOp = dyn_cast<tpu::ReshapeOp>(op)) {
      if (!reshapeOp->hasAttr("indexing_map_s2l")) {
        auto inputShape = module::getShape(reshapeOp.getInput());
        auto outputShape = module::getShape(reshapeOp.getOutput());
        auto map = buildReshapeAffineMap(inputShape, outputShape,
                                         /*isLoadDirection=*/true, ctx);
        if (map) {
          reshapeOp->setAttr("indexing_map_s2l",
                             AffineMapAttr::get(map.value()));
        }
      }
    }

    if (auto sliceOp = dyn_cast<tpu::SliceOp>(op)) {
      if (!sliceOp->hasAttr("indexing_map_s2l")) {
        auto offset = sliceOp.getOffset();
        auto steps = sliceOp.getSteps();
        auto rank = module::getShape(sliceOp.getOutput()).size();
        SmallVector<AffineExpr> exprs;
        for (int i = 0; i < rank; ++i) {
          int64_t off = offset[i].cast<IntegerAttr>().getInt();
          int64_t step = steps[i].cast<IntegerAttr>().getInt();
          exprs.push_back(getAffineConstantExpr(off, ctx) +
                          getAffineDimExpr(i, ctx) *
                              getAffineConstantExpr(step, ctx));
        }
        sliceOp->setAttr("indexing_map_s2l", AffineMapAttr::get(AffineMap::get(
                                                 rank, 0, exprs, ctx)));
      }
    }
  });
}
// Add indexing_map_l2s to Permute, Reshape, Slice ops for the Store path.
static void addIndexingMapsToOpsForStore(ModuleOp mOp) {
  auto ctx = mOp.getContext();

  mOp.walk([&](Operation *op) {
    // PadOp handled specially by traceAndComposeAffineMapsForStore()

    if (auto permuteOp = dyn_cast<tpu::PermuteOp>(op)) {
      if (!permuteOp->hasAttr("indexing_map_l2s")) {
        auto order = permuteOp.getOrder();
        auto outputShape = module::getShape(permuteOp.getOutput());
        auto rank = outputShape.size();

        int lastOrderVal = order[rank - 1].cast<IntegerAttr>().getInt();
        if (lastOrderVal != rank - 1) {
          return;
        }

        SmallVector<AffineExpr> exprs;
        for (int i = 0; i < rank; ++i) {
          exprs.push_back(
              getAffineDimExpr(order[i].cast<IntegerAttr>().getInt(), ctx));
        }
        permuteOp->setAttr(
            "indexing_map_l2s",
            AffineMapAttr::get(AffineMap::get(rank, 0, exprs, ctx)));
      }
    }

    if (auto reshapeOp = dyn_cast<tpu::ReshapeOp>(op)) {
      if (!reshapeOp->hasAttr("indexing_map_l2s")) {
        auto inputShape = module::getShape(reshapeOp.getInput());
        auto outputShape = module::getShape(reshapeOp.getOutput());
        auto map = buildReshapeAffineMap(inputShape, outputShape,
                                         /*isLoadDirection=*/false, ctx);
        if (map) {
          reshapeOp->setAttr("indexing_map_l2s",
                             AffineMapAttr::get(map.value()));
        }
      }
    }

    if (auto sliceOp = dyn_cast<tpu::SliceOp>(op)) {
      if (!sliceOp->hasAttr("indexing_map_l2s")) {
        auto offset = sliceOp.getOffset();
        auto steps = sliceOp.getSteps();
        auto rank = module::getShape(sliceOp.getOutput()).size();
        SmallVector<AffineExpr> exprs;
        for (int i = 0; i < rank; ++i) {
          int64_t off = offset[i].cast<IntegerAttr>().getInt();
          int64_t step = steps[i].cast<IntegerAttr>().getInt();
          auto diff =
              getAffineDimExpr(i, ctx) - getAffineConstantExpr(off, ctx);
          exprs.push_back(diff.floorDiv(getAffineConstantExpr(step, ctx)));
        }
        sliceOp->setAttr("indexing_map_l2s", AffineMapAttr::get(AffineMap::get(
                                                 rank, 0, exprs, ctx)));
      }
    }
  });
}

static tpu::GroupOp rebuildGroupOp(tpu::GroupOp oldGroupOp,
                                   const SmallVector<Type, 8> &newOutputTypes) {
  OpBuilder builder(oldGroupOp);

  llvm::DenseSet<Value> newInputsSet;
  SmallVector<Value, 8> newInputs;

  oldGroupOp.walk([&](tpu::LoadOp loadOp) {
    Value input = loadOp.getInput();
    if (input.getParentBlock() != &oldGroupOp.getBody().front()) {
      if (module::isWeight(input)) {
        return;
      }
      if (newInputsSet.insert(input).second) {
        newInputs.push_back(input);
      }
    }
  });

  SmallVector<Type, 8> resultTypes = newOutputTypes;

  SmallVector<Location, 8> locs;
  for (auto res : oldGroupOp.getResults()) {
    locs.push_back(res.getLoc());
  }
  auto groupLoc = builder.getFusedLoc(locs);

  auto newGroupOp = builder.create<tpu::GroupOp>(
      groupLoc, resultTypes, newInputs, oldGroupOp->getAttrs());

  for (auto it : llvm::enumerate(oldGroupOp.getResults())) {
    it.value().replaceAllUsesWith(newGroupOp.getResult(it.index()));
  }

  newGroupOp.getBody().takeBody(oldGroupOp.getBody());
  oldGroupOp->erase();

  return newGroupOp;
}

// Trace upward from Load's operand, collecting indexing_map_s2l from
// Permute/Reshape/Slice ops, then compose them back-to-front.
static bool traceAndComposeAffineMapsForLoad(Operation *startOp, Value operand,
                                             AffineMap &composedMap,
                                             Value &sourceOperand,
                                             MLIRContext *ctx) {
  SmallVector<AffineMap> maps;
  bool onlyReshape = true;
  Operation *currentOp = operand.getDefiningOp();
  Value lastOperand = operand;

  // Trace upward along the operand chain
  while (currentOp) {
    // Stop condition 1: op without indexing_map_s2l
    if (!currentOp->hasAttr("indexing_map_s2l")) {
      break;
    }

    // Stop if operand has unsupported users
    if (!operand.hasOneUse()) {
      bool has_unsupported_user = false;
      for (auto userop : operand.getUsers()) {
        if (!(isa<tpu::LoadOp>(userop) || isa<tpu::GroupOp>(userop))) {
          has_unsupported_user = true;
          break;
        }
      }
      if (has_unsupported_user)
        break;
    }

    if (!isa<tpu::PermuteOp, tpu::ReshapeOp, tpu::SliceOp>(currentOp)) {
      break;
    }

    if (!isa<tpu::ReshapeOp>(currentOp)) {
      onlyReshape = false;
    }

    auto mapAttr =
        currentOp->getAttr("indexing_map_s2l").dyn_cast<AffineMapAttr>();
    if (mapAttr) {
      maps.push_back(mapAttr.getValue());
    }

    if (currentOp->getNumOperands() > 0) {
      auto nextOperand = currentOp->getOperand(0);
      lastOperand = nextOperand;
      currentOp = nextOperand.getDefiningOp();
    } else {
      break;
    }
  }

  if (maps.empty()) {
    return false;
  }

  if (maps.size() == 1 && onlyReshape) {
    return false;
  }

  sourceOperand = lastOperand;

  composedMap = maps.back();
  for (int i = maps.size() - 2; i >= 0; --i) {
    composedMap = composedMap.compose(maps[i]);
  }

  return true;
}

// Trace forward from Store's output, collecting indexing_map_l2s from
// Permute/Reshape/Slice ops. PadOp acts as chain terminus — its offset
// is folded into the composed map.
static bool traceAndComposeAffineMapsForStore(
    Value groupOutput, AffineMap &composedMap, RankedTensorType &finalType,
    SmallVector<Operation *, 4> &opsToRemove, float &padConstVal, bool &hasPad,
    Attribute &paddingAttr, MLIRContext *ctx) {
  SmallVector<AffineMap> maps;
  bool onlyReshape = true;
  Value currentValue = groupOutput;
  hasPad = false;
  padConstVal = 0;
  SmallVector<int64_t> padBefore; // per-dimension front padding
  paddingAttr = nullptr;

  // Trace along user chain, fusing all transforms; Pad acts as terminus
  while (currentValue.hasOneUse()) {
    Operation *userOp = *currentValue.getUsers().begin();

    // Stop condition 1: PadOp
    if (isa<tpu::PadOp>(userOp)) {
      if (auto padOp = dyn_cast<tpu::PadOp>(userOp)) {
        hasPad = true;
        padConstVal = padOp.getVal().convertToDouble();
        auto paddings = padOp.getPaddings();
        auto rank = module::getShape(userOp->getResult(0)).size();

        padBefore.resize(rank);
        for (int i = 0; i < rank; ++i) {
          padBefore[i] = paddings[i].cast<IntegerAttr>().getInt();
        }

        finalType = userOp->getResult(0).getType().cast<RankedTensorType>();
        paddingAttr = padOp->getAttr("paddings");
      }
      opsToRemove.push_back(userOp);
      break;
    }

    // Stop condition 2: no indexing_map_l2s
    if (!userOp->hasAttr("indexing_map_l2s")) {
      break;
    }

    if (!isa<tpu::PermuteOp, tpu::ReshapeOp, tpu::SliceOp>(userOp)) {
      break;
    }

    if (!isa<tpu::ReshapeOp>(userOp)) {
      onlyReshape = false;
    }

    if (isa<tpu::ReshapeOp>(userOp)) {
      auto outputType = userOp->getResult(0).getType().cast<RankedTensorType>();
      if (outputType.getRank() != 4) {
        break;
      }
    }

    auto mapAttr =
        userOp->getAttr("indexing_map_l2s").dyn_cast<AffineMapAttr>();
    if (mapAttr) {
      maps.push_back(mapAttr.getValue());
    }

    opsToRemove.push_back(userOp);
    currentValue = userOp->getResult(0);
  }

  if (!hasPad && maps.empty()) {
    return false;
  }

  if (!hasPad && maps.size() == 1 && onlyReshape) {
    return false;
  }

  // Case 1: Transforms only (no Pad)
  if (!hasPad && !maps.empty()) {
    composedMap = maps.back();
    for (int i = maps.size() - 2; i >= 0; --i) {
      composedMap = composedMap.compose(maps[i]);
    }
    finalType = currentValue.getType().cast<RankedTensorType>();
    return true;
  }

  // Case 2: Has Pad (may also have transforms)
  AffineMap baseMap;
  if (maps.empty()) {
    // Pad only, no transforms - use identity map as base
    auto rank = module::getShape(groupOutput).size();
    SmallVector<AffineExpr> exprs;
    for (int i = 0; i < rank; ++i) {
      exprs.push_back(getAffineDimExpr(i, ctx));
    }
    baseMap = AffineMap::get(rank, 0, exprs, ctx);
  } else {
    // Fuse coordinate transforms (back-to-front)
    baseMap = maps.back();
    for (int i = maps.size() - 2; i >= 0; --i) {
      baseMap = baseMap.compose(maps[i]);
    }
  }

  // Pad map: output_coor = input_coor + pad_before
  auto rank = baseMap.getNumResults();
  SmallVector<AffineExpr> padExprs;
  for (int i = 0; i < rank; ++i) {
    auto dimExpr = getAffineDimExpr(i, ctx);
    int64_t pad = i < padBefore.size() ? padBefore[i] : 0;
    if (pad > 0) {
      padExprs.push_back(dimExpr + getAffineConstantExpr(pad, ctx));
    } else {
      padExprs.push_back(dimExpr);
    }
  }
  auto padMap = AffineMap::get(rank, 0, padExprs, ctx);

  composedMap = padMap.compose(baseMap);

  return true;
}

// Simulate codegen row-scan + stride fusion to estimate DMA quality.
// Returns false if avg transfer < min_continuous_bytes.
static bool
estimateAffineDmaQuality(AffineMap indexingMap, ArrayRef<int64_t> iShape,
                         ArrayRef<int64_t> oShape, group_type_t group_type,
                         int64_t bytesPerElem, int64_t gi_n_slice,
                         int64_t gi_c_slice, int64_t gi_h_slice,
                         int64_t gi_w_slice, int64_t min_continuous_bytes) {

  int64_t N, C, H, W;
  module::getNCHW(oShape, N, C, H, W, group_type);

  int64_t NPU_NUM = backend::Arch::NPU_NUM;
  if (NPU_NUM <= 0)
    return true;

  int64_t real_cslice = gi_c_slice;
  int64_t real_hslice = gi_h_slice;
  int64_t real_wslice = gi_w_slice;

  std::vector<int64_t> oShape_ext = {N, C, H, W};
  std::vector<int64_t> oStride = getCompactStrideFromShape(
      std::vector<int64_t>(oShape.begin(), oShape.end()));
  std::vector<int64_t> oStride_ext = getCompactStrideFromShape(oShape_ext);
  std::vector<int64_t> iShape_vec(iShape.begin(), iShape.end());
  std::vector<int64_t> iStride = getCompactStrideFromShape(iShape_vec);

  auto oShapeReal = std::vector<int64_t>(oShape.begin(), oShape.end());
  bool same_o_shape = (oShapeReal == oShape_ext);

  int64_t map_num_inputs = indexingMap.getNumInputs();

  std::vector<int64_t> lShape_ext = {gi_n_slice, real_cslice,
                                     real_hslice * real_wslice};
  if (real_cslice % NPU_NUM == 0) {
    lShape_ext[0] *= real_cslice / NPU_NUM;
    lShape_ext[1] = NPU_NUM;
  }
  std::vector<int64_t> lStride_ext = getCompactStrideFromShape(lShape_ext);

  bool hslice_as_wslice = (real_wslice == 1 && real_hslice > 1);
  int64_t scan_size = hslice_as_wslice ? real_hslice : real_wslice;
  int64_t rows_per_n =
      hslice_as_wslice ? real_cslice : real_cslice * real_hslice;
  int64_t total_rows = gi_n_slice * rows_per_n;

  // Compile affine map for fast evaluation (replaces recursive tree walking)
  const auto compiled_map = compileAffineMap(indexingMap);

  // Analyze contiguous block structure for fast scanning
  int scan_dim_idx = -1;
  if (hslice_as_wslice) {
    if (map_num_inputs == 4 && same_o_shape)
      scan_dim_idx = 2;
    else if (map_num_inputs >= 3)
      scan_dim_idx = map_num_inputs - 2;
  } else {
    scan_dim_idx = map_num_inputs - 1;
  }
  AffineBlockAnalysis block_analysis;
  if (scan_dim_idx >= 0) {
    block_analysis =
        analyzeContiguousBlocks(indexingMap, scan_dim_idx, iStride);
  }

  constexpr int64_t MAX_AFFINE_DMA_ENTRIES = 128;
  constexpr int64_t MAX_EST_ENTRIES = 200000;

  std::vector<FreeTensorDmaInfo> free_tensor_dma_infos;
  std::vector<int64_t> map_input_coords(map_num_inputs, 0);
  std::vector<int64_t> iCoords(compiled_map.results.size());

  int64_t lStride_HW = real_hslice * real_wslice;
  int64_t real_cslice_lStride = real_cslice * lStride_HW;

  bool early_exit = false;
  for (int64_t row = 0; row < total_rows && !early_exit; ++row) {
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

    auto get_mapped_gOffset = [&](int64_t cur_pos, int64_t &gOff) -> bool {
      int64_t oCoord_ext_0 = n_local;
      int64_t oCoord_ext_1 = c_local;
      int64_t oCoord_ext_2 = hslice_as_wslice ? cur_pos : h_local_base;
      int64_t oCoord_ext_3 = hslice_as_wslice ? 0 : cur_pos;

      if (map_num_inputs == 4 && same_o_shape) {
        map_input_coords[0] = oCoord_ext_0;
        map_input_coords[1] = oCoord_ext_1;
        map_input_coords[2] = oCoord_ext_2;
        map_input_coords[3] = oCoord_ext_3;
      } else {
        int64_t oOffset =
            oCoord_ext_0 * oStride_ext[0] + oCoord_ext_1 * oStride_ext[1] +
            oCoord_ext_2 * oStride_ext[2] + oCoord_ext_3 * oStride_ext[3];
        map_input_coords = offset_2_coords(oOffset, oStride);
        if ((int64_t)map_input_coords.size() != map_num_inputs)
          return false;
      }

      evalCompiledMapInto(compiled_map, map_input_coords, iCoords);
      if (!is_coords_valid(iCoords, iShape_vec))
        return false;
      gOff = coords_2_offset(iCoords, iStride);
      return true;
    };

    int64_t scan_pos = 0;
    while (scan_pos < scan_size) {
      int64_t gOffset = 0;
      if (!get_mapped_gOffset(scan_pos, gOffset)) {
        scan_pos++;
        continue;
      }

      int64_t entry_W = findContiguousLength(
          block_analysis, scan_pos, scan_size, gOffset, get_mapped_gOffset);

      int64_t h_local = hslice_as_wslice ? scan_pos : h_local_base;
      int64_t w_local = hslice_as_wslice ? 0 : scan_pos;
      int64_t lOffset = n_local * real_cslice_lStride + c_local * lStride_HW +
                        h_local * real_wslice + w_local;
      int64_t l_nidx = lOffset / lStride_ext[0];
      int64_t l_cidx = (lOffset % lStride_ext[0]) / lStride_ext[1];

      free_tensor_dma_infos.push_back(
          FreeTensorDmaInfo{gOffset, lOffset, 1, 1, 1, entry_W, entry_W,
                            entry_W, entry_W, l_nidx, l_cidx});

      scan_pos += entry_W;
    }

    // Early exit: too many entries, guaranteed to fail
    if ((int64_t)free_tensor_dma_infos.size() > MAX_EST_ENTRIES) {
      early_exit = true;
    }
  }

  if (free_tensor_dma_infos.empty())
    return true;

  // If early exit, evaluate based on current state
  if (!early_exit) {
    // Stride fusion: DIM_H -> DIM_C -> DIM_N
    free_tensor_dma_infos =
        fuse_dma_info_with_stride(free_tensor_dma_infos, DIM_H, lStride_ext);
    free_tensor_dma_infos =
        fuse_dma_info_with_stride(free_tensor_dma_infos, DIM_C, lStride_ext);
    if (real_cslice <= NPU_NUM) {
      const int64_t lse0 = lStride_ext[0];
      std::sort(free_tensor_dma_infos.begin(), free_tensor_dma_infos.end(),
                [lse0](const FreeTensorDmaInfo &a, const FreeTensorDmaInfo &b) {
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
                  int64_t a_cpos = a.lOffset % lse0;
                  int64_t b_cpos = b.lOffset % lse0;
                  if (a_cpos != b_cpos)
                    return a_cpos < b_cpos;
                  return a.l_nidx < b.l_nidx;
                });
    }
    free_tensor_dma_infos =
        fuse_dma_info_with_stride(free_tensor_dma_infos, DIM_N, lStride_ext);
  }

  // Quality evaluation
  int64_t total_entries = free_tensor_dma_infos.size();
  int64_t total_elements = 0;
  for (auto &info : free_tensor_dma_infos) {
    total_elements += info.entry_N * info.entry_C * info.entry_H * info.entry_W;
  }

  double avg_bytes_per_entry =
      total_entries > 0 ? (double)total_elements * bytesPerElem / total_entries
                        : 0;

  // Reject if too many entries or avg transfer too small
  if (total_entries > MAX_AFFINE_DMA_ENTRIES)
    return false;
  return avg_bytes_per_entry >= min_continuous_bytes;
}

// Compose indexing_map_s2l for LoadOps by tracing operand chains.
// Rebuilds GroupOp when Load operands change.
static void composeIndexingMapsForLoadOp(ModuleOp mOp,
                                         int64_t min_continuous_bytes) {
  auto ctx = mOp.getContext();
  llvm::DenseMap<tpu::GroupOp, SmallVector<Type, 8>> groupOpsToRebuild;

  mOp.walk([&](tpu::GroupOp groupOp) {
    if (!isSupportedGroupType(groupOp)) {
      return;
    }
    bool needRebuild = false;
    llvm::DenseMap<Operation *, Value> loadOpsToModify;

    groupOp.walk([&](tpu::LoadOp loadOp) {
      if (loadOp.getUse_3icOptimize() > 0) {
        return;
      }
      // loadOp.dump();
      Value operand = loadOp.getInput();
      Operation *defOp = operand.getDefiningOp();

      if (!defOp)
        return;

      if (!operand.hasOneUse()) {
        for (auto userop : operand.getUsers()) {
          if (!(isa<tpu::LoadOp>(userop) || isa<tpu::GroupOp>(userop))) {
            return;
          }
        }
      }

      AffineMap composedMap;
      Value sourceOperand;
      if (!traceAndComposeAffineMapsForLoad(loadOp, operand, composedMap,
                                            sourceOperand, ctx)) {
        return;
      }

      if (sourceOperand == operand)
        return;

      auto iShape = module::getShape(sourceOperand);
      auto oShape = module::getShape(loadOp.getOutput());
      auto groupTypeAttr =
          groupOp->getAttr("group_type").dyn_cast_or_null<IntegerAttr>();
      group_type_t gt =
          groupTypeAttr ? (group_type_t)groupTypeAttr.getInt() : GROUP_NORMAL;
      auto elemTy = module::getStorageType(loadOp.getOutput());
      int64_t bytesPerElem = elemTy.getIntOrFloatBitWidth() / 8;
      if (bytesPerElem <= 0)
        bytesPerElem = 1;

      auto gInfoAttr =
          loadOp->getAttr("ginfo").dyn_cast_or_null<tpu::LayerGroupAttr>();
      if (!gInfoAttr)
        return;
      auto nSliceArr = gInfoAttr.getNSlice();
      auto cSliceArr = gInfoAttr.getCSlice();
      auto hSliceArr = gInfoAttr.getHSlice();
      auto wSliceArr = gInfoAttr.getWSlice();
      if (nSliceArr.empty() || cSliceArr.empty() || hSliceArr.empty() ||
          wSliceArr.empty())
        return;
      int64_t gi_n_slice = nSliceArr[0];
      int64_t gi_c_slice = cSliceArr[0];
      int64_t gi_h_slice = hSliceArr[0];
      int64_t gi_w_slice = wSliceArr[0];

      if (!estimateAffineDmaQuality(
              composedMap, iShape, oShape, gt, bytesPerElem, gi_n_slice,
              gi_c_slice, gi_h_slice, gi_w_slice, min_continuous_bytes)) {
        return; // Quality too low, fall back to non-affine path
      }

      loadOp->setAttr("indexing_map_s2l", AffineMapAttr::get(composedMap));
      loadOpsToModify[loadOp.getOperation()] = sourceOperand;
      needRebuild = true;
    });

    for (auto &pair : loadOpsToModify) {
      pair.first->setOperand(0, pair.second);
    }

    if (needRebuild) {
      SmallVector<Type, 8> newOutputTypes;
      for (auto res : groupOp.getResults()) {
        newOutputTypes.push_back(res.getType());
      }
      groupOpsToRebuild[groupOp] = newOutputTypes;
    }
  });

  // Rebuild GroupOps
  for (auto &pair : groupOpsToRebuild) {
    rebuildGroupOp(pair.first, pair.second);
  }
}

// Compose indexing_map_l2s for Store ops
static void composeIndexingMapsForStoreOp(ModuleOp mOp,
                                          int64_t min_continuous_bytes) {
  auto ctx = mOp.getContext();
  llvm::DenseMap<tpu::GroupOp, SmallVector<Type, 8>> groupOpsToRebuild;
  SmallVector<Operation *, 16> opsToRemove;

  mOp.walk([&](tpu::GroupOp groupOp) {
    if (!isSupportedGroupType(groupOp)) {
      return;
    }
    bool needRebuild = false;
    SmallVector<Type, 8> newOutputTypes;

    for (auto res : groupOp.getResults()) {
      newOutputTypes.push_back(res.getType());
    }

    auto yieldOp =
        dyn_cast<tpu::YieldOp>(groupOp.getBody().front().getTerminator());
    if (!yieldOp)
      return;

    SmallVector<Location, 8> storeLocations;
    SmallVector<tpu::StoreOp, 8> storeOps;

    for (auto it : llvm::enumerate(groupOp.getResults())) {
      size_t idx = it.index();
      Value groupOutput = it.value();

      Value yieldOperand = yieldOp.getOperand(idx);
      auto storeOp =
          dyn_cast_or_null<tpu::StoreOp>(yieldOperand.getDefiningOp());
      if (!storeOp)
        continue;

      Location storeLoc = storeOp->getLoc();
      storeLocations.push_back(storeLoc);
      storeOps.push_back(storeOp);

      AffineMap composedMap;
      RankedTensorType finalType;
      SmallVector<Operation *, 4> currentOpsToRemove;
      float padConstVal = 0;
      bool hasPad = false;
      Attribute paddingAttr = nullptr;

      if (traceAndComposeAffineMapsForStore(groupOutput, composedMap, finalType,
                                            currentOpsToRemove, padConstVal,
                                            hasPad, paddingAttr, ctx)) {
        auto storeInputShape = module::getShape(storeOp.getInput());
        auto groupTypeAttr =
            groupOp->getAttr("group_type").dyn_cast_or_null<IntegerAttr>();
        group_type_t gt =
            groupTypeAttr ? (group_type_t)groupTypeAttr.getInt() : GROUP_NORMAL;
        auto elemTy = module::getStorageType(storeOp.getInput());
        int64_t bytesPerElem = elemTy.getIntOrFloatBitWidth() / 8;
        if (bytesPerElem <= 0)
          bytesPerElem = 1;

        auto gInfoAttr =
            storeOp->getAttr("ginfo").dyn_cast_or_null<tpu::LayerGroupAttr>();
        if (!gInfoAttr) {
          currentOpsToRemove.clear();
          continue;
        }
        auto nSliceArr = gInfoAttr.getNSlice();
        auto cSliceArr = gInfoAttr.getCSlice();
        auto hSliceArr = gInfoAttr.getHSlice();
        auto wSliceArr = gInfoAttr.getWSlice();
        if (nSliceArr.empty() || cSliceArr.empty() || hSliceArr.empty() ||
            wSliceArr.empty()) {
          currentOpsToRemove.clear();
          continue;
        }
        int64_t gi_n_slice = nSliceArr[0];
        int64_t gi_c_slice = cSliceArr[0];
        int64_t gi_h_slice = hSliceArr[0];
        int64_t gi_w_slice = wSliceArr[0];

        auto destShape = finalType.getShape();
        if (!estimateAffineDmaQuality(composedMap, destShape, storeInputShape,
                                      gt, bytesPerElem, gi_n_slice, gi_c_slice,
                                      gi_h_slice, gi_w_slice,
                                      min_continuous_bytes)) {
          currentOpsToRemove.clear();
          continue;
        }

        storeOp->setAttr("indexing_map_l2s", AffineMapAttr::get(composedMap));

        if (hasPad) {
          storeOp->setAttr("pad_const_val",
                           FloatAttr::get(FloatType::getF64(ctx), padConstVal));
          if (paddingAttr) {
            storeOp->setAttr("paddings", paddingAttr);
          }

          // Rename storeOp using Pad op's location
          for (auto op : currentOpsToRemove) {
            if (isa<tpu::PadOp>(op)) {
              storeOp->setLoc(op->getLoc());
              storeLocations[storeLocations.size() - 1] = op->getLoc();
              break;
            }
          }
        }

        // If no Pad, use the last fused op's location
        if (!hasPad && !currentOpsToRemove.empty()) {
          auto lastOp = currentOpsToRemove.back();
          storeOp->setLoc(lastOp->getLoc());
          storeLocations[storeLocations.size() - 1] = lastOp->getLoc();
        }

        auto storeOutputType = RankedTensorType::get(
            finalType.getShape(), finalType.getElementType());
        storeOp.getOutput().setType(storeOutputType);
        newOutputTypes[idx] = storeOutputType;

        opsToRemove.append(currentOpsToRemove);
        needRebuild = true;
      }
    }

    // Fuse store locations
    if (!storeLocations.empty()) {
      auto fusedLoc = OpBuilder(groupOp).getFusedLoc(storeLocations);
      yieldOp->setLoc(fusedLoc);
      groupOp->setLoc(fusedLoc);
    }

    if (needRebuild) {
      groupOpsToRebuild[groupOp] = newOutputTypes;
    }
  });

  // Rebuild GroupOps
  for (auto &pair : groupOpsToRebuild) {
    rebuildGroupOp(pair.first, pair.second);
  }

  // Clean up fused ops
  for (auto op : opsToRemove) {
    if (op->getNumOperands() > 0 && op->getNumResults() > 0) {
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
    }
  }

  for (auto op : opsToRemove) {
    if (op->use_empty()) {
      op->erase();
    }
  }
}

// Remove indexing_map attributes from non-Load/Store ops.
static void cleanupIndexingMaps(ModuleOp mOp) {
  mOp.walk([&](Operation *op) {
    if (!op->hasAttr("indexing_map") && !op->hasAttr("indexing_map_s2l") &&
        !op->hasAttr("indexing_map_l2s")) {
      return;
    }

    if (isa<tpu::LoadOp, tpu::StoreOp>(op)) {
      return;
    }

    if (op->hasAttr("indexing_map")) {
      op->removeAttr("indexing_map");
    }
    if (op->hasAttr("indexing_map_s2l")) {
      op->removeAttr("indexing_map_s2l");
    }
    if (op->hasAttr("indexing_map_l2s")) {
      op->removeAttr("indexing_map_l2s");
    }
  });
}

class AffineOptPass : public AffineOptBase<AffineOptPass> {
public:
  AffineOptPass() {}

  void runOnOperation() override {

    auto mOp = getOperation();
    if (!module::isBM1684XFamily() || module::isDynamic())
      return;

    bool hasDynamicFunc = false;
    mOp.walk([&](FuncOp op) {
      if (getRunMode(op) == tpu::RunMode::TPU_DYNAMIC) {
        hasDynamicFunc = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasDynamicFunc)
      return;

    addIndexingMapsToOpsForLoad(mOp);
    composeIndexingMapsForLoadOp(mOp, min_continuous_bytes);
    cleanupIndexingMaps(mOp);

    addIndexingMapsToOpsForStore(mOp);
    composeIndexingMapsForStoreOp(mOp, min_continuous_bytes);
    cleanupIndexingMaps(mOp);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAffineOptPass() {
  return std::make_unique<AffineOptPass>();
}

} // namespace tpu
} // namespace tpu_mlir
