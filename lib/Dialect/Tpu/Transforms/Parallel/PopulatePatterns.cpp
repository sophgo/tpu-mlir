//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Parallel.hpp"

namespace tpu_mlir {
namespace tpu {
using namespace bm1684x;

struct SliceParameters {
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
};

struct NdSlice {
  SliceParameters N;
  SliceParameters C;
  SliceParameters H;
  SliceParameters D;
  SliceParameters W;
};

LayerGroupAttr createGroupAttr(LayerGroupAttr inAttr, const NdSlice &ndSlice) {

  auto builder = Builder(inAttr.getContext());
  return LayerGroupAttr::get(
      inAttr.getContext(), inAttr.getOutAddr(), inAttr.getOutSize(),
      inAttr.getBufferAddr(), inAttr.getBufferSize(), inAttr.getEuAlign(),
      builder.getDenseI64ArrayAttr(ndSlice.N.offsets),
      builder.getDenseI64ArrayAttr(ndSlice.N.sizes),
      builder.getDenseI64ArrayAttr(ndSlice.C.offsets),
      builder.getDenseI64ArrayAttr(ndSlice.C.sizes),
      builder.getDenseI64ArrayAttr(ndSlice.D.offsets),
      builder.getDenseI64ArrayAttr(ndSlice.D.sizes),
      builder.getDenseI64ArrayAttr(ndSlice.H.offsets),
      builder.getDenseI64ArrayAttr(ndSlice.H.sizes),
      builder.getDenseI64ArrayAttr(ndSlice.W.offsets),
      builder.getDenseI64ArrayAttr(ndSlice.W.sizes), inAttr.getId(),
      inAttr.getStage(), inAttr.getGroupType());
}

NdSlice getNdSlice(LayerGroupAttr inAttr) {
  NdSlice ndSlice;
  ndSlice.N.offsets = SmallVector<int64_t>(inAttr.getNIdx().asArrayRef());
  ndSlice.N.sizes = SmallVector<int64_t>(inAttr.getNSlice().asArrayRef());
  ndSlice.C.offsets = SmallVector<int64_t>(inAttr.getCIdx().asArrayRef());
  ndSlice.C.sizes = SmallVector<int64_t>(inAttr.getCSlice().asArrayRef());
  ndSlice.H.offsets = SmallVector<int64_t>(inAttr.getHIdx().asArrayRef());
  ndSlice.H.sizes = SmallVector<int64_t>(inAttr.getHSlice().asArrayRef());
  ndSlice.D.offsets = SmallVector<int64_t>(inAttr.getDIdx().asArrayRef());
  ndSlice.D.sizes = SmallVector<int64_t>(inAttr.getDSlice().asArrayRef());
  ndSlice.W.offsets = SmallVector<int64_t>(inAttr.getWIdx().asArrayRef());
  ndSlice.W.sizes = SmallVector<int64_t>(inAttr.getWSlice().asArrayRef());
  return ndSlice;
}

template <>
LogicalResult
Parallel<tpu::GroupOp>::matchAndRewrite(tpu::GroupOp gOp,
                                        PatternRewriter &rewriter) const {
  if (isa_and_nonnull<tpu::ParallelOp>(gOp->getParentOp()))
    return failure();

  auto upOverlapOpAttrName =
      GroupOp::getOtherUpOverlapOpAttrName(gOp->getName());
  auto downOverlapOpAttrName =
      GroupOp::getOtherDownOverlapOpAttrName(gOp->getName());
  auto selfUpOverlapOpAttrName =
      GroupOp::getSelfUpOverlapOpAttrName(gOp->getName());
  auto selfDownOverlapOpAttrName =
      GroupOp::getSelfDownOverlapOpAttrName(gOp->getName());

  if (gOp->hasAttr(upOverlapOpAttrName) ||
      gOp->hasAttr(downOverlapOpAttrName) ||
      gOp->hasAttr(selfUpOverlapOpAttrName) ||
      gOp->hasAttr(selfDownOverlapOpAttrName)) {
    gOp->removeAttr(upOverlapOpAttrName);
    gOp->removeAttr(downOverlapOpAttrName);
    gOp->removeAttr(selfUpOverlapOpAttrName);
    gOp->removeAttr(selfDownOverlapOpAttrName);
    return success();
  }

  return failure();
};

void populateParalleBM1684XPatterns(RewritePatternSet *patterns, int coreNum) {
  // Add an Op-specific pattern if the generic IndexingMap fails to capture
  // the parallel semantics in this operation.
  patterns->add<Parallel<tpu::GroupOp>>(patterns->getContext(), coreNum);
};

} // namespace tpu
} // namespace tpu_mlir
