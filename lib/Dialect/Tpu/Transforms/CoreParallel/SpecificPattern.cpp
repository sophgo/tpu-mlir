//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/Transforms/CoreParallel/CoreParallel.hpp"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"
#include "tpu_mlir/Support/MathUtils.h"

namespace tpu_mlir {
namespace tpu {

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
      false, builder.getDenseI64ArrayAttr({}),
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
      inAttr.getStage(), 0, inAttr.getGroupType());
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

template <typename OpTy>
class CoreParallel : public OpRewriterPatternEx<OpTy> {
public:
  CoreParallel(mlir::MLIRContext *context)
      : OpRewriterPatternEx<OpTy>(context, "CoreParallel") {}

  LogicalResult matchAndRewriteImpl(OpTy gOp,
                                    PatternRewriter &rewriter) const override {
    if (isa_and_nonnull<tpu::CoreParallelOp>(gOp->getParentOp()))
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
  }
  bool shouldPrint(OpTy gOp) const override { return false; }
};

// CV84X6's floating-point grouped-convolution backend currently splits only N.
// For N=1, distribute complete groups instead. At this point the weights have
// already been reordered; restrict the rewrite to the standard OC-major packed
// layout so CoreSplit/CoreJoin can use contiguous views during address assign.
class GroupConvCoreParallel : public OpRewriterPatternEx<tpu::Conv2DOp> {
public:
  GroupConvCoreParallel(MLIRContext *context)
      : OpRewriterPatternEx<tpu::Conv2DOp>(context, "GroupConvCoreParallel") {
    if (const char *path = getenv("TPU_DIALECT_REWRITER_CONFIG")) {
      for (const auto &rule : loadRewriteConfig(path))
        if (rule.pattern_name == "GroupConvCoreParallel" &&
            getParam(rule.params, "use_pattern", 1) == 0)
          enabled = false;
    }
  }

  LogicalResult matchAndRewriteImpl(tpu::Conv2DOp op,
                                    PatternRewriter &rewriter) const override {
    auto cores = module::getCoreNum();
    // Do not repartition local groups, existing parallel regions, control flow
    // or dynamic subnets. N>1 channel slices are not contiguous in NCHW.
    if (!enabled || !module::isCV84X6() || cores < 2 ||
        !isa<FuncOp>(op->getParentOp()) ||
        getRunMode(op) != RunMode::TPU_STATIC || op->hasAttr("ginfo") ||
        !op.getMulticore().value_or(false) ||
        !op.getDoCoreParallel().value_or(true) || op.getGroup() <= 1 ||
        op.getCoeffMerged() || op.getWeightIsCoeff() != 1 ||
        op.getUse_3icOptimize() != 0 || op.getUseWinograd().value_or(0) != 0 ||
        op.getDoKernelRotate().value_or(false) ||
        op->hasAttr("compress_info") || op->hasAttr("weight_bits"))
      return failure();
    auto input = plainTensor(op.getInput());
    auto output = plainTensor(op.getOutput());
    auto filter = plainTensor(op.getFilter());
    if (!input || !output || !filter ||
        (input.getRank() != 3 && input.getRank() != 4) ||
        output.getRank() != input.getRank() || filter.getRank() != 4 ||
        input.getDimSize(0) != 1 || output.getDimSize(0) != 1 ||
        !module::isWeight(op.getFilter()))
      return failure();
    auto dtype = input.getElementType();
    if ((!dtype.isF16() && !dtype.isBF16()) ||
        output.getElementType() != dtype || filter.getElementType() != dtype)
      return failure();
    auto kernel = module::getI64Array(op.getKernelShape());
    auto stride = module::getI64Array(op.getStrides());
    auto pads = module::getI64Array(op.getPads());
    auto dilation = module::getI64Array(op.getDilations(), 2, 1);
    auto inserts = module::getI64Array(op.getInserts(), 2, 0);
    if (kernel->size() != 2 || stride->size() != 2 || pads->size() != 4 ||
        dilation->size() != 2 || inserts->size() != 2 ||
        llvm::any_of(*kernel, [](int64_t d) { return d <= 0; }) ||
        llvm::any_of(*stride, [](int64_t d) { return d <= 0 || d > 15; }) ||
        llvm::any_of(*dilation, [](int64_t d) { return d <= 0 || d > 15; }) ||
        llvm::any_of(*inserts, [](int64_t d) { return d != 0; }))
      return failure();
    int64_t groups = op.getGroup();
    int64_t ic = input.getDimSize(1), oc = output.getDimSize(1);
    if (ic % groups || oc % groups || (ic == groups && oc == groups))
      return failure();
    // Standard FP16/BF16 packing is [1, OC, ..., ...], with one independent
    // row of align_up(IC/group, IC_PARALLEL) * KH * KW elements per channel.
    // Large-stride cell packing, merged coefficients and depthwise packing
    // have different layouts and must not be sliced using this rule.
    int64_t icParallel = backend::BM168x::ic_num(2);
    if (icParallel <= 0)
      return failure();
    int64_t packedRow =
        align_up(ic / groups, icParallel) * (*kernel)[0] * (*kernel)[1];
    if (filter.getDimSize(0) != 1 || filter.getDimSize(1) != oc ||
        filter.getDimSize(2) * filter.getDimSize(3) != packedRow)
      return failure();
    auto weight = op.getFilter().getDefiningOp();
    if (weight->hasAttr("compress_info"))
      return failure();
    if (op.getWithBias()) {
      auto bias = plainTensor(op.getBias());
      if (!bias || bias.getRank() != 4 || !module::isWeight(op.getBias()) ||
          !bias.getElementType().isF32() ||
          bias.getShape() != ArrayRef<int64_t>({1, oc, 1, 1}))
        return failure();
    } else if (!module::isNone(op.getBias())) {
      return failure();
    }

    int64_t parts = std::min<int64_t>(cores, groups);
    SmallVector<int64_t> groupSizes;
    for (int64_t i = 0; i < parts; ++i)
      groupSizes.push_back(groups / parts + (i < groups % parts));
    auto typesFor = [&](Value value, int64_t channelsPerGroup) {
      auto type = value.getType().cast<RankedTensorType>();
      SmallVector<Type> types;
      for (auto size : groupSizes) {
        auto shape = type.getShape().vec();
        shape[1] = channelsPerGroup * size;
        types.push_back(RankedTensorType::get(shape, type.getElementType()));
      }
      return types;
    };
    auto name = module::getName(op.getOutput()).str();
    auto loc = [&](StringRef suffix) {
      return NameLoc::get(rewriter.getStringAttr(name + suffix.str()));
    };
    auto parallel = rewriter.create<tpu::CoreParallelOp>(
        op.getLoc(), op->getResultTypes(), op->getOperands(), 0, cores);
    parallel.getBody().push_back(new Block());
    rewriter.setInsertionPointToStart(&parallel.getBody().front());
    auto split = [&](Value value, int64_t channelsPerGroup, StringRef suffix) {
      auto types = typesFor(value, channelsPerGroup);
      return rewriter.create<tpu::CoreSplitOp>(
          loc(suffix), TypeRange(types), ValueRange{value},
          ArrayRef<NamedAttribute>{
              rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(1))});
    };
    auto inputSplit = split(op.getInput(), ic / groups, "_group_input");
    auto filterSplit = split(op.getFilter(), oc / groups, "_group_filter");
    tpu::CoreSplitOp biasSplit;
    if (op.getWithBias())
      biasSplit = split(op.getBias(), oc / groups, "_group_bias");
    auto outputTypes = typesFor(op.getOutput(), oc / groups);
    SmallVector<Value> outputs;
    for (int64_t i = 0; i < parts; ++i) {
      NamedAttrList attrs(op->getAttrs());
      attrs.set("group", rewriter.getI64IntegerAttr(groupSizes[i]));
      attrs.set("multicore", rewriter.getBoolAttr(false));
      attrs.set("do_core_parallel", rewriter.getBoolAttr(false));
      auto child = rewriter.create<tpu::Conv2DOp>(
          loc("_group_core" + std::to_string(i)), outputTypes[i],
          ValueRange{inputSplit->getResult(i), filterSplit->getResult(i),
                     biasSplit ? biasSplit->getResult(i) : op.getBias()},
          attrs.getAttrs());
      outputs.push_back(child.getOutput());
    }
    auto join = rewriter.create<tpu::CoreJoinOp>(
        loc("_group_join"), op.getOutput().getType(), outputs,
        ArrayRef<NamedAttribute>{
            rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(1))});
    rewriter.create<tpu::YieldOp>(op.getLoc(), join.getOutput());
    rewriter.replaceOp(op, parallel.getResults());
    return success();
  }

private:
  bool enabled = true;
  static RankedTensorType plainTensor(Value value) {
    auto type = value.getType().dyn_cast<RankedTensorType>();
    if (!type || !type.hasStaticShape() || type.getEncoding() ||
        llvm::any_of(type.getShape(), [](int64_t d) { return d <= 0; }))
      return {};
    return type;
  }
};

void doSpecificPattern(ModuleOp m) {
  // Add an Op-specific pattern if the generic IndexingMap fails to capture
  // the parallel semantics in this operation.
  module::applyPatternOnce<CoreParallel<tpu::GroupOp>>(m);
  module::applyPatternOnce<GroupConvCoreParallel>(m);
};

} // namespace tpu
} // namespace tpu_mlir
