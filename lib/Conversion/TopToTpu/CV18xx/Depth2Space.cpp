//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Conversion/TopToTpu/LoweringCV18xx.h"

#define DEBUG_TYPE "lowering-Depth2Space"

namespace tpu_mlir {
namespace cv18xx {

template <typename T>
static void shuffleFilterWeight(std::vector<T> &filter_data,
                                std::vector<T> &new_filter,
                                RankedTensorType &filter_type, int64_t stride) {
  std::vector<int64_t> shape(filter_type.getShape());
  int64_t size = std::accumulate(std::begin(shape), std::end(shape), 1,
                                 std::multiplies<>());
  assert(filter_data.size() == size && "filter size should be equal");

  int64_t oc, ic, frame_size;
  int64_t index = shape.size();
  assert((index == 4 || index == 5) && "filter shape size should be 4 or 5");

  frame_size = shape[index - 1] * shape[index - 2];
  ic = shape[index - 3];
  oc = shape[index - 4];
  if (index == 5) {
    oc *= shape[index - 5];
  }
  int64_t channel_shape = ic / stride / stride;
  // shuffle channel weight
  for (int64_t i = 0; i < oc; i++) {
    for (int64_t j = 0; j < ic; j++) {
      int64_t row = j / (stride * stride);
      int64_t col = j % (stride * stride);
      int64_t dst_c = col * channel_shape + row;
      T *in = filter_data.data() + (i * ic + j) * frame_size;
      T *out = new_filter.data() + (i * ic + dst_c) * frame_size;
      memcpy(out, in, frame_size * sizeof(T));
    }
  }
}

// when it is space2depthOp(ReorgOp) and its mode is "crd", convert it to "dcr"
// reorg + swap nextConv weight/insert ShuffleChannelOp because cv18xx reorgOp's
// backend only support "dcr" reorg.
template <typename T>
static bool convertDepth2Space(PatternRewriter &rewriter,
                               top::Depth2SpaceOp d2sOp) {
  if (!d2sOp.getIsInversed() || !d2sOp.getIs_CRD()) {
    return false;
  }
  std::string name = module::getName(d2sOp.getOutput()).str();
  auto input = d2sOp.getInput();
  auto output = d2sOp.getOutput();
  int64_t stride = d2sOp.getBlockH();
  assert(stride == d2sOp.getBlockW());
  // create new reorg op
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  operands.emplace_back(input);
  attrs.push_back(rewriter.getNamedAttr("is_CRD", rewriter.getBoolAttr(false)));
  attrs.push_back(
      rewriter.getNamedAttr("is_inversed", rewriter.getBoolAttr(true)));
  attrs.push_back(
      rewriter.getNamedAttr("block_h", rewriter.getI64IntegerAttr(stride)));
  attrs.push_back(
      rewriter.getNamedAttr("block_w", rewriter.getI64IntegerAttr(stride)));
  attrs.push_back(rewriter.getNamedAttr(
      "in_is_NCHW", rewriter.getBoolAttr(d2sOp.getInIs_NCHW())));
  attrs.push_back(rewriter.getNamedAttr(
      "out_is_NCHW", rewriter.getBoolAttr(d2sOp.getOutIs_NCHW())));
  attrs.push_back(rewriter.getNamedAttr(
      "swap_cr", rewriter.getBoolAttr(d2sOp.getSwapCr())));
  auto newloc = NameLoc::get(rewriter.getStringAttr(name + "_DCR"));
  rewriter.setInsertionPointAfterValue(input);
  auto newReorgOp = rewriter.create<top::Depth2SpaceOp>(
      newloc, output.getType(), operands, attrs);
  auto reorg_out = newReorgOp.getOutput();
  // swap nextConv weight/insert shuffle channels
  auto nextOp = module::getNextOp(d2sOp.getOperation());
  auto convOp = dyn_cast_or_null<tpu::Conv2DOp>(nextOp);
  // bool needSC = false;
  if (convOp && convOp.getGroup() == 1) {
    llvm::errs() << "Swap convOp Channel Weight.\n";
    assert(convOp.getNumOperands() == 3 && "Conv2D op should have 3 operands");
    // filter
    auto filterOp = cast<top::WeightOp>(convOp.getFilter().getDefiningOp());
    auto filter_data = *(filterOp.read<T>());
    auto filter_name = module::getName(filterOp.getOutput()).str();
    auto filter_type =
        convOp.getFilter().getType().template cast<RankedTensorType>();
    std::vector<T> new_filter_data(filter_data.size());
    shuffleFilterWeight<T>(filter_data, new_filter_data, filter_type, stride);
    auto newFilterOp = top::WeightOp::create(
        convOp, filter_name + "_shufflechannel", new_filter_data, filter_type);
    convOp.setOperand(1, newFilterOp);
    rewriter.replaceOp(d2sOp.getOperation(), newReorgOp);
  } else {
    // insert shuffle channel
    llvm::errs() << "insert shuffle channel.\n";
    operands.clear();
    attrs.clear();
    operands.emplace_back(reorg_out);
    attrs.emplace_back(rewriter.getNamedAttr(
        "group", rewriter.getI64IntegerAttr(stride * stride)));
    auto loc = NameLoc::get(rewriter.getStringAttr(name));
    rewriter.setInsertionPointAfterValue(reorg_out);
    auto scOp = rewriter.create<top::ShuffleChannelOp>(loc, reorg_out.getType(),
                                                       operands, attrs);
    rewriter.replaceOp(d2sOp.getOperation(), scOp);
  }
  return true;
}

void Depth2SpaceLowering::LoweringINT8(PatternRewriter &rewriter,
                                       top::Depth2SpaceOp d2sOp,
                                       bool asymmetric) const {
  if (convertDepth2Space<int8_t>(rewriter, d2sOp)) {
    return;
  }
  lowering_common_int8<tpu::Depth2SpaceOp>(rewriter, d2sOp.getOperation(),
                                           asymmetric);
}

void Depth2SpaceLowering::LoweringBF16(PatternRewriter &rewriter,
                                       top::Depth2SpaceOp d2sOp) const {
  if (convertDepth2Space<uint16_t>(rewriter, d2sOp)) {
    return;
  }
  lowering_common_bf16<tpu::Depth2SpaceOp>(rewriter, d2sOp.getOperation());
}
} // namespace cv18xx
} // namespace tpu_mlir
