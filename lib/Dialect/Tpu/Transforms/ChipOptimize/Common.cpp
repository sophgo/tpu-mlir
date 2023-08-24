//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

namespace tpu_mlir {
namespace tpu {

LogicalResult
LargePadConvPattern::matchAndRewrite(tpu::Conv2DOp op,
                                     PatternRewriter &rewriter) const {
  if (!(module::isBM1684Family() || module::isBM1684XFamily())) {
    return failure();
  }

  auto pads_v = module::getI64Array(op.getPads());
  auto pad_top = pads_v->at(0);
  auto pad_left = pads_v->size() > 2 ? pads_v->at(1) : 0;
  auto pad_bottom = pads_v->size() > 2 ? pads_v->at(2) : pads_v->at(1);
  auto pad_right = pads_v->size() > 2 ? pads_v->at(3) : 0;
  int64_t max_pad =
      std::max(std::max(pad_top, pad_bottom), std::max(pad_left, pad_right));
  const int64_t max_pad_threshold = 15;
  if (max_pad <= max_pad_threshold) {
    return failure();
  }

  if (module::isBM1684XFamily()) {
    auto strides = module::getI64Array(op.getStrides());
    auto dilations = module::getI64Array(op.getDilations(), 2, 1);
    bool h_support_large_pad =
        (strides->at(0) > 15) || (dilations->at(0) > 15) ||
        (std::max(pad_top, pad_bottom) <= max_pad_threshold);
    bool w_support_large_pad =
        (strides->at(1) > 15) || (dilations->at(1) > 15) ||
        (std::max(pad_left, pad_right) <= max_pad_threshold);
    if (h_support_large_pad && w_support_large_pad) {
      return failure();
    }
  }

  llvm::SmallVector<int64_t> conv_paddings = {pad_top, pad_bottom, pad_left,
                                              pad_right};
  Value input_value = op->getOperand(0);
  std::string output_name = module::getName(op->getResult(0)).str();
  auto input_ele_type = module::getElementType(input_value);

  for (int64_t i = 0; i < max_pad / max_pad_threshold; i++) {
    std::string name_pad = output_name + "$pad" + std::to_string(i);
    auto loc_pad = NameLoc::get(rewriter.getStringAttr(name_pad));
    std::vector<Value> operands_pad;
    operands_pad.push_back(input_value);
    operands_pad.push_back(module::getNoneOp(op));
    operands_pad.push_back(module::getNoneOp(op));
    operands_pad.push_back(module::getNoneOp(op));
    std::vector<NamedAttribute> attrs_pad;
    // pad_paddings[0/1/4/5]: n/c paddings for new pad layer, are always 0
    // pad_paddings[2/3/6/7]: h/w paddings for new pad layer
    auto input_shape = module::getShape(input_value);
    llvm::SmallVector<int64_t> pad_paddings(input_shape.size() * 2, 0);
    int64_t pad_limit = (input_shape.size() == 3 ? 2 : 4);
    for (size_t j = 0; j < pad_limit; j++) {
      int padding = std::min(conv_paddings[j], max_pad_threshold);
      pad_paddings[(j < 2 ? 2 : 3) + (j % 2 == 0 ? 0 : input_shape.size())] =
          padding;
      conv_paddings[j] -= padding;
    }
    attrs_pad.push_back(rewriter.getNamedAttr(
        "paddings", rewriter.getI64ArrayAttr(pad_paddings)));
    attrs_pad.push_back(rewriter.getNamedAttr(
        "mode",
        tpu::PaddingModeAttr::get(getContext(), tpu::PaddingMode::constant)));

    auto output_shape_pad = llvm::SmallVector<int64_t>(input_shape);
    if (input_shape.size() == 3) {
      output_shape_pad[2] += (pad_paddings[2] + pad_paddings[5]);
    }
    if (input_shape.size() == 4) {
      output_shape_pad[2] += (pad_paddings[2] + pad_paddings[6]);
      output_shape_pad[3] += (pad_paddings[3] + pad_paddings[7]);
    }

    auto op_pad = rewriter.create<tpu::PadOp>(
        loc_pad, RankedTensorType::get(output_shape_pad, input_ele_type),
        operands_pad, attrs_pad);
    input_value = op_pad.getResult();
  }
  op.setOperand(0, input_value);
  op.setPadsAttr(rewriter.getI64ArrayAttr(conv_paddings));
  return success();
}

void moveUnaryPermute(tpu::PermuteOp &op, Operation *nextOp,
                      PatternRewriter &rewriter,
                      std::vector<int64_t> *newUnaryShape = nullptr,
                      std::vector<int64_t> *newPermuteShape = nullptr) {
  auto oldNextOpName = module::getName(nextOp).str();

  auto input = op.getInput();
  auto inputType = input.getType();
  auto output = nextOp->getResult(0);
  auto outputType = output.getType();
  auto outputDtype = module::getElementType(output);

  // input -> unary
  rewriter.updateRootInPlace(nextOp, [&] {
    nextOp->setOperand(0, input);

    auto newType =
        newUnaryShape == nullptr
            ? RankedTensorType::get(module::getShape(op->getOperand(0)),
                                    outputDtype)
            : RankedTensorType::get(*newUnaryShape, outputDtype); // for pad
    nextOp->getResult(0).setType(newType);
    auto loc = NameLoc::get(
        rewriter.getStringAttr(module::getName(nextOp).str() + "_" +
                               module::getName(op.getOperation()).str()));
    nextOp->setLoc(loc);
  });

  // replace all uses of next to perm
  rewriter.replaceAllUsesWith(nextOp->getResult(0), op->getResult(0));

  // permute -> output
  rewriter.updateRootInPlace(op, [&] {
    op->setOperand(0, nextOp->getOpResult(0));
    if (newPermuteShape) {
    }

    auto newType = newPermuteShape == nullptr
                       ? RankedTensorType::get(
                             module::getShape(op->getResult(0)), outputDtype)
                       : RankedTensorType::get(*newPermuteShape, outputDtype);

    op->getResult(0).setType(newType);

    op->moveAfter(nextOp);
    auto loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(op.getOperation()).str() + "_" + oldNextOpName));
    op->setLoc(loc);
  });
  nextOp->dump();
  op.dump();
  return;
}

// reorder op when transpose is before unary and biary operation to optimize
// bert
LogicalResult
PermuteReorderPattern::matchAndRewrite(tpu::PermuteOp op,
                                       PatternRewriter &rewriter) const {
  auto nextOp = *op.getOutput().getUsers().begin();
  if (!nextOp->hasOneUse()) {
    return failure();
  }

  if (!op.getOutput().hasOneUse()) {
    return failure();
  }

  // NOTE: if remove this constrain, new_bi_out_shape should be dynamicly
  // calculated
  std::vector<int64_t> ps = {0, 2, 1, 3};

  auto order = module::getI64Array(op.getOrder());
  if (auto permute_op = dyn_cast<tpu::PermuteOp>(nextOp)) {
    // permute + permute with the same order
    auto sec_order = module::getI64Array(permute_op.getOrder());
    if (*sec_order != ps) {
      return failure();
    }

    permute_op.replaceAllUsesWith(op.getInput());
    rewriter.eraseOp(permute_op);
    rewriter.eraseOp(op);
    return success();
  }

  if (*order != ps) {
    return failure();
  }

  if (isa<tpu::AddOp, tpu::SubOp, tpu::MulOp>(nextOp)) {
    /**
     * binary op
     *
     * input1 -> permute1 \           =>    input1 -> \
     *                     =>  biop   =>               => biop -> permute1
     * input2 -> permute2 /           =>    input2 -> /
     */
    assert(nextOp->getNumOperands() == 2);

    if (nextOp->getOperand(0).getDefiningOp() !=
            op /**only do optimize when "this" permute op
  is the first input of nextOp*/
        ||
        !isa<tpu::PermuteOp>(
            nextOp->getOperand(1)
                .getDefiningOp()) /**second input should also be permute op*/) {
      return failure();
    }
    auto secOp =
        dyn_cast<tpu::PermuteOp>(nextOp->getOperand(1).getDefiningOp());

    const auto ps2 = module::getI64Array(secOp.getOrder());
    if (ps != *ps2) { /**number or elements not equal*/
      return failure();
    }

    auto bi_out = nextOp->getResult(0);
    auto bi_old_type = bi_out.getType();
    auto bi_out_shape = module::getShape(bi_out);
    std::vector<int64_t> new_bi_out_shape(
        {bi_out_shape[0], bi_out_shape[2], bi_out_shape[1], bi_out_shape[3]});
    auto newType =
        RankedTensorType::get(new_bi_out_shape, module::getElementType(bi_out));
    bi_out.setType(newType); // [0, 1, 2, 3]
    nextOp->setOperands(ValueRange{op.getInput(), secOp.getInput()});

    rewriter.setInsertionPointAfter(nextOp);

    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(ps)));
    // replace all uses of next to perm
    rewriter.replaceAllUsesWith(nextOp->getResult(0), op->getResult(0));

    rewriter.updateRootInPlace(op, [&] {
      op->setOperand(0, nextOp->getOpResult(0));
      // linear IR, tweak order
      op->moveAfter(nextOp);
      // rewrite loc for tests
      auto loc = NameLoc::get(rewriter.getStringAttr(
          module::getName(op.getOperation()).str() + "_after"));
      op->setLoc(loc);
    });

    secOp.erase();
    return success();
  } else if (isa<tpu::SoftmaxOp, tpu::CastOp, tpu::MulConstOp, tpu::AddConstOp,
                 tpu::MulShiftOp, tpu::ReluOp,
                 tpu::ActiveOp /** ex. tpu::SigmoidOp */
                 >(nextOp)) {
    /**
     * unary operation
     * input → permute → unaryOp → output
     **/
    if (auto softmax_op = dyn_cast<tpu::SoftmaxOp>(nextOp)) {
      auto softmax_axis = softmax_op.getAxis();
      softmax_axis =
          softmax_axis < 0 ? softmax_axis + order->size() : softmax_axis;
      auto new_axis = order->at(softmax_axis);
      softmax_op.setAxis(new_axis);
    }
    auto nextOp = *op.getOutput().user_begin();

    if (nextOp->getResults().size() != 1) {
      return failure();
    }

    moveUnaryPermute(op, nextOp, rewriter);
    return success();
  }
  return failure();
}

// permute + pad -> pad + permute
LogicalResult PermutePadSwap::matchAndRewrite(tpu::PermuteOp op,
                                              PatternRewriter &rewriter) const {
  auto out = op.getOutput();
  if (out.hasOneUse() == false) {
    return failure();
  }

  auto user = *out.getUsers().begin();
  auto pad_op = dyn_cast<tpu::PadOp>(user);
  if (!pad_op) {
    return failure();
  }
  auto permute_order = module::getI64Array(op.getOrder());
  auto padding = module::getI64Array(pad_op.getPaddings());
  std::size_t num_axis = permute_order->size();
  // should be like: paddings: [0, 2, 0, 2, 0, 2, 0, 2]; order: [0, 2, 1, 3]
  if (padding->size() != 2 * num_axis) {
    return failure();
  }

  std::vector<int64_t> new_paddings(2 * num_axis, 0);
  std::vector<int64_t> rev_order(num_axis, 0);
  new_paddings.assign(padding->begin(), padding->end());
  rev_order.assign(permute_order->begin(), permute_order->end());
  // get reverse operation of permute
  for (int i = 0; i < num_axis; i++) {
    rev_order[permute_order->at(i)] = i;
  }
  // adjust paddings accordingly
  for (int i = 0; i < num_axis; i++) {
    new_paddings[i] = padding->at(rev_order[i]);
    new_paddings[i + num_axis] = padding->at(rev_order[i] + num_axis);
  }
  pad_op->setAttr("paddings", rewriter.getI64ArrayAttr(new_paddings));

  // swap pad Op and permute Op
  auto permute_in = op.getInput();
  auto in_shape = module::getShape(permute_in);
  std::vector<int64_t> new_padded_shape(num_axis, 0);
  for (size_t i = 0; i < num_axis; ++i) {
    new_padded_shape[i] =
        in_shape[i] + new_paddings[i] + new_paddings[i + num_axis];
  }

  auto pad_out = pad_op.getOutput();
  std::vector<int64_t> new_permuted_shape(module::getShape(pad_out));
  moveUnaryPermute(op, pad_op, rewriter, &new_padded_shape,
                   &new_permuted_shape);
  return success();
}

} // namespace tpu
} // namespace tpu_mlir
