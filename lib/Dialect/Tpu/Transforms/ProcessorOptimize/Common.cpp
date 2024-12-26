//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "Common.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/DevParallel/DistributeUtils.h"
namespace tpu_mlir {
namespace tpu {

LogicalResult
LargePadConvPattern::matchAndRewriteImpl(tpu::Conv2DOp op,
                                         PatternRewriter &rewriter) const {
  if (!(module::isBM1684Family() || module::isBM1684XFamily() ||
        module::isBM1690Family())) {
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

  if (module::isBM1684XFamily() || module::isBM1690Family()) {
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

  // need exchange conv_paddings[1] and conv_paddings[2]
  auto swap_val = conv_paddings[1];
  conv_paddings[1] = conv_paddings[2];
  conv_paddings[2] = swap_val;
  op.setPadsAttr(rewriter.getI64ArrayAttr(conv_paddings));
  return success();
}

void moveUnaryPermute(tpu::PermuteOp &op, Operation *nextOp,
                      PatternRewriter &rewriter,
                      std::vector<int64_t> *newUnaryShape = nullptr,
                      std::vector<int64_t> *newPermuteShape = nullptr) {
  auto oldNextOpName = module::getName(nextOp).str();

  auto input = op.getInput();
  auto output = nextOp->getResult(0);
  auto outputDtype = module::getElementType(output);

  // input -> unary
  rewriter.updateRootInPlace(nextOp, [&] {
    nextOp->setOperand(0, input);
    if (nextOp->getOperands().size() == 2 &&
        module::isWeight(nextOp->getOperand(1))) {
      if (auto binaryshift_op = dyn_cast<tpu::BinaryShiftOp>(nextOp)) {
        auto binaryshift_weight_Op =
            dyn_cast<top::WeightOp>(nextOp->getOperand(1).getDefiningOp());
        // transpose the weight
        auto weight_type =
            module::getElementType(binaryshift_weight_Op.getOutput());
        auto weight_shape = module::getShape(binaryshift_weight_Op.getOutput());
        if (weight_shape.size() != 4) {
          return;
        }
        if (weight_type.isInteger(8)) {
          auto weight_data = binaryshift_weight_Op.read<uint8_t>();
          auto weight_trans =
              std::make_shared<std::vector<uint8_t>>(weight_data->size(), 0);
          function_permute(weight_data->data(), weight_trans->data(),
                           weight_shape, {0, 2, 1, 3});
          std::vector<int64_t> weight_new_shape = {
              weight_shape[0], weight_shape[2], weight_shape[1],
              weight_shape[3]};
          rewriter.setInsertionPointAfter(op);
          auto type = RankedTensorType::get(weight_new_shape, weight_type);
          auto new_weight = top::WeightOp::create<uint8_t>(
              op,
              module::getName(binaryshift_weight_Op.getOperation()).str() +
                  "transposed",
              *weight_trans, type);
          nextOp->setOperand(1, new_weight);
        } else if (weight_type.isInteger(32)) {
          auto weight_data = binaryshift_weight_Op.read<uint32_t>();
          auto weight_trans =
              std::make_shared<std::vector<uint32_t>>(weight_data->size(), 0);
          function_permute(weight_data->data(), weight_trans->data(),
                           weight_shape, {0, 2, 1, 3});
          std::vector<int64_t> weight_new_shape = {
              weight_shape[0], weight_shape[2], weight_shape[1],
              weight_shape[3]};
          rewriter.setInsertionPointAfter(op);
          auto type = RankedTensorType::get(weight_new_shape, weight_type);
          auto new_weight = top::WeightOp::create<uint32_t>(
              op,
              module::getName(binaryshift_weight_Op.getOperation()).str() +
                  "transposed",
              *weight_trans, type);
          nextOp->setOperand(1, new_weight);
        } else if (weight_type.isInteger(16)) {
          auto weight_data = binaryshift_weight_Op.read<uint16_t>();
          auto weight_trans =
              std::make_shared<std::vector<uint16_t>>(weight_data->size(), 0);
          function_permute(weight_data->data(), weight_trans->data(),
                           weight_shape, {0, 2, 1, 3});
          std::vector<int64_t> weight_new_shape = {
              weight_shape[0], weight_shape[2], weight_shape[1],
              weight_shape[3]};
          rewriter.setInsertionPointAfter(op);
          auto type = RankedTensorType::get(weight_new_shape, weight_type);
          auto new_weight = top::WeightOp::create<uint16_t>(
              op,
              module::getName(binaryshift_weight_Op.getOperation()).str() +
                  "transposed",
              *weight_trans, type);
          nextOp->setOperand(1, new_weight);
        } else {
          llvm_unreachable("Weight type error!");
        }
      }
    }

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
  // nextOp->dump();
  // op.dump();
  return;
} // namespace tpu

// reorder op when transpose is before unary and biary operation to optimize
// bert
LogicalResult
PermuteReorderPattern::matchAndRewriteImpl(tpu::PermuteOp op,
                                           PatternRewriter &rewriter) const {

  if (!op.getOutput().hasOneUse()) {
    return failure();
  }
  auto nextOp = *op.getOutput().getUsers().begin();
  if (!nextOp->hasOneUse()) {
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
                 tpu::MulShiftOp, tpu::ReluOp, tpu::RequantIntOp, tpu::ActiveOp,
                 tpu::BinaryShiftOp,
                 tpu::BinaryConstShiftOp /** ex. tpu::SigmoidOp */
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
LogicalResult
PermutePadSwap::matchAndRewriteImpl(tpu::PermuteOp op,
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

Value createSplitQuantizedMLP(mlir::PatternRewriter &rewriter,
                              mlir::Operation *op, Value arg0) {
  auto left1 = arg0;
  // split the pattern
  std::vector<Value> operands;
  for (int i = 0; i < 2; ++i) {
    auto cur_out = left1;
    Operation *next_op = op;
    auto suffix = std::to_string(i);
    next_op = tpu::cloneColParallelMatMul(rewriter, next_op, cur_out, 2, i, 0);
    next_op = tpu::cloneCommonOp(rewriter, next_op, cur_out, suffix);
    next_op = tpu::cloneRowParallelMatMul(rewriter, next_op, cur_out, 2, i, 0);
    operands.push_back(cur_out);
  }

  rewriter.setInsertionPointAfterValue(operands[0]);
  std::string suffix = std::string("add_");
  auto loc = module::getLocLike(operands[1], suffix);
  auto add = rewriter.create<tpu::AddOp>(
      loc, operands[0].getType(), mlir::ValueRange{operands[0], operands[1]});
  return add.getOutput();
}

Value weight_split(Value weight, int split_num, int idx, int axis, Type to_type,
                   std::string base_name) {
  auto op = weight.getDefiningOp();
  if (module::isWeight(weight)) {
    auto shape = module::getShape(weight);
    auto dim = shape.size();
    axis = axis < 0 ? dim + axis : axis;
    int begin = shape[axis] / split_num * idx;
    int end = shape[axis] / split_num * (idx + 1);
    end = end > shape[axis] ? shape[axis] : end;
    std::string suffix = base_name + "_split_" + std::to_string(idx);
    return dyn_cast<top::WeightOp>(op).split(begin, end, axis, to_type, suffix);
  } else {
    return top::NoneOp(op);
  }
}

Value createSplitQuantizedMLP2(mlir::PatternRewriter &rewriter,
                               mlir::Operation *op, Value arg0,
                               int num_devices) {
  std::vector<Value> operands;
  auto none_op = module::getNoneOp(op);
  std::vector<int64_t> m0_shape = module::getShape(op->getResult(0));
  m0_shape[m0_shape.size() - 1] /= num_devices;
  Value rq_out;
  for (int i = 0; i < num_devices; ++i) {
    auto cur_out = arg0;
    Operation *next_op = op;
    auto suffix = "split_" + std::to_string(i);
    // matmul split weight col
    auto m0 = dyn_cast<tpu::MatMulOp>(op);
    auto w0 = weight_split(m0.getRight(), num_devices, i, -1,
                           module::getStorageType(m0.getRight()), "");
    auto b0 = weight_split(m0.getBias(), num_devices, i, -1,
                           module::getStorageType(m0.getBias()), "");
    auto multi0 = weight_split(m0.getMulti(), num_devices, i, -1,
                               module::getStorageType(m0.getMulti()), "");
    auto new_loc = module::getLocLike(m0.getOutput(), suffix);
    auto m0_type = module::getTypeLike(m0.getOutput(), m0_shape);
    auto new_m0 = rewriter.create<tpu::MatMulOp>(
        new_loc, m0_type, ValueRange{arg0, w0, b0, multi0, none_op},
        op->getAttrs());

    next_op = *next_op->user_begin();
    auto new_common_op = rewriter.clone(*next_op);
    module::setLocSuffix(new_common_op, suffix);
    new_common_op->setOperand(0, new_m0.getOutput());
    module::setShape(new_common_op->getResult(0), m0_shape);
    cur_out = new_common_op->getResult(0);
    next_op = *next_op->user_begin();
    // matmul split weight row
    auto m1 = dyn_cast<tpu::MatMulOp>(next_op);
    auto w1 = weight_split(m1.getRight(), num_devices, i, -2,
                           module::getStorageType(m1.getRight()), "");
    auto new1_loc = module::getLocLike(m1.getOutput(), suffix);
    auto out_shape = module::getShape(m1.getOutput());
    auto newType = RankedTensorType::get(out_shape, rewriter.getI32Type());
    std::vector<Value> operands_m1 = {cur_out, w1};
    if (i == num_devices - 1) {
      operands_m1.push_back(m1.getBias());
    } else {
      operands_m1.push_back(none_op);
    }
    operands_m1.push_back(none_op);
    operands_m1.push_back(none_op);
    auto new_m1 = rewriter.create<tpu::MatMulOp>(new1_loc, newType, operands_m1,
                                                 op->getAttrs());
    new_m1.setFuseRqAttr(rewriter.getBoolAttr(false));

    operands.push_back(new_m1.getOutput());
    if (i > 0) {
      std::string suffix = std::string("add_") + std::to_string(i);
      auto loc = module::getLocLike(new_m1.getOutput(), suffix);
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("shift", rewriter.getSI32IntegerAttr(0)));
      attrs.push_back(
          rewriter.getNamedAttr("mode", rewriter.getStringAttr("Add")));
      auto add =
          rewriter.create<tpu::BinaryShiftOp>(loc, newType, operands, attrs);
      operands.clear();
      operands.push_back(add);
    }
    if (i == num_devices - 1) {
      operands.push_back(m1.getMulti());
      std::vector<NamedAttribute> attrs;
      int32_t shift = module::getI64Array(m1.getRshifts())->at(0);
      attrs.push_back(
          rewriter.getNamedAttr("shift", rewriter.getSI32IntegerAttr(-shift)));
      attrs.push_back(
          rewriter.getNamedAttr("mode", rewriter.getStringAttr("Mul")));
      rq_out = rewriter.create<tpu::BinaryShiftOp>(
          m1.getLoc(), m1.getOutput().getType(), operands, attrs);
    }
  }
  return rq_out;
}

// reshape (in == out)
LogicalResult RemoveReshape::matchAndRewrite(tpu::ReshapeOp op,
                                             PatternRewriter &rewriter) const {
  auto shape0 = module::getShape(op.getOutput());
  auto shape1 = module::getShape(op.getInput());
  if (shape0 != shape1) {
    return failure();
  }
  op.getOutput().replaceAllUsesWith(op.getInput());
  rewriter.eraseOp(op);
  return success();
}

} // namespace tpu
} // namespace tpu_mlir
