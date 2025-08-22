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
#define DIV_UP(a, b) ((a) == 0 ? 0 : ((a)-1) / (b) + 1)

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
        auto weight_shape =
            module::getShape(binaryshift_weight_Op.getOutput()).vec();
        if (module::getNumElements(binaryshift_weight_Op.getOutput()) == 1) {
          weight_shape = {1, 1, 1, 1};
        }
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
    if (nextOp->getOperands().size() == 2 &&
        !module::isWeight(nextOp->getOperand(1))) {
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

namespace tpu_dialect_tiling_primitives {

llvm::SmallVector<Value> split_value(Value in_value, int axis,
                                     std::vector<int64_t> &tile_lens,
                                     std::string suffix,
                                     PatternRewriter &rewriter) {
  if (in_value.getType().isa<NoneType>()) {
    return llvm::SmallVector<Value, 8>();
  }
  auto in_shape = module::getShape(in_value);
  auto dims = in_shape.size();
  axis = axis < 0 ? dims + axis : axis;
  assert(std::accumulate(tile_lens.begin(), tile_lens.end(), 0) ==
         in_shape[axis]);
  llvm::SmallVector<Value> split_values;
  if (module::isWeight(in_value)) {
    auto weight_op = dyn_cast<top::WeightOp>(in_value.getDefiningOp());
    for (int i = 0; i < tile_lens.size(); i++) {
      int begin = std::accumulate(tile_lens.begin(), tile_lens.begin() + i, 0);
      int end = begin + tile_lens[i];
      split_values.push_back(
          weight_op.split(begin, end, axis, module::getStorageType(in_value),
                          suffix + "_tile" + std::to_string(i)));
    }
  } else {
    std::vector<int64_t> offset(dims, 0);
    std::vector<int64_t> steps(dims, 1);
    std::vector<int64_t> ends(dims, std::numeric_limits<int64_t>::max());
    for (int i = 0; i < tile_lens.size(); i++) {
      int begin = std::accumulate(tile_lens.begin(), tile_lens.begin() + i, 0);
      int end = begin + tile_lens[i];
      offset[axis] = begin;
      ends[axis] = end;
      std::vector<NamedAttribute> attrs;
      attrs.push_back(
          rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(offset)));
      attrs.push_back(
          rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(steps)));
      attrs.push_back(
          rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(ends)));
      attrs.push_back(
          rewriter.getNamedAttr("axes", rewriter.getI64ArrayAttr({axis})));
      auto none = module::getNoneOp(in_value.getDefiningOp());
      std::vector<Value> operands = {in_value, none, none, none, none};
      auto slice_op = rewriter.create<tpu::SliceOp>(
          module::getLocLike(in_value, suffix + "_tile" + std::to_string(i)),
          in_value.getType(), operands, attrs);
      std::vector<int64_t> out_shape = in_shape;
      out_shape[axis] = tile_lens[i];
      module::setShape(slice_op.getOutput(), out_shape);
      split_values.push_back(slice_op.getOutput());
    }
  }
  return split_values;
}

llvm::SmallVector<Value> split_value(Value in_value, int axis, int tile_len,
                                     std::string suffix,
                                     PatternRewriter &rewriter) {
  if (in_value.getType().isa<NoneType>()) {
    return llvm::SmallVector<Value, 8>();
  }
  auto in_shape = module::getShape(in_value);
  auto dim = in_shape.size();
  axis = axis < 0 ? dim + axis : axis;
  int num_tiles = DIV_UP(in_shape[axis], tile_len);
  std::vector<int64_t> tile_lens(num_tiles, tile_len);
  tile_lens[num_tiles - 1] = in_shape[axis] - (num_tiles - 1) * tile_len;
  return split_value(in_value, axis, tile_lens, suffix, rewriter);
}

Value clone_matmul(tpu::MatMulOp op, Value input, Value right, Value bias,
                   Value multi, PatternRewriter &rewriter, std::string suffix) {
  auto new_matmul = rewriter.clone(*op);
  new_matmul->setOperand(0, input);
  new_matmul->setOperand(1, right);
  if (bias)
    new_matmul->setOperand(2, bias);
  if (multi)
    new_matmul->setOperand(3, multi);
  module::setLocSuffix(new_matmul, suffix);
  auto left_shape = module::getShape(input);
  auto right_shape = module::getShape(right);
  auto o_feat_len =
      op.getRightTranspose()
          ? (op.getHdimIsBatch() ? right_shape[right_shape.size() - 3]
                                 : right_shape[right_shape.size() - 2])
          : right_shape[right_shape.size() - 1];
  std::vector<int64_t> new_shape = left_shape;
  new_shape[left_shape.size() - 1] = o_feat_len;
  module::setShape(new_matmul->getResult(0), new_shape);
  return new_matmul->getResult(0);
}

Value clone_splitK_matmul(tpu::MatMulOp matMulOp, Value input, Value weight,
                          Value bias, PatternRewriter &rewriter,
                          std::string suffix) {
  auto new_loc = module::getLocLike(matMulOp.getOutput(), suffix);
  auto out_shape = module::getShape(matMulOp.getOutput());
  Type new_type;
  if (module::getStorageType(matMulOp.getOutput()).isInteger(8)) {
    new_type = RankedTensorType::get(out_shape, rewriter.getI32Type());
  } else {
    new_type = matMulOp.getOutput().getType();
    if (!module::isWeight(matMulOp.getOutput()))
      llvm::WithColor::warning()
          << "unsafe data type for matmul split K pattern."
          << "\n";
  }
  auto none = module::getNoneOp(matMulOp);
  auto new_matmul = rewriter.create<tpu::MatMulOp>(
      new_loc, new_type, ValueRange{input, weight, bias, none, none},
      matMulOp->getAttrs());
  new_matmul.setRshiftsAttr(rewriter.getI64ArrayAttr({0}));
  new_matmul.setFuseRqAttr(rewriter.getBoolAttr(false));
  return new_matmul.getResult();
}

Value clone_reshape(tpu::ReshapeOp op, Value input,
                    std::vector<int64_t> &oshape, PatternRewriter &rewriter,
                    std::string suffix) {
  auto new_reshape = rewriter.clone(*op);
  new_reshape->setOperand(0, input);
  module::setLocSuffix(new_reshape, suffix);
  new_reshape->setAttr("shape", rewriter.getI64ArrayAttr(oshape));
  module::setShape(new_reshape->getResult(0), oshape);
  assert(module::getNumElements(new_reshape->getResult(0)) ==
             module::getNumElements(new_reshape->getOperand(0)) &&
         "wrong shape.");
  return new_reshape->getResult(0);
}

Value clone_common_op(Operation *op, Value input, PatternRewriter &rewriter,
                      std::string suffix) {
  auto new_op = rewriter.clone(*op);
  new_op->setOperand(0, input);
  module::setShape(new_op->getResult(0), module::getShape(input));
  module::setLocSuffix(new_op, suffix);
  return new_op->getResult(0);
}

Value clone_common_ops_between(Operation *beg_op, Operation *end_op,
                               Value input, PatternRewriter &rewriter,
                               std::string suffix, int max_depth) {
  if (max_depth <= 0) {
    llvm::errs() << "Warning: [clone_common_ops] max depth reached.\n";
  }
  assert(beg_op->hasOneUse() && "use wrong api.");
  Operation *next_op = *beg_op->getUsers().begin();
  if (next_op == end_op) {
    return input;
  }
  Value new_input = clone_common_op(next_op, input, rewriter, suffix);
  return clone_common_ops_between(next_op, end_op, new_input, rewriter, suffix,
                                  max_depth - 1);
}

StringRef findLongestCommonPrefix(ArrayRef<StringRef> strs) {
  if (strs.empty())
    return StringRef();
  StringRef prefix = strs[0];
  for (size_t i = 1; i < strs.size(); ++i) {
    while (!strs[i].startswith(prefix)) {
      prefix = prefix.drop_back();
      if (prefix.empty())
        return StringRef();
    }
  }
  return prefix;
}

std::string concatNameLocsWithCommonPrefix(llvm::SmallVector<Value> values) {
  if (values.empty())
    return "";
  if (values.size() == 1)
    return module::getName(values[0]).str();
  llvm::SmallVector<StringRef, 8> strs;
  for (auto value : values) {
    strs.push_back(module::getName(value));
  }
  StringRef commonPrefix = findLongestCommonPrefix(strs);
  size_t prefixLen = commonPrefix.size();
  std::string result;
  for (size_t i = 0; i < strs.size(); ++i) {
    if (i != 0) {
      result += "_AND_";
    }
    result += strs[i].substr(prefixLen).str();
  }
  return (commonPrefix.str() + result);
}

Value concat_values(llvm::SmallVector<Value> values, int axis,
                    PatternRewriter &rewriter) {
  assert(values.size() > 0);
  if (values.size() == 1)
    return values[0];
  std::vector<int64_t> out_shape = module::getShape(values[0]);
  axis = axis < 0 ? out_shape.size() + axis : axis;
  for (int i = 1; i < values.size(); i++) {
    out_shape[axis] += module::getShape(values[i])[axis];
  }
  auto new_loc = NameLoc::get(rewriter.getStringAttr(
      concatNameLocsWithCommonPrefix(values) + "_concat"));
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(axis)));
  auto concat_op = rewriter.create<tpu::ConcatOp>(
      new_loc, module::getTypeLike(values[0], out_shape), values, attrs);
  return concat_op.getResult();
}

Value reduce_add(Value input1, Value input2, PatternRewriter &rewriter) {
  // reduce sum.
  auto element_type = module::getStorageType(input1);
  bool is_float_type =
      element_type.isF32() || element_type.isF16() || element_type.isBF16();
  bool is_integer_type =
      element_type.isInteger(8) || element_type.isInteger(32);
  auto new_loc = NameLoc::get(rewriter.getStringAttr(
      concatNameLocsWithCommonPrefix({input1, input2}) + "_ReduceSum"));
  if (is_integer_type) {
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("shift", rewriter.getSI32IntegerAttr(0)));
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr("Add")));
    std::vector<Value> operands = {input1, input2};
    auto binary_op = rewriter.create<tpu::BinaryShiftOp>(
        new_loc, input1.getType(), operands, attrs);
    return binary_op.getResult();
  } else if (is_float_type) {
    std::vector<Value> operands = {input1, input2};
    std::vector<NamedAttribute> attrs;
    auto add_op =
        rewriter.create<tpu::AddOp>(new_loc, input1.getType(), operands, attrs);
    return add_op.getResult();
  } else {
    assert(0 && "not implemented.");
    return nullptr;
  };
}

Value apply_multipliers(Value input, Value multipliers, int scale, int rshift,
                        PatternRewriter &rewriter) {

  if (!module::isNone(multipliers)) {
    if (scale != 1)
      llvm::errs() << "not support yet."
                   << "\n";
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("shift", rewriter.getSI32IntegerAttr(rshift)));
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr("Mul")));
    Value result = rewriter.create<tpu::BinaryShiftOp>(
        module::getLocLike(input, "_multi"), input.getType(),
        ValueRange{input, multipliers}, attrs);
    return result;
  } else if (scale != 1 || rshift != 0) {
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("scale", rewriter.getSI32IntegerAttr(scale)));
    attrs.push_back(
        rewriter.getNamedAttr("shift", rewriter.getSI32IntegerAttr(rshift)));
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr("Mul")));
    Value result = rewriter.create<tpu::BinaryConstShiftOp>(
        module::getLocLike(input, "_binaryconstshift"), input.getType(),
        ValueRange{input}, attrs);
    return result;
  }
  return input;
}

bool is_same_shape(const std::vector<int64_t> &shape1,
                   std::vector<int> exp_shape) {
  if (shape1.size() != exp_shape.size())
    return false;
  for (int i = 0; i < shape1.size(); i++) {
    if (shape1[i] != exp_shape[i] && exp_shape[i] != -1)
      return false;
  }
  return true;
}

} // namespace tpu_dialect_tiling_primitives

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
