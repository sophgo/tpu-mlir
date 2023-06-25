//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

// reshape1+permute+reshape2 ===> pixelshuffle
// reshape1:[1x128x64x64] -> [1x32x2x2x64x64]
// permute:[1x32x2x2x64x64] -> [1x32x64x2x64x2]
// reshape2:1x32x64x2x64x2] -> [1x32x128x128]
//==>pixelshuffle:[1x128x64x64] -> [1x32x128x128]

struct TopPermuteToPixelShuffle : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto input_shape = module::getShape(op.getInput());
    if (input_shape.size() != 6) {
      return failure();
    }

    std::vector<int64_t> ps_crd = {0, 1, 4, 2, 5, 3};
    std::vector<int64_t> ps_dcr = {0, 3, 4, 1, 5, 2};
    auto order = module::getI64Array(op.getOrder());
    bool crd = true;
    if (*order == ps_crd) {
      crd = true;
    } else if (*order == ps_dcr) {
      crd = false;
    } else {
      return failure();
    }
    auto reshape_before =
        dyn_cast_or_null<ReshapeOp>(op.getInput().getDefiningOp());
    if (!reshape_before) {
      return failure();
    }
    auto nextOp = *op.getOutput().getUsers().begin();
    auto reshape_after = dyn_cast_or_null<ReshapeOp>(nextOp);
    if (!reshape_after) {
      return failure();
    }
    auto output_shape = module::getShape(reshape_after.getOutput());
    int64_t upscale_factor = input_shape[2];
    int64_t on = input_shape[0];
    int64_t oc = crd ? input_shape[1]: input_shape[3];
    int64_t oh = upscale_factor * input_shape[4];
    int64_t ow = upscale_factor * input_shape[5];
    std::vector<int64_t> o_s = {on, oc, oh, ow};
    if (output_shape.vec() != o_s) {
      return failure();
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("is_CRD", rewriter.getBoolAttr(crd)));
    attrs.push_back(
        rewriter.getNamedAttr("is_inversed", rewriter.getBoolAttr(false)));
    attrs.push_back(rewriter.getNamedAttr(
        "block_h", rewriter.getI64IntegerAttr(upscale_factor)));
    attrs.push_back(rewriter.getNamedAttr(
        "block_w", rewriter.getI64IntegerAttr(upscale_factor)));
    attrs.push_back(
        rewriter.getNamedAttr("in_is_NCHW", rewriter.getBoolAttr(true)));
    attrs.push_back(
        rewriter.getNamedAttr("out_is_NCHW", rewriter.getBoolAttr(true)));
    attrs.push_back(
        rewriter.getNamedAttr("swap_cr", rewriter.getBoolAttr(false)));
    rewriter.replaceOpWithNewOp<Depth2SpaceOp>(
        reshape_after, reshape_after.getResult().getType(),
        ValueRange{reshape_before.getInput()}, attrs);
    rewriter.eraseOp(op);
    rewriter.eraseOp(reshape_before);
    return success();
  }
};

// reshape1+permute+reshape2 ===> reorg
// reshape1:[1x128x64x64] -> [1x128x32x2x32x2]
// permute:[1x128x32x2x32x2] -> [1x128x2x2x32x32]
// reshape2:[1x128x2x2x32x32] -> [1x512x32x32]
//==>reorg:[1x128x64x64] -> [1x512x32x32]
struct TopPermuteToReorg : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto input_shape = module::getShape(op.getInput());
    if (input_shape.size() != 6) {
      return failure();
    }

    std::vector<int64_t> ps_crd = {0, 1, 3, 5, 2, 4};
    std::vector<int64_t> ps_dcr = {0, 3, 5, 1, 2, 4};
    auto order = module::getI64Array(op.getOrder());
    bool crd = true;
    if (*order == ps_crd) {
      crd = true;
    } else if (*order == ps_dcr) {
      crd = false;
    } else {
      return failure();
    }
    auto reshape_before =
        dyn_cast_or_null<ReshapeOp>(op.getInput().getDefiningOp());
    if (!reshape_before) {
      return failure();
    }
    auto nextOp = *op.getOutput().getUsers().begin();
    auto reshape_after = dyn_cast_or_null<ReshapeOp>(nextOp);
    if (!reshape_after) {
      return failure();
    }
    auto output_shape = module::getShape(reshape_after.getOutput());
    int64_t stride = input_shape[3];
    int64_t on = input_shape[0];
    int64_t oc = input_shape[1] * stride * stride;
    int64_t oh = input_shape[2];
    int64_t ow = input_shape[4];
    std::vector<int64_t> o_s = {on, oc, oh, ow};
    if (output_shape.vec() != o_s) {
      return failure();
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("is_CRD", rewriter.getBoolAttr(crd)));
    attrs.push_back(
        rewriter.getNamedAttr("is_inversed", rewriter.getBoolAttr(true)));
    attrs.push_back(
        rewriter.getNamedAttr("block_h", rewriter.getI64IntegerAttr(stride)));
    attrs.push_back(
        rewriter.getNamedAttr("block_w", rewriter.getI64IntegerAttr(stride)));
    attrs.push_back(
        rewriter.getNamedAttr("in_is_NCHW", rewriter.getBoolAttr(true)));
    attrs.push_back(
        rewriter.getNamedAttr("out_is_NCHW", rewriter.getBoolAttr(true)));
    attrs.push_back(
        rewriter.getNamedAttr("swap_cr", rewriter.getBoolAttr(false)));
    rewriter.replaceOpWithNewOp<Depth2SpaceOp>(
        reshape_after, reshape_after.getResult().getType(),
        ValueRange{reshape_before.getInput()}, attrs);
    rewriter.eraseOp(op);
    rewriter.eraseOp(reshape_before);
    return success();
  }
};

template <typename T>
static int remove_value(std::vector<T> &v, T value) {
  int idx = 0;
  for (auto iter = v.begin(); iter != v.end(); iter++, idx++) {
    if (*iter == value) {
      v.erase(iter);
      return idx;
    }
  }
  return -1;
}

static void refresh(std::vector<int64_t> &order, int64_t idx) {
  for (auto &v : order) {
    if (v > idx) {
      v--;
    }
  }
}

static bool is_valid_order(std::vector<int64_t> shape,
                           std::vector<int64_t> order) {
  int num_dims = order.size();
  int target_dim = num_dims;
  bool valid_order = true;
  if (num_dims > 4) {
    valid_order = false;
    for (int i = 0; i < num_dims; ++i) {
      if (shape[i] == 1) {
        target_dim--;
        if (target_dim <= 4) {
          return true;
        }
        remove_value<int64_t>(order, i);
        refresh(order, i);
      }
    } // end for check any dim == 1
    num_dims = target_dim;
    for (int i = 0; i < num_dims - 1; ++i) {
      if (order[i] + 1 == order[i + 1]) {
        target_dim--;
        if (target_dim <= 4) {
          return true;
        }
      }
    } // end for check continous order
  }   // end num_dims > 4
  return valid_order;
}

static int indx(std::vector<int64_t> &v, int64_t value) {
  return find(v.begin(), v.end(), value) - v.begin();
}

static void left_continous(std::vector<int64_t> &order,
                           std::vector<int64_t> &lorder0,
                           std::vector<int64_t> &lorder1) {
  lorder0.clear();
  lorder1.clear();
  auto begin = order.front();
  if (begin + 1 > (int)order.size() - 1) {
    begin -= 1;
  }
  lorder0.push_back(begin);
  lorder0.push_back(begin + 1);
  for (uint32_t i = 1; i < order.size(); ++i) {
    if (order[i] != begin + 1 && order[i] != begin) {
      lorder0.push_back(order[i]);
    }
  }
  for (uint32_t i = 0; i < order.size(); ++i) {
    lorder1.push_back(indx(lorder0, order[i]));
  }
}

// convert unsupport permute5d order to double permute
struct Permute5dSplit : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    // implement
    std::vector<int64_t> order = *(module::getI64Array(op.getOrder()));
    auto input = op.getInput();
    std::string name = module::getName(op.getOutput()).str();
    std::vector<int64_t> order0;
    std::vector<int64_t> order1;
    std::vector<int64_t> input_shape = module::getShape(input);
    std::vector<int64_t> output_shape = module::getShape(op.getOutput());
    std::vector<int64_t> new_shape0(order.size()); // permute0 output_shape
    std::vector<int64_t> new_shape1(order.size()); // permute1 output_shape
    if (order.size() != 5 || is_valid_order(input_shape, order)) {
      return failure();
    }
    left_continous(order, order0, order1);
    for (int i = 0; i < order0.size(); i++) {
      new_shape0[i] = input_shape[order0[i]];
    }
    for (int i = 0; i < order1.size(); i++) {
      new_shape1[i] = new_shape0[order1[i]];
    }
    assert(new_shape1 == output_shape);
    // create permute0
    rewriter.setInsertionPointAfterValue(input);
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_expand0"));
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    operands.emplace_back(input);
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order0)));
    auto eltType =
        op.getResult().getType().cast<RankedTensorType>().getElementType();
    auto outType = RankedTensorType::get(new_shape0, eltType);
    auto permute0_op =
        rewriter.create<top::PermuteOp>(loc, outType, operands, attrs);
    auto permute0_out = permute0_op.getOutput();
    // create permute1
    operands.clear();
    attrs.clear();
    operands.emplace_back(permute0_out);
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order1)));
    auto outType1 = op.getResult().getType().cast<RankedTensorType>();
    auto permute1_op =
        rewriter.create<top::PermuteOp>(op.getLoc(), outType1, operands, attrs);
    rewriter.replaceOp(op, {permute1_op.getResult()});
    return success();
  }
};

// permute + permute or permute + reshape + permute
struct PermuteFuse : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    if (in.hasOneUse() == false) {
      return failure();
    }
    if (auto rop = dyn_cast<ReshapeOp>(in.getDefiningOp())) {
      in = rop.getInput();
      if (in.hasOneUse() == false) {
        return failure();
      }
    }
    auto permute_op = dyn_cast<PermuteOp>(in.getDefiningOp());
    if (!permute_op) {
      return failure();
    }
    // op order
    std::vector<int64_t> in0_shape = module::getShape(permute_op.getInput());
    auto in0_order = module::getI64Array(permute_op.getOrder());
    std::vector<int64_t> in1_shape = module::getShape(op.getInput());
    auto in1_order = module::getI64Array(op.getOrder());
    std::vector<int64_t> out1_shape = module::getShape(op.getOutput());
    std::vector<int64_t> in0_shape_fix;
    std::vector<int64_t> in0_order_fix;
    std::vector<int64_t> out0_shape_fix;
    std::vector<int64_t> in1_shape_fix;
    std::vector<int64_t> in1_order_fix;
    int to_dim;
    for (to_dim = 2; to_dim <= 5; to_dim++) {
      auto ret = permute_reset(in0_shape, *in0_order, in0_shape_fix,
                               in0_order_fix, to_dim);
      if (ret == false) {
        continue;
      }
      ret = permute_reset(in1_shape, *in1_order, in1_shape_fix, in1_order_fix,
                          to_dim);
      if (ret == false) {
        continue;
      }
      break;
    }
    if (to_dim > 5) {
      return failure();
    }
    for (auto o : in0_order_fix) {
      out0_shape_fix.push_back(in0_shape_fix[o]);
    }
    if (in1_shape_fix != out0_shape_fix) {
      return failure();
    }
    // test
    std::vector<int64_t> origin_data;
    for (int64_t i = 0; i < to_dim; i++) {
      origin_data.push_back(i);
    }
    std::vector<int64_t> result0_data;
    for (auto o : in0_order_fix) {
      result0_data.push_back(origin_data[o]);
    }
    std::vector<int64_t> result1_data;
    for (auto o : in1_order_fix) {
      result1_data.push_back(result0_data[o]);
    }
    if (result1_data != origin_data) {
      return failure();
    }
    // bingoo !
    if (out1_shape == in0_shape) {
      op.getOutput().replaceAllUsesWith(permute_op.getInput());
    } else {
      std::string in_name =
          module::getName(permute_op.getInput()).str() + "_Reshape";
      auto loc = NameLoc::get(rewriter.getStringAttr(in_name));
      rewriter.setInsertionPoint(op);
      auto rs_op = rewriter.create<ReshapeOp>(
          loc, op.getOutput().getType(), ValueRange{permute_op.getInput()});
      op.getOutput().replaceAllUsesWith(rs_op.getOutput());
    }
    return success();
  }
};

// Permute can convert to Reshape in some situations.
// For example:
// [4,3,28,1] => [4,3,1,28]
// [4,3,1,28] => [4,1,3,28]
struct TopPermuteToReshape : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    // todo
    std::vector<int64_t> shape = module::getShape(op.getInput());
    int dim_size = shape.size();
    int start = 0, end = dim_size - 1;
    auto order = module::getI64Array(op.getOrder());
    while (start < dim_size && start == order->at(start)) {
      start++;
    }
    while (end > start && end == order->at(end)) {
      end--;
    }
    bool do_reshape = true;
    int64_t sum = 1;
    for (int index = start; index <= end; index++) {
      sum *= shape[index];
      if (shape[index] != 1 && sum != shape[index]) {
        do_reshape = false;
        break;
      }
    }
    if (do_reshape == false) {
      return failure();
    }
    std::vector<Value> operands;
    operands.emplace_back(op.getInput());
    rewriter.replaceOpWithNewOp<top::ReshapeOp>(op, op.getResult().getType(),
                                                operands);
    return success();
  }
};

struct SoftmaxPermutePattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto out = op.getOutput();
    if (out.hasOneUse() == false) {
      return failure();
    }
    auto user = *out.getUsers().begin();
    auto softmax_op = dyn_cast<SoftmaxOp>(user);
    if (!softmax_op) {
      return failure();
    }
    // check param
    auto permute_attr = op->getAttrs();
    const auto permute_order = module::getI64Array(op.getOrder());
    auto softmax_axis = softmax_op.getAxis();
    softmax_axis =
        softmax_axis < 0 ? softmax_axis + permute_order->size() : softmax_axis;
    auto new_axis = permute_order->at(softmax_axis);
    auto from = op.getInput();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(new_axis)));
    attrs.push_back(rewriter.getNamedAttr(
        "log", rewriter.getBoolAttr(softmax_op.getLog())));
    auto loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(op.getInput()).str() + "_softmax"));
    rewriter.setInsertionPointAfterValue(from);
    auto ns_op = rewriter.create<SoftmaxOp>(loc, op.getInput().getType(),
                                            ValueRange{from}, attrs);
    rewriter.replaceOp(op, {ns_op});
    attrs.clear();
    auto new_permuted_shape = module::getShape(softmax_op.getOutput());
    rewriter.setInsertionPointAfterValue(ns_op.getOutput());
    auto newType = RankedTensorType::get(new_permuted_shape,
                                         module::getElementType(softmax_op));
    rewriter.replaceOpWithNewOp<PermuteOp>(softmax_op, newType,
                                           ns_op.getOutput(), permute_attr);

    return success();
  }
};

/*
Input0 + Permute ->               Input0           ->
                 -> MaksedFill =>                  -> MaskedFill + Permute
Input1           ->               Input1 + Permute ->
*/
struct MaskedFillPermutePattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    // check topo
    const auto &output = op.getOutput();
    if (!output.hasOneUse()) {
      return failure();
    }
    // get name
    auto name = module::getName(op.getOutput()).str();
    if (name.find("_permute") != std::string::npos) {
      return failure();
    }
    auto p_next_op = *output.getUsers().begin();
    if (!isa<MaskedFillOp>(p_next_op)) {
      return failure();
    }
    auto masked_fill_op = dyn_cast<MaskedFillOp>(p_next_op);
    // check param
    auto permute_attr = op->getAttrs();
    auto permute_order = *module::getI64Array(op.getOrder());
    std::vector<int64_t> inv_order(permute_order.size());
    for (int i = 0; i < permute_order.size(); ++i) {
      inv_order[permute_order[i]] = i;
    }
    assert(p_next_op->getNumOperands() == 2);
    PermuteOp permute_op = op;
    for (auto opd : p_next_op->getOperands()) {
      Operation *op_ = opd.getDefiningOp();
      if (op_ == op.getOperation())
        continue;
      if (!opd.hasOneUse()) {
        return failure();
      }
      auto name = module::getName(op_->getResults()[0]);
      // insert new permute op for another input
      std::vector<NamedAttribute> attrs;
      auto type = op.getInput().getType();
      auto permute_loc =
          NameLoc::get(rewriter.getStringAttr(name.str() + "_permute"));
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(inv_order)));
      permute_op = rewriter.create<PermuteOp>(
          permute_loc, type, ValueRange{op_->getResults()[0]}, attrs);
    }

    auto from = op.getInput();
    Value cond, brn;
    if(op.getOutput() == masked_fill_op.getCond()) {
      cond = from;
      brn = permute_op;
    } else {
      cond = permute_op;
      brn = from;
    }
    auto masked_fill_attrs = masked_fill_op->getAttrs();
    auto loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(op.getInput()).str() + "_masked_fill"));
    auto new_masked_fill = rewriter.create<MaskedFillOp>(
        loc, op.getInput().getType(), ValueRange{cond, brn},
        masked_fill_attrs);
    auto new_permuted_shape = module::getShape(masked_fill_op.getOutput());
    rewriter.setInsertionPointAfterValue(new_masked_fill.getOutput());
    auto newType = RankedTensorType::get(
        new_permuted_shape, module::getElementType(masked_fill_op));
    auto new_permute_op = rewriter.create<PermuteOp>(
        masked_fill_op.getLoc(), newType, ValueRange{new_masked_fill.getOutput()},
        permute_attr);
    masked_fill_op.replaceAllUsesWith(new_permute_op.getOperation());
    return success();
  }
};

/*
Input0 + Permute ->          Input0           ->
                 -> Concat =>                 -> Concat + Permute
Input1           ->          Input1 + Permute ->
*/
struct ConcatPermutePattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    // check topo
    const auto &output = op.getOutput();
    if (!output.hasOneUse()) {
      return failure();
    }
    // get name
    auto name = module::getName(op.getOutput()).str();
    if (name.find("_permute") != std::string::npos) {
      return failure();
    }
    auto p_next_op = *output.getUsers().begin();
    if (!isa<ConcatOp>(p_next_op)) {
      return failure();
    }
    auto concat_op = dyn_cast<ConcatOp>(p_next_op);
    // check param
    auto permute_attr = op->getAttrs();
    auto permute_order = *module::getI64Array(op.getOrder());
    std::vector<int64_t> inv_order(permute_order.size());
    for (int i = 0; i < permute_order.size(); ++i) {
      inv_order[permute_order[i]] = i;
    }
    // TODO: support more than 2 inputs
    if(p_next_op->getNumOperands() != 2){
      return failure();
    }
    PermuteOp permute_op = op;
    std::vector<Value> concat_operands;
    for (auto opd : p_next_op->getOperands()) {
      Operation *op_ = opd.getDefiningOp();
      if (op_ == op.getOperation())
        continue;
      if (!opd.hasOneUse()) {
        return failure();
      }
      auto name = module::getName(op_->getResults()[0]);
      // insert new permute op other inputs
      std::vector<NamedAttribute> attrs;
      auto type = op.getInput().getType();
      auto permute_loc =
          NameLoc::get(rewriter.getStringAttr(name.str() + "_permute"));
      attrs.push_back(
          rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(inv_order)));
      permute_op = rewriter.create<PermuteOp>(
          permute_loc, type, ValueRange{op_->getResults()[0]}, attrs);
    }
    auto from = op.getInput();
    Value concat_input0, concat_input1;
    if(op.getOutput() == concat_op.getInputs()[0]) {
      concat_operands.emplace_back(from);
      concat_operands.emplace_back(permute_op);
    } else {
      concat_operands.emplace_back(permute_op);
      concat_operands.emplace_back(from);
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("do_relu", rewriter.getBoolAttr(concat_op.getDoRelu())));
    attrs.push_back(rewriter.getNamedAttr("relu_limit", rewriter.getF64FloatAttr(concat_op.getReluLimit().convertToDouble())));
    attrs.push_back(rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(permute_order[concat_op.getAxis()])));
    auto loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(op.getInput()).str() + "_concat"));
    auto old_shape = module::getShape(concat_op.getOutput());
    std::vector<int64_t> new_shape;
    for(int i = 0; i< old_shape.size(); i++){
      new_shape.push_back(old_shape[inv_order[i]]);
    }
    auto newType = RankedTensorType::get(
        new_shape, module::getElementType(concat_op));
    auto new_concat = rewriter.create<ConcatOp>(
        loc, newType, concat_operands,
        attrs);
    auto new_permuted_shape = module::getShape(concat_op.getOutput());
    rewriter.setInsertionPointAfterValue(new_concat.getOutput());
    newType = RankedTensorType::get(
        new_permuted_shape, module::getElementType(concat_op));
    auto new_permute_op = rewriter.create<PermuteOp>(
        concat_op.getLoc(), newType, ValueRange{new_concat.getOutput()},
        permute_attr);
    concat_op.replaceAllUsesWith(new_permute_op.getOperation());
    return success();
  }
};

/**
 * Op1->NonZero->Permute->Op2 => Op1->NonZero->Op2
 **/
struct NonZeroPermutePattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    const auto &input = op.getInput();
    // check topo
    if (!input.hasOneUse()) {
      return failure();
    }
    auto in_op = input.getDefiningOp();
    if (!isa<NonZeroOp>(in_op)) {
      return failure();
    }
    auto nonzero_op = dyn_cast<NonZeroOp>(in_op);
    // check param
    const auto permute_order = module::getI64Array(op.getOrder());
    if (permute_order->size() != 2) {
      return failure();
    }
    if (permute_order->at(0) != 1 || permute_order->at(1) != 0) {
      return failure();
    }
    // rewrite now !
    const auto old_nz_order = nonzero_op.getOrder().str();
    const auto new_nz_order =
        old_nz_order == "ColMajor" ? "RowMajor" : "ColMajor";
    Value from = nonzero_op.getInput();
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getStringAttr(new_nz_order)));
    rewriter.replaceOpWithNewOp<NonZeroOp>(op, op.getResult().getType(),
                                           ValueRange{from}, attrs);
    return success();
  }
};

// permute + pad -> pad + permute
struct PermutePadSwap : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto out = op.getOutput();
    if (out.hasOneUse() == false) {
      return failure();
    }
    auto user = *out.getUsers().begin();
    auto pad_op = dyn_cast<PadOp>(user);
    if (!pad_op) {
      return failure();
    }
    auto permute_order = module::getI64Array(op.getOrder());
    auto padding = module::getI64Array(pad_op.getPaddings());
    std::size_t num_axis = permute_order->size();
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
    auto new_permute_attrs = op->getAttrs();
    auto new_pad_attrs = pad_op->getAttrs();
    auto permute_in = op.getInput();
    auto pad_out = pad_op.getOutput();
    auto in_shape = module::getShape(permute_in);
    rewriter.setInsertionPointAfterValue(permute_in);
    std::vector<int64_t> new_padded_shape(num_axis, 0);
    for (size_t i = 0; i < num_axis; ++i) {
      new_padded_shape[i] =
          in_shape[i] + new_paddings[i] + new_paddings[i + num_axis];
    }
    auto newType = RankedTensorType::get(new_padded_shape,
                                         module::getElementType(permute_in));
    auto loc = NameLoc::get(
        rewriter.getStringAttr(module::getName(permute_in).str() + "_pad"));
    auto new_pad_op = rewriter.create<PadOp>(
        loc, newType, ValueRange{permute_in}, new_pad_attrs);
    rewriter.replaceOp(op, {new_pad_op});

    auto new_permuted_shape = module::getShape(pad_out);
    auto new_pad_out = new_pad_op.getOutput();
    rewriter.setInsertionPointAfterValue(new_pad_out);
    newType = RankedTensorType::get(new_permuted_shape,
                                    module::getElementType(pad_out));
    rewriter.replaceOpWithNewOp<PermuteOp>(pad_op, newType, new_pad_out,
                                           new_permute_attrs);
    return success();
  }
};

/**
 * Op1 -> perm -> next  => Op1 -> next -> perm
 **/
struct PermuteMovePattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp permOp,
                                PatternRewriter &rewriter) const override {
    // check topo
    // have one user only
    if (!permOp.getOutput().hasOneUse()) {
      return failure();
    }
    // move trait
    auto nextOp = *permOp.getOutput().user_begin();
    if (!nextOp->hasTrait<SupportPermuteMove>()) {
      return failure();
    }
    // permute only accept one argument
    // thus the output of 'next' should be exactly one
    // otherwise, we need to construct new permutation op
    if (nextOp->getResults().size() != 1) {
      return failure();
    }

    // rewrite
    auto input = permOp.getInput();
    auto inputType = input.getType();
    // input -> next
    rewriter.updateRootInPlace(nextOp, [&] {
      nextOp->setOperands(input);
      // should be the same type as the input
      nextOp->getResult(0).setType(inputType);
      // rewrite loc for tests
      auto loc = NameLoc::get(
          rewriter.getStringAttr(module::getName(input).str() + "_" +
                                 nextOp->getName().getStringRef()));
      nextOp->setLoc(loc);
    });
    // replace all uses of next to perm
    rewriter.replaceAllUsesWith(nextOp->getResult(0), permOp->getResult(0));
    // next -> perm
    rewriter.updateRootInPlace(permOp, [&] {
      permOp->setOperands(nextOp->getResults());
      // linear IR, tweak order
      permOp->moveAfter(nextOp);
      // rewrite loc for tests
      auto loc = NameLoc::get(
          rewriter.getStringAttr(module::getName(nextOp).str() + "_" +
                                 permOp->getName().getStringRef()));
      permOp->setLoc(loc);
    });
    return success();
  }
};

template <typename OpTy>
void _permute_binary_rewrite(PermuteOp pm_op1, PermuteOp pm_op2, OpTy bi_op,
                             PatternRewriter &rewriter) {
  std::vector<NamedAttribute> attrs_bi;
  for (auto &attr : bi_op->getAttrs()) {
    attrs_bi.push_back(attr);
  }
  std::vector<Value> opds_bi = {pm_op1.getOperand(), pm_op2.getOperand()};
  auto loc = NameLoc::get(rewriter.getStringAttr(
      module::getName(bi_op.getOutput()).str() + "_PermuteBinary"));
  rewriter.setInsertionPointAfterValue(pm_op1.getOperand());
  rewriter.setInsertionPointAfterValue(pm_op2.getOperand());
  auto bi_out = bi_op.getOutput();
  auto bi_out_shape = module::getShape(bi_out);
  auto order = *module::getI64Array(pm_op1.getOrder());
  std::vector<int64_t> inv_order(order.size());
  for (int i = 0; i < order.size(); ++i) {
    inv_order[order[i]] = i;
  }
  std::vector<int64_t> new_bi_out_shape(bi_out_shape.size(), 0);
  for (auto i = 0; i < bi_out_shape.size(); ++i) {
    new_bi_out_shape[i] = bi_out_shape[inv_order[i]];
  }
  auto new_bi_out_type =
      RankedTensorType::get(new_bi_out_shape, module::getElementType(bi_out));
  auto new_bi_op =
      rewriter.create<OpTy>(loc, new_bi_out_type, opds_bi, attrs_bi);
  std::vector<NamedAttribute> attrs_pm;
  attrs_pm.emplace_back(
      rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
  rewriter.replaceOpWithNewOp<PermuteOp>(bi_op, bi_op.getResult().getType(),
                                         new_bi_op.getOutput(), attrs_pm);
}

/**
 * Permute(x2)->Binary => Binary->Permute
 **/
struct PermuteBinaryPattern : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    // check topo
    const auto &output = op.getOutput();
    if (!output.hasOneUse()) {
      return failure();
    }
    auto p_next_op = *output.getUsers().begin();
    if (!isa<AddOp, SubOp, MulOp>(p_next_op)) {
      return failure();
    }
    PermuteOp fd_op = op;
    assert(p_next_op->getNumOperands() == 2);
    for (auto opd : p_next_op->getOperands()) {
      Operation *op_ = opd.getDefiningOp();
      if (op_ == op.getOperation())
        continue;
      if (!opd.hasOneUse()) {
        return failure();
      }
      if (!isa<PermuteOp>(op_)) {
        return failure();
      }
      fd_op = dyn_cast<PermuteOp>(op_);
    }
    // check param
    if (op == fd_op) {
    } else {
      const auto order1 = module::getI64Array(op.getOrder());
      const auto order2 = module::getI64Array(fd_op.getOrder());
      if (order1->size() != order2->size()) {
        return failure();
      }
      for (auto i = 0; i < order1->size(); ++i) {
        if (order1->at(i) != order2->at(i)) {
          return failure();
        }
      }
    }
    // rewrite now !
    if (isa<AddOp>(p_next_op)) {
      _permute_binary_rewrite(op, fd_op, dyn_cast<AddOp>(p_next_op), rewriter);
    } else if (isa<SubOp>(p_next_op)) {
      _permute_binary_rewrite(op, fd_op, dyn_cast<SubOp>(p_next_op), rewriter);
    } else if (isa<MulOp>(p_next_op)) {
      _permute_binary_rewrite(op, fd_op, dyn_cast<MulOp>(p_next_op), rewriter);
    } else {
    }
    return success();
  }
};

void PermuteOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<TopPermuteToPixelShuffle, TopPermuteToReorg, Permute5dSplit,
                 PermuteFuse, PermuteMovePattern, TopPermuteToReshape,
                 SoftmaxPermutePattern, NonZeroPermutePattern, PermutePadSwap,
                 ConcatPermutePattern, MaskedFillPermutePattern, PermuteBinaryPattern>(context);
}
