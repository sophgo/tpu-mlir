//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

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
    auto nextOp = *op.getOutput().user_begin();
    auto reshape_after = dyn_cast_or_null<ReshapeOp>(nextOp);
    if (!reshape_after) {
      return failure();
    }
    auto output_shape = module::getShape(reshape_after.getOutput());
    int64_t upscale_factor = input_shape[2];
    int64_t on = input_shape[0];
    int64_t oc = crd ? input_shape[1] : input_shape[3];
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
    auto nextOp = *op.getOutput().user_begin();
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

template <typename T> static int remove_value(std::vector<T> &v, T value) {
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
    auto outType = module::getTypeLike(op.getOutput(), new_shape0);
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
      rewriter.eraseOp(op);
      rewriter.eraseOp(permute_op);
    } else {
      std::string in_name =
          module::getName(permute_op.getInput()).str() + "_Reshape";
      auto loc = NameLoc::get(rewriter.getStringAttr(in_name));
      rewriter.setInsertionPoint(op);
      auto rs_op = rewriter.create<ReshapeOp>(
          loc, op.getOutput().getType(), ValueRange{permute_op.getInput()});
      op.getOutput().replaceAllUsesWith(rs_op.getOutput());
      rewriter.eraseOp(op);
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
    if (do_reshape && order->size() == 2 && order->at(0) == 1 &&
        order->at(1) == 0) {
      auto nonzeroOp = dyn_cast<top::NonZeroOp>(op.getInput().getDefiningOp());
      if (nonzeroOp && nonzeroOp.getOrder().str() == "RowMajor")
        do_reshape = false;
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

// decomposed relative position embeddings
// this should be after MatMul.cpp:MatmulWithPermuteAndSplit
// wonder whether it should be in ChipOptimize
// topo
//                                           ...-MatMul-Reshape
//                                                         |
//                      ---------MatMul---------Unsqueeze-Add-
// ...-Permute-Reshape-{                                      }-Add-...
//                      -Permute-MatMul-Permute-Unsqueeze-----
// ==>
//
//                                                                                  ...MatMul
//      -Reshape-Permute-MatMul-Permute-Reshape-Unsqueeze- |
// ...-{ }-Add-Reshape-Permute-Reshape-Add-...
//      -Reshape-Permute-MatMul-Permute-Reshape-Unsqueeze-
struct TopDecomposedRelPosEmb : public OpRewritePattern<PermuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PermuteOp op,
                                PatternRewriter &rewriter) const override {
    // check topo
    if (!op->hasOneUse())
      return failure();
    auto reshape_op =
        dyn_cast_or_null<ReshapeOp>(*op.getOutput().getUsers().begin());
    if (!reshape_op)
      return failure();
    std::vector<mlir::Operation *> users;
    for (auto user : reshape_op.getOutput().getUsers()) {
      users.emplace_back(user);
    }
    if (users.size() != 2)
      return failure();
    auto matmul_h_op = dyn_cast_or_null<MatMulOp>(users[0]);
    auto permute_before_w_op = dyn_cast_or_null<PermuteOp>(users[1]);
    if (!(matmul_h_op && permute_before_w_op)) {
      matmul_h_op = dyn_cast_or_null<MatMulOp>(users[1]);
      permute_before_w_op = dyn_cast_or_null<PermuteOp>(users[0]);
      if (!(matmul_h_op && permute_before_w_op))
        return failure();
    }
    if (!permute_before_w_op->hasOneUse() || !matmul_h_op->hasOneUse())
      return failure();
    auto matmul_w_op = dyn_cast_or_null<MatMulOp>(
        *permute_before_w_op.getOutput().getUsers().begin());
    if (!matmul_w_op || !matmul_w_op->hasOneUse())
      return failure();
    if (!module::isWeight(matmul_h_op.getRight()) ||
        !module::isWeight(matmul_w_op.getRight()))
      return failure();
    auto permute_after_w_op = dyn_cast_or_null<PermuteOp>(
        *matmul_w_op.getOutput().getUsers().begin());
    if (!permute_after_w_op || !permute_after_w_op->hasOneUse())
      return failure();
    auto unsqueeze_h_op = dyn_cast_or_null<UnsqueezeOp>(
        *matmul_h_op.getOutput().getUsers().begin());
    if (!unsqueeze_h_op || !unsqueeze_h_op->hasOneUse())
      return failure();
    auto unsqueeze_w_op = dyn_cast_or_null<UnsqueezeOp>(
        *permute_after_w_op.getOutput().getUsers().begin());
    if (!unsqueeze_w_op || !unsqueeze_w_op->hasOneUse())
      return failure();
    auto add_h_op =
        dyn_cast_or_null<AddOp>(*unsqueeze_h_op.getOutput().getUsers().begin());
    if (!add_h_op || !add_h_op->hasOneUse())
      return failure();
    auto add_w_op =
        dyn_cast_or_null<AddOp>(*unsqueeze_w_op.getOutput().getUsers().begin());
    if (!add_w_op || !add_w_op->hasOneUse())
      return failure();
    auto reshape_other_out = add_h_op.getOperand(0);
    if (reshape_other_out == unsqueeze_h_op.getOutput()) {
      reshape_other_out = add_h_op.getOperand(1);
    }
    auto reshape_other_op =
        dyn_cast_or_null<ReshapeOp>(reshape_other_out.getDefiningOp());
    if (!reshape_other_op || !reshape_other_op->hasOneUse())
      return failure();
    auto matmul_other_op =
        dyn_cast_or_null<MatMulOp>(reshape_other_op.getInput().getDefiningOp());
    if (!matmul_other_op || !matmul_other_op->hasOneUse())
      return failure();

    // check params
    auto order = *module::getI64Array(op.getOrder());
    std::vector<int64_t> order_0213{0, 2, 1, 3};
    if (order != order_0213)
      return failure();
    auto permute_outshape = module::getShape(op.getOutput());
    int64_t batch, head_n, hw, head_sz;
    batch = permute_outshape[0];
    head_n = permute_outshape[1];
    hw = permute_outshape[2];
    head_sz = permute_outshape[3];
    auto reshape_outshape = module::getShape(reshape_op.getOutput());
    if (reshape_outshape.size() != 4 || reshape_outshape[0] != batch * head_n ||
        hw != reshape_outshape[1] * reshape_outshape[2] ||
        head_sz != reshape_outshape[3])
      return failure();
    int64_t h = reshape_outshape[1];
    int64_t w = reshape_outshape[2];
    auto matmul_h_output = matmul_h_op.getOutput();
    auto matmul_h_shape = module::getShape(matmul_h_output);
    if (h != matmul_h_shape[2])
      return failure();
    auto permute_before_w_order =
        *module::getI64Array(permute_before_w_op.getOrder());
    if (permute_before_w_order != order_0213)
      return failure();
    auto matmul_w_output = matmul_w_op.getOutput();
    auto matmul_w_shape = module::getShape(matmul_w_output);
    if (w != matmul_w_shape[2])
      return failure();
    auto permute_after_w_order =
        *module::getI64Array(permute_after_w_op.getOrder());
    if (permute_after_w_order != order_0213)
      return failure();
    if (*module::getI64Array(unsqueeze_h_op.getAxes()) !=
        std::vector<int64_t>{4})
      return failure();
    if (*module::getI64Array(unsqueeze_w_op.getAxes()) !=
        std::vector<int64_t>{3})
      return failure();
    if (module::getShape(matmul_other_op.getOutput()).vec() !=
        std::vector<int64_t>{batch * head_n, hw, hw})
      return failure();
    if (module::getShape(reshape_other_op.getOutput()).vec() !=
        std::vector<int64_t>{batch * head_n, h, w, h, w})
      return failure();

    // rewrite
    // squeeze_out: 25x196x12x64
    // get rid of this permute op
    auto output = op.getInput();
    auto name = module::getName(output);
    op.getOutput().replaceAllUsesWith(output);
    rewriter.eraseOp(op);
    // h part
    // reshape_h: (25x14)x14x12x64
    rewriter.setInsertionPointAfterValue(output);
    auto reshape_h_loc =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_reshape_h"));
    std::vector<int64_t> reshape_h_shape{batch * h, w, head_n, head_sz};
    auto reshape_h_type =
        RankedTensorType::get(reshape_h_shape, module::getElementType(output));
    auto new_reshape_h_op = rewriter.create<ReshapeOp>(
        reshape_h_loc, reshape_h_type, ValueRange{output});
    // permute_h: (25x14)x12x14x64
    rewriter.setInsertionPointAfter(new_reshape_h_op);
    auto permute_h_loc =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_permute_h"));
    std::vector<int64_t> permute_h_shape{batch * h, head_n, w, head_sz};
    auto permute_h_type = RankedTensorType::get(
        permute_h_shape, module::getElementType(new_reshape_h_op.getOutput()));
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order_0213)));
    auto new_permute_h_op = rewriter.create<PermuteOp>(
        permute_h_loc, permute_h_type, ValueRange{new_reshape_h_op.getOutput()},
        attrs);
    // rewrite h_weight: 300x14x14x64 => 25x(12)x(14)x14x64 =>
    // 25x(14)x(12)x14x64 => (25x14)x12x14x64
    auto h_weight_op = matmul_h_op.getRight().getDefiningOp<WeightOp>();
    auto h_weight_data = h_weight_op.read<float>();
    auto h_weight_trans =
        std::make_shared<std::vector<float>>(h_weight_data->size(), 0);
    function_permute(h_weight_data->data(), h_weight_trans->data(),
                     {batch, head_n, h, w, head_sz}, {0, 2, 1, 3, 4});
    std::vector<int64_t> h_weight_new_shape{batch * h, head_n, w, head_sz};
    auto h_weight_type =
        RankedTensorType::get(h_weight_new_shape, rewriter.getF32Type());
    auto new_weight_h = WeightOp::create(matmul_h_op, "rewrited",
                                         *h_weight_trans, h_weight_type);
    matmul_h_op->setOperand(0, new_permute_h_op.getOutput());
    matmul_h_op->setOperand(1, new_weight_h);
    // matmul_h_out: (25x14)x12x14x14
    matmul_h_output.setType(
        UnrankedTensorType::get(module::getElementType(matmul_h_output)));
    matmul_h_output.setLoc(NameLoc::get(rewriter.getStringAttr(
        module::getName(matmul_h_output).str() + "_new")));
    matmul_h_op.shape_inference();
    auto k_h = module::getShape(matmul_h_op.getOutput()).back();
    // permute_h_after: (25x14)x14x12x14
    auto permute_h_after_inshape =
        module::getShape(matmul_h_op.getOutput()).vec();
    std::vector<int64_t> permute_h_after_shape{batch * h, w, head_n, k_h};
    auto permute_h_after_type = RankedTensorType::get(
        permute_h_after_shape, module::getElementType(matmul_h_op.getOutput()));
    auto permute_h_after_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(matmul_h_op.getOutput()).str() + "_permute_h"));
    rewriter.setInsertionPointAfter(matmul_h_op);
    auto permute_h_after_op =
        rewriter.create<PermuteOp>(permute_h_after_loc, permute_h_after_type,
                                   ValueRange{matmul_h_output}, attrs);
    // reshape_h_after: 25x196x12x14
    std::vector<int64_t> reshape_h_after_shape{batch, h * w, head_n, k_h};
    auto reshape_h_after_type = RankedTensorType::get(
        reshape_h_after_shape,
        module::getElementType(permute_h_after_op.getOutput()));
    auto reshape_h_after_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(matmul_h_op.getOutput()).str() + "_reshape_h"));
    rewriter.setInsertionPointAfter(permute_h_after_op);
    auto reshape_h_after_op =
        rewriter.create<ReshapeOp>(reshape_h_after_loc, reshape_h_after_type,
                                   ValueRange{permute_h_after_op.getOutput()});
    // unsqueeze_h: 25x196x12x14x1
    unsqueeze_h_op->setOperand(0, reshape_h_after_op.getOutput());
    auto unsqueeze_h_output = unsqueeze_h_op.getOutput();
    unsqueeze_h_output.setType(
        UnrankedTensorType::get(module::getElementType(unsqueeze_h_output)));
    unsqueeze_h_output.setLoc(NameLoc::get(rewriter.getStringAttr(
        module::getName(unsqueeze_h_output).str() + "_new")));
    unsqueeze_h_op.shape_inference();
    // w part
    // reshape_w: 25x14x(14x12)x64
    rewriter.setInsertionPointAfterValue(output);
    auto reshape_w_loc =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_reshape_w"));
    std::vector<int64_t> reshape_w_shape{batch, h, w * head_n, head_sz};
    auto reshape_w_type =
        RankedTensorType::get(reshape_w_shape, module::getElementType(output));
    auto new_reshape_w_op = rewriter.create<ReshapeOp>(
        reshape_w_loc, reshape_w_type, ValueRange{output});
    // permute_w: 25x(14x12)x14x64
    rewriter.setInsertionPointAfter(new_reshape_w_op);
    auto permute_w_loc =
        NameLoc::get(rewriter.getStringAttr(name.str() + "_permute_w"));
    std::vector<int64_t> permute_w_shape{batch, w * head_n, h, head_sz};
    auto permute_w_type = RankedTensorType::get(
        permute_w_shape, module::getElementType(new_reshape_w_op.getOutput()));
    auto new_permute_w_op = rewriter.create<PermuteOp>(
        permute_w_loc, permute_w_type, ValueRange{new_reshape_w_op.getOutput()},
        attrs);
    // rewrite w_weight: 300x14x14x64 => 25x(12)x(14)x[14]x64 =>
    // 25x(14)x(12)x14x64 => 25x(14x12)x14x64
    auto w_weight_op = matmul_w_op.getRight().getDefiningOp<WeightOp>();
    auto w_weight_data = w_weight_op.read<float>();
    auto w_weight_trans =
        std::make_shared<std::vector<float>>(w_weight_data->size(), 0);
    function_permute(w_weight_data->data(), w_weight_trans->data(),
                     {batch, head_n, h, w, head_sz}, {0, 2, 1, 3, 4});
    std::vector<int64_t> w_weight_new_shape{batch, w * head_n, h, head_sz};
    auto w_weight_type =
        RankedTensorType::get(w_weight_new_shape, rewriter.getF32Type());
    auto new_weight_w = WeightOp::create(matmul_w_op, "rewrited",
                                         *w_weight_trans, w_weight_type);
    matmul_w_op->setOperand(0, new_permute_w_op.getOutput());
    matmul_w_op->setOperand(1, new_weight_w);
    // matmul_w_out: 25x(14x12)x14x14
    matmul_w_output.setType(
        UnrankedTensorType::get(module::getElementType(matmul_w_output)));
    matmul_w_output.setLoc(NameLoc::get(rewriter.getStringAttr(
        module::getName(matmul_w_output).str() + "_new")));
    matmul_w_op.shape_inference();
    auto k_w = module::getShape(matmul_w_op.getOutput()).back();
    // permute_w_after: 25x14x(14x12)x14
    auto permute_w_after_inshape =
        module::getShape(matmul_w_op.getOutput()).vec();
    std::vector<int64_t> permute_w_after_shape{batch, h, w * head_n, k_w};
    auto permute_w_after_type = RankedTensorType::get(
        permute_w_after_shape, module::getElementType(matmul_w_op.getOutput()));
    auto permute_w_after_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(matmul_w_op.getOutput()).str() + "_permute_w"));
    rewriter.setInsertionPointAfter(matmul_w_op);
    auto permute_w_after_op =
        rewriter.create<PermuteOp>(permute_w_after_loc, permute_w_after_type,
                                   ValueRange{matmul_w_output}, attrs);
    // reshape_w_after: 25x196x12x14
    std::vector<int64_t> reshape_w_after_shape{batch, h * w, head_n, k_w};
    auto reshape_w_after_type = RankedTensorType::get(
        reshape_w_after_shape,
        module::getElementType(permute_w_after_op.getOutput()));
    auto reshape_w_after_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(matmul_w_op.getOutput()).str() + "_reshape_w"));
    rewriter.setInsertionPointAfter(permute_w_after_op);
    auto reshape_w_after_op =
        rewriter.create<ReshapeOp>(reshape_w_after_loc, reshape_w_after_type,
                                   ValueRange{permute_w_after_op.getOutput()});
    // unsqueeze_w: 25x196x12x1x14
    unsqueeze_w_op->setOperand(0, reshape_w_after_op.getOutput());
    auto unsqueeze_w_output = unsqueeze_w_op.getOutput();
    unsqueeze_w_output.setType(
        UnrankedTensorType::get(module::getElementType(unsqueeze_w_output)));
    unsqueeze_w_output.setLoc(NameLoc::get(rewriter.getStringAttr(
        module::getName(unsqueeze_w_output).str() + "_new")));
    unsqueeze_w_op.shape_inference();
    // add part
    // add_hw: unsqueeze_h + unsqueeze_w => 25x196x12x14x14
    auto add_hw_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(add_h_op.getOutput()).str() + "_add_hw"));
    auto add_hw_type = RankedTensorType::get(
        {batch, h * w, head_n, k_h, k_w},
        module::getElementType(unsqueeze_h_op.getOutput()));
    rewriter.setInsertionPointAfter(unsqueeze_w_op);
    auto add_hw_op = rewriter.create<AddOp>(
        add_hw_loc, add_hw_type,
        ValueRange{unsqueeze_h_op.getOutput(), unsqueeze_w_op.getOutput()});
    // reshape_add_hw: 25x196x12x196
    auto reshape_add_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(add_h_op.getOutput()).str() + "_reshape_add_hw"));
    auto reshape_add_type =
        RankedTensorType::get({batch, h * w, head_n, k_h * k_w},
                              module::getElementType(add_hw_op.getOutput()));
    rewriter.setInsertionPointAfter(add_hw_op);
    auto reshape_add_op = rewriter.create<ReshapeOp>(
        reshape_add_loc, reshape_add_type, ValueRange{add_hw_op.getOutput()});
    // permute_add_hw: 25x12x196x196
    auto permute_add_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(add_h_op.getOutput()).str() + "_permute_add_hw"));
    auto permute_add_type = RankedTensorType::get(
        {batch, head_n, h * w, k_h * k_w},
        module::getElementType(reshape_add_op.getOutput()));
    rewriter.setInsertionPointAfter(reshape_add_op);
    auto permute_add_op = rewriter.create<PermuteOp>(
        permute_add_loc, permute_add_type,
        ValueRange{reshape_add_op.getOutput()}, attrs);
    // reshape_permute_add_hw: 300x196x196
    auto reshape_permute_add_loc = NameLoc::get(
        rewriter.getStringAttr(module::getName(add_h_op.getOutput()).str() +
                               "_reshape_permute_add_hw"));
    auto reshape_permute_add_type = RankedTensorType::get(
        {batch * head_n, h * w, k_h * k_w},
        module::getElementType(permute_add_op.getOutput()));
    rewriter.setInsertionPointAfter(permute_add_op);
    auto reshape_permute_add_op = rewriter.create<ReshapeOp>(
        reshape_permute_add_loc, reshape_permute_add_type,
        ValueRange{permute_add_op.getOutput()});
    // matmul_other_out: 300x196x196
    // add_qk: reshape_permute_add_hw + matmul_other_out: 300x196x196
    auto add_qk_loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(add_w_op.getOutput()).str() + "_add_qk"));
    auto add_qk_type = RankedTensorType::get(
        {batch * head_n, h * w, k_h * k_w},
        module::getElementType(reshape_permute_add_op.getOutput()));
    rewriter.setInsertionPointAfter(reshape_permute_add_op);
    auto add_qk_op =
        rewriter.create<AddOp>(add_qk_loc, add_qk_type,
                               ValueRange{reshape_permute_add_op.getOutput(),
                                          matmul_other_op.getOutput()});
    add_w_op.getOutput().replaceAllUsesExcept(add_qk_op.getOutput(), add_qk_op);
    return success();
  }
};

void PermuteOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<TopPermuteToPixelShuffle, TopPermuteToReorg, Permute5dSplit,
                 PermuteFuse, TopPermuteToReshape, NonZeroPermutePattern,
                 TopDecomposedRelPosEmb>(context);
}
