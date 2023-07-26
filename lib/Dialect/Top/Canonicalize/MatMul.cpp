//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

using namespace tpu_mlir::top;

// MatMul + Add(weight) => MatMul
struct MatMulWithBias : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto filter = op.getRight();
    if (module::isWeight(filter) == false) {
      return failure();
    }
    if (module::isNone(op.getBias()) == false) {
      return failure();
    }
    if (op->hasOneUse() == false) {
      return failure();
    }
    auto user = *op->user_begin();
    auto add_op = dyn_cast<AddOp>(user);
    if (!add_op) {
      return failure();
    }
    if (add_op.getNumOperands() != 2) {
      return failure();
    }
    Value bias = nullptr;
    bool bias_is_weight = false;
    for (auto v : add_op.getOperands()) {
      if (module::isWeight(v)) {
        bias = v;
        bias_is_weight = true;
        break;
      }
    }
    if (!bias_is_weight) {
      return failure();
    }
    auto p = op.parseParam();
    if (p.batch > 1) {
      // TODO: not support batch matmul; need to support
      return failure();
    }
    if (module::getNumElements(bias) != p.N) {
      // TODO: maybe == 1 is OK
      return failure();
    }
    auto bias_op = bias.getDefiningOp();
    if (!bias_op->isBeforeInBlock(op)) {
      bias_op->moveBefore(op);
    }
    op->setOperand(2, bias);
    op->setLoc(add_op.getLoc());
    add_op.replaceAllUsesWith(op.getOperation());
    rewriter.eraseOp(add_op);
    return success();
  }
};

// merge n and c if c is small and n is large
struct OptMatMulSmallCdim : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto left = op.getInput();
    auto right = op.getRight();
    auto out = op.getResult();

    auto mul_op = dyn_cast<MulConstOp>(left.getDefiningOp());
    if (!(mul_op && mul_op->hasOneUse())) {
      return failure();
    }
    auto mul_in = mul_op.getInput();
    auto mul_shape = module::getShape(mul_in);
    if (!mul_in.hasOneUse() || mul_shape.size() != 4 ||
        (mul_shape[1] >= mul_shape[2])) {
      return failure();
    }

    // 1. add ReshapeOp before MulConst
    rewriter.setInsertionPoint(mul_op);
    std::vector<int64_t> new_lshape(
        {mul_shape[0] * mul_shape[1], mul_shape[2], mul_shape[3]});
    auto type0 = RankedTensorType::get(new_lshape, rewriter.getF32Type());
    std::string in_name = module::getName(mul_in).str();
    std::string reshape0_name = in_name + "_reshape_left";
    auto loc0 = NameLoc::get(rewriter.getStringAttr(reshape0_name));
    auto reshape0_op =
        rewriter.create<ReshapeOp>(loc0, type0, ValueRange{mul_in});
    // mul_op->setOperand(0, reshape0_op);
    left.setType(type0);
    mul_in.replaceAllUsesExcept(reshape0_op.getOutput(), reshape0_op);

    // 2. add ReshapeOp before Right_in
    rewriter.setInsertionPoint(reshape0_op);
    auto rshape = module::getShape(right);
    std::vector<int64_t> new_rshape(
        {rshape[0] * rshape[1], rshape[2], rshape[3]});
    auto type1 = RankedTensorType::get(new_rshape, rewriter.getF32Type());
    std::string right_in_name = module::getName(right).str();
    std::string reshape1_name = right_in_name + "_reshape_right";
    auto loc1 = NameLoc::get(rewriter.getStringAttr(reshape1_name));
    auto reshape1_op =
        rewriter.create<ReshapeOp>(loc1, type1, ValueRange{right});
    // op->setOperand(1, reshape1_op);
    right.replaceAllUsesExcept(reshape1_op.getOutput(), reshape1_op);

    // 3. add ReshapeOp after MatMul out
    rewriter.setInsertionPointAfterValue(out);
    auto oshape = module::getShape(out);
    std::vector<int64_t> new_oshape(
        {oshape[0] * oshape[1], oshape[2], oshape[3]});
    auto type2 = RankedTensorType::get(new_oshape, rewriter.getF32Type());
    std::string out_name = module::getName(right).str();
    std::string reshape2_name = out_name + "_reshape_matmul";
    auto loc2 = NameLoc::get(rewriter.getStringAttr(reshape2_name));
    auto reshape2_op =
        rewriter.create<ReshapeOp>(loc2, out.getType(), ValueRange{out});
    out.setType(type2);
    out.replaceAllUsesExcept(reshape2_op.getOutput(), reshape2_op);

    return success();
  }
};

// Add Reshape op after non-keepdims MatMul to make layergroup easier
struct NoKeepDimsAddReshape : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {

    if (op.getKeepDims()) {
      return failure();
    }

    // specail case for bert
    auto output = op.getResult();
    for (auto user : output.getUsers()) {
      if (auto packop = dyn_cast<top::PackOp>(user)) {
        return failure();
      }
    }

    // cache the output type and loc
    auto reshape_out = output.getType();
    auto out_loc = output.getLoc();

    // change the MatMul op into keepdims and recalculate the output shape
    op.setKeepDims(true);
    output.setType(UnrankedTensorType::get(module::getElementType(output)));
    output.setLoc(NameLoc::get(
        rewriter.getStringAttr(module::getName(output).str() + "_keepdims")));
    op.shape_inference();

    // add reshape op after Matmul
    rewriter.setInsertionPointAfter(op);
    auto reshape_op =
        rewriter.create<ReshapeOp>(out_loc, reshape_out, ValueRange{output});
    output.replaceAllUsesExcept(reshape_op.getOutput(), reshape_op);

    return success();
  }
};

// Matmul + Reshape + Permute0 + (Permute1) + (Reshape2) + n*(slice + squeeze) => Matmul + Reshape + n*(slice + squeeze + Permute2 + (Reshape3))
struct MatmulWithPermuteAndSplit : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op,
                                PatternRewriter &rewriter) const override {

    // check topo
    auto nextOp = *op->user_begin();
    auto reshape_after_matmul = dyn_cast_or_null<ReshapeOp>(nextOp);
    if (!reshape_after_matmul) {
      return failure();
    }
    if (!reshape_after_matmul.getOutput().hasOneUse()) {
      return failure();
    }
    auto permute0 = dyn_cast<PermuteOp>(
        *reshape_after_matmul.getOutput().getUsers().begin());
    if (!permute0) {
      return failure();
    }
    auto order0 = *module::getI64Array(permute0.getOrder());
    auto permute1 =
        dyn_cast<PermuteOp>(*permute0.getOutput().getUsers().begin());
    mlir::TypedValue<mlir::TensorType> permute_output;
    if (permute1) {
      permute_output = permute1.getOutput();
    } else {
      permute_output = permute0.getOutput();
    }

    mlir::TypedValue<mlir::TensorType> before_slice;
    auto reshape2 = dyn_cast<ReshapeOp>(*permute_output.getUsers().begin());
    std::vector<int64_t> reshape2_inshape;
    std::vector<int64_t> reshape2_outshape;
    if (reshape2) {
      if (!permute_output.hasOneUse())
        return failure();
      if (module::getShape(reshape2.getOutput()).size() + 1 != module::getShape(permute_output).size())
        return failure();
      before_slice = reshape2.getOutput();
      reshape2_inshape = module::getShape(reshape2.getInput()).vec();
      reshape2_outshape = module::getShape(reshape2.getOutput()).vec();
    } else {
      before_slice = permute_output;
    }
    std::vector<SliceOp> slice_vec;
    int slice_axis;
    for (auto user : before_slice.getUsers()) {
      auto slice = dyn_cast<SliceOp>(user);
      if (!slice) {
        return failure();
      } else {
        auto squeeze = dyn_cast<SqueezeOp>(
             *slice.getOutput().getUsers().begin());
        if (!squeeze) {
          return failure();
        }
      }
      auto slice_in_shape = module::getShape(slice.getInput());
      auto slice_out_shape = module::getShape(slice.getOutput());
      std::vector<std::pair<int, int>> diff;
      for (int i = 0; i < slice_in_shape.size(); ++i) {
        if (slice_in_shape[i] != slice_out_shape[i]) {
          diff.push_back(std::make_pair(i, slice_out_shape[i]));
        }
      }
      if (diff.size() > 1 || diff[0].second != 1) {
        return failure();
      }
      slice_axis = diff[0].first;
      slice_vec.push_back(slice);
    }
    int mixed_idx = 0;
    if (reshape2) {
      for(int i = 0; i < reshape2_outshape.size(); ++i) {
        if (reshape2_inshape[i] != reshape2_outshape[i]) {
          mixed_idx = i;
          break;
        }
      }
      if (mixed_idx < slice_axis)
        slice_axis ++;
    }

    // Check Param
    auto matmul_output_shape = module::getShape(op.getOutput());
    auto reshape_output_shape =
        module::getShape(reshape_after_matmul.getOutput());
    // check-1.1: 64x49x288 -> 64x49x3x3x32: 3x3 is from 288
    if ((matmul_output_shape.size() + 2 != reshape_output_shape.size() ||
         !std::equal(matmul_output_shape.begin(), matmul_output_shape.end() - 1,
                    reshape_output_shape.begin())) &&
    // check-1.2: 25x14x14x2304 -> 25x196x3x12x64
        (matmul_output_shape.size() + 1 != reshape_output_shape.size() ||
         !(matmul_output_shape.size() >= 3 && reshape_output_shape.size() > 2 &&
           matmul_output_shape[0] == reshape_output_shape[0] &&
           matmul_output_shape[1] * matmul_output_shape[2] == reshape_output_shape[1]))) {
      return failure();
    }
    if (reshape_output_shape[order0[slice_axis]] != slice_vec.size()) {
      return failure();
    }

    // check-2: trans_dim is the same as split_dim
    std::vector<int> order_final(order0.size());
    if (permute1) {
      auto order1 = *module::getI64Array(permute1.getOrder());
      for (int i = 0; i < order0.size(); ++i) {
        order_final[i] = order0[order1[i]];
      }
    } else {
      for (int i = 0; i < order0.size(); ++i) {
        order_final[i] = order0[i];
      }
    }

    if (order_final[slice_axis] == slice_axis) {
      return failure();
    }

    // rewrite
    int slice_num = slice_vec.size();
    auto num_dims = reshape_output_shape.size();
    int new_slice_axis = order_final[slice_axis];
    for (int i = 0; i < slice_num; i++) {
      auto slice_op = slice_vec[i];
      auto old_offset = module::getI64Array(slice_op.getOffsetAttr());
      auto old_ends = module::getI64Array(slice_op.getEndsAttr());
      std::vector<int64_t> new_offset(num_dims, 0);
      std::vector<int64_t> new_ends(num_dims, 0);
      std::vector<int64_t> new_steps(num_dims, 0);
      auto in_steps = module::getI64Array(slice_op.getSteps());
      if (reshape2) {
        old_offset->insert(old_offset->begin()+mixed_idx+1, 0);
        in_steps->insert(in_steps->begin()+mixed_idx+1, 1);
        old_ends->insert(old_ends->begin()+mixed_idx+1, reshape2_inshape[mixed_idx+1]);
        old_ends->at(mixed_idx) = reshape2_inshape[mixed_idx];
      }
      for (int j = 0; j < num_dims; j++) {
        new_offset[order_final[j]] = old_offset->at(j);
        new_ends[order_final[j]] = old_ends->at(j);
        new_steps[order_final[j]] = in_steps->at(j);
      }
      slice_op->setAttr("offset", rewriter.getI64ArrayAttr(new_offset));
      slice_op->setAttr("ends", rewriter.getI64ArrayAttr(new_ends));
      slice_op->setAttr("steps", rewriter.getI64ArrayAttr(new_steps));
      slice_op->setOperand(0, reshape_after_matmul.getOutput());
      auto slice_output = slice_op.getResult();
      slice_output.setType(
          UnrankedTensorType::get(module::getElementType(slice_output)));
      slice_output.setLoc(NameLoc::get(rewriter.getStringAttr(
          module::getName(slice_output).str() + "_new")));
      slice_op.shape_inference();
      auto squeeze_op =
          dyn_cast<SqueezeOp>(*slice_op.getOutput().getUsers().begin());
      squeeze_op->setAttr("axes", rewriter.getI64ArrayAttr(new_slice_axis));

      auto squeeze_out = squeeze_op.getOutput();
      auto squeeze_out_type = squeeze_out.getType();
      auto squeeze_out_shape = module::getShape(squeeze_out).vec();
      if (reshape2) {
        squeeze_out_shape = reshape2_inshape;
        squeeze_out_shape.erase(squeeze_out_shape.begin() + slice_axis);
        squeeze_out_type = RankedTensorType::get(squeeze_out_shape, module::getElementType(squeeze_out));
      }
      auto inv_order_size = squeeze_out_shape.size();
      // caculate permute order
      std::vector<int64_t> inv_order(inv_order_size);
      std::iota(inv_order.begin(), inv_order.end(), 0);
      auto permute_in_shape = reshape_output_shape.vec();
      permute_in_shape.erase(permute_in_shape.begin() + new_slice_axis);

      for (int64_t i = 0; i < inv_order_size; ++i) {
        for (int64_t j = 0; j < inv_order_size; ++j) {
          if (permute_in_shape[i] == squeeze_out_shape[j] &&
              inv_order[j] != i) {
            std::swap(inv_order[j], inv_order[i]);
            break;
          }
        }
      }
      squeeze_out.setType(
          UnrankedTensorType::get(module::getElementType(squeeze_out)));
      squeeze_out.setLoc(NameLoc::get(
          rewriter.getStringAttr(module::getName(squeeze_out).str() + "_new")));
      squeeze_op.shape_inference();

      std::vector<mlir::Operation *> users;
      for (auto user: squeeze_out.getUsers()) {
        users.emplace_back(user);
      }

      // don't worry, if the new permute is not reused, it will be folded.
      for(auto user : users ) {
        std::vector<NamedAttribute> attrs;
        attrs.push_back(
            rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(inv_order)));
        auto name = module::getName(squeeze_op->getResults()[0]);
        auto permute_loc =
            NameLoc::get(rewriter.getStringAttr(name.str() + "_permute_for_" + module::getName(user->getResult(0))));
        std::vector<Value> operands;
        operands.emplace_back(squeeze_out);
        // rewriter.setInsertionPointAfterValue(squeeze_out);
        rewriter.setInsertionPoint(user);
        auto new_permute_op = rewriter.create<PermuteOp>(
            permute_loc, squeeze_out_type, operands, attrs);

        // squeeze_op.getOutput().replaceAllUsesExcept(new_permute_op.getOutput(),
        //                                             new_permute_op);

        auto last_output = new_permute_op.getOutput();

        if (reshape2) {
          rewriter.setInsertionPoint(user);
          auto reshape3_outshape = reshape2_outshape;
          reshape3_outshape.erase(reshape3_outshape.begin() + slice_axis);
          auto reshape3_type = RankedTensorType::get(reshape3_outshape, module::getElementType(last_output));
          auto reshape3_loc = NameLoc::get(rewriter.getStringAttr(permute_loc.getName().str() + "_reshape3"));
          auto reshape3_op =
              rewriter.create<ReshapeOp>(reshape3_loc, reshape3_type, ValueRange{last_output});
          // new_permute_op.getOutput().replaceAllUsesExcept(reshape3_op.getOutput(), reshape3_op);
          last_output = reshape3_op.getOutput();
        }

        std::vector<Value> user_operands;
        for(auto opd : user->getOperands()) {
          if (opd == squeeze_out) {
            user_operands.emplace_back(last_output);
          } else {
            user_operands.emplace_back(opd);
          }
        }
        user->setOperands(user_operands);
      }
    }

    return success();
  }
};

void MatMulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<MatMulWithBias, NoKeepDimsAddReshape, MatmulWithPermuteAndSplit>(context);
}
