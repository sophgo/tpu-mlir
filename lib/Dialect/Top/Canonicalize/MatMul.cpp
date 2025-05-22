//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

// MatMul + Add(weight) => MatMul
struct MatMulWithBias : public OpRewriterPatternEx<MatMulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MatMulWithBias(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MatMulOp>(context, "MatMulWithBias") {}

  LogicalResult matchAndRewriteImpl(MatMulOp op,
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
struct OptMatMulSmallCdim : public OpRewriterPatternEx<MatMulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  OptMatMulSmallCdim(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MatMulOp>(context, "OptMatMulSmallCdim") {}

  LogicalResult matchAndRewriteImpl(MatMulOp op,
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
struct NoKeepDimsAddReshape : public OpRewriterPatternEx<MatMulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  NoKeepDimsAddReshape(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MatMulOp>(context, "NoKeepDimsAddReshape") {}

  LogicalResult matchAndRewriteImpl(MatMulOp op,
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
        rewriter.getStringAttr(module::getName(output).str() + "_r_keepdims")));
    op.shape_inference();

    // add reshape op after Matmul
    rewriter.setInsertionPointAfter(op);
    auto reshape_op =
        rewriter.create<ReshapeOp>(out_loc, reshape_out, ValueRange{output});
    output.replaceAllUsesExcept(reshape_op.getOutput(), reshape_op);

    return success();
  }
};

// Matmul + Reshape + Permute0 + (Permute1) + (Reshape2) + n*(slice + squeeze)
// => Matmul + Reshape + n*(slice + squeeze + Permute2 + (Reshape3))
struct MatmulWithPermuteAndSplit : public OpRewriterPatternEx<MatMulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MatmulWithPermuteAndSplit(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MatMulOp>(context, "MatmulWithPermuteAndSplit") {}

  LogicalResult matchAndRewriteImpl(MatMulOp op,
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
      if (module::getShape(reshape2.getOutput()).size() + 1 !=
          module::getShape(permute_output).size())
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
        auto squeeze =
            dyn_cast<SqueezeOp>(*slice.getOutput().getUsers().begin());
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
      for (int i = 0; i < reshape2_outshape.size(); ++i) {
        if (reshape2_inshape[i] != reshape2_outshape[i]) {
          mixed_idx = i;
          break;
        }
      }
      if (mixed_idx < slice_axis)
        slice_axis++;
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
           matmul_output_shape[1] * matmul_output_shape[2] ==
               reshape_output_shape[1]))) {
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
        old_offset->insert(old_offset->begin() + mixed_idx + 1, 0);
        in_steps->insert(in_steps->begin() + mixed_idx + 1, 1);
        old_ends->insert(old_ends->begin() + mixed_idx + 1,
                         reshape2_inshape[mixed_idx + 1]);
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
      slice_op->setAttr("hasparamConvert_axes",
                        rewriter.getI64ArrayAttr(order_final[slice_axis]));
      slice_op->setOperand(0, reshape_after_matmul.getOutput());
      auto slice_output = slice_op.getResult();
      slice_output.setType(
          UnrankedTensorType::get(module::getElementType(slice_output)));
      slice_output.setLoc(NameLoc::get(rewriter.getStringAttr(
          module::getName(slice_output).str() + "_r_new")));
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
        squeeze_out_type = RankedTensorType::get(
            squeeze_out_shape, module::getElementType(squeeze_out));
      }
      auto inv_order_size = squeeze_out_shape.size();
      // caculate permute order
      std::vector<int64_t> inv_order(inv_order_size);
      std::iota(inv_order.begin(), inv_order.end(), 0);
      auto permute_in_shape = reshape_output_shape.vec();
      permute_in_shape.erase(permute_in_shape.begin() + new_slice_axis);

      for (int64_t i = 0; i < inv_order_size; ++i) {
        if (order_final[i + 1] < order_final[0]) {
          inv_order[i] = order_final[i + 1];
        } else {
          inv_order[i] = order_final[i + 1] - 1;
        }
      }
      squeeze_out.setType(
          UnrankedTensorType::get(module::getElementType(squeeze_out)));
      squeeze_out.setLoc(NameLoc::get(rewriter.getStringAttr(
          module::getName(squeeze_out).str() + "_r_new")));
      squeeze_op.shape_inference();

      std::vector<mlir::Operation *> users;
      for (auto user : squeeze_out.getUsers()) {
        users.emplace_back(user);
      }

      // don't worry, if the new permute is not reused, it will be folded.
      for (auto user : users) {
        std::vector<NamedAttribute> attrs;
        attrs.push_back(rewriter.getNamedAttr(
            "order", rewriter.getI64ArrayAttr(inv_order)));
        auto name = module::getName(squeeze_op->getResults()[0]);
        auto permute_loc = NameLoc::get(
            rewriter.getStringAttr(name.str() + "_r_permute_for_" +
                                   module::getName(user->getResult(0))));
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
          auto reshape3_type = RankedTensorType::get(
              reshape3_outshape, module::getElementType(last_output));
          auto reshape3_loc = NameLoc::get(rewriter.getStringAttr(
              permute_loc.getName().str() + "_r_reshape3"));
          auto reshape3_op = rewriter.create<ReshapeOp>(
              reshape3_loc, reshape3_type, ValueRange{last_output});
          // new_permute_op.getOutput().replaceAllUsesExcept(reshape3_op.getOutput(),
          // reshape3_op);
          last_output = reshape3_op.getOutput();
        }

        std::vector<Value> user_operands;
        for (auto opd : user->getOperands()) {
          if (opd == squeeze_out) {
            user_operands.emplace_back(last_output);
          } else {
            user_operands.emplace_back(opd);
          }
        }
        user->setOperands(user_operands);
      }
    }
    if (reshape2) {
      rewriter.eraseOp(reshape2);
    }
    if (permute1) {
      rewriter.eraseOp(permute1);
    }
    rewriter.eraseOp(permute0);
    return success();
  }
};

Value get_weight(Value weight, int begin, int end, int axis, Type to_type,
                 std::string suffix) {
  auto op = weight.getDefiningOp();
  if (module::isWeight(weight)) {
    return dyn_cast<top::WeightOp>(op).split(begin, end, axis, to_type, suffix);
  } else {
    return top::NoneOp(op);
  }
}

// matmul + slice => matmul + matmul
struct MatMulWithSlice : public OpRewriterPatternEx<MatMulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MatMulWithSlice(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MatMulOp>(context, "MatMulWithSlice") {}

  LogicalResult matchAndRewriteImpl(MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    auto filter = op.getRight();
    if (module::isWeight(filter) == false) {
      return failure();
    }
    if (op->hasOneUse() == true) {
      return failure();
    }
    auto shape = module::getShape(op.getOutput());
    for (auto user = op->getUsers().begin(); user != op->getUsers().end();
         ++user) {
      if (auto slice_op = dyn_cast<SliceOp>(*user)) {
        if (!module::isNone(slice_op.getOffsetT()) ||
            !module::isNone(slice_op.getEndsT()) ||
            !module::isNone(slice_op.getStepsT())) {
          return failure();
        }
        auto s_shape = module::getShape(slice_op.getOutput());
        if (shape[0] != s_shape[0] || shape[1] != s_shape[1] ||
            s_shape[s_shape.size() - 1] < 128) {
          return failure();
        }
        auto offset = module::getI64Array(slice_op.getOffsetAttr());
        if (offset->at(0) != 0 || offset->at(1) != 0) {
          return failure();
        }
        auto step = module::getI64Array(slice_op.getStepsAttr());
        if (step->at(0) != 1 || step->at(1) != 1 || step->at(2) != 1) {
          return failure();
        }
      } else {
        return failure();
      }
    }

    std::string out_name = module::getName(op.getOutput()).data();
    for (auto user = op->getUsers().begin(); user != op->getUsers().end();
         ++user) {
      auto slice_op = dyn_cast<SliceOp>(*user);
      std::vector<Value> operands;
      operands.push_back(op.getInput());
      auto s_shape = module::getShape(slice_op.getOutput());
      auto offset =
          module::getI64Array(slice_op.getOffsetAttr())->at(s_shape.size() - 1);
      auto size = module::getShape(slice_op.getOutput())[s_shape.size() - 1];
      auto input_size = shape[s_shape.size() - 1];
      if (offset < 0) {
        offset += input_size;
      }
      operands.push_back(get_weight(op.getRight(), offset, offset + size, -1,
                                    rewriter.getF32Type(),
                                    "_offset" + std::to_string(offset)));
      operands.push_back(get_weight(op.getBias(), offset, offset + size, -1,
                                    rewriter.getF32Type(),
                                    "_offset" + std::to_string(offset)));
      rewriter.replaceOpWithNewOp<top::MatMulOp>(
          slice_op, slice_op.getOutput().getType(), operands, op->getAttrs());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// Permute0 + Matmul + Permute1 => Matmul
struct PermuteMatMulPermute : public OpRewriterPatternEx<MatMulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  PermuteMatMulPermute(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MatMulOp>(context, "PermuteMatMulPermute") {}

  LogicalResult matchAndRewriteImpl(MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    auto in0 = op.getInput();
    auto in1 = op.getRight();
    if (!module::isWeight(in0) || module::isWeight(in1)) {
      return failure();
    }
    if (!op->hasOneUse()) {
      return failure();
    }
    auto pre_op = in1.getDefiningOp();
    auto next_op = *op->user_begin();
    if (!isa<top::PermuteOp>(pre_op) || !isa<top::PermuteOp>(next_op)) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }
    auto permute0 = cast<top::PermuteOp>(pre_op);
    auto permute1 = cast<top::PermuteOp>(next_op);
    auto order0 = module::getI64Array(permute0.getOrder());
    auto order1 = module::getI64Array(permute1.getOrder());
    // order = {0, 2, 1}
    if (order0->size() != 3 || order1->size() != 3) {
      return failure();
    }
    if (!(order0->at(0) == 0 && order0->at(1) == 2 && order0->at(2) == 1 &&
          order1->at(0) == 0 && order1->at(1) == 2 && order1->at(2) == 1)) {
      return failure();
    }
    // create new weight
    auto weight = cast<top::WeightOp>(in0.getDefiningOp());
    auto ori_data = weight.read_as_float();
    std::vector<float> to_data(ori_data->size(), 0);
    std::vector<int64_t> shape = module::getShape(in0);
    std::vector<int64_t> order{1, 0};
    function_permute(ori_data->data(), to_data.data(), shape, order);
    auto new_weight = WeightOp::create_float(
        weight, "reordered", to_data, {1, shape[1], shape[0]}, storage_type);
    // update matmul output shape
    op.getOutput().setType(permute1.getOutput().getType());

    int64_t idx = 0;
    Value out = permute1.getOutput();
    auto next2_op = *permute1->user_begin();
    for (; idx < next2_op->getNumOperands(); idx++) {
      Value opd = next2_op->getOperand(idx);
      if (out == opd) {
        break;
      }
    }
    next2_op->setOperand(idx, op.getOutput());
    auto loc = module::getLoc(permute1.getOutput());

    op.setOperand(0, permute0.getInput());
    op.setOperand(1, new_weight);
    rewriter.eraseOp(weight);
    rewriter.eraseOp(permute0);
    rewriter.eraseOp(permute1);
    module::setLoc(op.getOutput(), loc);
    return success();
  }
};

// in eva02 and gpt2, there is matmul and slice to 3 branches. split matmul to
// three branch for better cali. now, this one is for eva2 only matmul + reshape
// + 3*(slice + squeeze) => 3*(matmul + reshape + squeeze) or 3*(matmul+reshape)
// when squeeze the slice axes
// 2024.01.30 pp_ocr_rec match here, too. :(
struct SplitMatMulEva2 : public OpRewriterPatternEx<MatMulOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  SplitMatMulEva2(mlir::MLIRContext *context)
      : OpRewriterPatternEx<MatMulOp>(context, "SplitMatMulEva2") {}

  LogicalResult matchAndRewriteImpl(MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    if (!op.getOutput().hasOneUse())
      return failure();
    ReshapeOp reshape_op = NULL;

    for (auto out : op.getOutput().getUsers()) {
      if (isa<ReshapeOp>(out))
        reshape_op = dyn_cast_or_null<ReshapeOp>(out);
      else
        return failure();
    }
    if (reshape_op == NULL)
      return failure();
    int pos[3] = {0};
    SliceOp slice_op[3] = {NULL};
    SqueezeOp squeeze_ops[3] = {NULL};
    int i = 0;
    int squ_axes = -1;
    for (auto u : reshape_op.getOutput().getUsers()) {
      if (!isa<SliceOp>(u)) {
        if (isa<SplitOp>(u))
          continue;
        else
          return failure();
      }
      if (i > 3)
        return failure();
      slice_op[i] = dyn_cast_or_null<SliceOp>(u);
      if (slice_op[i] == NULL || !slice_op[i].getOutput().hasOneUse())
        return failure();
      for (auto out : slice_op[i].getOutput().getUsers()) {
        if (isa<SqueezeOp>(out))
          squeeze_ops[i] = dyn_cast_or_null<SqueezeOp>(out);
        else
          return failure();
      }
      if (squeeze_ops[i] == NULL)
        return failure();
      auto axes_ = module::getI64Array(squeeze_ops[i].getAxes());
      if (squ_axes < 0)
        squ_axes = axes_->at(0);
      else if (squ_axes != axes_->at(0))
        return failure();
      auto off = module::getI64Array(slice_op[i].getOffset());
      auto steps = module::getI64Array(slice_op[i].getSteps());
      auto ends = module::getI64Array(slice_op[i].getEnds());
      auto axes = module::getI64Array(slice_op[i].getAxes());
      if (axes->size() == 0) {
        if (steps->size() != off->size() || steps->size() != ends->size() ||
            steps->size() <= squ_axes)
          return failure();
        for (int idx = 0; idx < steps->size(); idx++) {
          if (steps->at(idx) != 1)
            return failure();
        }
        pos[i] = off->at(squ_axes);
      } else {
        return failure(); // don't know how it looks if axes is set
      }
      if (module::getShape(op->getOperands()[0]).size() <= squ_axes)
        return failure();
      i++;
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }

    if (i != 3 || slice_op[0] == NULL || slice_op[1] == NULL ||
        slice_op[2] == NULL)
      return failure();
    WeightOp weight_op = NULL;
    WeightOp bias_op = NULL;
    std::vector<int64_t> weight_shape;
    std::vector<int64_t> bias_shape;
    for (auto in : op.getOperands()) {
      if (isa<WeightOp>(in.getDefiningOp())) {
        if (weight_op == NULL) {
          weight_op = dyn_cast<WeightOp>(in.getDefiningOp());
          auto shape = in.getType().dyn_cast<TensorType>().getShape();
          if (shape.size() != 2)
            return failure();
          for (auto s : shape)
            weight_shape.push_back(s);
        } else if (bias_op == NULL) {
          bias_op = dyn_cast<WeightOp>(in.getDefiningOp());
          auto shape = in.getType().dyn_cast<TensorType>().getShape();
          for (auto s : shape)
            bias_shape.push_back(s);
        }
      }
    }
    if (weight_shape[1] != bias_shape[bias_shape.size() - 1] ||
        weight_shape[1] % 3 != 0)
      return failure();

    auto weight = weight_op.read_as_float();
    for (int i = 0; i < 3; i++) {
      auto w = std::make_shared<std::vector<float>>(weight_shape[0] *
                                                    weight_shape[1] / 3);
      for (int k = 0; k < weight_shape[0]; k++) {
        for (int m = 0; m < weight_shape[1] / 3; m++) {
          w->data()[k * weight_shape[1] / 3 + m] = weight->at(
              pos[i] * weight_shape[1] / 3 + k * weight_shape[1] + m);
        }
      }
      auto b = std::make_shared<std::vector<float>>(weight_shape[1] / 3);
      if (bias_op != NULL) {
        auto bias = bias_op.read_as_float();
        for (int m = 0; m < weight_shape[1] / 3; m++) {
          b->data()[m] = bias->at(pos[i] * weight_shape[1] / 3 + m);
        }
      }

      std::vector<NamedAttribute> attrs;
      for (auto &attr : op->getAttrs()) {
        attrs.push_back(attr);
      }
      std::vector<NamedAttribute> rsattrs;
      std::vector<int64_t> mmout_shape;
      int idx = 0;
      size_t shape_last = 1;
      for (auto s : module::getShape(slice_op[i].getOutput())) {
        if (idx < squ_axes)
          mmout_shape.push_back(s);
        else if (idx >= squ_axes) // skip squ_axes
          shape_last *= s;
        idx++;
      }
      mmout_shape.push_back(shape_last);

      auto mmout_type =
          RankedTensorType::get(mmout_shape, rewriter.getF32Type());
      auto w_op = WeightOp::create_float(
          op,
          module::getName(slice_op[i].getOutput()).str() + "_w" +
              std::to_string(i),
          *w, {weight_shape[0], weight_shape[1] / 3}, storage_type);
      if (bias_op != NULL) {
        auto bias_type =
            RankedTensorType::get({weight_shape[1] / 3}, rewriter.getF32Type());
        auto b_op = WeightOp::create<float>(
            op,
            module::getName(slice_op[i].getOutput()).str() + "_b" +
                std::to_string(i),
            *b, bias_type);
        rewriter.replaceOpWithNewOp<MatMulOp>(
            slice_op[i], mmout_type, ValueRange{op.getInput(), w_op, b_op},
            attrs);
        rewriter.replaceOpWithNewOp<ReshapeOp>(
            squeeze_ops[i], squeeze_ops[i].getOutput().getType(),
            ValueRange{squeeze_ops[i].getInput()}, rsattrs);
      } else {
        rewriter.replaceOpWithNewOp<MatMulOp>(
            slice_op[i], mmout_type,
            ValueRange{op.getInput(), w_op, module::getNoneOp(op)}, attrs);
        rewriter.replaceOpWithNewOp<ReshapeOp>(
            squeeze_ops[i], squeeze_ops[i].getOutput().getType(),
            ValueRange{squeeze_ops[i].getInput()}, rsattrs);
      }
    }
    return success();
  }
};

// test case: [5, 128]x[7, 128, 64] => [7, 5, 64]
struct MatMulReverse : public OpRewriterPatternEx<MatMulOp> {
  MatMulReverse(MLIRContext *context, PatternBenefit benefit = 9)
      : OpRewriterPatternEx<MatMulOp>(context, "MatMulReverse", benefit) {}
  using OpRewriterPatternEx::OpRewriterPatternEx;
  LogicalResult matchAndRewriteImpl(MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    auto in0 = op.getInput();
    auto in1 = op.getRight();
    auto out = op.getOutput();
    if (!module::isActive(in0) || !module::isActive(in1) ||
        !module::isNone(op.getBias())) {
      return failure();
    }
    auto in0_shape = module::getShape(in0);
    auto in1_shape = module::getShape(in1);
    int in1_ndims = in1_shape.size();
    auto out_type = out.getType();
    auto out_loc = out.getLoc();
    if (!(in0_shape.size() == 2 && in1_ndims > 2)) {
      return failure();
    }
    if (op.getLeftTranspose() || op.getRightTranspose() ||
        op.getOutputTranspose()) {
      return failure();
    }
    rewriter.setInsertionPointAfterValue(in1);
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    operands.emplace_back(in1);
    std::vector<int64_t> order1(in1_ndims);
    std::iota(order1.begin(), order1.end(), 0);
    std::swap(order1[in1_ndims - 1], order1[in1_ndims - 2]);
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order1)));
    std::vector<int64_t> new_in1_shape = in1_shape;
    std::swap(new_in1_shape[in1_ndims - 1], new_in1_shape[in1_ndims - 2]);
    auto outType = module::getTypeLike(out, new_in1_shape);
    std::string in1_name = module::getName(in1).str();
    auto newLoc =
        NameLoc::get(rewriter.getStringAttr(in1_name + "_r_permute1"));
    // auto newLoc = module::getLocLike(in1, "order");
    auto p1_op =
        rewriter.create<top::PermuteOp>(newLoc, outType, operands, attrs);
    op.setOperand(0, p1_op.getOutput());
    op.setOperand(1, in0);
    op.setRightTranspose(true);
    std::vector<int64_t> new_out_shape = module::getShape(out);
    std::swap(new_out_shape[in1_ndims - 1], new_out_shape[in1_ndims - 2]);
    module::setShape(out, new_out_shape);
    std::string out_name = module::getName(out).str();
    auto fix_loc = NameLoc::get(rewriter.getStringAttr(out_name + "_r_order"));
    // auto fix_loc = module::getLocLike(out, "order");
    out.setLoc(fix_loc);
    rewriter.setInsertionPointAfterValue(out);
    auto p2_op = rewriter.create<top::PermuteOp>(out_loc, out_type,
                                                 ValueRange{out}, attrs);
    rewriter.replaceAllUsesExcept(out, p2_op.getOutput(), p2_op);
    return success();
  }
};

void MatMulOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  // clang-format off
  results.insert<
                MatMulWithBias,
                NoKeepDimsAddReshape,
                MatmulWithPermuteAndSplit,
                MatMulWithSlice,
                PermuteMatMulPermute,
                MatMulReverse
                >(context);
  // clang-format on
}
