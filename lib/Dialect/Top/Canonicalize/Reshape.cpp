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
#include "tpu_mlir/Support/Patterns.h"

using namespace tpu_mlir::top;

// reshape (in == out)
struct TopFuseReshape2 : public OpRewriterPatternEx<ReshapeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopFuseReshape2(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ReshapeOp>(context, "TopFuseReshape2") {}

  LogicalResult matchAndRewriteImpl(ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    auto shape0 = module::getShape(op.getOutput());
    auto shape1 = module::getShape(op.getInput());
    if (shape0 != shape1) {
      return failure();
    }
    op.getOutput().replaceAllUsesWith(op.getInput());
    rewriter.eraseOp(op);
    return success();
  }
};

// add + reshape + add + reshape
struct TopFuseReshape3 : public OpRewriterPatternEx<ReshapeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  TopFuseReshape3(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ReshapeOp>(context, "TopFuseReshape3") {}

  LogicalResult matchAndRewriteImpl(ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    auto add_op = dyn_cast<AddOp>(in.getDefiningOp());
    if (!(add_op && add_op->hasOneUse() && in.hasOneUse())) {
      return failure();
    }
    if (add_op.getNumOperands() != 2) {
      return failure();
    }
    auto a_in = add_op.getInputs()[0];
    auto b_in = add_op.getInputs()[1];
    if (!module::isWeight(b_in)) {
      return failure();
    }
    if (!a_in.hasOneUse()) {
      return failure();
    }
    if (!b_in.hasOneUse()) {
      return failure();
    }
    if (!isa<ReshapeOp>(a_in.getDefiningOp())) {
      return failure();
    }
    std::vector<int64_t> shape0 = module::getShape(op.getInput());
    std::vector<int64_t> shape1 = module::getShape(op.getOutput());
    if (shape0.size() != 1 + shape1.size()) {
      return failure();
    }
    if (!std::equal(shape0.begin() + 1, shape0.end(), shape1.begin())) {
      return failure();
    }
    if (shape0[0] != 1) {
      return failure();
    }
    std::vector<int64_t> a_shape = module::getShape(a_in);
    std::vector<int64_t> b_shape = module::getShape(b_in);
    if (a_shape[0] != 1 || b_shape[0] != 1) {
      return failure();
    }
    a_shape.erase(a_shape.begin());
    b_shape.erase(b_shape.begin());
    shape0.erase(shape0.begin());
    auto b_type = RankedTensorType::get(b_shape, module::getElementType(b_in));
    b_in.setType(b_type);
    auto a_type = RankedTensorType::get(a_shape, module::getElementType(a_in));
    a_in.setType(a_type);
    auto in_type = RankedTensorType::get(shape0, module::getElementType(in));
    in.setType(in_type);
    return success();
  }
};

// reshape<(0,ng,-1)> + instance_norm -> group_norm<ng> + reshape
struct ReshapeInstanceNormPattern : public OpRewriterPatternEx<ReshapeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  ReshapeInstanceNormPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ReshapeOp>(context, "ReshapeInstanceNormPattern") {}

  LogicalResult matchAndRewriteImpl(ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    // check param
    auto output = op.getOutput();
    if (!output.hasOneUse())
      return failure();
    auto next_op_ = *output.user_begin();
    if (!isa<InstanceNormOp>(next_op_))
      return failure();
    auto next_op = dyn_cast<InstanceNormOp>(next_op_);
    auto ishape = module::getShape(op.getInput());
    auto oshape = module::getShape(op.getOutput());
    if (ishape[0] != oshape[0])
      return failure();
    if (ishape[1] < oshape[1])
      return failure();
    // rewrite now !
    const auto num_groups = oshape[1];
    auto input = op.getInput();
    std::vector<NamedAttribute> attrs;
    next_op->setAttr("num_groups", rewriter.getI64IntegerAttr(num_groups));
    for (auto &attr : next_op->getAttrs()) {
      attrs.push_back(attr);
    }

    auto gn_out_type =
        RankedTensorType::get(ishape, module::getElementType(input));
    auto loc = NameLoc::get(rewriter.getStringAttr(
        module::getName(next_op.getResult()).str() + "_r_GroupNorm"));

    auto groupnorm_filter_broadcast =
        [](const std::vector<int64_t> &filter_shape, const void *filter_orig,
           void *filter_trans, const int num_groups) -> void {
      int c = filter_shape[1];
      for (int kc = 0; kc < c; kc++) {
        *((float *)filter_trans + kc) =
            *((float *)filter_orig + kc / (c / num_groups));
      }
    };
    // broadcast for weight and bias
    std::vector<Value> gn_opds = {input, next_op->getOperand(1),
                                  next_op->getOperand(2)};
    int new_filter_count = ishape[1];
    auto out_type = module::getStorageType(next_op.getOutput());
    if (ishape.size() <= 2)
      return failure();
    std::vector<int64_t> new_filter_shape(ishape.size(), 1);
    new_filter_shape[1] = ishape[1];
    if (!module::isNone(next_op.getWeight())) {
      auto filterOp = next_op.getWeight().getDefiningOp<top::WeightOp>();
      auto weight_data = filterOp.read_as_byte();
      auto new_weight =
          std::make_shared<std::vector<float>>(new_filter_count, 0);
      groupnorm_filter_broadcast(new_filter_shape, weight_data->data(),
                                 new_weight->data(), num_groups);
      auto new_w_type = RankedTensorType::get(new_filter_shape, out_type);
      auto new_weightOp =
          top::WeightOp::create(next_op.getWeight().getDefiningOp(), "reorderd",
                                *new_weight, new_w_type);
      gn_opds[1] = new_weightOp;
    }

    if (!module::isNone(next_op.getBias())) {
      auto biasOp = next_op.getBias().getDefiningOp<top::WeightOp>();
      auto bias_data = biasOp.read_as_byte();
      auto new_bias = std::make_shared<std::vector<float>>(new_filter_count, 0);
      groupnorm_filter_broadcast(new_filter_shape, bias_data->data(),
                                 new_bias->data(), num_groups);
      auto new_b_type = RankedTensorType::get(new_filter_shape, out_type);
      auto new_biasOp = top::WeightOp::create(
          next_op.getBias().getDefiningOp(), "reorderd", *new_bias, new_b_type);
      gn_opds[2] = new_biasOp;
    }

    Value insertpoint = next_op.getOutput();
    rewriter.setInsertionPointAfterValue(insertpoint);

    auto gn_op = rewriter.create<GroupNormOp>(loc, gn_out_type, gn_opds, attrs);
    rewriter.replaceOp(op, gn_op);
    auto gn_output = gn_op.getOutput();
    rewriter.setInsertionPointAfterValue(gn_output);
    auto new_reshape_out_type = next_op.getResult().getType();
    rewriter.replaceOpWithNewOp<ReshapeOp>(next_op, new_reshape_out_type,
                                           gn_output,
                                           std::vector<NamedAttribute>());
    return success();
  }
};

// merge some tanh and power(x,3) comprised gelu to gelu, first found in pytorch
// traced gpt2
struct MergeGeluPattern : public OpRewriterPatternEx<ReshapeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MergeGeluPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ReshapeOp>(context, "MergeGeluPattern") {}

  LogicalResult matchAndRewriteImpl(ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    MulOp mul_op = dyn_cast<MulOp>(op.getInput().getDefiningOp());
    if (mul_op == NULL || !mul_op.getOutput().hasOneUse())
      return failure();

    MulConstOp mulconst_op = NULL;
    AddConstOp addconst_op = NULL;

    for (auto in : mul_op.getInputs()) {
      if (isa<MulConstOp>(in.getDefiningOp()))
        mulconst_op = dyn_cast<MulConstOp>(in.getDefiningOp());
      else if (isa<AddConstOp>(in.getDefiningOp()))
        addconst_op = dyn_cast<AddConstOp>(in.getDefiningOp());
      else
        return failure();
    }
    if (!mulconst_op.getOutput().hasOneUse() ||
        !addconst_op.getOutput().hasOneUse())
      return failure();

    TanhOp tanh_op = NULL;
    if (!isa<TanhOp>(addconst_op.getInput().getDefiningOp()))
      return failure();
    else
      tanh_op = dyn_cast<TanhOp>(addconst_op.getInput().getDefiningOp());
    if (!tanh_op.getOutput().hasOneUse())
      return failure();

    MulConstOp mulconst_op1 = NULL;
    AddOp add_op = NULL;
    if (!isa<MulConstOp>(tanh_op.getInput().getDefiningOp()))
      return failure();
    else
      mulconst_op1 = dyn_cast<MulConstOp>(tanh_op.getInput().getDefiningOp());
    if (!isa<AddOp>(mulconst_op1.getInput().getDefiningOp()))
      return failure();
    else
      add_op = dyn_cast<AddOp>(mulconst_op1.getInput().getDefiningOp());
    if (!mulconst_op1.getOutput().hasOneUse() ||
        !add_op.getOutput().hasOneUse())
      return failure();

    MulConstOp mulconst_op2 = NULL;
    PowOp pow_op = NULL;
    ReshapeOp reshape_op = NULL;
    for (auto in : add_op.getInputs()) {
      if (isa<MulConstOp>(in.getDefiningOp()))
        mulconst_op2 = dyn_cast<MulConstOp>(in.getDefiningOp());
      else if (isa<ReshapeOp>(in.getDefiningOp()))
        reshape_op = dyn_cast<ReshapeOp>(in.getDefiningOp());
      else
        return failure();
    }
    if (!isa<PowOp>(mulconst_op2.getInput().getDefiningOp()))
      return failure();
    else
      pow_op = dyn_cast<PowOp>(mulconst_op2.getInput().getDefiningOp());
    if (!mulconst_op2.getOutput().hasOneUse() ||
        !pow_op.getOutput().hasOneUse())
      return failure();

    if (pow_op.getInput().getDefiningOp() != reshape_op ||
        mulconst_op.getInput().getDefiningOp() != reshape_op)
      return failure();
    int cnt = 0;
    int all = 0;
    for (auto out : reshape_op.getOutput().getUsers()) {
      if (out == mulconst_op || out == pow_op || out == add_op)
        cnt++;
      all++;
    }
    if (cnt != 3 || all != 3)
      return failure();
    if (pow_op.getExponent().convertToDouble() != 3.0 ||
        fabs(mulconst_op2.getConstVal().convertToDouble() -
             0.044714998453855515) > 1e-4 ||
        addconst_op.getConstVal().convertToDouble() != 1.0 ||
        fabs(mulconst_op1.getConstVal().convertToDouble() -
             0.79788458347320556) > 1e-4 ||
        fabs(mulconst_op.getConstVal().convertToDouble() - 0.5) > 1e-4)
      return failure();
    rewriter.replaceOpWithNewOp<GELUOp>(op, op.getResult().getType(),
                                        ValueRange{reshape_op.getInput()});
    return success();
  }
};

/**
 * Reshape(tensor<1xf32>) -> tensor<f32>
 * Unsqueeze(tensor<f32>) -> tensor<1xf32>
 **/
struct InValidReshapeMergePattern : public OpRewriterPatternEx<ReshapeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  InValidReshapeMergePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ReshapeOp>(context, "InValidReshapeMergePattern") {}

  LogicalResult matchAndRewriteImpl(ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    // check topo
    // have one user only
    if (!op.getOutput().hasOneUse()) {
      return failure();
    }

    auto shape = module::getShape(op.getOutput());
    if (shape.size() > 0) {
      return failure();
    }

    // move trait
    for (auto nextOp : op.getResult().getUsers()) {
      // ops that support permute move should also support reshape move
      if (auto unsqueezeOp = dyn_cast<top::UnsqueezeOp>(nextOp)) {
        unsqueezeOp.replaceAllUsesWith(op.getInput());
        rewriter.eraseOp(nextOp);
      } else {
        return failure();
      }
      // if (!isa<top::UnsqueezeOp>(nextOp)) {
      // }
    }

    rewriter.eraseOp(op);

    return success();
  }
};

//  Do:
//     A                                          A + Reshape
//       + Add + Reshape + LayerNorm/Matmul -->>              + Add +
//       LayerNorm/Matmul
//     B                                          B + Reshape
// swint
struct TopAddReshapeSwap : public OpRewriterPatternEx<ReshapeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopAddReshapeSwap(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ReshapeOp>(context, "TopAddReshapeSwap") {}

  LogicalResult matchAndRewriteImpl(ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }
    auto in = op.getInput();
    auto add_op = dyn_cast<AddOp>(in.getDefiningOp());
    if (!add_op || !add_op.getOutput().hasOneUse()) {
      return failure();
    }
    bool add_can_merge = false;
    for (auto nextOp : op.getOutput().getUsers()) {
      if (isa<LayerNormOp, MatMulOp>(nextOp)) {
        add_can_merge = true;
        break;
      }
    }
    if (!add_can_merge) {
      return failure();
    }
    auto add_out_elements = module::getNumElements(add_op.getOutput());
    for (auto add_in : add_op.getInputs()) {
      if (add_in.hasOneUse() &&
          isa<LayerNormOp, MatMulOp>(add_in.getDefiningOp())) {
        return failure();
      }
      auto add_in_elements = module::getNumElements(add_in);
      if (add_in_elements != add_out_elements) {
        return failure();
      }
    }

    // fix bug for qwen
    auto in_shape = module::getShape(op.getInput());
    auto out_shape = module::getShape(op.getOutput());
    if (in_shape.size() == 4 && out_shape.size() == 4 && in_shape[0] == 1 &&
        in_shape[1] == 1 && out_shape[0] == 1 && out_shape[2] == 1) {
      return failure();
    }

    std::vector<Value> operands;
    for (auto add_in : add_op.getInputs()) {
      std::string in_name = module::getName(add_in).str() + "_r_reshape";
      auto loc = NameLoc::get(rewriter.getStringAttr(in_name));
      rewriter.setInsertionPoint(add_op);
      auto reshape_op = rewriter.create<ReshapeOp>(
          loc, op.getOutput().getType(), ValueRange{add_in});
      operands.push_back(reshape_op);
    }
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getType(), operands,
                                       add_op->getAttrs());
    rewriter.eraseOp(add_op);
    return success();
  }
};

// Reshape + Reshape -->> Reshape
// swint
struct TopReshapeFuse : public OpRewriterPatternEx<ReshapeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  TopReshapeFuse(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ReshapeOp>(context, "TopReshapeFuse") {}

  LogicalResult matchAndRewriteImpl(ReshapeOp op,
                                    PatternRewriter &rewriter) const override {

    auto in = op.getInput();
    auto pre_op = dyn_cast<ReshapeOp>(in.getDefiningOp());
    if (!pre_op) {
      return failure();
    }
    if (!in.hasOneUse()) {
      return failure();
    }
    op.setOperand(0, pre_op.getInput());
    rewriter.eraseOp(pre_op);
    return success();
  }
};

//           OP            Reshape + Op
// Reshape + Reshape  -->> Reshape + Reshape
struct TopReshapeFuse2 : public OpRewriterPatternEx<ReshapeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  TopReshapeFuse2(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ReshapeOp>(context, "TopReshapeFuse2") {}
  LogicalResult matchAndRewriteImpl(ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    auto in = op.getInput();
    auto pre_op = dyn_cast<ReshapeOp>(in.getDefiningOp());
    if (!pre_op) {
      return failure();
    }
    if (in.hasOneUse()) {
      return failure();
    }
    auto shape0 = module::getShape(op.getOutput());
    auto shape1 = module::getShape(pre_op.getInput());
    if (shape0 != shape1) {
      return failure();
    }
    int32_t index = 0;
    for (auto &use : llvm::make_early_inc_range(pre_op.getResult().getUses())) {
      auto *nextOp = use.getOwner();
      std::string in_name = module::getName(in).str() + "_reshape_fuse_" +
                            std::to_string(index++);
      auto loc = NameLoc::get(rewriter.getStringAttr(in_name));
      rewriter.setInsertionPoint(pre_op);
      auto reshape_op = rewriter.create<ReshapeOp>(
          loc, pre_op.getOutput().getType(), ValueRange{pre_op.getInput()});
      nextOp->setOperand(use.getOperandNumber(), reshape_op.getOutput());
    }
    rewriter.eraseOp(pre_op);
    return success();
  }
};

// Reshape + Permute + Reshape -->> Reshape + Depth2Space
// Reshape + Permute + Reshape -->> Depth2Space
// Swint PatchMerging
struct Reshape4Depth2SpacePattern : public OpRewriterPatternEx<ReshapeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  /*
  Case 1:
    Reshape:  input like: 1,3136,96
              output like: 1,28,2,28,2,96

    Permute:  input like 1,28,2,28,2,96
              order [0,1,3,4,2,5]
              output like:  1,28,28,2,2,96

    Reshape:  input like: 1,28,28,2,2,96
              output like: 1,28,28,384
    For Case 1:
      attrs = {
        is_CRD        f
        is_inversed   t
        in_is_NCHW    f
        out_is_NCHW   f
        swap_cr       t
      }
  */
  /*
  Case 2:
    Reshape:  input like: 1,56,56,96
              output like: 1,28,2,28,2,96

    Permute:  input like 1,28,2,28,2,96
              order [0,1,3,4,2,5]
              output like:  1,28,28,2,2,96

    Reshape:  input like: 1,28,28,2,2,96
              output like: 1,28,28,384
    For Case 2:
    => Depth2Space

  */
  /*
  Case 3:
    Reshape:  input like: 1,56,56,96
              output like: 1,8,7,8,7,96

    Permute:  input like: 1,8,7,8,7,96
              order [0, 1, 3, 2, 4, 53]
              output like: 1,8,8,7,7,96

    Reshape:  input like: 1,8,8,7,7,96
              output like: 64,49,96
  This is a Reshape-Permute-Reshape => Depth2Space+Reshape Structure. Not in
  this pattern.
  */
  LogicalResult matchAndRewriteImpl(ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    // check output and next op
    auto output = op.getOutput();
    if (!output.hasOneUse())
      return failure();
    auto output_shape = module::getShape(output);

    // check input and the previous permute op
    auto in = op.getInput();
    // permute: input shape like 1,28,2,28,2,96 and order [0,1,3,4,2,5]
    auto permute_op = dyn_cast<PermuteOp>(in.getDefiningOp());
    if (!permute_op) {
      return failure();
    }
    if (!in.hasOneUse()) {
      return failure();
    }
    // check permute order
    auto order = module::getI64Array(permute_op.getOrder());
    std::vector<int64_t> valid_permute_order1 = {0, 1, 3, 4, 2, 5};
    if (order->size() != valid_permute_order1.size() ||
        !std::equal(order->begin(), order->end(),
                    valid_permute_order1.begin())) {
      return failure();
    }

    // check the input of previous permute op and the previous reshape op
    auto permute_op_in = permute_op.getInput();
    auto pre_reshape_op = dyn_cast<ReshapeOp>(permute_op_in.getDefiningOp());
    if (!pre_reshape_op) {
      return failure();
    }
    // if the input of permute_op has other used, then those ops cannot be fused
    // into depth2space
    if (!permute_op_in.hasOneUse()) {
      return failure();
    }
    auto permute_op_in_shape = module::getShape(permute_op_in);

    // define the new output shape for the first reshape
    std::vector<int64_t> new_out_shape_pre_reshape = {
        permute_op_in_shape[0], permute_op_in_shape[1] * permute_op_in_shape[2],
        permute_op_in_shape[3] * permute_op_in_shape[4],
        permute_op_in_shape[5]};

    // check the input shape of previous reshape
    // shape of the input of pre reshape, usually like [1,3196,96], or
    // [1,56,56,96]
    auto input_shape_pre = module::getShape(pre_reshape_op.getInput());
    // shape of the output of pre reshape, usually like [1, 28, 2, 28, 2, 96];
    // and final output will be [1,28,28,96*4]
    auto output_shape_pre = module::getShape(pre_reshape_op.getOutput());

    int64_t block_h = output_shape_pre[2];
    int64_t block_w = output_shape_pre[4];

    // if the output channel is input channel * bh * bw
    if (input_shape_pre[input_shape_pre.size() - 1] * block_h * block_w !=
        output_shape[output_shape.size() - 1]) {
      return failure();
    }

    // setup attributes for Depth2SpaceOp
    std::vector<NamedAttribute> attrs;
    // First Reshape ouput, like 1*28*2*28*2*96. Permute to 1*28*28*2*2*96. Thus
    // CRD
    attrs.push_back(
        rewriter.getNamedAttr("is_CRD", rewriter.getBoolAttr(false)));
    attrs.push_back(
        rewriter.getNamedAttr("is_inversed", rewriter.getBoolAttr(true)));
    attrs.push_back(
        rewriter.getNamedAttr("in_is_NCHW", rewriter.getBoolAttr(false)));
    attrs.push_back(
        rewriter.getNamedAttr("out_is_NCHW", rewriter.getBoolAttr(false)));
    attrs.push_back(
        rewriter.getNamedAttr("block_h", rewriter.getI64IntegerAttr(block_h)));
    attrs.push_back(
        rewriter.getNamedAttr("block_w", rewriter.getI64IntegerAttr(block_w)));
    // if swap_cr is false, npz comparison can not pass.
    attrs.push_back(
        rewriter.getNamedAttr("swap_cr", rewriter.getBoolAttr(true)));
    auto depth2space_output_type = op.getResult().getType();

    // if the input of previous reshape is [1, 3196, 96]
    if (input_shape_pre.size() + 1 == output_shape.size()) {
      // update the output shape of first reshape
      module::setShape(pre_reshape_op->getResult(0), new_out_shape_pre_reshape);
      pre_reshape_op.setShapeAttr(
          rewriter.getI64ArrayAttr(new_out_shape_pre_reshape));
      rewriter.replaceOpWithNewOp<Depth2SpaceOp>(
          op, depth2space_output_type, ValueRange{pre_reshape_op.getOutput()},
          attrs);
      rewriter.eraseOp(permute_op);

      return success();
    }
    // if the input is [1,56,56,96]
    else if (input_shape_pre.size() == output_shape.size()) {
      rewriter.replaceOpWithNewOp<Depth2SpaceOp>(
          op, depth2space_output_type, ValueRange{pre_reshape_op.getInput()},
          attrs);
      rewriter.eraseOp(permute_op);
      rewriter.eraseOp(pre_reshape_op);

      return success();
    }

    return failure();
  }
};

struct PermuteBeforeGridSampler : public OpRewriterPatternEx<ReshapeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  LogicalResult matchAndRewriteImpl(ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    auto before_op = op.getInput().getDefiningOp();
    if (module::getShape(op.getInput()).size() == 5) {
      if (!isa<top::PermuteOp>(before_op))
        return failure();
    } else if (module::getShape(op.getInput()).size() == 4) {
      if (!isa<top::MatMulOp>(before_op))
        return failure();
    } else {
      return failure();
    }
    SmallVector<Operation *> grid_samplers;
    auto reshape_users = op->getResult(0).getUsers();
    for (auto user : reshape_users) {
      if (isa<top::AvgPoolOp>(user)) {
        for (auto avgpool_user : user->getResult(0).getUsers()) {
          if (auto grid_sampler = dyn_cast<top::GridSamplerOp>(avgpool_user)) {
            std::vector<int64_t> grid_sampler_shape =
                module::getShape(grid_sampler);
            if (grid_sampler_shape.size() != 4 || grid_sampler_shape[2] != 1)
              return failure();
            grid_samplers.push_back(grid_sampler);
          } else {
            return failure();
          }
        }
      } else if (auto grid_sampler = dyn_cast<top::GridSamplerOp>(user)) {
        std::vector<int64_t> grid_sampler_shape =
            module::getShape(grid_sampler);
        if (grid_sampler_shape.size() != 4 || grid_sampler_shape[2] != 1)
          return failure();
        grid_samplers.push_back(grid_sampler);
      } else {
        return failure();
      }
    }

    auto reshape_loc = module::getName(op.getOutput()).str();
    if (auto permute_op = dyn_cast<top::PermuteOp>(before_op)) {
      auto in_shape = module::getShape(permute_op.getInput());
      std::vector<int64_t> reshape_shape{in_shape[0], in_shape[1], in_shape[2],
                                         in_shape[3] * in_shape[4]};
      auto reshape_type = RankedTensorType::get(
          reshape_shape, module::getElementType(permute_op.getInput()));
      op->setOperand(0, permute_op.getInput());
      op->setLoc(
          NameLoc::get(rewriter.getStringAttr(reshape_loc + "_permute")));
      op.getResult().setType(reshape_type);
      op->setAttr("shape", rewriter.getI64ArrayAttr(reshape_shape));
    } else if (auto matmul_op = dyn_cast<top::MatMulOp>(before_op)) {
      std::vector<int64_t> permute_order{0, 3, 1, 2};
      auto input_value = matmul_op.getOutput();
      auto input_type = input_value.getType().cast<RankedTensorType>();
      auto input_shape = input_type.getShape();
      SmallVector<int64_t> output_shape;
      for (auto dim : permute_order) {
        output_shape.push_back(input_shape[dim]);
      }
      auto output_type =
          RankedTensorType::get(output_shape, input_type.getElementType());
      auto matmul_loc = module::getName(matmul_op.getOutput()).str();
      auto permute_op = rewriter.create<top::PermuteOp>(
          NameLoc::get(rewriter.getStringAttr(matmul_loc + "_permute")),
          output_type, input_value,
          rewriter.getNamedAttr("order",
                                rewriter.getI64ArrayAttr(permute_order)));
      auto in_shape = module::getShape(permute_op.getOutput());
      std::vector<int64_t> reshape_shape{in_shape[0], 1, in_shape[1],
                                         in_shape[2] * in_shape[3]};
      auto reshape_type = RankedTensorType::get(
          reshape_shape, module::getElementType(permute_op.getOutput()));
      op->setOperand(0, permute_op.getOutput());
      op.getResult().setType(reshape_type);
      op->setAttr("shape", rewriter.getI64ArrayAttr(reshape_shape));
      op->setLoc(
          NameLoc::get(rewriter.getStringAttr(reshape_loc + "_permute")));
    }

    for (auto user : reshape_users) {
      if (auto avg_pool = dyn_cast<top::AvgPoolOp>(user)) {
        auto input_type =
            avg_pool.getInput().getType().cast<RankedTensorType>();
        auto input_shape = input_type.getShape();

        SmallVector<int64_t> old_kernel;
        for (auto dim : avg_pool.getKernelShape().getValue()) {
          old_kernel.push_back(dim.cast<IntegerAttr>().getInt());
        }

        SmallVector<int64_t> kernel = {old_kernel[1], old_kernel[0]};
        SmallVector<int64_t> strides = kernel;

        int64_t in_height = input_shape[2];
        int64_t kernel_height = kernel[0];
        int64_t strides_height = strides[0];
        int64_t out_height = (in_height - kernel_height) / strides_height + 1;

        SmallVector<int64_t> output_shape{input_shape[0], input_shape[1],
                                          out_height, input_shape[3]};

        auto output_type =
            RankedTensorType::get(output_shape, input_type.getElementType());
        auto avg_pool_loc = module::getName(avg_pool.getOutput()).str();
        rewriter.updateRootInPlace(avg_pool, [&]() {
          avg_pool->setLoc(
              NameLoc::get(rewriter.getStringAttr(avg_pool_loc + "_hwtrans")));
          avg_pool->setAttr("kernel_shape", rewriter.getI64ArrayAttr(kernel));
          avg_pool->setAttr("strides", rewriter.getI64ArrayAttr(strides));
          avg_pool.getResult().setType(output_type);
        });
      }
    }
    std::vector<int64_t> permute_order{3, 1, 0, 2};
    for (auto user : grid_samplers) {
      auto grid_sampler = dyn_cast<top::GridSamplerOp>(user);
      rewriter.setInsertionPoint(grid_sampler);
      auto grid_sampler_loc = module::getName(grid_sampler.getOutput()).str();

      auto input_value = grid_sampler.getOperand(0);
      auto input_type = input_value.getType().cast<RankedTensorType>();
      auto input_shape = input_type.getShape();
      SmallVector<int64_t> output_shape;
      for (auto dim : permute_order) {
        output_shape.push_back(input_shape[dim]);
      }
      auto output_type =
          RankedTensorType::get(output_shape, input_type.getElementType());
      auto permute_op = rewriter.create<top::PermuteOp>(
          NameLoc::get(rewriter.getStringAttr(grid_sampler_loc + "_permute")),
          output_type, input_value,
          rewriter.getNamedAttr("order",
                                rewriter.getI64ArrayAttr(permute_order)));
      grid_sampler->setOperand(0, permute_op.getOutput());
    }
    return success();
  }
};

struct ReshapeBeforeGridSampler : public OpRewriterPatternEx<ReshapeOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  LogicalResult matchAndRewriteImpl(ReshapeOp op,
                                    PatternRewriter &rewriter) const override {
    std::vector<int64_t> reshape_shape = module::getShape(op);
    if (reshape_shape.size() != 4 ||
        reshape_shape[1] * reshape_shape[2] * reshape_shape[3] != 1) {
      return failure();
    }

    auto onlyUserOf = [](Value v) -> Operation * {
      Operation *u = nullptr;
      for (auto &use : v.getUses()) {
        if (u && use.getOwner() != u)
          return nullptr;
        u = use.getOwner();
      }
      return u;
    };
    SmallVector<Operation *, 4> mulconst_ops;
    SmallVector<Operation *, 4> sub_ops;
    SmallVector<Operation *, 4> add_ops;

    for (auto user : op->getUsers()) {
      Operation *cur_op = user;
      if (isa<top::AddOp>(cur_op)) {
        add_ops.push_back(cur_op);
      } else if (isa<top::SubOp>(cur_op)) {
        sub_ops.push_back(cur_op);
        auto next = onlyUserOf(cur_op->getResult(0));
        if (!next)
          return failure();
        if (auto add_op = dyn_cast_or_null<top::AddOp>(next)) {
          add_ops.push_back(next);
        } else {
          return failure();
        }
      } else if (isa<top::MulConstOp>(cur_op)) {
        mulconst_ops.push_back(cur_op);
        auto next_op = cur_op->getResult(0);
        for (auto next : next_op.getUsers()) {
          if (auto add_op = dyn_cast<top::AddOp>(next)) {
            add_ops.push_back(next);
          } else if (auto sub_op = dyn_cast<top::SubOp>(next)) {
            sub_ops.push_back(next);
            Operation *next2 = onlyUserOf(sub_op->getResult(0));
            if (!next2)
              return failure();
            if (auto add_op = dyn_cast<top::AddOp>(next2)) {
              add_ops.push_back(next2);
            } else {
              return failure();
            }
          } else {
            return failure();
          }
        }
      } else {
        return failure();
      }
    }
    for (auto add_op : add_ops) {
      Operation *next = onlyUserOf(add_op->getResult(0));
      if (!next)
        return failure();
      auto concat_head = dyn_cast<top::ConcatOp>(next);
      if (!concat_head)
        return failure();

      top::ConcatOp concat_tail = nullptr;
      if (!concat_head->hasOneUse())
        return failure();
      auto slice_op = dyn_cast<top::SplitOp>(*concat_head->getUsers().begin());
      if (!slice_op)
        return failure();
      for (auto user : slice_op->getUsers()) {
        if (auto concat_op = dyn_cast<top::ConcatOp>(user)) {
          if (concat_tail && concat_op != concat_tail)
            return failure();
          concat_tail = concat_op;
          continue;
        }
        auto mulconst_op = dyn_cast<top::MulOp>(user);
        if (mulconst_op) {
          // mulconst_op->dump();
          auto after_mulconst = *mulconst_op->getUsers().begin();
          // after_mulconst->dump();
          auto div_op = dyn_cast<top::MulConstOp>(after_mulconst);
          if (!div_op)
            return failure();
          auto after_div = *div_op->getUsers().begin();
          // after_div->dump();
          auto addconst_op = dyn_cast<top::AddConstOp>(after_div);
          if (!addconst_op)
            return failure();
          auto after_addconst = *addconst_op->getUsers().begin();
          // after_addconst->dump();
          auto concat_op = dyn_cast<top::ConcatOp>(after_addconst);
          if (!concat_op)
            return failure();
          if (concat_tail && concat_op != concat_tail)
            return failure();
          concat_tail = concat_op;
          continue;
        }
        return failure();
      }
      if (!concat_tail)
        return failure();
      auto after_concat_tail = onlyUserOf(concat_tail.getResult());
      if (!after_concat_tail)
        return failure();
      if (!isa<top::GridSamplerOp>(after_concat_tail))
        return failure();
    }

    std::vector<int64_t> new_shape = {reshape_shape[3], reshape_shape[1],
                                      reshape_shape[2], reshape_shape[0]};
    std::vector<int64_t> permute_order = {3, 1, 2, 0};
    auto elem_type = module::getElementType(op.getInput());
    auto reshape_loc = module::getName(op.getOutput()).str();
    auto new_reshape = rewriter.create<top::ReshapeOp>(
        NameLoc::get(rewriter.getStringAttr(reshape_loc + "_nwtrans")),
        RankedTensorType::get(new_shape, elem_type), op.getInput(),
        rewriter.getNamedAttr("shape", rewriter.getI64ArrayAttr(new_shape)));
    auto new_reshape_out = new_reshape.getOutput();
    rewriter.replaceOp(op, new_reshape_out);

    for (auto opd : mulconst_ops) {
      auto mulconst_op = dyn_cast<top::MulConstOp>(opd);
      auto in_type =
          mulconst_op.getInput().getType().dyn_cast<RankedTensorType>();
      auto elem_type = in_type.getElementType();
      rewriter.setInsertionPoint(mulconst_op);
      auto mulconst_loc = module::getName(mulconst_op.getOutput()).str();
      auto const_val = mulconst_op.getConstVal().convertToDouble();
      auto new_mulconst = rewriter.create<top::MulConstOp>(
          NameLoc::get(rewriter.getStringAttr(mulconst_loc + "_nwtrans")),
          RankedTensorType::get(new_shape, elem_type), mulconst_op.getInput(),
          rewriter.getNamedAttr("const_val",
                                rewriter.getF64FloatAttr(const_val)));
      mulconst_op.replaceAllUsesWith(new_mulconst.getResult());
      rewriter.eraseOp(mulconst_op);
    }

    for (auto opd : sub_ops) {
      auto sub_op = dyn_cast<top::SubOp>(opd);
      auto sub_in_1 = sub_op.getInputs()[0];
      auto sub_in_2 = sub_op.getInputs()[1];
      bool w_is_in1 = dyn_cast<top::WeightOp>(sub_in_1.getDefiningOp());
      Value input = w_is_in1 ? sub_in_2 : sub_in_1;
      Value weight = w_is_in1 ? sub_in_1 : sub_in_2;
      auto weight_op = dyn_cast<top::WeightOp>(weight.getDefiningOp());
      auto weight_shape = module::getShape(weight_op);
      if (weight_shape[1] * weight_shape[2] * weight_shape[3] != 1) {
        return failure();
      }
      auto weight_f32 = weight_op.read<float>();
      auto reorder = std::make_shared<std::vector<float>>(weight_shape[0], 0);
      function_permute(weight_f32->data(), reorder->data(), weight_shape,
                       {3, 1, 2, 0});
      auto elem_type = module::getStorageType(weight_op);
      auto out_type = RankedTensorType::get(new_shape, elem_type);
      auto reorder_wieght = top::WeightOp::create<float>(weight_op, "reordered",
                                                         *reorder, out_type);
      SmallVector<Value, 2> operands;
      operands.resize(2);
      if (w_is_in1) {
        operands[0] = reorder_wieght;
        operands[1] = input;
      } else {
        operands[0] = input;
        operands[1] = reorder_wieght;
      }
      auto sub_loc = module::getName(sub_op.getOutput()).str();
      rewriter.setInsertionPoint(sub_op);
      auto new_sub = rewriter.create<top::SubOp>(
          NameLoc::get(rewriter.getStringAttr(sub_loc + "_nwtrans")),
          RankedTensorType::get(new_shape, module::getStorageType(input)),
          operands);
      sub_op.replaceAllUsesWith(new_sub.getResult());
      rewriter.eraseOp(sub_op);
    }

    for (auto add : add_ops) {
      auto add_op = dyn_cast<top::AddOp>(add);
      auto add_in_1 = add_op.getInputs()[0];
      auto add_in_2 = add_op.getInputs()[1];
      Value input = nullptr;
      Value weight = nullptr;
      if (dyn_cast<top::WeightOp>(add_in_1.getDefiningOp())) {
        weight = add_in_1;
        input = add_in_2;
      } else {
        input = add_in_1;
        weight = add_in_2;
      }
      auto add_shape = module::getShape(add_op);
      auto new_add_shape = {add_shape[3], add_shape[1], add_shape[2],
                            add_shape[0]};
      std::vector<Value> operands;
      operands.push_back(input);
      operands.push_back(weight);
      auto add_loc = module::getName(add_op.getOutput()).str();
      rewriter.setInsertionPoint(add_op);
      auto new_add = rewriter.create<top::AddOp>(
          NameLoc::get(rewriter.getStringAttr(add_loc + "_nwtrans")),
          RankedTensorType::get(new_add_shape, module::getStorageType(input)),
          operands);
      auto old_out = add_op.getOutput();
      auto old_type = old_out.getType().dyn_cast<RankedTensorType>();
      rewriter.setInsertionPointAfter(new_add);
      std::vector<int64_t> permute_order{3, 1, 2, 0};
      auto permute_op = rewriter.create<top::PermuteOp>(
          NameLoc::get(rewriter.getStringAttr(add_loc)), old_type,
          new_add.getResult(),
          rewriter.getNamedAttr("order",
                                rewriter.getI64ArrayAttr(permute_order)));

      old_out.replaceAllUsesWith(permute_op.getOutput());
      rewriter.eraseOp(add_op);
    }
    return success();
  }
};

void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.insert<Reshape4Depth2SpacePattern,
                 patterns::FuseRepeatPattern<top::ReshapeOp>, TopFuseReshape2,
                 TopFuseReshape3, ReshapeInstanceNormPattern, MergeGeluPattern,
                 InValidReshapeMergePattern, TopAddReshapeSwap, TopReshapeFuse,
                 TopReshapeFuse2, PermuteBeforeGridSampler,
                 ReshapeBeforeGridSampler>(context);
}

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  Operation *op = *this;
  auto weightOp = op->getOperand(0).getDefiningOp<top::WeightOp>();
  if (weightOp) {
    if (!weightOp.getOperation()->hasOneUse()) {
      return {};
    }
    auto data = weightOp.read_as_float();
    auto shape = module::getShape(this->getOutput());
    auto storage_type = module::getStorageType(getOutput());
    auto new_op = WeightOp::create_float(weightOp.getOperation(), "folder",
                                         *data, shape.vec(), storage_type);
    return new_op;
  } else {
    return {};
  }
}
