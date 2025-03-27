//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;
using namespace tpu_mlir::trait;

static LogicalResult find_slice_order(ConcatOp concat_op, int ex_dims,
                                      bool is_NCHW, std::vector<int64_t> &order,
                                      Value &from, int64_t &bh, int64_t &bw) {
  // idx of n,c,h,w
  int ci, hi, wi;
  if (is_NCHW) {
    ci = 1 + ex_dims;
    hi = 2 + ex_dims;
    wi = 3 + ex_dims;
  } else {
    hi = 1 + ex_dims;
    wi = 2 + ex_dims;
    ci = 3 + ex_dims;
  }
  bh = 0, bw = 0;
  const auto &inputs = concat_op.getInputs();
  int num_inputs = inputs.size();
  order.clear();
  order.reserve(num_inputs);
  for (int i = 0; i < num_inputs; i++) {
    auto in_op = inputs[i].getDefiningOp();
    if (!isa<SliceOp>(in_op)) {
      return failure();
    }
    auto slice_op = dyn_cast<SliceOp>(in_op);
    auto offset = module::getI64Array(slice_op.getOffset());
    for (int e = 0; e <= ex_dims; e++) {
      if (offset->at(e) != 0) {
        return failure();
      }
    }
    if (offset->at(ci) != 0) {
      return failure();
    }
    auto steps = module::getI64Array(slice_op.getSteps());
    for (int e = 0; e <= ex_dims; e++) {
      if (steps->at(e) != 1) {
        return failure();
      }
    }
    if (steps->at(ci) != 1) {
      return failure();
    }
    if (i == 0) {
      bh = steps->at(hi);
      bw = steps->at(wi);
      if (bh * bw != num_inputs) {
        return failure();
      }
      from = slice_op.getInput();
    } else {
      if (bh != steps->at(hi) || bw != steps->at(wi)) {
        return failure();
      }
      if (from != slice_op.getInput()) {
        return failure();
      }
    }
    int64_t begin_order = offset->at(hi) * bw + offset->at(wi);
    if (std::find(order.begin(), order.end(), begin_order) != order.end()) {
      return failure();
    }
    order.push_back(begin_order);
  }
  return success();
}

static void replaceOpWithDepth2SpaceOp(PatternRewriter &rewriter, ConcatOp &op,
                                       ValueRange &&args, int64_t bh,
                                       int64_t bw, bool is_CRD,
                                       bool is_inversed, bool in_is_NCHW,
                                       bool out_is_NCHW, bool swap_cr) {
  std::vector<NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("block_h", rewriter.getI64IntegerAttr(bh)));
  attrs.push_back(
      rewriter.getNamedAttr("block_w", rewriter.getI64IntegerAttr(bw)));
  attrs.push_back(
      rewriter.getNamedAttr("is_CRD", rewriter.getBoolAttr(is_CRD)));
  attrs.push_back(
      rewriter.getNamedAttr("is_inversed", rewriter.getBoolAttr(is_inversed)));
  attrs.push_back(
      rewriter.getNamedAttr("in_is_NCHW", rewriter.getBoolAttr(in_is_NCHW)));
  attrs.push_back(
      rewriter.getNamedAttr("out_is_NCHW", rewriter.getBoolAttr(out_is_NCHW)));
  attrs.push_back(
      rewriter.getNamedAttr("swap_cr", rewriter.getBoolAttr(swap_cr)));
  rewriter.replaceOpWithNewOp<Depth2SpaceOp>(op, op.getResult().getType(), args,
                                             attrs);
}

// concat slices to Depth2Space.
// test by yolov5s
struct ConcatToDepth2SpacePattern : public OpRewriterPatternEx<ConcatOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  ConcatToDepth2SpacePattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ConcatOp>(context, "ConcatToDepth2SpacePattern") {}

  LogicalResult matchAndRewriteImpl(ConcatOp concat_op,
                                    PatternRewriter &rewriter) const override {
    if (concat_op.getDoRelu()) {
      return failure();
    }
    auto shape = module::getShape(concat_op.getOutput());
    int num_dims = shape.size();
    if (num_dims < 3) {
      return failure();
    }
    int ex_dims = num_dims - 4;
    if (concat_op.getAxis() - ex_dims != 1) {
      return failure();
    }
    if (concat_op->hasOneUse() == false) {
      return failure();
    }
    auto use_op = *concat_op->user_begin();
    if (!isa<ConvOp>(use_op)) {
      return failure();
    }
    Value from;
    int64_t bh;
    int64_t bw;
    std::vector<int64_t> order;
    auto ret = find_slice_order(concat_op, ex_dims, true, order, from, bh, bw);
    if (ret.failed()) {
      return failure();
    }
    bool need_reorder = false;
    for (size_t i = 0; i < order.size(); ++i) {
      if (order[i] != i && false == need_reorder) {
        need_reorder = true;
      }
    }
    if (need_reorder) {
      if (ex_dims != 0) {
        return failure();
      }
      if (concat_op->hasOneUse() == false) {
        return failure();
      }
      auto use_op = *concat_op->user_begin();
      if (!isa<ConvOp>(use_op)) {
        return failure();
      }
      auto conv_op = dyn_cast<ConvOp>(use_op);
      if (conv_op.getGroup() != 1) {
        return failure();
      }
      auto storage_type = module::getStorageType(conv_op.getOutput());
      if (!storage_type.isF32() && !storage_type.isF16()) {
        return failure();
      }
      auto filter_op = conv_op.getFilter().getDefiningOp<WeightOp>();
      // TODO: maybe filter is i8 in Top Dialect
      auto filter_old = filter_op.read_as_float();
      auto filter_new =
          std::make_shared<std::vector<float>>(filter_old->size(), 0.0);
      int64_t oc, ic, kh, kw;
      module::getNCHW(conv_op.getFilter(), oc, ic, kh, kw);
      int64_t block = bh * bw;
      int64_t inner_dim = (ic / block) * kh * kw;
      int64_t outer_dim = oc;
      for (int o = 0; o < outer_dim; o++) {
        for (int i = 0; i < block; i++) {
          auto begin = filter_old->begin() + (o * block + order[i]) * inner_dim;
          auto end = begin + inner_dim;
          auto to = filter_new->begin() + (o * block + i) * inner_dim;
          std::copy(begin, end, to);
        }
      }
      auto filter_shape = module::getShape(filter_op.getOutput());
      auto new_filter_op = WeightOp::create_float(
          use_op, "filter_S2D", *filter_new, filter_shape, storage_type);
      use_op->setOperand(1, new_filter_op);
      // change name of new op to avoid wrong comparison
      concat_op->setLoc(NameLoc::get(rewriter.getStringAttr(
          module::getName(concat_op.getOperation()).str() + "_r_Depth2Space")));
    }
    replaceOpWithDepth2SpaceOp(rewriter, concat_op, ValueRange(from), bh, bw,
                               false, true, true, true, false);
    return success();
  }
};

template <typename T>
void ProcessRopeWeights(std::vector<T> &new_weight0,
                        std::vector<T> &new_weight1, std::vector<T> &new_w0,
                        std::vector<T> &new_w1,
                        const std::vector<T> &left_weight,
                        const std::vector<T> &right_weight,
                        const std::vector<int64_t> &weight_shape,
                        int mul1_shift, int add_shift) {

  // 4D weight_shape is {A=1, B=1, C=weight_shape[2], D=weight_shape[3]};
  int64_t C = weight_shape[2];
  int64_t D = weight_shape[3];

  for (int j = 0; j < D; j++) {
    new_weight0[j] = 0;
    new_weight1[j] = static_cast<T>(pow(2, (mul1_shift + add_shift)));
  }

  int cnt = 0;
  for (int i = 0; i < C; i++) {
    for (int j = 0; j < D; j++) {
      int index = (i + 1) * D + j;
      new_weight0[index] = left_weight[cnt];
      new_weight1[index] = right_weight[cnt];
      cnt += 1;
    }
  }

  int count = 0;
  for (int i = 0; i < C + 1; i++) {
    for (int j = 0; j < D; j++) {
      new_w0[count] = new_weight0[i * D + j];
      new_w1[count] = new_weight1[i * D + j];
      count += 1;
    }
  }
}

struct ConcatToRope : public OpRewriterPatternEx<ConcatOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  ConcatToRope(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ConcatOp>(context, "ConcatToRope") {}

  LogicalResult matchAndRewriteImpl(ConcatOp op,
                                    PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 2) {
      return failure();
    }

    int indx = -1;
    for (int i = 0; i < 2; ++i) {
      if (dyn_cast<RopeOp>(op.getInputs()[i].getDefiningOp())) {
        indx = i;
        break;
      }
    }

    if (indx == -1)
      return failure();

    auto rope_op = dyn_cast<RopeOp>(op.getInputs()[indx].getDefiningOp());
    auto slice0_op =
        dyn_cast<SliceOp>(op.getInputs()[1 - indx].getDefiningOp());
    if (!rope_op || !slice0_op) {
      return failure();
    }

    auto slice1_op = dyn_cast<SliceOp>(rope_op.getInput1().getDefiningOp());
    if (!slice1_op) {
      return failure();
    }

    auto weight0 = rope_op.getInput2();
    auto weight1 = rope_op.getInput3();
    auto weight_shape = module::getShape(weight0);
    auto W0 = dyn_cast<WeightOp>(weight0.getDefiningOp());
    auto W1 = dyn_cast<WeightOp>(weight1.getDefiningOp());
    if (!W0 || !W1) {
      return failure();
    }

    auto storage_type = module::getStorageType(op.getOutput());
    std::vector<int64_t> new_weight_shape = {
        weight_shape[0], weight_shape[1], weight_shape[2] + 1, weight_shape[3]};

    if (storage_type.isF32() || storage_type.isF16()) {
      auto left_weight = *(W0.read_as_float());
      auto right_weight = *(W1.read_as_float());
      std::vector<float> new_weight0(weight_shape[0] * weight_shape[1] *
                                     (weight_shape[2] + 1) * weight_shape[3]);
      std::vector<float> new_weight1(weight_shape[0] * weight_shape[1] *
                                     (weight_shape[2] + 1) * weight_shape[3]);
      std::vector<float> new_w0((weight_shape[2] + 1) * weight_shape[3]);
      std::vector<float> new_w1((weight_shape[2] + 1) * weight_shape[3]);

      ProcessRopeWeights(new_weight0, new_weight1, new_w0, new_w1, left_weight,
                         right_weight, weight_shape, 0, 0);
      auto new_Weight0 = WeightOp::create_float(op, "weight0", new_w0,
                                                new_weight_shape, storage_type);
      auto new_Weight1 = WeightOp::create_float(op, "weight1", new_w1,
                                                new_weight_shape, storage_type);
      return handleSliceReplacement(rewriter, op, slice0_op, slice1_op,
                                    new_Weight0, new_Weight1);
    } else {
      auto left_weight = *W0.read<int8_t>();
      auto right_weight = *W1.read<int8_t>();
      std::vector<int8_t> new_weight0(weight_shape[0] * weight_shape[1] *
                                      (weight_shape[2] + 1) * weight_shape[3]);
      std::vector<int8_t> new_weight1(weight_shape[0] * weight_shape[1] *
                                      (weight_shape[2] + 1) * weight_shape[3]);
      std::vector<int8_t> new_w0((weight_shape[2] + 1) * weight_shape[3]);
      std::vector<int8_t> new_w1((weight_shape[2] + 1) * weight_shape[3]);

      int mul1_shift = rope_op.getMul1Shift();
      int add_shift = rope_op.getAddShift();
      ProcessRopeWeights(new_weight0, new_weight1, new_w0, new_w1, left_weight,
                         right_weight, weight_shape, mul1_shift, add_shift);

      auto new_type = RankedTensorType::get(
          new_weight_shape, module::getElementType(op.getOutput()));
      auto new_Weight0 =
          top::WeightOp::create<int8_t>(op, "weight0", new_w0, new_type);
      auto new_Weight1 =
          top::WeightOp::create<int8_t>(op, "weight1", new_w1, new_type);
      return handleSliceReplacement(rewriter, op, slice0_op, slice1_op,
                                    new_Weight0, new_Weight1);
    }
  }

private:
  LogicalResult handleSliceReplacement(PatternRewriter &rewriter, ConcatOp op,
                                       SliceOp slice0_op, SliceOp slice1_op,
                                       Value new_Weight0,
                                       Value new_Weight1) const {
    Value in_value;
    if (slice0_op.getInput().getDefiningOp() ==
        slice1_op.getInput().getDefiningOp()) {
      in_value = slice0_op.getInput();
    } else {
      return failure();
    }
    std::vector<NamedAttribute> attrs;
    rewriter.replaceOpWithNewOp<RopeOp>(
        op, op.getResult().getType(),
        ValueRange{in_value, new_Weight0, new_Weight1}, attrs);
    return success();
  }
};
struct ConcatToDepth2SpacePattern2 : public OpRewriterPatternEx<ConcatOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  ConcatToDepth2SpacePattern2(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ConcatOp>(context, "ConcatToDepth2SpacePattern2") {}

  LogicalResult matchAndRewriteImpl(ConcatOp concat_op,
                                    PatternRewriter &rewriter) const override {
    if (concat_op.getDoRelu()) {
      return failure();
    }
    const auto &shape = module::getShape(concat_op.getOutput());
    int num_dims = shape.size();
    if (num_dims < 3) {
      return failure();
    }
    int ex_dims = num_dims - 4;
    if (concat_op.getAxis() - ex_dims != 1 &&
        concat_op.getAxis() - ex_dims != 3) {
      return failure();
    }
    if (concat_op->hasOneUse()) {
      auto use_op = *concat_op->user_begin();
      if (isa<ConvOp>(use_op)) {
        return failure();
      }
    }
    bool in_is_NCHW = (concat_op.getAxis() - ex_dims) == 1;
    bool out_is_NCHW = in_is_NCHW;
    Value from;
    int64_t bh;
    int64_t bw;
    std::vector<int64_t> order;
    auto ret =
        find_slice_order(concat_op, ex_dims, in_is_NCHW, order, from, bh, bw);
    if (ret.failed()) {
      return failure();
    }
    bool flag0 = true;
    bool flag1 = true;
    for (int64_t i = 0; i < bh * bw; ++i) {
      if (order[i] != i) {
        flag0 = false;
        break;
      }
    }
    if (!flag0) {
      for (int64_t i = 0; i < bw; ++i) {
        for (int64_t j = 0; j < bh; ++j) {
          if (order[j * bw + i] != i * bh + j) {
            flag1 = false;
            break;
          }
        }
      }
    }
    if (!flag0 && !flag1)
      return failure();
    bool swap_cr = flag1;
    replaceOpWithDepth2SpaceOp(rewriter, concat_op, ValueRange(from), bh, bw,
                               false, true, in_is_NCHW, out_is_NCHW, swap_cr);
    return success();
  }
};

/**
 *       -- Slice --
 *      /           \
 * Op1->|            |->Concat->Op2 => Op1->Slice->Op2
 *      \           /
 *       -- Slice --
 **/
struct MergeSliceConcatPattern : public OpRewriterPatternEx<ConcatOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  MergeSliceConcatPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ConcatOp>(context, "MergeSliceConcatPattern") {}

  LogicalResult matchAndRewriteImpl(ConcatOp concat_op,
                                    PatternRewriter &rewriter) const override {
    const auto &inputs = concat_op.getInputs();
    if (concat_op.getDoRelu()) {
      return failure();
    }
    const int num_inputs = inputs.size();
    Value from;
    // check topo
    for (int i = 0; i < num_inputs; i++) {
      auto in_op = inputs[i].getDefiningOp();
      if (!isa<SliceOp>(in_op)) {
        return failure();
      }
      auto slice_op = dyn_cast<SliceOp>(in_op);
      if (i == 0) {
        from = slice_op.getInput();
      } else {
        if (from != slice_op.getInput()) {
          return failure();
        }
      }
    }
    // check param
    int64_t start = -1;
    int64_t end = 0;
    i64_array_t steps0, offset0, ends0;
    const auto axis = concat_op.getAxis();
    for (int i = 0; i < num_inputs; i++) {
      auto in_op = inputs[i].getDefiningOp();
      auto slice_op = dyn_cast<SliceOp>(in_op);
      const auto steps = module::getI64Array(slice_op.getSteps());
      const auto offset = module::getI64Array(slice_op.getOffset());
      const auto ends = module::getI64Array(slice_op.getEnds());
      std::vector<int64_t> in_shape = module::getShape(slice_op.getInput());
      const size_t slice_dims = offset->size();
      for (int i = 0; i < slice_dims; ++i) {
        if (offset->at(i) < 0) {
          offset->at(i) += in_shape[i];
        }
        if (ends->at(i) < 0) {
          ends->at(i) += in_shape[i];
        }
        offset->at(i) = steps->at(i) > 0
                            ? std::clamp(offset->at(i), 0L, in_shape[i])
                            : std::clamp(offset->at(i), 0L, in_shape[i] - 1);
        ends->at(axis) =
            steps->at(axis) > 0
                ? std::clamp(ends->at(axis), 0L, in_shape[axis])
                : std::clamp(ends->at(axis), -1L, in_shape[axis] - 1);
      }
      if (steps->at(axis) != 1) {
        return failure();
      }
      if (i == 0) {
        start = offset->at(axis);
        end = ends->at(axis);
      } else {
        if (offset->at(axis) != end) {
          return failure();
        }
      }
      if (i == 0) {
        steps0 = steps;
        offset0 = offset;
        ends0 = ends;
      } else {
        for (size_t i = 0; i < steps->size(); ++i) {
          if (i == axis)
            continue;
          if (steps->at(i) != steps0->at(i)) {
            return failure();
          }
          if (offset->at(i) != offset0->at(i)) {
            return failure();
          }
        }
      }
      const auto &output_shape = module::getShape(slice_op.getOutput());
      end = offset->at(axis) + output_shape[axis];
    }
    // rewrite now !
    offset0->at(axis) = start;
    ends0->at(axis) = end;
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("offset", rewriter.getI64ArrayAttr(*offset0)));
    attrs.push_back(
        rewriter.getNamedAttr("steps", rewriter.getI64ArrayAttr(*steps0)));
    attrs.push_back(
        rewriter.getNamedAttr("ends", rewriter.getI64ArrayAttr(*ends0)));
    auto none = module::getNoneOp(concat_op);
    std::vector<Value> operands;
    operands.push_back(from);
    operands.push_back(none);
    operands.push_back(none);
    operands.push_back(none);
    rewriter.replaceOpWithNewOp<SliceOp>(
        concat_op, concat_op.getResult().getType(), operands, attrs);
    return success();
  }
};

struct ConvertLoadWeightConcatToLoadWeightPattern
    : public OpRewriterPatternEx<ConcatOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  ConvertLoadWeightConcatToLoadWeightPattern(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ConcatOp>(
            context, "ConvertLoadWeightConcatToLoadWeightPattern") {}

  LogicalResult matchAndRewriteImpl(ConcatOp concat_op,
                                    PatternRewriter &rewriter) const override {
    if (concat_op.getDoRelu()) {
      return failure();
    }
    auto input_num = concat_op.getNumOperands();
    for (uint32_t i = 0; i < input_num; ++i) {
      auto formerOp = concat_op.getOperand(i).getDefiningOp();
      if (!isa<WeightOp>(formerOp)) {
        return failure();
      }
    }
    auto storage_type = module::getStorageType(concat_op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }
    uint32_t h, w;
    int tmp_w = 0;

    auto o_shape = module::getShape(concat_op.getOutput());
    std::vector<float> resultT;

    std::vector<std::shared_ptr<std::vector<float>>> input_load_weight(
        input_num);

    for (uint32_t i = 0; i < input_num; ++i) {
      auto weight_op = cast<WeightOp>(concat_op.getOperand(i).getDefiningOp());
      input_load_weight[i] = weight_op.read_as_float();
    }

    for (uint32_t i = 0; i < input_num; ++i) {
      auto w_shape = module::getShape(concat_op.getOperand(i));
      assert(3 == w_shape.size());
      h = w_shape[1];
      w = w_shape[2];

      float *input_data = (float *)input_load_weight[i]->data();
      for (uint32_t idx_h = 0; idx_h < h; ++idx_h) {
        std::vector<float> shapeT(w);
        int64_t insert_offset = ((idx_h + 1) * tmp_w) + idx_h * w;
        shapeT.assign(&input_data[idx_h * w], &input_data[(idx_h + 1) * w]);
        resultT.insert(resultT.begin() + insert_offset, shapeT.begin(),
                       shapeT.end());
      }
      tmp_w += w;
    }
    auto tensor_name = module::getName(concat_op, 0).str() + "loadweight";
    auto weight_operand = WeightOp::create_float(
        concat_op, tensor_name, resultT, o_shape, storage_type);
    rewriter.replaceOp(concat_op, weight_operand);
    return success();
  }
};

struct RemoveInvaidShapeConcatInput : public OpRewriterPatternEx<ConcatOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;
  RemoveInvaidShapeConcatInput(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ConcatOp>(context, "RemoveInvaidShapeConcatInput") {
  }

  LogicalResult matchAndRewriteImpl(ConcatOp concat_op,
                                    PatternRewriter &rewriter) const override {
    if (concat_op.getDoRelu()) {
      return failure();
    }

    if (concat_op.getNumOperands() != 1) {
      return failure();
    }
    rewriter.replaceAllUsesWith(concat_op.getOutput(), concat_op.getOperand(0));
    return success();
  }
};

/***
 * Remove meaningless Struct
 *  Concat (0, X) --> Concat(X); Concat (None, X) => Concat(X)
 * ***/
struct RemoveInvaidConcatSlice : public OpRewriterPatternEx<ConcatOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  RemoveInvaidConcatSlice(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ConcatOp>(context, "RemoveInvaidConcatSlice") {}

  LogicalResult matchAndRewriteImpl(ConcatOp concat_op,
                                    PatternRewriter &rewriter) const override {
    auto inputs = concat_op.getInputs();
    int num_inputs = inputs.size();

    std::vector<Value> new_operands;
    bool ret = false;
    for (int i = 0; i < num_inputs; i++) {
      auto in = concat_op.getInputs()[i];
      if (0 == module::getNumElements(in)) {
        ret = true;
      } else {
        new_operands.push_back(in);
      }
    }
    if (ret == false) {
      return failure();
    }
    concat_op->setOperands(new_operands);
    return success();
  }
};

void ConcatOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.insert<ConvertLoadWeightConcatToLoadWeightPattern,
                 ConcatToDepth2SpacePattern, ConcatToRope,
                 ConcatToDepth2SpacePattern2, MergeSliceConcatPattern,
                 RemoveInvaidConcatSlice, RemoveInvaidShapeConcatInput>(
      context);
}
