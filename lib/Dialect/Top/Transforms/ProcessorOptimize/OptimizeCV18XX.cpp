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

namespace cv18xx {

class ConvertAddConstOp : public OpRewriterPatternEx<top::AddConstOp> {
public:
  ConvertAddConstOp(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::AddConstOp>(context, "ConvertAddConstOp") {}

protected:
  LogicalResult
  matchAndRewriteImpl(top::AddConstOp op,
                      mlir::PatternRewriter &rewriter) const override {
    std::vector<Value> operands;
    std::vector<float> weight_data;
    std::string weight_name =
        module::getName(op.getOutput()).str() + "_const_val";
    weight_data.emplace_back(op.getConstVal().convertToDouble());
    auto weight_type = RankedTensorType::get({1}, rewriter.getF32Type());
    auto weight_operand =
        top::WeightOp::create(op, weight_name, weight_data, weight_type);
    operands.emplace_back(op.getInput());
    operands.emplace_back(weight_operand);

    std::vector<NamedAttribute> attrs;
    for (auto &attr : op->getAttrs()) {
      attrs.push_back(attr);
    }
    rewriter.replaceOpWithNewOp<top::AddOp>(
        op, op.getOutput().getType().cast<RankedTensorType>(), operands, attrs);
    return success();
  }
};

class ConvertArgmaxOp : public OpRewriterPatternEx<top::ArgOp> {
public:
  ConvertArgmaxOp(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::ArgOp>(context, "ConvertArgmaxOp") {}

protected:
  LogicalResult matchAndRewriteImpl(top::ArgOp op,
                                    PatternRewriter &rewriter) const override {
    if (op.getMode() == "ArgMin") {
      return failure();
    }
    auto axis = op.getAxis();
    auto shape = module::getShape(op.getInput());
    if (axis == shape.size() - 1) {
      return failure();
    }
    assert(axis < shape.size());

    std::vector<int64_t> order(shape.size());
    std::iota(order.begin(), order.end(), 0);
    order.erase(order.begin() + axis);
    order.push_back(axis);

    auto op_name = module::getName(op);

    // add transposeOp
    std::vector<int64_t> output_shape;
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    for (uint32_t i = 0; i < shape.size(); ++i) {
      output_shape.emplace_back(shape[order[i]]);
    }

    operands.emplace_back(op.getInput());
    auto loc = NameLoc::get(rewriter.getStringAttr(op_name + "_permute"));
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
    auto type = module::getTypeLike(op.getInput(), output_shape);
    auto permute_op =
        rewriter.create<top::PermuteOp>(loc, type, operands, attrs);

    // add argmax
    op.setOperand(permute_op.getResult());
    op.setAxis(output_shape.size() - 1);
    return success();
  }
};

class ConvertAvgPoolOp : public OpRewriterPatternEx<top::AvgPoolOp> {
public:
  ConvertAvgPoolOp(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::AvgPoolOp>(context, "ConvertAvgPoolOp") {}

protected:
  LogicalResult matchAndRewriteImpl(top::AvgPoolOp op,
                                    PatternRewriter &rewriter) const override {
    const size_t kernel_size = op.getKernelShape().size();
    if (kernel_size != 2) {
      return failure();
    }
    Value input_val = op.getOperand();
    Value output_val = op.getResult();
    int64_t on, oc, oh, ow;
    module::getNCHW(output_val, on, oc, oh, ow, false);

    uint64_t lmem_size = 32 * 1024;
    int64_t n, c, ih, iw;
    module::getNCHW(input_val, n, c, ih, iw, false);

    int64_t output_bytes = 2 * on * oh * ow;
    int64_t input_bytes = 2 * n * ih * iw;

    if ((uint64_t)(input_bytes + output_bytes) < lmem_size ||
        !(oh == 1 && ow == 1)) {
      return failure();
    }
    std::string name = module::getName(output_val).str();
    auto elementType_ =
        output_val.getType().cast<TensorType>().getElementType();
    std::vector<int> h_slices;
    int h_slice_size = (int)((lmem_size - output_bytes) / (2 * n * iw));
    int total_h = ih;
    while (total_h > 0) {
      if (total_h > h_slice_size) {
        total_h -= h_slice_size;
        h_slices.push_back(h_slice_size);
      } else {
        h_slices.push_back(total_h);
        break;
      }
    }

    rewriter.setInsertionPointAfterValue(input_val);
    int offset = 0;
    int ends = 0;
    std::vector<Value> concat_operands;
    for (auto &slice : h_slices) {
      std::vector<Value> slice_operands;
      slice_operands.emplace_back(input_val);
      auto none = module::getNoneOp(op);
      slice_operands.push_back(none);
      slice_operands.push_back(none);
      slice_operands.push_back(none);
      ends += slice;
      std::vector<NamedAttribute> slice_attrs;
      slice_attrs.emplace_back(rewriter.getNamedAttr(
          "offset", rewriter.getI64ArrayAttr({0, 0, offset, 0})));
      slice_attrs.emplace_back(rewriter.getNamedAttr(
          "steps", rewriter.getI64ArrayAttr({1, 1, 1, 1})));
      slice_attrs.emplace_back(rewriter.getNamedAttr(
          "ends",
          rewriter.getI64ArrayAttr({std::numeric_limits<int64_t>::max(),
                                    std::numeric_limits<int64_t>::max(), ends,
                                    std::numeric_limits<int64_t>::max()})));
      offset += slice;
      std::string slice_name = "slice_" + name + std::to_string(offset);
      auto slice_loc = NameLoc::get(rewriter.getStringAttr(slice_name));
      auto slice_type = RankedTensorType::get({n, c, slice, iw}, elementType_);
      auto slice_op = rewriter.create<top::SliceOp>(
          slice_loc, slice_type, slice_operands, slice_attrs);
      auto slice_out = slice_op.getResult();

      rewriter.setInsertionPointAfterValue(slice_out);
      std::vector<Value> small_pool_operands;
      small_pool_operands.emplace_back(slice_out);
      std::vector<NamedAttribute> small_pool_attrs;
      small_pool_attrs.emplace_back(rewriter.getNamedAttr(
          "kernel_shape", rewriter.getI64ArrayAttr({slice, iw})));
      small_pool_attrs.emplace_back(
          rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({1, 1})));
      small_pool_attrs.emplace_back(rewriter.getNamedAttr(
          "pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
      std::string small_pool_name = "pool_" + name + std::to_string(offset);
      auto small_pool_loc =
          NameLoc::get(rewriter.getStringAttr(small_pool_name));
      auto small_pool_type = RankedTensorType::get({n, c, 1, 1}, elementType_);
      auto small_pool_op = rewriter.create<top::AvgPoolOp>(
          small_pool_loc, small_pool_type, small_pool_operands,
          small_pool_attrs);
      auto small_pool_out = small_pool_op.getResult();
      concat_operands.emplace_back(small_pool_out);
      rewriter.setInsertionPointAfterValue(small_pool_out);
    }

    std::vector<NamedAttribute> concat_attrs;
    concat_attrs.emplace_back(
        rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(2)));
    int h_slices_num = h_slices.size();
    std::vector<int64_t> multilpier_arr(h_slices_num, 1);
    std::vector<int64_t> rshifts_arr(h_slices_num, 0);
    std::string concat_name = "concat_" + name;
    auto concat_loc = NameLoc::get(rewriter.getStringAttr(concat_name));
    auto concat_type =
        RankedTensorType::get({n, c, h_slices_num, 1}, elementType_);
    auto concat_op = rewriter.create<top::ConcatOp>(
        concat_loc, concat_type, concat_operands, concat_attrs);
    auto concat_out = concat_op.getResult();
    rewriter.setInsertionPointAfterValue(concat_out);

    std::vector<NamedAttribute> final_attrs;
    final_attrs.emplace_back(rewriter.getNamedAttr(
        "kernel_shape", rewriter.getI64ArrayAttr({h_slices_num, 1})));
    final_attrs.emplace_back(
        rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr({1, 1})));
    final_attrs.emplace_back(
        rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
    auto final_type = RankedTensorType::get({n, c, 1, 1}, elementType_);
    rewriter.replaceOpWithNewOp<top::AvgPoolOp>(
        op.getOperation(), final_type, ValueRange{concat_out}, final_attrs);
    return success();
  }
};

class ConvertConvDilation : public OpRewriterPatternEx<top::ConvOp> {
public:
  ConvertConvDilation(mlir::MLIRContext *context, int benefit)
      : OpRewriterPatternEx<top::ConvOp>(context, "ConvertConvDilation") {}

protected:
  LogicalResult matchAndRewriteImpl(top::ConvOp op,
                                    PatternRewriter &rewriter) const override {
    const int DILATION_H_MAX = 15;
    const int DILATION_W_MAX = 15;
    auto attr = op.parseParam();
    if (attr.dh <= DILATION_H_MAX && attr.dw <= DILATION_W_MAX)
      return failure();
    // filter
    auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
    auto filter_f32 = filterOp.read<float>();
    std::vector<int64_t> filterShape = module::getShape(op.getFilter());
    int64_t oc = 0;
    int64_t ic = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    if (filterShape.size() == 4) {
      oc = filterShape[0];
      ic = filterShape[1];
      kh = filterShape[2];
      kw = filterShape[3];
    } else if (filterShape.size() == 5) {
      // g, oc/g, ic/g, kh, kw
      oc = filterShape[0] * filterShape[1];
      ic = filterShape[2];
      kh = filterShape[3];
      kw = filterShape[4];
    } else {
      llvm_unreachable("Not support now.");
    }

    int insertNumH = 0;
    int insertNumW = 0;
    int newDilationH = attr.dh;
    int newDilationW = attr.dw;
    while (1) {
      insertNumH++;
      newDilationH = (attr.dh - 1 - insertNumH) / (insertNumH + 1) + 1;
      if (((attr.dh - 1 - insertNumH) % (insertNumH + 1) == 0) &&
          newDilationH < DILATION_H_MAX)
        break;
    }

    if (attr.dw > 1) {
      while (1) {
        insertNumW++;
        newDilationW = (attr.dw - 1 - insertNumW) / (insertNumW + 1) + 1;
        if (((attr.dw - 1 - insertNumW) % (insertNumW + 1) == 0) &&
            newDilationW < DILATION_W_MAX)
          break;
      }
    }

    int k_ext_h = (insertNumH + 1) * (kh - 1) + 1;
    int k_ext_w = (insertNumW + 1) * (kw - 1) + 1;
    filterShape[2] = k_ext_h;
    filterShape[3] = k_ext_w;
    auto filterSize = oc * ic * k_ext_h * k_ext_w;
    std::vector<float> newFilter(filterSize, 0);
    for (int i = 0; i < oc * ic; i++) {
      for (int j = 0; j < kh; j++) {
        for (int k = 0; k < kw; k++) {
          auto old_offset = i * kh * kw + j * kw + k;
          auto new_offset = i * k_ext_h * k_ext_w +
                            j * (insertNumH + 1) * k_ext_w +
                            k * (insertNumW + 1);
          newFilter[new_offset] = filter_f32->data()[old_offset];
        }
      }
    }

    // update filter op
    auto new_type = RankedTensorType::get(filterShape, rewriter.getF32Type());
    auto new_filter_op =
        top::WeightOp::create(op, "dilation", newFilter, new_type);
    op->setOperand(1, new_filter_op);
    // update convOp attr
    std::vector<int64_t> new_kernel_shape, new_dilations;
    auto kernel_shape = module::getI64Array(op.getKernelShape());
    auto dilations =
        module::getI64Array(op.getDilations(), kernel_shape->size(), 1);
    new_kernel_shape.assign(kernel_shape->begin(), kernel_shape->end());
    new_dilations.assign(dilations->begin(), dilations->end());
    auto kernel_size = new_kernel_shape.size();
    new_kernel_shape[kernel_size - 2] = k_ext_h;
    new_kernel_shape[kernel_size - 1] = k_ext_w;

    new_dilations[kernel_size - 2] = newDilationH;
    new_dilations[kernel_size - 1] = newDilationW;

    op->setAttr("kernel_shape", rewriter.getI64ArrayAttr(new_kernel_shape));
    op->setAttr("dilations", rewriter.getI64ArrayAttr(new_dilations));
    auto convOp = rewriter.create<top::ConvOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands(), op->getAttrs());
    rewriter.replaceOp(op, {convOp.getOutput()});
    return success();
  }
};

class ConvertConvPading : public OpRewriterPatternEx<top::ConvOp> {
public:
  ConvertConvPading(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::ConvOp>(context, "ConvertConvPading",
                                         benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::ConvOp op,
                                    PatternRewriter &rewriter) const override {
    // deal with pad > 16
    auto attr = op.parseParam();
    bool insert_pad = false;
    std::vector<int64_t> input_shape = module::getShape(op.getInput());
    auto kernel_size = input_shape.size() - 2;
    auto _pads = module::getI64Array(op.getPads());
    std::vector<int64_t> pad_v;
    std::vector<int64_t> new_pad_v;
    new_pad_v.resize(input_shape.size() * 2);
    pad_v.assign(_pads->begin(), _pads->end());
    for (auto p : pad_v) {
      if (p > 15) {
        insert_pad = true;
        break;
      }
    }
    if (!insert_pad) {
      return failure();
    }
    if (kernel_size == 3) {
      if (attr.pht > 15) {
        assert(attr.pht == pad_v[1]);
        pad_v[1] = 0;
        new_pad_v[3] = attr.pht;
        input_shape[3] += attr.pht;
      }
      if (attr.pwl > 15) {
        assert(attr.pwl == pad_v[2]);
        pad_v[2] = 0;
        new_pad_v[4] = attr.pwl;
        input_shape[4] += attr.pwl;
      }
      if (attr.phb > 15) {
        assert(attr.phb == pad_v[4]);
        pad_v[4] = 0;
        new_pad_v[8] = attr.phb;
        input_shape[3] += attr.phb;
      }
      if (attr.pwr > 15) {
        assert(attr.pwr == pad_v[5]);
        pad_v[5] = 0;
        new_pad_v[9] = attr.pwr;
        input_shape[4] += attr.pwr;
      }
    } else if (kernel_size == 2) {
      if (attr.pht > 15) {
        assert(attr.pht == pad_v[0]);
        pad_v[0] = 0;
        new_pad_v[2] = attr.pht;
        input_shape[2] += attr.pht;
      }
      if (attr.pwl > 15) {
        assert(attr.pwl == pad_v[1]);
        pad_v[1] = 0;
        new_pad_v[3] = attr.pwl;
        input_shape[3] += attr.pwl;
      }
      if (attr.phb > 15) {
        assert(attr.phb == pad_v[2]);
        pad_v[2] = 0;
        new_pad_v[6] = attr.phb;
        input_shape[2] += attr.phb;
      }
      if (attr.pwr > 15) {
        assert(attr.pwr == pad_v[3]);
        pad_v[3] = 0;
        new_pad_v[7] = attr.pwr;
        input_shape[3] += attr.pwr;
      }
    } else if (kernel_size == 1) {
      if (attr.pht > 15) {
        assert(attr.pht == pad_v[0]);
        pad_v[0] = 0;
        new_pad_v[2] = attr.pht;
        input_shape[2] += attr.pht;
      }
      if (attr.phb > 15) {
        // conv1d convert to conv2d in pre common pass, but input and output
        // shape is still conv1d mannaer pads = [25, 0, 25, 0]
        // (tensor<1x256x1592xf32>, tensor<256x256x11x1xf32>, tensor<256xf32>)
        // -> tensor<1x256x1592xf32>
        assert(attr.phb == pad_v[pad_v.size() / 2]);
        pad_v[pad_v.size() / 2] = 0;
        new_pad_v[5] = attr.phb;
        input_shape[2] += attr.phb;
      }
    } else {
      llvm_unreachable("Not support now.");
    }
    std::vector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr(
        "paddings", rewriter.getI64ArrayAttr(ArrayRef<int64_t>{new_pad_v})));
    attrs.push_back(
        rewriter.getNamedAttr("mode", rewriter.getStringAttr("constant")));
    auto op_name = module::getName(op.getOperation()).str();
    auto loc = NameLoc::get(rewriter.getStringAttr(op_name + "_pad"));
    auto type = op.getInput().getType().cast<RankedTensorType>();
    auto new_type = RankedTensorType::get(input_shape, type.getElementType());
    auto padOp = rewriter.create<top::PadOp>(loc, new_type,
                                             ValueRange{
                                                 op.getInput(),
                                                 module::getNoneOp(op),
                                             },
                                             attrs);
    op->setAttr("pads", rewriter.getI64ArrayAttr(pad_v));
    op->setOperand(0, padOp);
    auto convOp = rewriter.create<top::ConvOp>(
        op->getLoc(), op->getResultTypes(), op->getOperands(), op->getAttrs());
    rewriter.replaceOp(op, {convOp.getOutput()});
    return success();
  }
};

class ConvertConv2dToMatMul : public OpRewriterPatternEx<top::ConvOp> {
public:
  ConvertConv2dToMatMul(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::ConvOp>(context, "ConvertConv2dToMatMul",
                                         benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::ConvOp op,
                                    PatternRewriter &rewriter) const override {
    auto attr = op.parseParam();
    // support hua'an pose_res model
    auto kernel = module::getI64Array(op.getKernelShape());
    if (kernel->size() != 2) {
      return failure();
    }
    int64_t n = attr.n, ic = attr.ic, ih = attr.ih, iw = attr.iw;
    int64_t kh = attr.kh, kw = attr.kw, sh = attr.sh, sw = attr.sw;
    if ((kh != sh || kw != sw) || (sh < 16 || sw < 16) ||
        (ih % kh || iw % kw)) {
      return failure();
    }
    if (attr.pht || attr.phb || attr.pwl || attr.pwr) {
      return failure();
    }
    auto input = op.getInput();
    auto input_type = module::getElementType(input);
    auto out_type = module::getElementType(op.getOutput());
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    std::string op_name = module::getName(op.getResult()).str();
    // reshape0 8x3x224x224 --> 8x3x14x16x14x16
    rewriter.setInsertionPointAfterValue(input);
    operands.emplace_back(input);
    auto loc0 = NameLoc::get(rewriter.getStringAttr(op_name + "_reshape0"));
    auto reshape0_type =
        RankedTensorType::get({n, ic, ih / kh, kh, iw / kw, kw}, input_type);
    auto reshape0_op =
        rewriter.create<top::ReshapeOp>(loc0, reshape0_type, operands, attrs);
    auto reshape0_out = reshape0_op.getResult();
    // permute0 [0 1 2 4 3 5] (8x3x14x16x14x16 --> 8x3x14x14x16x16)
    rewriter.setInsertionPointAfterValue(reshape0_out);
    operands.clear();
    attrs.clear();
    operands.emplace_back(reshape0_out);
    attrs.emplace_back(rewriter.getNamedAttr(
        "order", rewriter.getI64ArrayAttr({0, 1, 2, 4, 3, 5})));
    auto loc1 = NameLoc::get(rewriter.getStringAttr(op_name + "_permute0"));
    auto permute0_type =
        RankedTensorType::get({n, ic, ih / kh, iw / kw, kh, kw}, input_type);
    auto permute0_op =
        rewriter.create<top::PermuteOp>(loc1, permute0_type, operands, attrs);
    auto permute0_out = permute0_op.getResult();
    // permute1 [0 2 3 1 4 5] ( 8x3x14x14x16x16 --> 8x14x14x3x16x16)
    rewriter.setInsertionPointAfterValue(permute0_out);
    operands.clear();
    attrs.clear();
    operands.emplace_back(permute0_out);
    attrs.emplace_back(rewriter.getNamedAttr(
        "order", rewriter.getI64ArrayAttr({0, 2, 3, 1, 4, 5})));
    auto loc2 = NameLoc::get(rewriter.getStringAttr(op_name + "_permute1"));
    auto permute1_type =
        RankedTensorType::get({n, ih / kh, iw / kw, ic, kh, kw}, input_type);
    auto permute1_op =
        rewriter.create<top::PermuteOp>(loc2, permute1_type, operands, attrs);
    auto permute1_out = permute1_op.getResult();
    // reshape1 8x14x14x3x16x16 -->  MxK(8x14x14, 3x16x16)
    rewriter.setInsertionPointAfterValue(permute1_out);
    operands.clear();
    attrs.clear();
    operands.emplace_back(permute1_out);
    auto loc3 = NameLoc::get(rewriter.getStringAttr(op_name + "_reshape1"));
    auto reshape1_type =
        RankedTensorType::get({n, ih / kh, iw / kw, ic * kh * kw}, input_type);
    auto reshape1_op =
        rewriter.create<top::ReshapeOp>(loc3, reshape1_type, operands, attrs);
    auto reshape1_out = reshape1_op.getResult();

    // insert matmulOp
    rewriter.setInsertionPointAfterValue(reshape1_out);
    operands.clear();
    attrs.clear();
    auto noneOp = module::getNoneOp(op);
    operands.emplace_back(reshape1_out);
    operands.emplace_back(noneOp);
    operands.emplace_back(noneOp);
    // reshape filter 768x3x16x16 --> NxK(768, 3x16x16)
    auto filterOp = cast<top::WeightOp>(op.getFilter().getDefiningOp());
    auto filter_f32 = filterOp.read<float>();
    std::vector<int64_t> filter_shape = module::getShape(op.getFilter());
    if (filter_shape.size() != 4) {
      return failure();
    }
    int64_t N = filter_shape[0];
    int64_t K = std::accumulate(filter_shape.begin() + 1, filter_shape.end(), 1,
                                std::multiplies<int64_t>());
    // filter weight transpose
    std::vector<float> new_filter_f32(filter_f32->size());
    for (int64_t i = 0; i < N; i++) {
      for (int64_t j = 0; j < K; j++) {
        new_filter_f32[j * N + i] = filter_f32->at(i * K + j);
      }
    }
    attrs.emplace_back(
        rewriter.getNamedAttr("right_transpose", rewriter.getBoolAttr(false)));
    auto loc4 = NameLoc::get(rewriter.getStringAttr(op_name + "_matmul"));
    auto matmul_type =
        RankedTensorType::get({n, ih / kh, iw / kw, N}, out_type);
    auto matmulOp =
        rewriter.create<top::MatMulOp>(loc4, matmul_type, operands, attrs);
    auto new_filter_type = RankedTensorType::get({K, N}, rewriter.getF32Type());
    auto new_filter = top::WeightOp::create(matmulOp, op_name + "_filter",
                                            new_filter_f32, new_filter_type);
    matmulOp.setOperand(1, new_filter);
    if (attr.has_bias) {
      auto biasOp = cast<top::WeightOp>(op.getBias().getDefiningOp());
      auto bias_f32 = biasOp.read<float>();
      auto new_bias_type = RankedTensorType::get({N}, rewriter.getF32Type());
      auto new_bias = top::WeightOp::create(matmulOp, op_name + "_bias",
                                            *bias_f32, new_bias_type);
      matmulOp.setOperand(2, new_bias);
    }

    auto matmul_out = matmulOp.getResult();
    // permute2 [0,3,1,2] --> 8x768x14x14
    rewriter.setInsertionPointAfterValue(matmul_out);
    operands.clear();
    attrs.clear();
    operands.emplace_back(matmul_out);
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr({0, 3, 1, 2})));
    auto permute2_type =
        RankedTensorType::get({n, N, ih / kh, iw / kw}, out_type);
    rewriter.replaceOpWithNewOp<top::PermuteOp>(op, permute2_type, operands,
                                                attrs);
    return success();
  }
};

class ConvertGatherOp : public OpRewriterPatternEx<top::GatherOp> {
public:
  ConvertGatherOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::GatherOp>(context, "ConvertGatherOp",
                                           benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::GatherOp op,
                                    PatternRewriter &rewriter) const override {
    // for transform decode's index op
    Value input = op.getInput();
    Value indices = op.getIndices();
    Value ori_out = op.getOutput();
    std::string name = module::getName(ori_out).str();
    uint64_t axis = op.getAxis();
    std::vector<int64_t> input_shape = module::getShape(input);
    std::vector<int64_t> output_shape = module::getShape(ori_out);
    std::vector<int64_t> indices_shape = module::getShape(indices);
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;

    // convert to slice
    auto weight_op = dyn_cast_or_null<top::WeightOp>(indices.getDefiningOp());
    if (weight_op) {
      auto weight_data = weight_op.read_as_float();
      // indices size == 1
      if (weight_data->size() == 1) {
        assert(input_shape.size() == output_shape.size());
        int offset = static_cast<int>(weight_data->at(0));
        if (offset < 0) {
          offset = input_shape[axis] + offset;
        }
        std::vector<int64_t> slice_offsets(input_shape.size(), 0);
        std::vector<int64_t> slice_steps(input_shape.size(), 1);
        std::vector<int64_t> slice_ends(input_shape.size(), -1);
        slice_offsets[axis] = offset;
        operands.emplace_back(input);
        auto none = module::getNoneOp(op);
        operands.push_back(none);
        operands.push_back(none);
        operands.push_back(none);
        attrs.emplace_back(rewriter.getNamedAttr(
            "offset", rewriter.getI64ArrayAttr(slice_offsets)));
        attrs.emplace_back(rewriter.getNamedAttr(
            "steps", rewriter.getI64ArrayAttr(slice_steps)));
        attrs.emplace_back(rewriter.getNamedAttr(
            "ends", rewriter.getI64ArrayAttr(slice_ends)));
        rewriter.replaceOpWithNewOp<top::SliceOp>(op, ori_out.getType(),
                                                  operands, attrs);
        return success();
      } else {
        // indices is an arithmetic progression
        int diff = static_cast<int>(weight_data->at(1)) -
                   static_cast<int>(weight_data->at(0));
        if (diff == 0) {
          return failure();
        }
        int last = static_cast<int>(weight_data->at(1));
        bool is_arithmetic_progression = true;
        for (int i = 2; i < weight_data->size(); ++i) {
          int tmp = static_cast<int>(weight_data->at(i)) - last;
          if (tmp != diff) {
            is_arithmetic_progression = false;
            break;
          }
          last = static_cast<int>(weight_data->at(i));
        }
        if (is_arithmetic_progression) {
          int _axis = axis >= 0 ? axis : axis + input_shape.size();
          std::vector<int64_t> slice_offsets(input_shape.size(), 0);
          std::vector<int64_t> slice_steps(input_shape.size(), 1);
          std::vector<int64_t> slice_ends(input_shape.size(), -1);
          slice_offsets[_axis] = diff > 0
                                     ? static_cast<int>(*weight_data->begin())
                                     : static_cast<int>(*weight_data->rbegin());
          slice_steps[_axis] = std::abs(diff);
          operands.emplace_back(input);
          auto none = module::getNoneOp(op);
          operands.push_back(none);
          operands.push_back(none);
          operands.push_back(none);
          attrs.emplace_back(rewriter.getNamedAttr(
              "offset", rewriter.getI64ArrayAttr(slice_offsets)));
          attrs.emplace_back(rewriter.getNamedAttr(
              "steps", rewriter.getI64ArrayAttr(slice_steps)));
          attrs.emplace_back(rewriter.getNamedAttr(
              "ends", rewriter.getI64ArrayAttr(slice_ends)));
          rewriter.replaceOpWithNewOp<top::SliceOp>(op, ori_out.getType(),
                                                    operands, attrs);
          return success();
        }
      }
    } else {
      // convert for embedding
      bool need_convert =
          (axis == 1 && indices_shape.size() == 1 && input_shape.size() == 3 &&
           input_shape[0] == 1 && !(module::isWeight(input)));
      if (need_convert) {
        // conver to reshapeOp + new GatherOp
        rewriter.setInsertionPointAfterValue(ori_out);
        double in_thr, out_thr;
        RankedTensorType type1, type2;
        if (module::isCalibratedType(ori_out)) {
          auto caliType1 = quant::CalibratedQuantizedType::get(
              rewriter.getF32Type(), -in_thr, in_thr);
          auto caliType2 = quant::CalibratedQuantizedType::get(
              rewriter.getF32Type(), -out_thr, out_thr);
          type1 = RankedTensorType::get({input_shape[1], input_shape[2]},
                                        caliType1);
          type2 = RankedTensorType::get(output_shape, caliType2);
        } else {
          type1 = RankedTensorType::get({input_shape[1], input_shape[2]},
                                        rewriter.getF32Type());
          type2 = ori_out.getType().cast<RankedTensorType>();
        }
        operands.emplace_back(input);
        auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_reshape"));
        auto reshapeOp =
            rewriter.create<top::ReshapeOp>(loc1, type1, operands, attrs);
        auto out1 = reshapeOp.getOutput();
        operands.clear();
        operands.emplace_back(out1);
        operands.emplace_back(indices);
        attrs.emplace_back(
            rewriter.getNamedAttr("axis", rewriter.getSI32IntegerAttr(0)));
        auto loc2 = NameLoc::get(rewriter.getStringAttr(name));
        auto newOp =
            rewriter.create<top::GatherOp>(loc2, type2, operands, attrs);
        auto newOut = newOp.getOutput();
        rewriter.replaceAllUsesWith(ori_out, newOut);
        rewriter.eraseOp(op);
        return success();
      }
    }
    return failure();
  }
};

class ConvertInterpOp : public OpRewriterPatternEx<top::InterpOp> {
public:
  ConvertInterpOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::InterpOp>(context, "ConvertInterpOp",
                                           benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::InterpOp op,
                                    PatternRewriter &rewriter) const override {
    // implement
    std::string mode = op.getMode().str();
    double scale_h = op.getScaleH().convertToDouble();
    double scale_w = op.getScaleW().convertToDouble();
    if (mode == "nearest" && std::ceil(scale_h) == std::floor(scale_h) &&
        std::ceil(scale_w) == std::floor(scale_w)) {
      llvm::errs() << "Warning, if model is onnx format, it should be already "
                      "converted in onnx_convert\n";
      // from torch
      std::vector<NamedAttribute> attrs;
      attrs.emplace_back(rewriter.getNamedAttr(
          "scale_h", rewriter.getI64IntegerAttr((int64_t)scale_h)));
      attrs.emplace_back(rewriter.getNamedAttr(
          "scale_w", rewriter.getI64IntegerAttr((int64_t)scale_w)));
      std::vector<Value> operands;
      operands.emplace_back(op.getInput());
      rewriter.replaceOpWithNewOp<top::UpsampleOp>(op, op.getType(), operands,
                                                   attrs);
      return success();
    }
    return failure();
  }
};

class ConvertMaskedFillOp : public OpRewriterPatternEx<top::MaskedFillOp> {
public:
  ConvertMaskedFillOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::MaskedFillOp>(context, "ConvertMaskedFillOp",
                                               benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::MaskedFillOp op,
                                    PatternRewriter &rewriter) const override {
    bool inverse = op.getInversed();
    double const_val = op.getConstVal().convertToDouble();
    Value input0 = op.getOperand(0);
    Value input1 = op.getOperand(1);
    Value ori_out = op.getOutput();
    std::string name = module::getName(ori_out).str();
    std::vector<int64_t> output_shape = module::getShape(ori_out);
    std::vector<int64_t> input0_shape = module::getShape(input0);
    std::vector<int64_t> input1_shape = module::getShape(input1);
    auto out_type = ori_out.getType().cast<RankedTensorType>();
    bool isCali = false;
    double out_thr, in0_thr, in1_thr;
    if (module::isCalibratedType(out_type)) {
      isCali = true;
      auto otype = module::getCalibratedType(ori_out);
      auto in0_type = module::getCalibratedType(input0);
      auto in1_type = module::getCalibratedType(input1);
      out_thr = otype.getMax();
      in0_thr = in0_type.getMax();
      in1_thr = in1_type.getMax();
    }
    // cv18xx only support one operand broadcast now.
    assert((input0_shape == output_shape || input1_shape == output_shape));
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    rewriter.setInsertionPointAfterValue(ori_out);
    if (inverse) {
      // out = input[0] * const_val + (1 - input[0]) * input[1]

      // create input[0] * const_val
      operands.emplace_back(input0);
      attrs.emplace_back(rewriter.getNamedAttr(
          "const_val", rewriter.getF64FloatAttr(const_val)));
      auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_mulconst1"));
      RankedTensorType type1;
      if (isCali) {
        auto caliType = quant::CalibratedQuantizedType::get(
            rewriter.getF32Type(), -std::abs(const_val) * in0_thr,
            std::abs(const_val) * in0_thr);
        type1 = RankedTensorType::get(input0_shape, caliType);
      } else {
        type1 = RankedTensorType::get(input0_shape, rewriter.getF32Type());
      }
      auto mulconstOp1 =
          rewriter.create<top::MulConstOp>(loc1, type1, operands, attrs);
      auto out1 = mulconstOp1.getOutput();

      // create (input[0]) * input[1]
      operands.clear();
      attrs.clear();
      rewriter.setInsertionPointAfterValue(out1);
      operands.emplace_back(input0);
      operands.emplace_back(input1);
      auto loc2 = NameLoc::get(rewriter.getStringAttr(name + "_mul1"));
      RankedTensorType type2;
      if (isCali) {
        auto caliType = quant::CalibratedQuantizedType::get(
            rewriter.getF32Type(), -in1_thr, in1_thr);
        type2 = RankedTensorType::get(output_shape, caliType);
      } else {
        type2 = RankedTensorType::get(output_shape, rewriter.getF32Type());
      }
      auto mulOp1 = rewriter.create<top::MulOp>(loc2, type2, operands, attrs);
      auto out2 = mulOp1.getOutput();

      // create input[1] - input[0] * input[1]
      attrs.clear();
      operands.clear();
      rewriter.setInsertionPointAfterValue(out2);
      operands.emplace_back(input1);
      operands.emplace_back(out2);
      auto loc3 = NameLoc::get(rewriter.getStringAttr(name + "_sub1"));
      auto subOp1 = rewriter.create<top::SubOp>(loc3, type2, operands, attrs);
      auto out3 = subOp1.getOutput();

      // create (input[0] * const_val)+ (input[1] - input[0] * input[1])
      attrs.clear();
      operands.clear();
      rewriter.setInsertionPointAfterValue(out3);
      operands.emplace_back(out3);
      operands.emplace_back(out1);
      RankedTensorType type4;
      if (isCali) {
        auto caliType = quant::CalibratedQuantizedType::get(
            rewriter.getF32Type(), -out_thr, out_thr);
        type4 = RankedTensorType::get(output_shape, caliType);
      } else {
        type4 = RankedTensorType::get(output_shape, rewriter.getF32Type());
      }
      auto loc4 = NameLoc::get(rewriter.getStringAttr(name));
      auto addOp2 = rewriter.create<top::AddOp>(loc4, type4, operands, attrs);
      rewriter.replaceAllUsesWith(ori_out, addOp2.getOutput());
      rewriter.eraseOp(op);

    } else {
      // out = input[0] * input[1] + (1 - input[0]) * const_val

      // create input[0] * input[1]
      operands.emplace_back(input0);
      operands.emplace_back(input1);
      auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_mul1"));
      RankedTensorType type1;
      if (isCali) {
        auto caliType = quant::CalibratedQuantizedType::get(
            rewriter.getF32Type(), -in1_thr, in1_thr);
        type1 = RankedTensorType::get(output_shape, caliType);
      } else {
        type1 = RankedTensorType::get(output_shape, rewriter.getF32Type());
      }
      auto mulOp1 = rewriter.create<top::MulOp>(loc1, type1, operands, attrs);
      auto out1 = mulOp1.getOutput();
      // out1.setLoc(op.getLoc());

      // create -const_val * input[0]
      operands.clear();
      attrs.emplace_back(rewriter.getNamedAttr(
          "const_val", rewriter.getF64FloatAttr(-const_val)));
      operands.emplace_back(input0);
      auto loc2 = NameLoc::get(rewriter.getStringAttr(name + "mulconst1"));
      RankedTensorType type2;
      if (isCali) {
        auto caliType = quant::CalibratedQuantizedType::get(
            rewriter.getF32Type(), -std::abs(const_val) * in0_thr,
            std::abs(const_val) * in0_thr);
        type2 = RankedTensorType::get(input0_shape, caliType);
      } else {
        type2 = RankedTensorType::get(input0_shape, rewriter.getF32Type());
      }
      rewriter.setInsertionPointAfterValue(out1);
      auto mulconstOp1 =
          rewriter.create<top::MulConstOp>(loc2, type2, operands, attrs);
      auto out2 = mulconstOp1.getOutput();

      // create (-const_val * input[0]) + const_val
      operands.clear();
      attrs.clear();
      operands.emplace_back(out2);
      attrs.emplace_back(rewriter.getNamedAttr(
          "const_val", rewriter.getF64FloatAttr(const_val)));
      auto loc3 = NameLoc::get(rewriter.getStringAttr(name + "addconst1"));
      rewriter.setInsertionPointAfterValue(out2);
      auto addconstOp1 =
          rewriter.create<top::AddConstOp>(loc3, type2, operands, attrs);
      auto out3 = addconstOp1.getOutput();

      // create (input[0] * input[1]) + ((-const_val * input[0]) + const_val)
      operands.clear();
      attrs.clear();
      operands.emplace_back(out1);
      operands.emplace_back(out3);
      auto loc4 = NameLoc::get(rewriter.getStringAttr(name));
      RankedTensorType type4;
      if (isCali) {
        auto caliType = quant::CalibratedQuantizedType::get(
            rewriter.getF32Type(), -out_thr, out_thr);
        type4 = RankedTensorType::get(output_shape, caliType);
      } else {
        type4 = RankedTensorType::get(output_shape, rewriter.getF32Type());
      }
      rewriter.setInsertionPointAfterValue(out3);
      auto addOp1 = rewriter.create<top::AddOp>(loc4, type4, operands, attrs);
      rewriter.replaceAllUsesWith(ori_out, addOp1.getOutput());
      rewriter.eraseOp(op);
    }
    return success();
  }
};

class ConvertMatMulWithRightTranspose
    : public OpRewriterPatternEx<top::MatMulOp> {
public:
  ConvertMatMulWithRightTranspose(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::MatMulOp>(
            context, "ConvertMatMulWithRightTranspose", benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::MatMulOp op,
                                    PatternRewriter &rewriter) const override {
    auto left = op.getInput();
    auto right = op.getRight();
    std::vector<int64_t> lshape = module::getShape(left);
    std::vector<int64_t> rshape = module::getShape(right);
    auto out = module::getNextOp(op);

    // if right is weight and need transpose, do it here.
    if (op.getRightTranspose() && module::isWeight(right)) {
      std::string filter_name = module::getName(right).str();
      auto filterOp = dyn_cast<top::WeightOp>(right.getDefiningOp());
      auto filter_f32 = filterOp.read<float>();
      int64_t filter_dims = rshape.size();
      auto p = op.parseParam();
      int64_t K = p.K, N = p.N;
      int64_t batch = std::accumulate(rshape.begin(), rshape.end() - 2, 1,
                                      std::multiplies<int64_t>());
      assert(rshape[filter_dims - 2] == N && rshape[filter_dims - 1] == K &&
             batch == p.batch);
      // transpose filter of last two dims
      std::vector<float> new_filter(filter_f32->size());
      for (int64_t b = 0; b < batch; b++) {
        for (int i = 0; i < N; i++) {
          for (int j = 0; j < K; j++) {
            int64_t idx1 = b * N * K + i * K + j;
            int64_t idx2 = b * N * K + j * N + i;
            new_filter[idx1] = filter_f32->at(idx2);
          }
        }
      }
      // swap last two dims to get new_rshape
      std::swap(rshape[filter_dims - 2], rshape[filter_dims - 1]);
      auto new_filter_type =
          RankedTensorType::get(rshape, rewriter.getF32Type());
      auto new_filter_op =
          top::WeightOp::create(op, filter_name, new_filter, new_filter_type);
      op.setOperand(1, new_filter_op);
      op.setRightTranspose(false);
      return success();
    }

    if (lshape.size() != 4 || rshape.size() != 4) {
      return failure();
    }
    if (op.getRightTranspose() != false || op.getLeftTranspose() != false ||
        op.getOutputTranspose() != false) {
      return failure();
    }

    bool match = false;
    std::vector<int64_t> pattern = {0, 2, 1, 3};
    std::vector<int64_t> shape_4, order_4;
    auto leftOp = left.getDefiningOp();
    if (isa<top::PermuteOp>(leftOp) && left.hasOneUse()) {
      auto ltrans_op = dyn_cast<top::PermuteOp>(leftOp);
      order_4 = *module::getI64Array(ltrans_op.getOrder());
      if (order_4 == pattern) {
        op.setOperand(0, ltrans_op.getInput());
        op.setLeftTranspose(true);
        op.setHdimIsBatch(true);
        rewriter.eraseOp(ltrans_op);
        match = true;
      }
    }
    auto rightOp = right.getDefiningOp();
    if (isa<top::PermuteOp>(rightOp) && right.hasOneUse()) {
      auto rtrans_op = dyn_cast<top::PermuteOp>(rightOp);
      order_4 = *module::getI64Array(rtrans_op.getOrder());
      if (order_4 == pattern) {
        op.setOperand(1, rtrans_op.getInput());
        op.setRightTranspose(true);
        op.setHdimIsBatch(true);
        rewriter.eraseOp(rtrans_op);
        match = true;
      }
    }
    if (out != nullptr && isa<top::PermuteOp>(out) &&
        op.getResult().hasOneUse()) {
      auto otrans_op = dyn_cast<top::PermuteOp>(out);
      order_4 = *module::getI64Array(otrans_op.getOrder());
      if (order_4 == pattern) {
        op.setOutputTranspose(true);
        op.setHdimIsBatch(true);
        op->setLoc(otrans_op->getLoc());
        op.getResult().setType(otrans_op.getResult().getType());
        otrans_op.getOutput().replaceAllUsesWith(otrans_op.getInput());
        rewriter.eraseOp(otrans_op);
        match = true;
      }
    }
    return match ? success() : failure();
  }
};

class convertMaxPool3D : public OpRewriterPatternEx<top::MaxPoolOp> {
public:
  convertMaxPool3D(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::MaxPoolOp>(context, "convertMaxPool3D",
                                            benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::MaxPoolOp op,
                                    PatternRewriter &rewriter) const override {
    std::vector<Value> operands;
    std::vector<int64_t> tmp_shape0(4, 1);
    std::vector<int64_t> tmp_shape1;
    std::vector<int64_t> _kernel;
    std::vector<int64_t> _strides;
    std::vector<int64_t> _pad;
    if (op.getKernelShape().size() != 3) {
      return failure();
    }

    auto input_shape = module::getShape(op.getInput());
    auto output_shape = module::getShape(op.getOutput());
    auto kernel = module::getI64Array(op.getKernelShape());
    auto strides = module::getI64Array(op.getStrides());
    auto pads = module::getI64Array(op.getPads());
    auto op_name = module::getName(op.getOperation()).str();
    auto type = module::getElementType(op.getOutput());
    // 0. reshape [n c f h w] -> [n*c h w f].
    module::getNCHW(input_shape, tmp_shape0[0], tmp_shape0[1], tmp_shape0[2],
                    tmp_shape0[3], false);
    auto newType = RankedTensorType::get(tmp_shape0, type);
    auto name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_reshape"));
    auto reshapeOp =
        rewriter.create<top::ReshapeOp>(name_loc, newType, op->getOperands());
    // 1. do pool at last 2 dim
    for (int i = 1; i < 3; i++) {
      _kernel.push_back(kernel->at(i));
      _strides.push_back(strides->at(i));
      _pad.push_back(pads->at(i));
    }
    for (int i = 4; i < 6; i++) {
      _pad.push_back(pads->at(i));
    }
    auto dims = input_shape.size();
    tmp_shape0[2] = output_shape[dims - 2];
    tmp_shape0[3] = output_shape[dims - 1];
    newType = RankedTensorType::get(tmp_shape0, type);
    name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_0"));
    auto newOp0 = rewriter.create<top::MaxPoolOp>(
        name_loc, newType, ValueRange{reshapeOp.getOutput()}, op->getAttrs());
    newOp0->setAttr("kernel_shape", rewriter.getI64ArrayAttr(_kernel));
    newOp0->setAttr("strides", rewriter.getI64ArrayAttr(_strides));
    newOp0->setAttr("pads", rewriter.getI64ArrayAttr(_pad));
    // 2. trans [n*c f h w] -> [n*c h w f]
    std::vector<int64_t> order(tmp_shape0.size());
    std::iota(order.begin(), order.end(), 0);
    order.erase(order.begin() + tmp_shape0.size() - 3);
    order.push_back(tmp_shape0.size() - 3);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
    for (auto i : order) {
      tmp_shape1.push_back(tmp_shape0[i]);
    }
    newType = RankedTensorType::get(tmp_shape1, type);
    name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_trans1"));
    auto newOp1 = rewriter.create<top::PermuteOp>(
        name_loc, newType, ValueRange{newOp0.getOutput()}, attrs);
    // 3. do pool last dim
    tmp_shape1[tmp_shape1.size() - 1] = output_shape[output_shape.size() - 3];
    newType = RankedTensorType::get(tmp_shape1, type);
    name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_1"));
    auto newOp2 = rewriter.create<top::MaxPoolOp>(
        name_loc, newType, ValueRange{newOp1.getOutput()}, op->getAttrs());
    newOp2->setAttr("kernel_shape",
                    rewriter.getI64ArrayAttr({1, kernel->at(0)}));
    newOp2->setAttr("strides", rewriter.getI64ArrayAttr({1, strides->at(0)}));
    newOp2->setAttr("pads",
                    rewriter.getI64ArrayAttr({0, pads->at(0), 0, pads->at(3)}));
    // 4. trans back  [n c h w f] -> [n c f h w]
    name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_2"));
    newType = RankedTensorType::get(output_shape, type);
    std::iota(order.begin(), order.end(), 0);
    order.pop_back();
    order.insert(order.begin() + tmp_shape1.size() - 3, tmp_shape1.size() - 1);
    attrs.clear();
    attrs.push_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
    auto newOp3 = rewriter.create<top::PermuteOp>(
        name_loc, newType, ValueRange{newOp2.getOutput()}, attrs);
    // 5. reshape back
    newType = RankedTensorType::get(output_shape, type);
    auto reshape_backOp = rewriter.create<top::ReshapeOp>(
        op->getLoc(), newType, ValueRange{newOp3.getOutput()});

    rewriter.replaceOp(op, {reshape_backOp.getOutput()});
    return success();
  }
};

class ConvertMaxPoolWithMaskOp
    : public OpRewriterPatternEx<top::MaxPoolWithMaskOp> {
public:
  ConvertMaxPoolWithMaskOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::MaxPoolWithMaskOp>(
            context, "ConvertMaxPoolWithMaskOp", benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::MaxPoolWithMaskOp op,
                                    PatternRewriter &rewriter) const override {
    auto kernel_shape = module::getI64Array(op.getKernelShape());
    assert(kernel_shape->size() == 2 &&
           kernel_shape->at(0) == kernel_shape->at(1));
    std::vector<NamedAttribute> attrs;
    for (auto &attr : op->getAttrs()) {
      attrs.emplace_back(attr);
    }

    // create max_pool op
    auto max_pool_op = rewriter.create<top::MaxPoolOp>(
        op->getLoc(), op.getOutput().getType().cast<RankedTensorType>(),
        ValueRange{op.getInput()}, attrs);
    rewriter.setInsertionPointAfter(max_pool_op);

    // create pool mask op
    attrs.clear();
    attrs.emplace_back(rewriter.getNamedAttr(
        "scale", rewriter.getI64IntegerAttr(kernel_shape->at(0))));
    std::string name = module::getName(op.getMask()).str() + "_convert";
    auto loc = NameLoc::get(rewriter.getStringAttr(name));
    auto input_shape = module::getShape(op.getInput());
    std::vector<int64_t> mask_shape = input_shape.vec();
    mask_shape[2] = align_up(mask_shape[2], kernel_shape->at(0));
    mask_shape[3] = align_up(mask_shape[3], kernel_shape->at(0));

    auto pool_mask_type =
        RankedTensorType::get(mask_shape, rewriter.getF32Type());
    auto pool_mask_op = rewriter.create<top::PoolMaskOp>(
        loc, pool_mask_type, ValueRange{op.getInput()}, attrs);
    op.getMask().replaceAllUsesWith(pool_mask_op.getOutput());
    rewriter.replaceOp(op, {max_pool_op.getResult(), pool_mask_op.getResult()});
    return success();
  }
};

class ConvertMaxUnpoolOp : public OpRewriterPatternEx<top::MaxUnpoolOp> {
public:
  ConvertMaxUnpoolOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::MaxUnpoolOp>(context, "ConvertMaxUnpoolOp",
                                              benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::MaxUnpoolOp op,
                                    PatternRewriter &rewriter) const override {
    auto mask_op = op.getMask().getDefiningOp();
    if (!isa<top::PoolMaskOp>(mask_op)) {
      return failure();
    }
    auto output_shape = module::getShape(op.getOutput());
    std::vector<int64_t> mask_shape;
    mask_shape = module::getShape(op.getMask());
    bool need_crop = false;
    if (mask_shape[3] != output_shape[3] || mask_shape[2] != output_shape[2]) {
      need_crop = true;
    }
    std::string max_unpool_name = module::getName(op.getOutput()).str();
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    std::string name = max_unpool_name + "_nearst";

    // create upsample op
    auto loc = NameLoc::get(rewriter.getStringAttr(name));
    operands.emplace_back(op.getInput());
    attrs.emplace_back(rewriter.getNamedAttr("scale_h", op.getScaleHAttr()));
    attrs.emplace_back(rewriter.getNamedAttr("scale_w", op.getScaleWAttr()));
    auto new_type = module::getTypeLike(op.getOutput(), mask_shape);
    auto upsample_op =
        rewriter.create<top::UpsampleOp>(loc, new_type, operands, attrs);

    // create mul op
    attrs.clear();
    operands.clear();
    if (need_crop) {
      name = max_unpool_name + "_multi";
    } else {
      name = max_unpool_name;
    }

    loc = NameLoc::get(rewriter.getStringAttr(name));
    operands.emplace_back(upsample_op);
    operands.emplace_back(op.getMask());
    auto mul_op = rewriter.create<top::MulOp>(loc, new_type, operands, attrs);

    if (need_crop) {
      // create crop op
      attrs.clear();
      operands.clear();
      name = max_unpool_name;
      loc = NameLoc::get(rewriter.getStringAttr(name));
      std::vector<int64_t> crop_offset(4, 0);
      std::vector<int64_t> steps(4, 1);
      std::vector<int64_t> ends(4, -1);
      attrs.emplace_back(rewriter.getNamedAttr(
          "offset",
          rewriter.getI64ArrayAttr(ArrayRef<int64_t>({crop_offset}))));
      attrs.emplace_back(rewriter.getNamedAttr(
          "steps", rewriter.getI64ArrayAttr(ArrayRef<int64_t>({steps}))));
      attrs.emplace_back(rewriter.getNamedAttr(
          "ends", rewriter.getI64ArrayAttr(ArrayRef<int64_t>({ends}))));
      operands.emplace_back(mul_op);
      auto none = module::getNoneOp(mul_op);
      operands.push_back(none);
      operands.push_back(none);
      operands.push_back(none);
      auto crop_op = rewriter.create<top::SliceOp>(
          loc, op.getOutput().getType().cast<RankedTensorType>(), operands,
          attrs);
      rewriter.replaceOp(op, crop_op);
    } else {
      rewriter.replaceOp(op, mul_op);
    }
    return success();
  }
};

class ConvertPixelNormOp : public OpRewriterPatternEx<top::PixelNormOp> {
public:
  ConvertPixelNormOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::PixelNormOp>(context, "ConvertPixelNormOp",
                                              benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::PixelNormOp op,
                                    PatternRewriter &rewriter) const override {
    auto shape = module::getShape(op.getInput());
    bool has_weight = true, has_bias = true;
    if (!op.getWeight().getType().isa<mlir::NoneType>()) {
      has_weight = false;
    }
    if (!op.getBias().getType().isa<mlir::NoneType>()) {
      has_bias = false;
    }
    std::vector<int64_t> new_shape;
    // (NCHW) -> (NHWC)
    std::vector<int64_t> _order(shape.size());
    std::vector<int64_t> order;
    std::iota(_order.begin(), _order.end(), 0);
    int32_t axis = 1;
    for (int32_t i = 0; i < _order.size(); i++) {
      if (i == axis) {
        continue;
      }
      order.emplace_back(_order[i]);
    }
    order.emplace_back(_order[axis]);
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    operands.emplace_back(op.getInput());
    auto inEltType = module::getElementType(op.getInput());
    auto outEltType = module::getElementType(op.getOutput());
    std::string op_name = module::getName(op.getOutput()).str();
    auto name0 = NameLoc::get(rewriter.getStringAttr(op_name + "_permute0"));
    auto name1 = NameLoc::get(rewriter.getStringAttr(op_name + "_permute1"));
    auto name2 = NameLoc::get(rewriter.getStringAttr(op_name + "_transposed"));
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(order)));
    for (uint32_t i = 0; i < order.size(); ++i) {
      new_shape.push_back(shape[order[i]]);
    }
    auto permuteType = RankedTensorType::get(new_shape, inEltType);
    auto permuteOp =
        rewriter.create<top::PermuteOp>(name0, permuteType, operands, attrs);
    operands.clear();
    attrs.clear();
    operands.emplace_back(permuteOp.getOutput());
    auto ic = new_shape[new_shape.size() - 1];
    if (has_weight) {
      std::vector<int64_t> wshape = module::getShape(op.getWeight());
      assert(ic == wshape[1]);
      op.getWeight().setType(
          RankedTensorType::get({ic}, module::getStorageType(op.getWeight())));
    } else {
      operands.emplace_back(op.getWeight());
    }
    if (has_bias) {
      std::vector<int64_t> bshape = module::getShape(op.getBias());
      assert(ic == bshape[1]);
      op.getBias().setType(
          RankedTensorType::get({ic}, module::getStorageType(op.getBias())));
    } else {
      operands.emplace_back(op.getBias());
    }
    attrs.emplace_back(rewriter.getNamedAttr("eps", op.getEpsAttr()));
    attrs.emplace_back(rewriter.getNamedAttr(
        "axis", rewriter.getSI32IntegerAttr(new_shape.size() - 1)));
    attrs.emplace_back(rewriter.getNamedAttr("normalized_shape",
                                             rewriter.getI64ArrayAttr({ic})));
    auto outType = RankedTensorType::get(new_shape, outEltType);
    auto layerNormOp =
        rewriter.create<top::LayerNormOp>(name2, outType, operands, attrs);
    operands.clear();
    attrs.clear();
    operands.emplace_back(layerNormOp.getOutput());
    // (NHWC) -> (NCHW)
    axis = _order.back();
    _order.pop_back();
    _order.insert(_order.begin() + 1, axis);
    attrs.emplace_back(
        rewriter.getNamedAttr("order", rewriter.getI64ArrayAttr(_order)));
    permuteType = RankedTensorType::get(shape, outEltType);
    permuteOp =
        rewriter.create<top::PermuteOp>(name1, permuteType, operands, attrs);
    rewriter.replaceOp(op, permuteOp);
    return success();
  }
};

class ConvertDivOp : public OpRewriterPatternEx<top::DivOp> {
public:
  static std::map<std::string, Operation *> reciprocal_name_ops;

  ConvertDivOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::DivOp>(context, "ConvertDivOp", benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::DivOp op,
                                    PatternRewriter &rewriter) const override {
    std::vector<Value> operands;
    auto input_shape2 = module::getShape(op.getInputs()[1]);
    auto input1Op = op.getInputs()[1].getDefiningOp();
    auto weight_op = dyn_cast<top::WeightOp>(input1Op);
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
    attrs.emplace_back(
        rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
    operands.emplace_back(op.getInputs()[0]);
    if (weight_op) {
      assert(weight_op);
      auto const_f32 = weight_op.read<float>();
      for (auto &const_value : *const_f32) {
        const_value = 1 / const_value;
      }
      auto weight_type = weight_op.getType().cast<RankedTensorType>();
      auto new_weight_operand =
          top::WeightOp::create(op, "weight", *const_f32, weight_type);
      operands.emplace_back(new_weight_operand);
      rewriter.replaceOpWithNewOp<top::MulConstOp>(
          op.getOperation(), op.getOutput().getType().cast<RankedTensorType>(),
          operands, attrs);
    } else {
      rewriter.setInsertionPointAfterValue(op.getInputs()[1]);
      std::string name =
          module::getName(op.getInputs()[1]).str() + "_reciprocal";
      if (reciprocal_name_ops.find(name) == reciprocal_name_ops.end()) {
        auto loc = NameLoc::get(rewriter.getStringAttr(name));
        std::vector<NamedAttribute> reci_attrs;
        reci_attrs.emplace_back(
            rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(1.0)));
        auto reciprocal_type =
            RankedTensorType::get(input_shape2, rewriter.getF32Type());
        auto reciprocal_op = rewriter.create<top::ReciprocalOp>(
            loc, reciprocal_type, ValueRange{op.getInputs()[1]}, reci_attrs);
        reciprocal_name_ops[name] = reciprocal_op.getOperation();
        operands.emplace_back(reciprocal_op.getOutput());
      } else {
        auto reciprocal_op =
            dyn_cast_or_null<top::ReciprocalOp>(reciprocal_name_ops[name]);
        assert(reciprocal_op);
        operands.emplace_back(reciprocal_op.getOutput());
      }
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<top::MulOp>(
          op.getOperation(), op.getOutput().getType().cast<RankedTensorType>(),
          operands, attrs);
    }
    return success();
  }
};
std::map<std::string, Operation *> ConvertDivOp::reciprocal_name_ops;

class ConvertSqrtOp : public OpRewriterPatternEx<top::SqrtOp> {
public:
  ConvertSqrtOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::SqrtOp>(context, "ConvertSqrtOp", benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::SqrtOp op,
                                    PatternRewriter &rewriter) const override {
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    Value input = op.getInput();
    operands.emplace_back(input);
    attrs.emplace_back(
        rewriter.getNamedAttr("exponent", rewriter.getF64FloatAttr(0.5)));
    rewriter.replaceOpWithNewOp<top::PowOp>(
        op, op.getOutput().getType().cast<RankedTensorType>(), operands, attrs);
    return success();
  }
};

static int is_bcast(top::SubOp op) {
  int bcast = 0;
  auto shape0 = module::getShape(op.getInputs()[0]);
  auto shape1 = module::getShape(op.getInputs()[1]);
  auto prod0 = std::accumulate(shape0.begin(), shape0.end(), 1,
                               std::multiplies<int64_t>());
  auto prod1 = std::accumulate(shape1.begin(), shape1.end(), 1,
                               std::multiplies<int64_t>());
  auto sub = prod0 - prod1;
  if (sub < 0) {
    bcast = 1; // left bcast
  } else if (sub > 0) {
    bcast = 2; // right bcast
  }
  if (bcast) {
    auto len = std::min(shape0.size(), shape1.size());
    for (int i = 0; i < len; i++) {
      int dim_a = shape0[shape0.size() - 1 - i];
      int dim_b = shape1[shape1.size() - 1 - i];
      if (dim_a != dim_b &&
          ((sub > 0 && dim_b != 1) || (sub < 0 && dim_a != 1))) {
        llvm_unreachable("Broadcast dim should be 1");
      }
    }
  }
  return bcast;
}

class ConvertSubOp : public OpRewriterPatternEx<top::SubOp> {
public:
  ConvertSubOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::SubOp>(context, "ConvertSubOp", benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::SubOp op,
                                    PatternRewriter &rewriter) const override {
    // if not bcast,convert it to AddOp
    // if left_bcast, convert it to MulConstOp + AddOp
    // if right_bcast, not convert
    assert(op.getNumOperands() == 2);
    int bcast = is_bcast(op);
    auto coeff_v = module::getF64Array(op.getCoeff(), 2, 1.0);
    assert(coeff_v->at(0) == 1 && coeff_v->at(1) == 1);
    std::vector<NamedAttribute> attrs;
    if (bcast == 0) {
      attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
      attrs.push_back(
          rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
      attrs.push_back(
          rewriter.getNamedAttr("coeff", rewriter.getF64ArrayAttr({1., -1.})));
      rewriter.replaceOpWithNewOp<top::AddOp>(
          op, op.getOutput().getType().cast<RankedTensorType>(),
          op.getOperands(), attrs);
      return success();
    } else if (bcast == 1) {
      auto left_operand = op.getOperands()[0];
      auto right_operand = op.getOperands()[1];
      assert(!module::isWeight(right_operand));
      rewriter.setInsertionPointAfterValue(right_operand);
      std::vector<Value> operands;
      attrs.emplace_back(
          rewriter.getNamedAttr("const_val", rewriter.getF64FloatAttr(-1.0)));
      operands.emplace_back(right_operand);
      std::string name = module::getName(op.getOutput()).str();
      auto loc1 = NameLoc::get(rewriter.getStringAttr(name + "_mulconst"));
      auto type1 = right_operand.getType().cast<RankedTensorType>();
      auto mulconstOp =
          rewriter.create<top::MulConstOp>(loc1, type1, operands, attrs);
      auto out1 = mulconstOp.getOutput();
      attrs.clear();
      operands.clear();
      attrs.push_back(rewriter.getNamedAttr("do_relu", op.getDoReluAttr()));
      attrs.push_back(
          rewriter.getNamedAttr("relu_limit", op.getReluLimitAttr()));
      operands.emplace_back(left_operand);
      operands.emplace_back(out1);
      rewriter.setInsertionPoint(op);
      rewriter.replaceOpWithNewOp<top::AddOp>(
          op, op.getOutput().getType().cast<RankedTensorType>(), operands,
          attrs);
      return success();

    } else if (bcast == 2) {
      return failure();
    }
    return success();
  }
};

class ConvertUpsampleOp : public OpRewriterPatternEx<top::UpsampleOp> {
public:
  ConvertUpsampleOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::UpsampleOp>(context, "ConvertUpsampleOp",
                                             benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::UpsampleOp op,
                                    PatternRewriter &rewriter) const override {
    int64_t scale_h = op.getScaleH();
    int64_t scale_w = op.getScaleW();

    if (scale_h >= 16 || scale_w >= 16) {
      return failure();
    }
    auto input_shape = module::getShape(op.getInput());
    int64_t g = input_shape[1];
    int64_t oc = input_shape[1] / g;
    int64_t ic = input_shape[1] / g;
    int64_t h = scale_h;
    int64_t w = scale_w;

    int64_t count = g * oc * ic * h * w;
    std::vector<float> filter(count, 1);
    std::vector<int64_t> filter_shape;
    if (g != 1) {
      filter_shape.emplace_back(g);
    }
    filter_shape.emplace_back(oc);
    filter_shape.emplace_back(ic);
    filter_shape.emplace_back(h);
    filter_shape.emplace_back(w);

    std::string op_name = module::getName(op.getOutput()).str();
    auto filter_type =
        RankedTensorType::get(filter_shape, rewriter.getF32Type());
    auto filter_op =
        top::WeightOp::create(op, op_name + "filter", filter, filter_type);

    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(rewriter.getNamedAttr("kernel_shape",
                                             rewriter.getI64ArrayAttr({h, w})));
    attrs.emplace_back(rewriter.getNamedAttr(
        "strides", rewriter.getI64ArrayAttr({scale_h, scale_w})));
    attrs.emplace_back(
        rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr({0, 0, 0, 0})));
    attrs.emplace_back(
        rewriter.getNamedAttr("dilations", rewriter.getI64ArrayAttr({1, 1})));
    attrs.emplace_back(
        rewriter.getNamedAttr("inserts", rewriter.getI64ArrayAttr({0, 0})));
    attrs.emplace_back(
        rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(g)));

    std::vector<Value> operands;
    operands.emplace_back(op.getInput());
    operands.emplace_back(filter_op);
    operands.emplace_back(module::getNoneOp(op));
    rewriter.replaceOpWithNewOp<top::DeconvOp>(
        op, op.getOutput().getType().cast<RankedTensorType>(), operands, attrs);
    return success();
  }
};

class ConvertWhereOp : public OpRewriterPatternEx<top::WhereOp> {
public:
  Value genLeftOpd(top::WhereOp &op, Value left, Value right, bool is_const,
                   double const_v, double out_thr, bool isCali,
                   std::vector<int64_t> &out_shape, std::string &name,
                   std::string ext, PatternRewriter &rewriter) const {
    if (module::isWeight(left) && module::isWeight(right)) {
      auto constOp = dyn_cast<top::WeightOp>(left.getDefiningOp());
      auto constF32 = constOp.read<float>();
      std::vector<float> data(constF32->size());
      if (is_const) {
        if (std::isinf(const_v) && const_v < 0) {
          const_v = -std::pow(2, 22);
        }
        for (int i = 0; i < constF32->size(); ++i) {
          data[i] = const_v * constF32->at(i);
        }
      } else {
        auto const_x = dyn_cast<top::WeightOp>(right.getDefiningOp());
        auto x_f32 = const_x.read<float>();
        assert(x_f32->size() == constF32->size());
        for (int i = 0; i < constF32->size(); ++i) {
          float x = x_f32->at(i);
          if (std::isinf(x) && x < 0) {
            x = -std::pow(2, 22);
          }
          data[i] = x * constF32->at(i);
        }
      }
      auto type = RankedTensorType::get({out_shape}, rewriter.getF32Type());
      auto newWeight = top::WeightOp::create(
          op, module::getName(op.getOutput()).str() + ext, data, type);
      return newWeight;
    } else {
      Type out_type;
      if (isCali) {
        double r_thr;
        if (module::isWeight(right)) {
          r_thr = out_thr;
        } else {
          auto r_type = module::getCalibratedType(right);
          r_thr = r_type.getMax();
        }
        auto caliType = quant::CalibratedQuantizedType::get(
            rewriter.getF32Type(), -r_thr, r_thr);
        out_type = RankedTensorType::get(out_shape, caliType);
      } else {
        out_type = RankedTensorType::get(out_shape, rewriter.getF32Type());
      }
      std::vector<Value> operands;
      std::vector<NamedAttribute> attrs;
      operands.emplace_back(left);
      operands.emplace_back(right);
      auto loc = NameLoc::get(rewriter.getStringAttr(name + ext));
      auto mulOp = rewriter.create<top::MulOp>(loc, out_type, operands, attrs);
      return mulOp.getOutput();
    }
  }

  Value genRightOpd(top::WhereOp &op, Value left, Value right, bool is_const,
                    double const_v, double out_thr, bool isCali,
                    std::vector<int64_t> &out_shape, std::string &name,
                    std::string ext, PatternRewriter &rewriter) const {
    if (module::isWeight(left) && module::isWeight(right)) {
      auto constOp = dyn_cast<top::WeightOp>(left.getDefiningOp());
      auto constF32 = constOp.read<float>();
      std::vector<float> data(constF32->size());
      if (is_const) {
        if (std::isinf(const_v) && const_v < 0) {
          const_v = -std::pow(2, 22) + 1;
        }
        for (int i = 0; i < constF32->size(); ++i) {
          data[i] = (1 - const_v) * constF32->at(i);
        }
      } else {
        auto const_x = dyn_cast<top::WeightOp>(right.getDefiningOp());
        auto x_f32 = const_x.read<float>();
        assert(x_f32->size() == constF32->size());
        for (int i = 0; i < constF32->size(); ++i) {
          float x = x_f32->at(i);
          if (std::isinf(x) && x < 0) {
            x = -std::pow(2, 22) + 1;
          }
          data[i] = (1 - x) * constF32->at(i);
        }
      }
      auto type = RankedTensorType::get({out_shape}, rewriter.getF32Type());
      auto newWeight = top::WeightOp::create(
          op, module::getName(op.getOutput()).str() + ext, *constF32, type);
      return newWeight;
    } else {
      Type out_type;
      if (isCali) {
        double r_thr;
        if (module::isWeight(right)) {
          r_thr = out_thr;
        } else {
          auto r_type = module::getCalibratedType(right);
          r_thr = r_type.getMax();
        }
        auto caliType = quant::CalibratedQuantizedType::get(
            rewriter.getF32Type(), -r_thr, r_thr);
        out_type = RankedTensorType::get(out_shape, caliType);
      } else {
        out_type = RankedTensorType::get(out_shape, rewriter.getF32Type());
      }
      std::vector<Value> operands;
      std::vector<NamedAttribute> attrs;
      operands.emplace_back(left);
      operands.emplace_back(right);
      auto loc = NameLoc::get(rewriter.getStringAttr(name + ext));
      auto mulOp = rewriter.create<top::MulOp>(loc, out_type, operands, attrs);
      auto out = mulOp.getOutput();

      // create input[2] - input[0] * input[2]
      attrs.clear();
      operands.clear();
      operands.emplace_back(right);
      operands.emplace_back(out);
      auto loc3 = NameLoc::get(rewriter.getStringAttr(name + "_sub1"));
      auto subOp1 =
          rewriter.create<top::SubOp>(loc3, out_type, operands, attrs);

      return subOp1.getOutput();
    }
  }

  ConvertWhereOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::WhereOp>(context, "ConvertWhereOp", benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::WhereOp op,
                                    PatternRewriter &rewriter) const override {
    // out = input[0] * input[1] + (1 - input[0]) * input[2]
    Value ori_out = op.getOutput();
    std::vector<int64_t> output_shape = module::getShape(ori_out);
    int32_t out_num = module::getNumElements(ori_out);
    auto add_weight = [&](float val, std::string name, int32_t idx) {
      auto type = RankedTensorType::get({output_shape}, rewriter.getF32Type());
      auto newWeight = top::WeightOp::create(
          op, module::getName(op.getOutput()).str() + name,
          std::vector<float>(out_num, val), type);
      op.setOperand(idx, newWeight);
    };
    if (op.getXIsConst()) {
      add_weight(op.getXConstVal().convertToDouble(), "_x", 1);
    }
    if (op.getYIsConst()) {
      add_weight(op.getYConstVal().convertToDouble(), "_y", 2);
    }
    Value input0 = op.getOperand(0); // cond
    Value input1 = op.getOperand(1); // true branch
    Value input2 = op.getOperand(2); // false branch
    std::string name = module::getName(ori_out).str();
    std::vector<int64_t> input0_shape = module::getShape(input0);
    std::vector<int64_t> input1_shape = module::getShape(input1);
    std::vector<int64_t> input2_shape = module::getShape(input2);
    // cv18xx only support one operand broadcast now.
    assert((input0_shape == output_shape || input1_shape == output_shape ||
            input2_shape == output_shape));
    bool isCali = false;
    double out_thr;
    if (module::isCalibratedType(ori_out)) {
      isCali = true;
      auto otype = module::getCalibratedType(ori_out);
      out_thr = otype.getMax();
    }
    rewriter.setInsertionPointAfterValue(ori_out);
    auto out1 = genLeftOpd(op, input0, input1, op.getXIsConst(),
                           op.getXConstVal().convertToDouble(), out_thr, isCali,
                           output_shape, name, "_true", rewriter);

    auto out2 = genRightOpd(op, input0, input2, op.getYIsConst(),
                            op.getYConstVal().convertToDouble(), out_thr,
                            isCali, output_shape, name, "_false", rewriter);

    // create (input[0] * input[1]) + (input[2] - input[0] * input[2])
    std::vector<Value> operands;
    std::vector<NamedAttribute> attrs;
    operands.emplace_back(out1);
    operands.emplace_back(out2);
    rewriter.setInsertionPointAfterValue(out2);
    auto loc4 = NameLoc::get(rewriter.getStringAttr(name));
    RankedTensorType type4;
    if (isCali) {
      auto caliType = quant::CalibratedQuantizedType::get(rewriter.getF32Type(),
                                                          -out_thr, out_thr);
      type4 = RankedTensorType::get(output_shape, caliType);
    } else {
      type4 = RankedTensorType::get(output_shape, rewriter.getF32Type());
    }
    auto add2Op = rewriter.create<top::AddOp>(loc4, type4, operands, attrs);
    auto out4 = add2Op.getOutput();
    rewriter.replaceAllUsesWith(ori_out, out4);
    rewriter.eraseOp(op);

    return success();
  }
};

class ConvertClipOp : public OpRewriterPatternEx<top::ClipOp> {
public:
  ConvertClipOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::ClipOp>(context, "ConvertClipOp", benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::ClipOp op,
                                    PatternRewriter &rewriter) const override {
    auto input = op.getOperand();
    if (module::isCalibratedType(input)) {
      auto formerOp = input.getDefiningOp();
      double min = op.getMin().convertToDouble();
      double max = op.getMax().convertToDouble();
      if (min != 0.0 || !formerOp->hasAttr("do_relu")) {
        return failure();
      }
      auto input_type = input.getType().cast<RankedTensorType>();
      double input_max = module::getCalibratedType(input).getMax();
      double new_max = std::min(max, input_max);
      auto newCaliType = quant::CalibratedQuantizedType::get(
          rewriter.getF32Type(), -new_max, new_max);
      auto newType = RankedTensorType::get(input_type.getShape(), newCaliType);
      auto out_name = module::getName(op.getOutput()).str();
      auto input_loc = NameLoc::get(rewriter.getStringAttr(out_name));
      formerOp->setLoc(input_loc);
      formerOp->setAttr("do_relu", rewriter.getBoolAttr(true));
      input.setType(newType);
      rewriter.replaceOp(op, {input});
      return success();
    }
    return failure();
  }
};

class SplitReduceOp : public OpRewriterPatternEx<top::ReduceOp> {
public:
  SplitReduceOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::ReduceOp>(context, "SplitReduceOp", benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::ReduceOp op,
                                    PatternRewriter &rewriter) const override {
    auto axes_val = module::getI64Array(op.getAxes());
    int num_axes = axes_val->size();
    if (num_axes == 0)
      return failure();
    std::vector<std::vector<int64_t>> axes_slice;
    std::vector<int64_t> _axes = {axes_val->at(0)};
    for (int i = 1; i < num_axes; i++) {
      auto pre_axis = axes_val->at(i - 1);
      auto cur_axis = axes_val->at(i);
      if (cur_axis != pre_axis + 1) {
        axes_slice.push_back(_axes);
        _axes.clear();
      }
      _axes.push_back(cur_axis);
    }
    axes_slice.push_back(_axes);
    if (axes_slice.size() <= 1)
      return failure();

    auto input = op.getInput();
    auto input_shape = module::getShape(op.getInput()).vec();
    auto keep_dims = op.getKeepdims();
    std::string name = module::getName(op.getResult()).str();
    auto elt_type = module::getElementType(op.getOutput());
    for (int i = axes_slice.size() - 1; i >= 0; i--) {
      auto _axes = axes_slice[i];
      auto out_shape = input_shape;
      for (int j = _axes.size() - 1; j >= 0; j--) {
        auto idx = _axes[j];
        if (keep_dims)
          out_shape[idx] = 1;
        else
          out_shape.erase(out_shape.begin() + idx);
      }
      // creat ReduceOp
      auto loc = NameLoc::get(rewriter.getStringAttr(
          name + (i != 0 ? "_" + std::to_string(i) : "")));
      auto type = RankedTensorType::get(out_shape, elt_type);
      auto reduce_op = rewriter.create<top::ReduceOp>(
          loc, type, ValueRange{input}, op->getAttrs());
      reduce_op->setAttr("axes", rewriter.getI64ArrayAttr(_axes));
      input = reduce_op.getResult();
      input_shape = out_shape;
    }
    rewriter.replaceAllUsesWith(op.getResult(), input);
    return success();
  }
};

class ReshapeArgOp : public OpRewriterPatternEx<top::ArgOp> {
public:
  ReshapeArgOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::ArgOp>(context, "ReshapeArgOp", benifit) {}

protected:
  LogicalResult matchAndRewriteImpl(top::ArgOp op,
                                    PatternRewriter &rewriter) const override {
    auto succ = false;
    for (auto out : op->getResults()) {
      if (!module::isNone(out) && out.hasOneUse()) {
        auto nextOp = module::getNextOp(op);
        if (isa<top::ReshapeOp, top::UnsqueezeOp, top::SqueezeOp>(nextOp)) {
          out.setType(nextOp->getResult(0).getType());
          rewriter.replaceAllUsesWith(nextOp->getResult(0), out);
          succ = true;
        }
      }
    }
    return succ ? success() : failure();
  }
};

template <typename OpTy>
struct ConvertPoolOp : public OpRewriterPatternEx<OpTy> {
public:
  ConvertPoolOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<OpTy>(context, "ConvertPoolOp", benifit) {}

protected:
  mlir::LogicalResult
  matchAndRewriteImpl(OpTy op, mlir::PatternRewriter &rewriter) const override {
    if (!op.getDoRelu()) {
      return failure();
    }
    rewriter.setInsertionPointAfter(op);
    std::vector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("relu_limit", rewriter.getF64FloatAttr(-1.)));
    auto newOp = rewriter.create<top::ReluOp>(op->getLoc(), op.getType(),
                                              op->getResults(), attrs);
    auto op_name = module::getName(op.getResult()).str();
    std::string lower = op_name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    size_t pos = lower.find("relu");
    if (pos != std::string::npos) {
      op_name.erase(pos, op_name.length() - pos);
    }
    op->setAttr("do_relu", rewriter.getBoolAttr(false));
    op->setLoc(NameLoc::get(rewriter.getStringAttr(op_name + "_pool")));
    rewriter.replaceAllUsesExcept(op->getResult(0), newOp.getOutput(), newOp);

    return success();
  }
};

class RemoveUnuseOutput : public OpRewriterPatternEx<top::TopKOp> {
public:
  RemoveUnuseOutput(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::TopKOp>(context, "RemoveUnuseOutput",
                                         benifit) {}

  LogicalResult matchAndRewriteImpl(top::TopKOp op,
                                    PatternRewriter &rewriter) const override {
    module::getModuleOp()->dump();
    for (Value out : op.getResults()) {
      if (out.getUsers().empty()) {
        out.setType(mlir::NoneType::get(rewriter.getContext()));
      }
    }
    module::getModuleOp()->dump();
    return success();
  }
};

class convertScale5dOp : public OpRewriterPatternEx<top::ScaleOp> {
public:
  convertScale5dOp(mlir::MLIRContext *context, int benifit)
      : OpRewriterPatternEx<top::ScaleOp>(context, "convertScale5dOp",
                                          benifit) {}

  LogicalResult matchAndRewriteImpl(top::ScaleOp op,
                                    PatternRewriter &rewriter) const override {
    auto shape = module::getShape(op.getInput());
    if (shape.size() <= 4) {
      return failure();
    }

    int64_t in, ic, ih, iw;
    module::getNCHW(op.getOutput(), in, ic, ih, iw);
    auto op_name = module::getName(op.getResult()).str();
    auto type = module::getElementType(op.getOutput());
    auto newType = RankedTensorType::get({in, ic, ih, iw}, type);
    auto name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_reshape"));
    rewriter.setInsertionPoint(op);
    auto reshapeOp =
        rewriter.create<top::ReshapeOp>(name_loc, newType, op->getOperand(0));
    name_loc = NameLoc::get(rewriter.getStringAttr(op_name + "_scale4d"));
    std::vector<Value> operands;
    operands.emplace_back(reshapeOp.getOutput());
    for (int i = 1; i < op.getNumOperands(); i++) {
      operands.emplace_back(op.getOperand(i));
    }
    auto new_scaleOp = rewriter.create<top::ScaleOp>(name_loc, newType,
                                                     operands, op->getAttrs());
    rewriter.replaceOpWithNewOp<top::ReshapeOp>(
        op, op.getOutput().getType().cast<RankedTensorType>(),
        ValueRange{new_scaleOp.getOutput()});
    return mlir::success();
  }
};

} // namespace cv18xx

namespace top {
using namespace cv18xx;
void populateOptimizeCV18XXPatterns(RewritePatternSet *patterns) {
  patterns->add<MergeScale2Conv>(patterns->getContext(),
                                 /*PatternBenefit*/ 9);
  patterns->add<
      ConvertArgmaxOp, ReshapeArgOp, ConvertConvPading, ConvertConvDilation,
      ConvertConv2dToMatMul, ConvertAddConstOp, ConvertDivOp, ConvertGatherOp,
      convertScale5dOp, ConvertMaskedFillOp, ConvertMaxPoolWithMaskOp,
      ConvertMaxUnpoolOp, ConvertScaleOp, ConvertSubOp, ConvertInterpOp,
      ConvertUpsampleOp, ConvertWhereOp, ConvertMatMulWithRightTranspose,
      ConvertPixelNormOp, convertMaxPool3D, ConvertSqrtOp, ConvertAvgPoolOp,
      SplitReduceOp, ConvertPoolOp<top::AvgPoolOp>,
      ConvertPoolOp<top::MaxPoolOp>, ConvertPoolOp<top::MulConstOp>,
      patterns::SqueezeToReshapePattern, patterns::UnsqueezeToReshapePattern,
      ConvertClipOp, RemoveUnuseOutput>(patterns->getContext(), 8);
}
} // namespace top
} // namespace tpu_mlir
