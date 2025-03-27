//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace tpu_mlir::top;

// Conv1dTo2d.cpp
struct Conv1dTo2d : public OpRewriterPatternEx<ConvOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  Conv1dTo2d(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ConvOp>(context, "Conv1dTo2d") {}

  LogicalResult matchAndRewriteImpl(ConvOp op,
                                    PatternRewriter &rewriter) const override {
    auto kernel = module::getI64Array(op.getKernelShape());
    if (kernel->size() != 1) {
      return failure();
    }
    std::vector<int64_t> vfilterShape = module::getShape(op.getFilter());
    vfilterShape.push_back(1);
    auto new_type = RankedTensorType::get(vfilterShape, rewriter.getF32Type());
    op.getFilter().setType(new_type);

    std::vector<int64_t> kernel_shape =
        *module::getI64Array(op.getKernelShape());
    kernel_shape.push_back(1);
    op.setKernelShapeAttr(rewriter.getI64ArrayAttr(kernel_shape));
    std::vector<int64_t> strides = *module::getI64Array(op.getStrides());
    strides.push_back(1);
    op.setStridesAttr(rewriter.getI64ArrayAttr(strides));

    auto pads_v = module::getI64Array(op.getPads());
    std::vector<int64_t> pads = {pads_v->at(0), 0, pads_v->at(1), 0};
    op.setPadsAttr(rewriter.getI64ArrayAttr(pads));

    std::vector<int64_t> dilations =
        *module::getI64Array(op.getDilations(), 1, 1);
    dilations.push_back(1);
    op.setDilationsAttr(rewriter.getI64ArrayAttr(dilations));

    return success();
  }
};

struct Conv3dTo2d : public OpRewriterPatternEx<ConvOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  Conv3dTo2d(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ConvOp>(context, "Conv3dTo2d") {}

  LogicalResult matchAndRewriteImpl(ConvOp op,
                                    PatternRewriter &rewriter) const override {
    auto p = op.parseParam();
    if (op.getKernelShape().size() != 3 || p.id != p.kd || p.od != 1) {
      return failure();
    }
    static int callCount = 0;
    auto in = op.getInput();
    auto out = op.getOutput();
    // in reshape to 4dim
    std::vector<int64_t> in_shape = {p.n, p.ic * p.id, p.ih, p.iw};
    auto newType = RankedTensorType::get(in_shape, module::getElementType(in));
    std::string in_name =
        module::getName(in).str() + "_r_To4Dim" + std::to_string(callCount);
    auto loc = NameLoc::get(rewriter.getStringAttr(in_name));
    rewriter.setInsertionPoint(op);
    auto rs1_op = rewriter.create<ReshapeOp>(loc, newType, ValueRange{in});
    op.setOperand(0, rs1_op.getOutput());
    // out reshape to 5dim
    auto outType = out.getType();
    std::string out_name =
        module::getName(out).str() + "_r_To5Dim" + std::to_string(callCount);
    loc = NameLoc::get(rewriter.getStringAttr(out_name));
    rewriter.setInsertionPointAfter(op);
    auto rs2_op = rewriter.create<ReshapeOp>(loc, outType, ValueRange{out});
    out.replaceAllUsesExcept(rs2_op.getOutput(), rs2_op);
    // conv 5d to 4d
    newType = RankedTensorType::get({p.n, p.oc * p.od, p.oh, p.ow},
                                    module::getElementType(out));
    out.setType(newType);
    op.setKernelShapeAttr(rewriter.getI64ArrayAttr({p.kh, p.kw}));
    op.setStridesAttr(rewriter.getI64ArrayAttr({p.sh, p.sw}));
    op.setDilationsAttr(rewriter.getI64ArrayAttr({p.dh, p.dw}));
    op.setPadsAttr(rewriter.getI64ArrayAttr({p.pht, p.pwl, p.phb, p.pwr}));
    auto kernel = op.getFilter();
    newType = RankedTensorType::get({p.oc, p.ic * p.kd / p.groups, p.kh, p.kw},
                                    module::getElementType(kernel));
    kernel.setType(newType);
    callCount++;
    return success();
  }
};

struct Conv3dTranspose : public OpRewriterPatternEx<ConvOp> {

  Conv3dTranspose(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ConvOp>(context, "Conv3dTranspose") {}

  LogicalResult matchAndRewriteImpl(ConvOp op,
                                    PatternRewriter &rewriter) const override {
    /* make sure it is a Conv3dOp and id == kd and ic == filter ic */
    auto p = op.parseParam();
    auto filter = op.getFilter();
    auto f_shape = module::getShape(filter);
    if (op.getKernelShape().size() != 3 || p.id != p.kd || f_shape[1] != p.ic) {
      return failure();
    }
    /* make sure the input comes from PermuteOp */
    auto in = op.getInput();
    auto tp = dyn_cast<PermuteOp>(in.getDefiningOp());
    if (!tp) {
      return failure();
    }
    /* make sure the input is the only output of PermuteOp */
    if (!tp.getOutput().hasOneUse()) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }
    /* make sure the PermuteOp is between dim 1 and 2 */
    std::vector<int64_t> ps = {0, 2, 1, 3, 4};
    auto order = module::getI64Array(tp.getOrder());
    if (*order != ps) {
      return failure();
    }
    /* transpose the filter */
    auto filter_op = filter.getDefiningOp<top::WeightOp>();
    auto filter_data = filter_op.read_as_float();
    auto filter_tp =
        std::make_shared<std::vector<float>>(filter_data->size(), 0);
    function_permute(filter_data->data(), filter_tp->data(), f_shape, ps);
    std::vector<int64_t> f_shape_tp = {f_shape[0], f_shape[2], f_shape[1],
                                       f_shape[3], f_shape[4]};
    /* get rid of PermuteOp */
    tp.getOutput().replaceAllUsesWith(
        tp.getInput()); // this replaces op.getInput() with tp.getInput().
    rewriter.eraseOp(tp);
    /* create a new weight for the transposed filter */
    rewriter.setInsertionPointAfter(op);
    auto weight = WeightOp::create_float(op, "transposed_weight", *filter_tp,
                                         f_shape_tp, storage_type);
    /* change the attr of conv3d op */
    op.setOperand(
        1,
        weight); // op.setOperand vs op->setOperand: in this case both OK. This
                 // replaces op.getFilter() with the transposed filter $weight.
    rewriter.eraseOp(filter_op); // remove unused WeightOp manually, optional
    op.setKernelShapeAttr(rewriter.getI64ArrayAttr({p.ic, p.kh, p.kw}));
    return success();
  }
};

struct Conv1x1Convkxk2dMerge : public OpRewriterPatternEx<ConvOp> {
  using OpRewriterPatternEx::OpRewriterPatternEx;

  Conv1x1Convkxk2dMerge(mlir::MLIRContext *context)
      : OpRewriterPatternEx<ConvOp>(context, "Conv1x1Convkxk2dMerge") {}

  LogicalResult matchAndRewriteImpl(ConvOp op,
                                    PatternRewriter &rewriter) const override {
    if (module::isUniformQuantized(op.getOutput())) {
      return failure();
    }
    auto kernel = module::getI64Array(op.getKernelShape());
    if (kernel->size() != 2) {
      return failure();
    }
    auto storage_type = module::getStorageType(op.getOutput());
    if (!storage_type.isF32() && !storage_type.isF16()) {
      return failure();
    }

    auto inputOperand = op.getInput();
    auto prevOp = inputOperand.getDefiningOp();
    auto prevConvOp = dyn_cast<ConvOp>(prevOp);
    if (!prevConvOp) {
      // There is no previous ConvOp
      return failure();
    } else {
      // 'prevConvOp' is the previous ConvOp
      std::vector<int64_t> pre_kernel_shape =
          *module::getI64Array(prevConvOp.getKernelShape());
      if (pre_kernel_shape.size() != 2 || pre_kernel_shape[0] != 1 ||
          pre_kernel_shape[1] != 1) {
        return failure();
      }

      if (!prevConvOp.getResult().hasOneUse()) {
        return failure();
      }

      if (prevConvOp.getGroup() != 1 || op.getGroup() != 1) {
        return failure();
      }

      // can't have padding now
      auto p = op.parseParam();
      auto prep = prevConvOp.parseParam();
      if (p.pht != 0 || p.phb != 0 || p.pwl != 0 || p.pwr != 0 ||
          prep.pht != 0 || prep.phb != 0 || prep.pwl != 0 || prep.pwr != 0) {
        return failure();
      }

      if (p.do_relu || prep.do_relu) {
        return failure();
      }

      auto Filterop = op.getFilter().getDefiningOp<top::WeightOp>();
      auto Filterop_f32 = Filterop.read_as_float();
      std::vector<int64_t> filterShape = module::getShape(op.getFilter());

      auto preFilterop = prevConvOp.getFilter().getDefiningOp<top::WeightOp>();
      auto preFilterop_f32 = preFilterop.read_as_float();
      std::vector<int64_t> prefilterShape =
          module::getShape(prevConvOp.getFilter());

      // transform prefilterShape, exchange dim ic and oc
      std::vector<int64_t> ps = {1, 0, 2, 3};
      auto prefilter_f32_tp =
          std::make_shared<std::vector<float>>(preFilterop_f32->size(), 0);
      function_permute(preFilterop_f32->data(), prefilter_f32_tp->data(),
                       prefilterShape, ps);

      // calculate merge filter's weight
      //  Convkxk = [E, D, K, K], Conv1x1_trans = [C, D, 1, 1], Conv1x1 = [D, C,
      //  1, 1]
      int E = filterShape.at(0), D = filterShape.at(1), K = filterShape.at(2),
          C = prefilterShape.at(1);
      int eckk = E * C * K * K;
      int ckk = C * K * K;
      int kk = K * K;
      auto filter_merge = std::make_shared<std::vector<float>>(size_t(eckk), 0);
      // TODO. Use pass to set option attr to the op before Canonicalize.
      // For the rare case: w0(63, 1536, 1) w1(18507, 63, 1) -> [18507, 1536, 1]
      int edkk = E * D * K * K;
      if ((float)eckk / (edkk + C * D) > 20 /* for this case */) {
        return failure();
      }

#pragma omp parallel for schedule(static, omp_schedule(eckk))
      for (int i = 0; i < eckk; ++i) {
        int e = i / ckk;
        int c = (i % ckk) / kk;
        int k1 = (i % kk) / K;
        int k2 = i % K;
        float sum = 0.0;
        for (int d = 0; d < D; ++d) {
          sum += (*Filterop_f32)[e * D * K * K + d * K * K + k1 * K + k2] *
                 (*prefilter_f32_tp)[c * D + d];
        }
        (*filter_merge)[e * C * K * K + c * K * K + k1 * K + k2] = sum;
      }

      auto bias_merge = std::make_shared<std::vector<float>>((size_t)E, 0);
      if (p.has_bias) {
        auto Biasop = op.getBias().getDefiningOp<top::WeightOp>();
        auto Biasop_f32 = Biasop.read_as_float();
        bias_merge = Biasop_f32;
      }

      if (prep.has_bias) {
        // calcute merge filter's bias
        auto preBiasop = prevConvOp.getBias().getDefiningOp<top::WeightOp>();
        auto preBiasop_f32 = preBiasop.read_as_float();
#pragma omp parallel for schedule(static, omp_schedule(E))
        for (int e = 0; e < E; ++e) {
          float sum = 0.0;
          for (int d = 0; d < D; ++d) {
            for (int u = 0; u < K; ++u) {
              for (int v = 0; v < K; ++v) {
                sum += (*preBiasop_f32)[d] *
                       (*Filterop_f32)[e * D * K * K + d * K * K + u * K + v];
              }
            }
          }
          (*bias_merge)[e] += sum;
        }
      }

      // remove prevConvOp
      prevConvOp.getOutput().replaceAllUsesWith(prevConvOp.getInput());
      rewriter.eraseOp(prevConvOp);

      // Set op
      // setup merige_filter_shape
      std::vector<int64_t> mfilterShape = module::getShape(op.getFilter());
      mfilterShape[1] = C;
      auto mnew_type =
          RankedTensorType::get(mfilterShape, rewriter.getF32Type());
      op.getFilter().setType(mnew_type);

      // setup merige_filter_weight
      auto mnew_op = WeightOp::create_float(
          op, "merge_filter_weight", *filter_merge, mfilterShape, storage_type);
      op->setOperand(1, mnew_op);

      // setup merige_bias_weight
      if (p.has_bias || prep.has_bias) {
        std::vector<int64_t> biasShape = {E};
        if (p.has_bias) {
          biasShape = module::getShape(op.getBias());
        }
        std::vector<int64_t> mbiasShape = biasShape;
        auto mbnew_type =
            RankedTensorType::get(mbiasShape, rewriter.getF32Type());
        auto mbnew_op = top::WeightOp::create(op, "merge_bias_weight",
                                              *bias_merge, mbnew_type);
        op->setOperand(2, mbnew_op);
      }
    }
    return success();
  }
};

void ConvOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
  results
      .insert<Conv3dTranspose, Conv3dTo2d, Conv1dTo2d, Conv1x1Convkxk2dMerge>(
          context);
}
