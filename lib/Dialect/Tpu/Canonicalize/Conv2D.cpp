#include "tpu_mlir/Backend/BM168x/MARS3.h"
#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/OpRewriterPatternEx.h"

using namespace llvm;

namespace tpu_mlir {
namespace tpu {

class ConvSliceICPattern : public OpRewriterPatternEx<tpu::Conv2DOp> {
public:
  ConvSliceICPattern(MLIRContext *context)
      : OpRewriterPatternEx<tpu::Conv2DOp>(context, "ConvSliceICPattern") {}

  LogicalResult matchAndRewriteImpl(tpu::Conv2DOp op,
                                    PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto filter = op.getFilter();
    auto originalBias = op.getOperand(2);
    auto output = op.getOutput();
    auto name = module::getName(op.getOutput()).str();

    auto inputShape = module::getShape(input);
    auto filterShape = module::getShape(filter);
    const int64_t cDim = 1;
    if (!(module::isMARS3() || module::isSGTPUV8()))
      return failure();
    // is depthwise conv
    int64_t groups = op.getGroup();
    int64_t ic = inputShape[cDim];
    int64_t oc = module::getShape(output)[cDim];
    if (!(groups == ic && ic == oc))
      return failure();
    if (op.getDoLeakyRelu())
      return failure();
    if (inputShape.size() != 4 || inputShape[cDim] <= 512)
      return failure();

    // cut input tensor close to 512
    const int64_t expectedPart = 512;
    const int64_t splitNumber =
        (inputShape[cDim] + expectedPart - 1) / expectedPart;
    const int64_t NPU_NUM = backend::Arch::NPU_NUM;

    auto splitAndAlign = [&](int64_t total, int split) -> std::vector<int64_t> {
      std::vector<int64_t> parts;
      int64_t base = total / split;
      int64_t remainder = total % split;

      for (int i = 0; i < split - 1; ++i) {
        int64_t part = base + (i < remainder ? 1 : 0);
        part = align_up(part, NPU_NUM);
        total -= part;
        parts.push_back(part);
      }
      parts.push_back(total);
      return parts;
    };

    auto inPerPart = splitAndAlign(inputShape[cDim], splitNumber);
    auto ocPerPart = splitAndAlign(filterShape[0], splitNumber);

    auto none = module::getNoneOp(op);

    rewriter.setInsertionPoint(op);
    // 切分输入tensor
    std::vector<Value> slicedInputs;
    int64_t inPerPartSum = 0;
    for (int i = 0; i < splitNumber; ++i) {
      auto loc = NameLoc::get(
          rewriter.getStringAttr(name + "_slice_input_" + std::to_string(i)));
      auto newType = input.getType().cast<RankedTensorType>().clone(
          {inputShape[0], inPerPart[i], inputShape[2], inputShape[3]});

      std::vector<Value> operands;
      operands.push_back(input);
      operands.push_back(none);
      operands.push_back(none);
      operands.push_back(none);
      operands.push_back(none);

      NamedAttrList attrs;
      attrs.set("offset", rewriter.getI64ArrayAttr({0, inPerPartSum, 0, 0}));
      attrs.set("ends", rewriter.getI64ArrayAttr(
                            {inputShape[0], inPerPartSum + inPerPart[i],
                             inputShape[2], inputShape[3]}));
      attrs.set("steps", rewriter.getI64ArrayAttr({1, 1, 1, 1}));
      attrs.set("axes", rewriter.getI64ArrayAttr({}));
      attrs.set("hasparamConvert_axes", rewriter.getI64ArrayAttr({}));

      auto slice = rewriter.create<tpu::SliceOp>(loc, newType, operands, attrs);
      slicedInputs.push_back(slice.getOutput());

      inPerPartSum += inPerPart[i];
      rewriter.setInsertionPointAfterValue(slice.getOutput());
    }

    // Filter切片
    std::vector<Value> slicedFilters;
    int64_t ocPerPartSum = 0;
    for (int i = 0; i < splitNumber; ++i) {
      auto loc = NameLoc::get(
          rewriter.getStringAttr(name + "_slice_filter_" + std::to_string(i)));
      auto newType = filter.getType().cast<RankedTensorType>().clone(
          {ocPerPart[i], filterShape[1], filterShape[2], filterShape[3]});

      std::vector<Value> operands;
      operands.push_back(filter);
      operands.push_back(none);
      operands.push_back(none);
      operands.push_back(none);
      operands.push_back(none);

      NamedAttrList attrs;
      attrs.set("offset", rewriter.getI64ArrayAttr({ocPerPartSum, 0, 0, 0}));
      attrs.set("ends", rewriter.getI64ArrayAttr(
                            {ocPerPartSum + ocPerPart[i], filterShape[1],
                             filterShape[2], filterShape[3]}));
      attrs.set("steps", rewriter.getI64ArrayAttr({1, 1, 1, 1}));
      attrs.set("axes", rewriter.getI64ArrayAttr({}));
      attrs.set("hasparamConvert_axes", rewriter.getI64ArrayAttr({}));

      auto slice = rewriter.create<tpu::SliceOp>(loc, newType, operands, attrs);
      slicedFilters.push_back(slice.getOutput());

      ocPerPartSum += ocPerPart[i];
      rewriter.setInsertionPointAfterValue(slice.getOutput());
    }

    // Bias切片
    auto biasOp = op.getBias().getDefiningOp();
    std::vector<Value> slicedBiases;
    if (!isa<top::NoneOp>(biasOp)) {
      ocPerPartSum = 0;
      for (int i = 0; i < splitNumber; ++i) {
        auto loc = NameLoc::get(
            rewriter.getStringAttr(name + "_slice_bias_" + std::to_string(i)));
        auto newType = originalBias.getType().cast<RankedTensorType>().clone({
            1, ocPerPart[i], 1, 1 // 输出通道匹配 ocPerPart
        });

        std::vector<Value> operands;
        operands.push_back(originalBias);
        operands.push_back(none);
        operands.push_back(none);
        operands.push_back(none);
        operands.push_back(none);

        NamedAttrList attrs;
        attrs.set("offset", rewriter.getI64ArrayAttr({0, ocPerPartSum, 0, 0}));
        attrs.set("ends", rewriter.getI64ArrayAttr(
                              {1, ocPerPartSum + ocPerPart[i], 1, 1}));
        attrs.set("steps", rewriter.getI64ArrayAttr({1, 1, 1, 1}));

        auto slice =
            rewriter.create<tpu::SliceOp>(loc, newType, operands, attrs);
        slicedBiases.push_back(slice.getOutput());

        ocPerPartSum += ocPerPart[i];
        rewriter.setInsertionPointAfterValue(slice.getOutput());
      }
    } else {
      for (int i = 0; i < splitNumber; ++i) {
        slicedBiases.push_back(none);
      }
    }

    inPerPartSum = 0;
    auto old_rshift = op.getRshift();
    auto old_rshift_v = module::getI64Array(old_rshift.value());
    auto old_multiplier = op.getMultiplier();
    auto old_multiplier_v = module::getI64Array(old_multiplier.value());
    // 创建分组卷积
    std::vector<Value> convResults;
    for (int i = 0; i < splitNumber; ++i) {
      auto loc = NameLoc::get(
          rewriter.getStringAttr(name + "_slice_conv_" + std::to_string(i)));
      auto outputType = output.getType().cast<RankedTensorType>().clone(
          {inputShape[0], ocPerPart[i], module::getShape(output)[2],
           module::getShape(output)[3]});

      std::vector<Value> operands;
      operands.push_back(slicedInputs[i]);
      operands.push_back(slicedFilters[i]);
      operands.push_back(slicedBiases[i]);

      NamedAttrList attrs = op->getAttrs();
      attrs.set("group", rewriter.getI64IntegerAttr(inPerPart[i]));

      // rshift also need to be sliced
      if (old_rshift.has_value()) {
        std::vector<int64_t> new_rshift_v(old_rshift_v->begin() + inPerPartSum,
                                          old_rshift_v->begin() + inPerPartSum +
                                              inPerPart[i]);
        attrs.set("rshift", rewriter.getI64ArrayAttr(new_rshift_v));
      }

      if (old_multiplier.has_value()) {
        std::vector<int64_t> new_multiplier_v(
            old_multiplier_v->begin() + inPerPartSum,
            old_multiplier_v->begin() + inPerPartSum + inPerPart[i]);
        attrs.set("multiplier", rewriter.getI64ArrayAttr(new_multiplier_v));
      }

      auto newConv =
          rewriter.create<tpu::Conv2DOp>(loc, outputType, operands, attrs);
      convResults.push_back(newConv.getOutput());

      inPerPartSum += inPerPart[i];
      rewriter.setInsertionPointAfterValue(newConv.getOutput());
    }

    // 拼接结果
    auto loc = NameLoc::get(rewriter.getStringAttr(name + "_concat"));
    NamedAttrList attrs;
    attrs.set("axis", rewriter.getSI32IntegerAttr(1));
    attrs.set("do_relu", rewriter.getBoolAttr(false));
    attrs.set("relu_limit", rewriter.getF64FloatAttr(-1.0));
    attrs.set("round_mode", rewriter.getStringAttr("HalfAwayFromZero"));

    auto concat = rewriter.create<tpu::ConcatOp>(loc, output.getType(),
                                                 convResults, attrs);

    rewriter.replaceOp(op, concat.getOutput());
    return success();
  }

  bool shouldPrint(tpu::Conv2DOp op) const override { return true; }
};

void tpu::Conv2DOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<ConvSliceICPattern>(context);
}

} // namespace tpu
} // namespace tpu_mlir
