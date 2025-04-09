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
    auto bias = op.getBias();
    auto rshitf = op.getRshift();
    auto multiplier = op.getMultiplier();
    auto output = op.getOutput();
    auto name = module::getName(op.getOutput()).str();

    auto inputShape = module::getShape(input);
    auto filterShape = module::getShape(filter);
    if (!(module::isMARS3() || module::isSGTPUV8()))
      return failure();
    // is depthwise conv
    int64_t groups = op.getGroup();
    int64_t ic = inputShape[1];
    int64_t oc = module::getShape(output)[1];
    if (!(groups == ic && ic == oc))
      return failure();
    if (op.getDoLeakyRelu())
      return failure();
    if (!(inputShape.size() == 4 && inputShape[1] >= 1024 &&
          (inputShape[2] <= 8 || inputShape[3] <= 8)))
      return failure();

    // cut input tensor close to 512
    const int64_t expectedPart = 512;
    const int64_t splitNumber =
        (inputShape[1] + expectedPart - 1) / expectedPart;
    const int64_t NPU_NUM = backend::Arch::NPU_NUM;

    auto splitAndAlign = [&](int64_t total, int split) -> std::vector<int64_t> {
      std::vector<int64_t> parts;
      int64_t base = total / split;
      int64_t part = align_up(base, NPU_NUM);
      // int64_t remainder = total % split;

      for (int i = 0; i < split - 1; ++i) {
        // int64_t part = base + (i < remainder ? 1 : 0);
        total -= part;
        parts.push_back(part);
      }
      parts.push_back(total);
      return parts;
    };

    auto inPerPart = splitAndAlign(inputShape[1], splitNumber);
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
      attrs.set("axes", rewriter.getI64ArrayAttr({1}));

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
      attrs.set("axes", rewriter.getI64ArrayAttr({1}));

      auto slice = rewriter.create<tpu::SliceOp>(loc, newType, operands, attrs);
      slicedFilters.push_back(slice.getOutput());

      ocPerPartSum += ocPerPart[i];
      rewriter.setInsertionPointAfterValue(slice.getOutput());
    }

    // Bias切片
    std::vector<Value> slicedBiases;
    if (module::isNone(bias)) {
      for (int i = 0; i < splitNumber; ++i) {
        slicedBiases.push_back(none);
      }
    } else {
      ocPerPartSum = 0;
      for (int i = 0; i < splitNumber; ++i) {
        auto loc = NameLoc::get(
            rewriter.getStringAttr(name + "_slice_bias_" + std::to_string(i)));

        auto newType = bias.getType().cast<RankedTensorType>().clone({
            1, ocPerPart[i], 1, 1 // 输出通道匹配 ocPerPart
        });

        std::vector<Value> operands;
        operands.push_back(bias);
        operands.push_back(none);
        operands.push_back(none);
        operands.push_back(none);
        operands.push_back(none);

        NamedAttrList attrs;
        attrs.set("offset", rewriter.getI64ArrayAttr({0, ocPerPartSum, 0, 0}));
        attrs.set("ends", rewriter.getI64ArrayAttr(
                              {1, ocPerPartSum + ocPerPart[i], 1, 1}));
        attrs.set("steps", rewriter.getI64ArrayAttr({1, 1, 1, 1}));
        attrs.set("axes", rewriter.getI64ArrayAttr({1}));

        if (module::getShape(bias).size() == 1) {
          newType = RankedTensorType::get(
              {ocPerPart[i]},
              bias.getType().cast<RankedTensorType>().getElementType());
          attrs.set("offset", rewriter.getI64ArrayAttr({ocPerPartSum}));
          attrs.set("ends",
                    rewriter.getI64ArrayAttr({ocPerPartSum + ocPerPart[i]}));
          attrs.set("steps", rewriter.getI64ArrayAttr({1}));
          attrs.set("axes", rewriter.getI64ArrayAttr({0}));
        }

        auto slice =
            rewriter.create<tpu::SliceOp>(loc, newType, operands, attrs);
        slicedBiases.push_back(slice.getOutput());

        ocPerPartSum += ocPerPart[i];
        rewriter.setInsertionPointAfterValue(slice.getOutput());
      }
    }

    i64_array_t old_rshift_v;
    if (rshitf.has_value()) {
      old_rshift_v = module::getI64Array(rshitf.value());
    }

    i64_array_t old_multiplier_v;
    if (multiplier.has_value()) {
      old_multiplier_v = module::getI64Array(multiplier.value());
    }

    // 创建分组卷积
    inPerPartSum = 0;
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
      if (rshitf.has_value()) {
        std::vector<int64_t> new_rshift_v(old_rshift_v->begin() + inPerPartSum,
                                          old_rshift_v->begin() + inPerPartSum +
                                              inPerPart[i]);
        attrs.set("rshift", rewriter.getI64ArrayAttr(new_rshift_v));
      }

      if (multiplier.has_value()) {
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

  bool shouldPrint(tpu::Conv2DOp op) const override { return false; }
};

void tpu::Conv2DOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<ConvSliceICPattern>(context);
}

} // namespace tpu
} // namespace tpu_mlir
