//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

using namespace llvm;

namespace tpu_mlir {

namespace bm168x {

class LargePadConvPattern : public OpRewritePattern<tpu::Conv2DOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tpu::Conv2DOp op,
                                PatternRewriter &rewriter) const override {
    if (!(module::isBM1684Family() || module::isBM1684XFamily())) {
      return failure();
    }
    auto pads_v = module::getI64Array(op.getPads());
    auto pad_top = pads_v->at(0);
    auto pad_left = pads_v->size() > 2 ? pads_v->at(1) : 0;
    auto pad_bottom = pads_v->size() > 2 ? pads_v->at(2) : pads_v->at(1);
    auto pad_right = pads_v->size() > 2 ? pads_v->at(3) : 0;
    int64_t max_pad =
        std::max(std::max(pad_top, pad_bottom), std::max(pad_left, pad_right));
    const int64_t max_pad_threshold = 15;
    if (max_pad <= max_pad_threshold) {
      return failure();
    }
    llvm::SmallVector<int64_t> conv_paddings = {pad_top, pad_bottom, pad_left,
                                                pad_right};
    Value input_value = op->getOperand(0);
    std::string output_name = module::getName(op->getResult(0)).str();
    auto input_ele_type = module::getElementType(input_value);

    for (int64_t i = 0; i < max_pad / max_pad_threshold; i++) {
      std::string name_pad = output_name + "$pad" + std::to_string(i);
      auto loc_pad = NameLoc::get(rewriter.getStringAttr(name_pad));
      std::vector<Value> operands_pad;
      operands_pad.push_back(input_value);
      operands_pad.push_back(module::getNoneOp(op));
      operands_pad.push_back(module::getNoneOp(op));
      operands_pad.push_back(module::getNoneOp(op));
      std::vector<NamedAttribute> attrs_pad;
      // pad_paddings[0/1/4/5]: n/c paddings for new pad layer, are always 0
      // pad_paddings[2/3/6/7]: h/w paddings for new pad layer
      auto input_shape = module::getShape(input_value);
      llvm::SmallVector<int64_t> pad_paddings(input_shape.size() * 2, 0);
      int64_t pad_limit = (input_shape.size() == 3 ? 2 : 4);
      for (size_t j = 0; j < pad_limit; j++) {
        int padding = std::min(conv_paddings[j], max_pad_threshold);
        pad_paddings[(j < 2 ? 2 : 3) + (j % 2 == 0 ? 0 : input_shape.size())] = padding;
        conv_paddings[j] -= padding;
      }
      attrs_pad.push_back(rewriter.getNamedAttr(
          "paddings", rewriter.getI64ArrayAttr(pad_paddings)));
      attrs_pad.push_back(rewriter.getNamedAttr(
          "mode", tpu::PaddingModeAttr::get(getContext(),tpu::PaddingMode::constant)));

      auto output_shape_pad = llvm::SmallVector<int64_t>(input_shape);
      if (input_shape.size() == 3){
        output_shape_pad[2] += (pad_paddings[2] + pad_paddings[5]);
      }
      if (input_shape.size() == 4){
        output_shape_pad[2] += (pad_paddings[2] + pad_paddings[6]);
        output_shape_pad[3] += (pad_paddings[3] + pad_paddings[7]);
      }

      auto op_pad = rewriter.create<tpu::PadOp>(
          loc_pad, RankedTensorType::get(output_shape_pad, input_ele_type),
          operands_pad, attrs_pad);
      input_value = op_pad.getResult();
    }
    op.setOperand(0, input_value);
    op.setPadsAttr(rewriter.getI64ArrayAttr(conv_paddings));
    return success();
  }
};

} // namespace bm168x

namespace tpu {
using namespace bm168x;
void populateOptimizeBM168xPatterns(RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<
    LargePadConvPattern
  >(patterns->getContext());
  // clang-format on
};
} // namespace tpu

} // namespace tpu_mlir
