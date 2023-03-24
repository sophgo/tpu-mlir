//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Support/Module.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace tpu_mlir {
namespace top {

class FusePreprocessPass : public FusePreprocessBase<FusePreprocessPass> {
public:
  FusePreprocessPass() {}
  void runOnOperation() override {
    llvm::errs() << "Entering FusePreprocessPass.\n";
    auto module_ = getOperation();
    auto ctx_ = &getContext();
    auto fn = module::getMainFuncOp();
    auto builder = OpBuilder(ctx_);
    std::vector<mlir::Type> returnTypes;
    Block &entryBlock = fn.front();
    auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
    for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
      returnTypes.push_back(returnOp->getOperand(i).getType());
    }
    std::vector<mlir::Type> argumentTypes;

    std::map<std::string, std::pair<std::string, std::string>> attributes_map =
        {{"RGB_PLANAR", {"rgb", "nchw"}},  {"RGB_PACKED", {"rgb", "nhwc"}},
         {"BGR_PLANAR", {"bgr", "nchw"}},  {"BGR_PACKED", {"bgr", "nhwc"}},
         {"GRAYSCALE", {"bgr", "nchw"}},   {"YUV420_PLANAR", {"bgr", "nchw"}},
         {"YUV_NV21", {"bgr", "nchw"}},    {"YUV_NV12", {"bgr", "nchw"}},
         {"RGBA_PLANAR", {"rgba", "nchw"}}};
    std::string quant_mode = this->mode;
    std::string pixel_format = this->customization_format;
    fn.walk([&](top::InputOp inputOp) {
      double max, min;
      // check if there is calibration table (for mix precision case)
      if (module::isCalibratedType(inputOp.getOutput())) {
        auto itype = module::getCalibratedType(inputOp.getOutput());
        max = itype.getMax();
        min = itype.getMin();
      } else {
        max = 127; // a random threshold value for Type
        min = -128;
      }

      auto name = module::getName(inputOp.getOutput()).str();
      auto resized_dims = module::getI64Array(inputOp.getResizeDims().value());
      // Get the original channel_order(rgb,bgr,etc..) and save it to
      // preprocessOp.
      auto channel_order = inputOp.getPixelFormat().value().str();
      module::getNCHW(inputOp.getResult(), n, c, h, w, false);
      std::vector<int64_t> dims;
      if (resized_dims->size() == 2) {
        resize_h = resized_dims->at(0);
        resize_w = resized_dims->at(1);
      } else {
        resize_h = h;
        resize_w = w;
      }
      auto color = std::get<0>(attributes_map[pixel_format]);
      auto layout = std::get<1>(attributes_map[pixel_format]);
      std::vector<Operation *> uses;
      for (auto &use : inputOp.getResult().getUses()) {
        auto opd = use.getOwner();
        uses.emplace_back(opd);
      }

      // set the real shape of function's args.
      std::vector<int64_t> arg_shape{n, c, resize_h, resize_w};
      std::vector<int64_t> input_shape{n, c, resize_h, resize_w};
      if (layout == "nhwc") {
        arg_shape[1] = resize_h;
        arg_shape[2] = resize_w;
        arg_shape[3] = c;
        input_shape[1] = resize_h;
        input_shape[2] = resize_w;
        input_shape[3] = c;
      }

      auto arg_type =
          RankedTensorType::get(arg_shape, builder.getIntegerType(8, false));
      argumentTypes.push_back(arg_type);

      auto input_loc = NameLoc::get(builder.getStringAttr(name + "_raw"));
      inputOp.getOperand().setType(arg_type);
      inputOp.getResult().setLoc(input_loc);
      inputOp.setPixelFormat(color);
      inputOp.setChannelFormat(layout);
      inputOp.setCustomizationFormat(pixel_format);
      auto uni_type = quant::UniformQuantizedType::get(
          0, IntegerType::get(ctx_, 8), builder.getF32Type(), 1.0, 0, 0, 255);
      auto input_type = RankedTensorType::get(input_shape, uni_type);
      inputOp.getResult().setType(input_type);

      mlir::Value currentOut = inputOp.getResult();
      builder.setInsertionPointAfterValue(currentOut);

      // insert preprocessOp
      std::vector<NamedAttribute> attrs;
      auto mean = module::getF64Array(inputOp.getMeanAttr());
      bool sign = false;
      attrs.emplace_back(builder.getNamedAttr(
          "quant_mode", builder.getStringAttr(quant_mode)));
      attrs.emplace_back(builder.getNamedAttr(
          "customization_format", builder.getStringAttr(pixel_format)));
      attrs.emplace_back(builder.getNamedAttr(
          "channel_order", builder.getStringAttr(channel_order)));
      attrs.emplace_back(
          builder.getNamedAttr("resize_dims", inputOp.getResizeDimsAttr()));
      attrs.emplace_back(builder.getNamedAttr("scale", inputOp.getScaleAttr()));
      attrs.emplace_back(builder.getNamedAttr("mean", inputOp.getMeanAttr()));
      auto loc = NameLoc::get(builder.getStringAttr(name + "_preprocess"));
      for (int i = 0; i < mean->size(); i++) {
        if (mean->at(i) != 0) {
          sign = true;
          break;
        }
      }
      attrs.emplace_back(
          builder.getNamedAttr("sign", builder.getBoolAttr(sign)));
      auto cali_type =
          quant::CalibratedQuantizedType::get(builder.getF32Type(), min, max);
      auto type = RankedTensorType::get({n, c, h, w}, cali_type);
      auto newOp = builder.create<top::PreprocessOp>(
          loc, type, ArrayRef<Value>{currentOut}, attrs);
      currentOut = newOp.getResult();
      // reset inputOp's scale and mean
      inputOp.setScaleAttr(builder.getF64ArrayAttr({1.0, 1.0, 1.0}));
      inputOp.setMeanAttr(builder.getF64ArrayAttr({0.0, 0.0, 0.0}));
      // update operand of all inputOp's uses
      for (auto use_op : uses) {
        for (int i = 0; i < (int)use_op->getNumOperands(); i++) {
          if (use_op->getOperand(i) == inputOp.getResult()) {
            use_op->setOperand(i, currentOut);
          }
        }
      }
    });

    // alter the function type to match the real type
    // of InputOp and ReturnOp
    auto fnType =
        builder.getFunctionType(llvm::ArrayRef<mlir::Type>{argumentTypes},
                                llvm::ArrayRef<mlir::Type>{returnTypes});
    fn.setType(fnType);
  }

private:
  int64_t n, c, h, w;
  int64_t resize_h, resize_w;
};

std::unique_ptr<OperationPass<ModuleOp>> createFusePreprocessPass() {
  return std::make_unique<FusePreprocessPass>();
}
} // namespace top
} // namespace tpu_mlir
