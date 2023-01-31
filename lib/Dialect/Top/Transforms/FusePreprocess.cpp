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
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace llvm;

namespace tpu_mlir {
namespace top {
template<typename T>
void swapInputChannelOfFilter(
      std::vector<T> &filter_data, std::vector<T> &new_filter,
      RankedTensorType &filter_type) {

  std::vector<int64_t> shape(filter_type.getShape());
  int64_t size = std::accumulate(std::begin(shape), std::end(shape), 1,
                      std::multiplies<>());
  assert(filter_data.size() == size && "filter size should be equal");

  int64_t oc, ic, frame_size;
  int64_t index = shape.size();
  assert((index == 4 || index == 5) && "filter shape size should be 4 or 5");

  frame_size = shape[index - 1] * shape[index - 2];
  ic = shape[index - 3];
  oc = shape[index - 4];
  if (index == 5) {
    oc *= shape[index - 5];
  }
  std::vector<int> order {2, 1, 0};
  //std::vector<T> new_filter(size);
  T *filter = (T *)filter_data.data();
  for (int i = 0; i < oc; ++i) {
    for (int j = 0; j < ic; ++j) {
      assert(order[j] < ic);
      T *in = filter + i * ic * frame_size + order[j] * frame_size;
      T *out =
          (T *)new_filter.data() + i * ic * frame_size + j * frame_size;
      memcpy(out, in, frame_size * sizeof(T));
    }
  }
}

template<typename OpTy>
struct FoldSwapAxisOpPattern : public RewritePattern {
  FoldSwapAxisOpPattern(MLIRContext *context)
      : RewritePattern(OpTy::getOperationName(), 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    //return success();
    auto swapOp = cast<OpTy>(op);
    auto nextOp = module::getNextOp(swapOp);
    if (!nextOp) {
      return failure();
    }
    auto convOp = dyn_cast_or_null<top::ConvOp>(nextOp);
    if (!convOp) {
      return failure();
    }
    if (convOp.getGroup() != 1 || convOp.getKernelShape().size() != 2) {
      return failure();
    }
    assert(convOp.getNumOperands() == 3 && "Conv2D op should have 3 operands");
    // filter
    auto filterOp = cast<top::WeightOp>(convOp.getFilter().getDefiningOp());
    auto filter_fp32 = *(filterOp.read_as_float());
    //filter_fp32 = *(filterOp.read<float>());
    auto filter_name = module::getName(filterOp.getOutput()).str();
    auto filter_type = convOp.getFilter().getType().template cast<RankedTensorType>();
    //auto filter_type = convOp.getFilter().getType();
    std::vector<float> new_filter_data(filter_fp32.size());
    //swap filter weight
    swapInputChannelOfFilter<float>(filter_fp32, new_filter_data, filter_type);
    auto newFilterOp = top::WeightOp::create(convOp, filter_name + "_swap_channel",
                                             new_filter_data, filter_type);
    convOp.setOperand(1, newFilterOp);
    //rewriter.replaceOp(filterOp.getResult().getDefiningOp(), {newFilterOp.getResult()});
    rewriter.replaceOp(op, {swapOp.getOperand()});
    return success();
  }
};


class FusePreprocessPass
    : public FusePreprocessBase<FusePreprocessPass> {
public:
  FusePreprocessPass() {}
  void runOnOperation() override {
    llvm::errs()<<"Entering FusePreprocessPass.\n";
    auto module_ = getOperation();
    auto ctx_ = &getContext();
    auto fn = module::getMainFuncOp();
    auto builder = OpBuilder(ctx_);
    // fn.walk([&](Operation *op) {
    //   op->dump();
    // });
    std::vector<mlir::Type> returnTypes;
    Block &entryBlock = fn.front();
    auto returnOp = dyn_cast<ReturnOp>(entryBlock.back()).getOperation();
    for (uint32_t i = 0; i < returnOp->getNumOperands(); ++i) {
      returnTypes.push_back(returnOp->getOperand(i).getType());
    }
    std::vector<mlir::Type> argumentTypes;

    std::map<std::string,
             std::pair<std::string, std::string>> attributes_map = {
      {"RGB_PLANAR",    {"rgb", "nchw"}},
      {"RGB_PACKED",    {"rgb", "nhwc"}},
      {"BGR_PLANAR",    {"bgr", "nchw"}},
      {"BGR_PACKED",    {"bgr", "nhwc"}},
      {"GRAYSCALE",     {"bgr", "nchw"}},
      {"YUV420_PLANAR", {"bgr", "nchw"}},
      {"YUV_NV21",      {"bgr", "nchw"}},
      {"YUV_NV12",      {"bgr", "nchw"}},
      {"RGBA_PLANAR", {"rgba", "nchw"}}
    };
    std::string quant_mode = this->mode;
    std::string pixel_format = this->customization_format;
    fn.walk([&](top::InputOp inputOp) {
      double threshold;
      if (quant_mode == "BF16") {
        threshold = 0.0001;
      } else if (quant_mode == "INT8") {
        auto itype = module::getCalibratedType(inputOp.getOutput());
        threshold = itype.getMax();
      }
      auto name = module::getName(inputOp.getOutput()).str();
      auto resized_dims = module::getI64Array(inputOp.getResizeDims().value());
      //rgb,bgr,etc..
      auto channel_order = inputOp.getPixelFormat().value().str();
      this->mean = *(module::getF64Array(inputOp.getMean().value()));
      this->scale = *(module::getF64Array(inputOp.getScale().value()));
      // this->type = inputOp.getOutput().getType().cast<RankedTensorType>().getElementType();
      // llvm::errs()<<"this->type:"<<this->type<<"\n";
      std::vector<int64_t> model_shape;
      module::getShapeVec(inputOp.getResult(), model_shape);
      module::getNCHW(model_shape, n, c, h, w, false);
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
      bool swap_channel = (color != channel_order) ? true : false;
      llvm::errs() << "pixel_format:" << pixel_format
                  << ", color:" << color
                  << ", layout:" << layout
                  << ", swap_channel:" << swap_channel
                  << "\n";
      std::vector<Operation *> uses;
      for (auto &use : inputOp.getResult().getUses()) {
        auto opd = use.getOwner();
        uses.emplace_back(opd);
      }

      //set the real shape of function's args.
      std::vector<int64_t> arg_shape {n, c, resize_h, resize_w};
      std::vector<int64_t> input_shape {n, c, resize_h, resize_w};
      if (layout == "nhwc") {
        arg_shape[1] = resize_h;
        arg_shape[2] = resize_w;
        arg_shape[3] = c;
        input_shape[1] = resize_h;
        input_shape[2] = resize_w;
        input_shape[3] = c;

      }

      auto arg_type = RankedTensorType::get(arg_shape, builder.getIntegerType(8, false));
      argumentTypes.push_back(arg_type);

      //change the shape of inputOp
      //auto cali_type = quant::CalibratedQuantizedType::get(builder.getIntegerType(8, false), 0, 255);
      auto input_loc = NameLoc::get(builder.getStringAttr(name + "_raw"));
      inputOp.getOperand().setType(arg_type);
      inputOp.getResult().setLoc(input_loc);
      inputOp.setPixelFormat(color);
      inputOp.setChannelFormat(layout);
      auto input_type = RankedTensorType::get(input_shape, builder.getIntegerType(8, false));
      inputOp.getResult().setType(input_type);

      mlir::Value currentOut = inputOp.getResult();
      builder.setInsertionPointAfterValue(currentOut);

      // create transpose Op if need
      if (layout == "nhwc") {
        currentOut = this->insertTransposeOp(builder, name, currentOut);
        builder.setInsertionPointAfterValue(currentOut);
      }

      // create SliceOp
      if (resize_h != h || resize_w != w) {
        //llvm::errs()<<"this->type:"<<this->type<<"\n";
        currentOut = this->insertSliceOp(builder, name, currentOut);
        builder.setInsertionPointAfterValue(currentOut);
      }

      if (quant_mode == "INT8") {
        currentOut = this->insertScaleLutOp(builder, name, currentOut, threshold, swap_channel);
        builder.setInsertionPointAfterValue(currentOut);
      } else {
        //BF16
        currentOut = this->insertScaleOp(builder, name, currentOut, threshold, swap_channel);
      }

      if (swap_channel) {
        currentOut = this->insertSwapAxisOp(builder, name, currentOut, threshold);
      }
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
    auto fnType = builder.getFunctionType(
          llvm::ArrayRef<mlir::Type>{argumentTypes},
          llvm::ArrayRef<mlir::Type>{returnTypes});
    fn.setType(fnType);

    RewritePatternSet patterns(ctx_);
    patterns.insert<
        FoldSwapAxisOpPattern<top::SwapChannelOp>
        >(ctx_);
    applyPatternsAndFoldGreedily(fn, std::move(patterns));
  }

private:
  int64_t n, c, h, w;
  int64_t resize_h, resize_w;
  std::vector<double> mean, scale;
  //mlir::Type type;

  Value insertTransposeOp(OpBuilder &builder, std::string &name, Value opd) {
    auto loc = NameLoc::get(builder.getStringAttr(name + "_preprocess_tranpose"));
    std::vector<int64_t> order{0, 3, 1, 2};
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(builder.getNamedAttr("order", builder.getI64ArrayAttr(order)));
    //auto cali_type = quant::CalibratedQuantizedType::get(builder.getIntegerType(8, false), 0, 255);
    auto cali_type = quant::CalibratedQuantizedType::get(builder.getF32Type(), 0, 255);
    auto type = RankedTensorType::get({n, c, resize_h, resize_w}, cali_type);
    auto newOp = builder.create<top::PermuteOp>(loc, type, ArrayRef<Value>{opd}, attrs);
    return newOp.getOutput();
  }

  Value insertSliceOp(OpBuilder &builder, std::string &name, Value opd) {
    auto loc = NameLoc::get(builder.getStringAttr(name + "_preprocess_slice"));
    int64_t start_h = resize_h / 2 - h / 2;
    int64_t start_w = resize_w / 2 - w / 2;
    std::vector<int64_t> slice_offset{0, 0, start_h, start_w};
    std::vector<int64_t> slice_step{1, 1, 1, 1};
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(builder.getNamedAttr("offset", builder.getI64ArrayAttr(slice_offset)));
    attrs.emplace_back(builder.getNamedAttr("steps", builder.getI64ArrayAttr(slice_step)));
    //auto cali_type = quant::CalibratedQuantizedType::get(builder.getIntegerType(8, false), 0, 255);
    auto cali_type = quant::CalibratedQuantizedType::get(builder.getF32Type(), 0, 255);
    auto type = RankedTensorType::get({n, c, h, w}, cali_type);
    auto newOp = builder.create<top::SliceOp>(loc, type, ArrayRef<Value>{opd}, attrs);
    return newOp.getOutput();
  }

  Value insertScaleLutOp(OpBuilder &builder, std::string &name, Value opd,
                          double threshold, bool swap_channel) {
    auto loc = NameLoc::get(builder.getStringAttr(name + "_preprocess_scale_lut"));
    double qscale = 128.0 / threshold;
    std::vector<double> scales;
    std::vector<double> bias;
    for (int i = 0; i < c; i++) {
      scales.push_back(this->scale[i]);
      bias.push_back(-1 * this->scale[i] * this->mean[i]);
    }
    if (swap_channel) {
      // keep order bgr
      std::swap(scales[0], scales[2]);
      std::swap(bias[0], bias[2]);
    }
    //quant
    for (int i = 0; i < c; i++) {
      scales[i] *= qscale;
      bias[i] *= qscale;
    }
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(builder.getNamedAttr("scale", builder.getF64ArrayAttr(scales)));
    attrs.emplace_back(builder.getNamedAttr("bias", builder.getF64ArrayAttr(bias)));
    //auto cali_type = quant::CalibratedQuantizedType::get(builder.getIntegerType(8, true), -threshold, threshold);
    auto cali_type = quant::CalibratedQuantizedType::get(builder.getF32Type(), -threshold, threshold);
    auto type = RankedTensorType::get({n, c, h, w}, cali_type);
    auto newOp = builder.create<top::ScaleLutOp>(loc, type, ArrayRef<Value>{opd}, attrs);
    return newOp.getOutput();
  }

  Value insertScaleOp(OpBuilder &builder, std::string &name, Value opd,
                          double threshold, bool swap_channel) {
    auto loc = NameLoc::get(builder.getStringAttr(name + "_preprocess_scale"));
    auto none = module::getNoneOp(opd.getDefiningOp());
    std::vector<float> scales;
    std::vector<float> bias;
    for (int i = 0; i < c; i++) {
      scales.push_back(this->scale[i]);
      bias.push_back(-1 * this->scale[i] * this->mean[i]);
    }

    llvm::errs() << "scale:";
    for (auto s : scales)
      llvm::errs() << " " << s;
    llvm::errs() << "\n";
    llvm::errs() << "bias:";
    for (auto b : bias)
      llvm::errs() << " " << b;
    llvm::errs() << "\n";

    if (c == 3 && swap_channel) {
      std::swap(scales[0], scales[2]);
      std::swap(bias[0], bias[2]);
    }

    auto scale_type = RankedTensorType::get({1, c, 1, 1}, builder.getF32Type());
    auto bias_type = RankedTensorType::get({1, c, 1, 1}, builder.getF32Type());
    std::vector<Value> operands;
    operands.emplace_back(opd);
    operands.emplace_back(none);
    operands.emplace_back(none);
    std::vector<NamedAttribute> attrs;
    attrs.emplace_back(builder.getNamedAttr("do_relu", builder.getBoolAttr(false)));
    auto cali_type = quant::CalibratedQuantizedType::get(builder.getF32Type(), -threshold, threshold);
    auto type = RankedTensorType::get({n, c, h, w}, cali_type);
    auto newOp = builder.create<top::ScaleOp>(loc, type, operands, attrs);
    auto scale_weight = top::WeightOp::create(newOp, name + "_preprocess_scale_0", scales, scale_type);
    auto bias_Weight = top::WeightOp::create(newOp, name + "_preprocess_scale_1", bias, bias_type);
    newOp.setOperand(1, scale_weight);
    newOp.setOperand(2, bias_Weight);
    return newOp.getOutput();
  }

  Value insertSwapAxisOp(OpBuilder &builder, std::string &name, Value opd, float threshold) {
    llvm::errs()<<"Entering insertSwapAxisOp.\n";
    auto loc = NameLoc::get(builder.getStringAttr(name + "_preprocess_swapaxis"));
    std::vector<NamedAttribute> attrs;
    std::vector<int64_t> orders {2, 1, 0};
    attrs.emplace_back(builder.getNamedAttr("channel_order", builder.getI64ArrayAttr(orders)));
    auto cali_type = quant::CalibratedQuantizedType::get(builder.getF32Type(), -threshold, threshold);
    auto type = RankedTensorType::get({n, c, h, w}, cali_type);
    auto newOp = builder.create<top::SwapChannelOp>(loc, type, ArrayRef<Value>{opd}, attrs);
    return newOp.getOutput();
  }
// protected:
//   ModuleOp module_;
//   FuncOp fn;
//   MLIRContext *ctx_;

};
std::unique_ptr<OperationPass<ModuleOp>> createFusePreprocessPass() {
  return std::make_unique<FusePreprocessPass>();
}
}
}
