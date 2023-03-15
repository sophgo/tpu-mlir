//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"

using namespace llvm;
using namespace mlir;

namespace tpu_mlir {
namespace top {

class PostHandlePass : public PostHandleBase<PostHandlePass> {
public:
  PostHandlePass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    std::string type = this->type;
    for (auto&& func : mOp.getOps<FuncOp>()) {
      if (!func.back().empty()
          && func.back().back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        if (type == "") continue;
        Operation *terminator = func.getBody().back().getTerminator();
        auto&& ctx_ = func.getContext();
        auto builder = OpBuilder(ctx_);
        std::vector<Value> inputs;
        for (auto&& in: terminator->getOperands())
          inputs.emplace_back(in);
        terminator->erase();
        auto functionType = func.getFunctionType();
        BitVector erasedResultIndices(functionType.getNumResults(), true);
        func.eraseResults(erasedResultIndices);
        builder.setInsertionPointToEnd(&func.back());
        std::vector<NamedAttribute> attrs;
        std::string new_name;
        int64_t keep_topk= 200;
        int64_t batch_num = module::getShape(inputs[0])[0];
        int64_t dims = module::getShape(inputs[0]).size();
        //insert posthandleOp
        if (type == "yolo") {
          new_name = "yolo_post";
          int64_t h,w;
          int64_t flag = 1;//used to distinguish between BM16xx and CV18xx
          std::vector<int64_t> scale{8,16,32};
          std::vector<int64_t> mask{0, 1, 2, 3, 4, 5, 6, 7, 8};
          h = scale[0] * module::getShape(inputs[0])[dims-2];
          w = scale[0] * module::getShape(inputs[0])[dims-1];
          attrs.emplace_back(builder.getNamedAttr("net_input_h",builder.getI64IntegerAttr(h)));
          attrs.emplace_back(builder.getNamedAttr("net_input_w",builder.getI64IntegerAttr(w)));
          attrs.emplace_back(builder.getNamedAttr("nms_threshold",builder.getF64FloatAttr(0.5)));
          attrs.emplace_back(builder.getNamedAttr("obj_threshold",builder.getF64FloatAttr(0.7)));
          attrs.emplace_back(builder.getNamedAttr("keep_topk",builder.getI64IntegerAttr(keep_topk)));
          attrs.emplace_back(builder.getNamedAttr("anchors", builder.getStringAttr("10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326")));
          attrs.emplace_back(builder.getNamedAttr("scale", builder.getI64ArrayAttr(scale)));
          attrs.emplace_back(builder.getNamedAttr("mask", builder.getI64ArrayAttr(mask)));
          attrs.emplace_back(builder.getNamedAttr("flag", builder.getI64IntegerAttr(flag)));
        } else if (type == "ssd") {
          new_name = "ssd_post";
          int top_k = 200;
          int background_label_id = 0;
          int num_classes = module::getShape(inputs[1])[1];
          float eta = 1.0;
          float variance_encoded_in_target = 0;
          bool share_location = true;
          std::string code_type = "CENTER_SIZE";
          attrs.emplace_back(builder.getNamedAttr("num_classes",builder.getI64IntegerAttr(num_classes)));
          attrs.emplace_back(builder.getNamedAttr("background_label_id",builder.getI64IntegerAttr(background_label_id)));
          attrs.emplace_back(builder.getNamedAttr("nms_threshold",builder.getF64FloatAttr(0.5)));
          attrs.emplace_back(builder.getNamedAttr("top_k",builder.getI64IntegerAttr(top_k)));
          attrs.emplace_back(builder.getNamedAttr("code_type",builder.getStringAttr(code_type)));
          attrs.emplace_back(builder.getNamedAttr("keep_top_k",builder.getI64IntegerAttr(keep_topk)));
          attrs.emplace_back(builder.getNamedAttr("confidence_threshold",builder.getF64FloatAttr(0.05)));
          attrs.emplace_back(builder.getNamedAttr("share_location",builder.getBoolAttr(share_location)));
          attrs.emplace_back(builder.getNamedAttr("variance_encoded_in_target",builder.getF64FloatAttr(variance_encoded_in_target)));
          attrs.emplace_back(builder.getNamedAttr("eta",builder.getF64FloatAttr(eta)));
        }

        llvm::ArrayRef<int64_t> shape = {1, 1, keep_topk * batch_num, 7};
        auto new_type = RankedTensorType::get(shape, module::getElementType(inputs[0]));
        [&]() {
          if (type == "yolo") {
            auto post_op =  builder.create<top::YoloDetectionOp>(NameLoc::get(builder.getStringAttr(new_name)),
                                    new_type, inputs, attrs);
            auto retOp = builder.create<ReturnOp>(module::getLoc(), post_op.getOutput());
            func.insertResult(0, post_op.getOutput().getType(), nullptr);
          }}();

        [&]() {
          if (type == "ssd") {
            auto post_op = builder.create<top::DetectionOutputOp>(NameLoc::get(builder.getStringAttr(new_name)),
                                    new_type, inputs, attrs);
            auto retOp = builder.create<ReturnOp>(module::getLoc(), post_op.getOutput());
            func.insertResult(0, post_op.getOutput().getType(), nullptr);
          }}();
      }
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createPostHandlePass() {
  return std::make_unique<PostHandlePass>();
}
} // namespace top
} // namespace tpu_mlir
