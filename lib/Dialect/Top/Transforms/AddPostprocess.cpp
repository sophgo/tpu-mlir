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

namespace tpu_mlir {
namespace top {

class AddPostprocessPass : public AddPostprocessBase<AddPostprocessPass> {
public:
  AddPostprocessPass() {}
  void runOnOperation() override {
    auto mOp = getOperation();
    auto func = module::getMainFuncOp();
    StringRef type = this->type;
    if (type == "") {
      return;
    }

    Operation *terminator = func.getBody().back().getTerminator();
    OpBuilder builder(&getContext());
    auto operands = terminator->getOperands();
    auto arg_type = func.getArgumentTypes();
    auto in_shape = arg_type[0].cast<RankedTensorType>().getShape();
    int64_t topk = 200;
    auto batch_num = in_shape[0];
    auto h = in_shape[2];
    auto w = in_shape[3];

    // insert posthandleOp
    builder.setInsertionPoint(terminator);
    std::vector<NamedAttribute> attrs;
    auto new_type = RankedTensorType::get({1, 1, batch_num * topk, 7},
                                          builder.getF32Type());
    if (type.starts_with("yolo")) {
      auto loc = NameLoc::get(builder.getStringAttr("yolo_post"));
      std::vector<int64_t> scale{8, 16, 32};
      std::vector<int64_t> mask{0, 1, 2, 3, 4, 5, 6, 7, 8};
      attrs.emplace_back(
          builder.getNamedAttr("net_input_h", builder.getI64IntegerAttr(h)));
      attrs.emplace_back(
          builder.getNamedAttr("net_input_w", builder.getI64IntegerAttr(w)));
      attrs.emplace_back(
          builder.getNamedAttr("nms_threshold", builder.getF64FloatAttr(0.5)));
      attrs.emplace_back(
          builder.getNamedAttr("obj_threshold", builder.getF64FloatAttr(0.7)));
      attrs.emplace_back(
          builder.getNamedAttr("keep_topk", builder.getI64IntegerAttr(topk)));
      attrs.emplace_back(builder.getNamedAttr(
          "anchors",
          builder.getI64ArrayAttr({10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59,
                                   119, 116, 90, 156, 198, 373, 326})));
      attrs.emplace_back(
          builder.getNamedAttr("scale", builder.getI64ArrayAttr(scale)));
      attrs.emplace_back(
          builder.getNamedAttr("mask", builder.getI64ArrayAttr(mask)));
      attrs.emplace_back(
          builder.getNamedAttr("version", builder.getStringAttr(type)));
      auto post_op =
          builder.create<top::YoloDetectionOp>(loc, new_type, operands, attrs);
      terminator->setOperands({post_op.getOutput()});
    } else if (type == "ssd") {
      auto loc = NameLoc::get(builder.getStringAttr("ssd_post"));
      int background_label_id = 0;
      int num_classes = module::getShape(operands[1])[1];
      float eta = 1.0;
      float variance_encoded_in_target = 0;
      bool share_location = true;
      std::string code_type = "CENTER_SIZE";
      attrs.emplace_back(builder.getNamedAttr(
          "num_classes", builder.getI64IntegerAttr(num_classes)));
      attrs.emplace_back(
          builder.getNamedAttr("background_label_id",
                               builder.getI64IntegerAttr(background_label_id)));
      attrs.emplace_back(
          builder.getNamedAttr("nms_threshold", builder.getF64FloatAttr(0.5)));
      attrs.emplace_back(
          builder.getNamedAttr("top_k", builder.getI64IntegerAttr(topk)));
      attrs.emplace_back(
          builder.getNamedAttr("code_type", builder.getStringAttr(code_type)));
      attrs.emplace_back(
          builder.getNamedAttr("keep_top_k", builder.getI64IntegerAttr(topk)));
      attrs.emplace_back(builder.getNamedAttr("confidence_threshold",
                                              builder.getF64FloatAttr(0.05)));
      attrs.emplace_back(builder.getNamedAttr(
          "share_location", builder.getBoolAttr(share_location)));
      attrs.emplace_back(builder.getNamedAttr(
          "variance_encoded_in_target",
          builder.getF64FloatAttr(variance_encoded_in_target)));
      attrs.emplace_back(
          builder.getNamedAttr("eta", builder.getF64FloatAttr(eta)));
      auto post_op = builder.create<top::DetectionOutputOp>(loc, new_type,
                                                            operands, attrs);
      terminator->setOperands({post_op.getOutput()});
    }
    module::updateModuleTypes();
    module::setPostprocess(type);
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createAddPostprocessPass() {
  return std::make_unique<AddPostprocessPass>();
}
} // namespace top
} // namespace tpu_mlir
