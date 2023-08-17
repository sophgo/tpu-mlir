//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/Transforms/Passes.h"
#include "tpu_mlir/Support/MathUtils.h"

using namespace llvm;

namespace tpu_mlir {
namespace top {

static const std::vector<int64_t> YOLOV3_ANCHORS = {
    10,  13, 16,  30,  33,  23,  // 8
    30,  61, 62,  45,  59,  119, // 16
    116, 90, 156, 198, 373, 326  // 32
};

static const std::vector<int64_t> YOLOV3_TINY_ANCHORS = {
    10, 14, 23,  27,  37,  58, // 16
    81, 82, 135, 169, 344, 319 // 32
};

static const std::vector<int64_t> YOLOV4_ANCHORS = {
    12,  16,  19,  36,  40,  28,  // 8
    36,  75,  76,  55,  72,  146, // 16
    142, 110, 192, 243, 459, 401  // 32
};

static const std::vector<int64_t> YOLOV5_ANCHORS = {
    10,  13, 16,  30,  33,  23,  // 8
    30,  61, 62,  45,  59,  119, // 16
    116, 90, 156, 198, 373, 326  // 32
};

class AddPostprocessPass : public AddPostprocessBase<AddPostprocessPass> {
public:
  AddPostprocessPass() {}
  void runOnOperation() override {
    post_type = this->type;
    if (post_type.empty()) {
      return;
    }
    auto mOp = getOperation();
    auto func = module::getMainFuncOp(mOp);
    terminator = func.getBody().back().getTerminator();
    auto arg_type = func.getArgumentTypes();
    in_shape = arg_type[0].cast<RankedTensorType>().getShape();
    batch = in_shape[0];
    height = in_shape[2];
    width = in_shape[3];
    topk = 200;
    // insert posthandleOp
    OpBuilder builder(&getContext());
    builder.setInsertionPoint(terminator);
    if (post_type.starts_with("yolo")) {
      insertYoloOp(builder);
    } else if (post_type == "ssd") {
      insertSsdOp(builder);
    } else if (post_type == "bnr") {
      insertDepackRawOp(builder);
    }
    module::updateModuleTypes();
    module::setPostprocess(post_type);
  }

protected:
  void getYoloOperandsAndAnchors(std::vector<Value> &operands,
                                 std::vector<int64_t> &anchors);
  void insertYoloOp(OpBuilder &builder);
  void insertSsdOp(OpBuilder &builder);
  void insertDepackRawOp(OpBuilder &builder);

protected:
  std::vector<int64_t> in_shape;
  int64_t batch, height, width, topk;
  StringRef post_type;
  Operation *terminator;
};

void AddPostprocessPass::getYoloOperandsAndAnchors(
    std::vector<Value> &operands, std::vector<int64_t> &anchors) {
  std::vector<int64_t> widths;
  auto opds = terminator->getOperands();
  auto num_opds = opds.size();
  // TODO: Maybe yolov5 has only 1 yolo layer
  if (post_type == "yolov5" && num_opds == 1 &&
      module::getShape(opds[0]).size() == 3) {
    operands.push_back(opds[0]);
    anchors = {0};
    return;
  }
  // yolov8
  if (post_type == "yolov8" && num_opds == 1){
    operands.push_back(opds[0]);
    anchors = {0};
    return;
  }
  for (auto opd : opds) {
    auto s = module::getShape(opd);
    if (s.size() != 4 || width % s[3] != 0 || num_opds > 3) {
      terminator->dump();
      llvm_unreachable("outputs are not correct");
    }
    widths.push_back(s[3]);
  }
  std::vector<std::pair<int, int64_t>> result;
  topk_indices(result, widths.data(), widths.size(), widths.size(), true);
  std::vector<int64_t> scales;
  for (int i = 0; i < num_opds; i++) {
    auto idx = result[i].first;
    auto w = result[i].second;
    operands.push_back(opds[idx]);
    scales.push_back(width / w);
  }
  // TODO: refine this mess
  if (post_type == "yolov5" && module::getShape(opds[0]).size() == 4) {
    anchors = YOLOV5_ANCHORS;
    return;
  }
  if (num_opds == 3) {
    anchors = post_type == "yolov4" ? YOLOV4_ANCHORS : YOLOV3_ANCHORS;
    return;
  }
  if (post_type == "yolov3_tiny" && num_opds == 2) {
    anchors = YOLOV3_TINY_ANCHORS;
    return;
  }
  // anchors by scale
  anchors.assign(num_opds * 6, 0);
  for (int i = 0; i < num_opds; i++) {
    auto s = scales[i];
    int idx = (s % 8 != 0 ? s / 8 - 1 : i);
    if (idx > num_opds) {
      idx = i;
    }
    std::copy(YOLOV3_ANCHORS.begin() + idx * 6,
              YOLOV3_ANCHORS.begin() + idx * 6 + 6, anchors.begin() + i * 6);
  }
}

void AddPostprocessPass::insertYoloOp(OpBuilder &builder) {
  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  std::vector<int64_t> anchors;
  getYoloOperandsAndAnchors(operands, anchors);
  auto loc = NameLoc::get(builder.getStringAttr("yolo_post"));
  attrs.emplace_back(
      builder.getNamedAttr("net_input_h", builder.getI64IntegerAttr(height)));
  attrs.emplace_back(
      builder.getNamedAttr("net_input_w", builder.getI64IntegerAttr(width)));
  attrs.emplace_back(
      builder.getNamedAttr("nms_threshold", builder.getF64FloatAttr(0.5)));
  attrs.emplace_back(
      builder.getNamedAttr("obj_threshold", builder.getF64FloatAttr(0.5)));
  attrs.emplace_back(
      builder.getNamedAttr("keep_topk", builder.getI64IntegerAttr(topk)));
  attrs.emplace_back(
      builder.getNamedAttr("anchors", builder.getI64ArrayAttr(anchors)));
  attrs.emplace_back(
      builder.getNamedAttr("agnostic_nms", builder.getBoolAttr(false)));
  attrs.emplace_back(
      builder.getNamedAttr("version", builder.getStringAttr(post_type)));
  auto new_type =
      RankedTensorType::get({1, 1, batch * topk, 7}, builder.getF32Type());
  auto post_op =
      builder.create<top::YoloDetectionOp>(loc, new_type, operands, attrs);
  terminator->setOperands({post_op.getOutput()});
}

void AddPostprocessPass::insertSsdOp(OpBuilder &builder) {
  auto operands = terminator->getOperands();
  auto new_type =
      RankedTensorType::get({1, 1, batch * topk, 7}, builder.getF32Type());
  auto loc = NameLoc::get(builder.getStringAttr("ssd_post"));
  int background_label_id = 0;
  int num_classes = module::getShape(operands[1])[1];
  float eta = 1.0;
  float variance_encoded_in_target = 0;
  bool share_location = true;
  std::string code_type = "CENTER_SIZE";
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(builder.getNamedAttr(
      "num_classes", builder.getI64IntegerAttr(num_classes)));
  attrs.emplace_back(builder.getNamedAttr(
      "background_label_id", builder.getI64IntegerAttr(background_label_id)));
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
  attrs.emplace_back(builder.getNamedAttr("share_location",
                                          builder.getBoolAttr(share_location)));
  attrs.emplace_back(builder.getNamedAttr(
      "variance_encoded_in_target",
      builder.getF64FloatAttr(variance_encoded_in_target)));
  attrs.emplace_back(builder.getNamedAttr("eta", builder.getF64FloatAttr(eta)));
  auto post_op =
      builder.create<top::DetectionOutputOp>(loc, new_type, operands, attrs);
  terminator->setOperands({post_op.getOutput()});
}

void AddPostprocessPass::insertDepackRawOp(OpBuilder &builder) {
  auto mOp = getOperation();
  auto func = module::getMainFuncOp(mOp);
  float white_level, black_level;
  std::string pixel_format;
  func.walk([&](top::InputOp inputOp){
    if ( inputOp.getDoPreprocess() )
    {
      white_level = inputOp.getWhiteLevel()->convertToDouble();
      black_level = inputOp.getBlackLevel()->convertToDouble();
      pixel_format = inputOp.getPixelFormat()->str();
      return;
    }
  });
  auto operands = terminator->getOperands();
  auto opd = operands[0];
  auto shape = module::getShape(opd);
  int padding_h = 0;
  int padding_w = 0;
  int oh = ( shape[2] - padding_h ) * 2;
  int ow = ( shape[3] - padding_w ) * 3;
  std::vector<int64_t> channel_order;
  // RGBG->(3, 2, 0, 1)->GBRG
  // RGBG->(1, 0, 2, 3)->GRBG
  // RGBG->(0, 1, 3, 2)->RGGB
  // RGBG->(2, 3, 1, 0)->BGGR
  if ( pixel_format == "gbrg" )      channel_order = {3, 2, 0, 1};
  else if ( pixel_format == "grbg" ) channel_order = {1, 0, 2, 3};
  else if ( pixel_format == "rggb" ) channel_order = {0, 1, 3, 2};
  else if ( pixel_format == "bggr" ) channel_order = {2, 3, 1, 0};
  else llvm_unreachable ("raw format not support current type");
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(builder.getNamedAttr("padding_h", builder.getI64IntegerAttr(padding_h)));
  attrs.emplace_back(builder.getNamedAttr("padding_w", builder.getI64IntegerAttr(padding_w)));
  attrs.emplace_back(builder.getNamedAttr("white_level", builder.getF64FloatAttr(white_level)));
  attrs.emplace_back(builder.getNamedAttr("black_level", builder.getF64FloatAttr(black_level)));
  attrs.emplace_back(builder.getNamedAttr("channel_order", builder.getI64ArrayAttr(channel_order)));
  auto loc = NameLoc::get(builder.getStringAttr("depack_raw"));
  auto new_type = RankedTensorType::get({batch, 1, oh, ow}, builder.getIntegerType(8, false));
  auto post_op = builder.create<top::DepackRawOp>(loc, new_type, opd, attrs);
  terminator->setOperand(0, post_op.getOutput());
}

std::unique_ptr<OperationPass<ModuleOp>> createAddPostprocessPass() {
  return std::make_unique<AddPostprocessPass>();
}
} // namespace top
} // namespace tpu_mlir
