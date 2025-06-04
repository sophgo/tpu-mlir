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

static const std::vector<double> YOLOV3_ANCHORS = {
    10,  13, 16,  30,  33,  23,  // 8
    30,  61, 62,  45,  59,  119, // 16
    116, 90, 156, 198, 373, 326  // 32
};

static const std::vector<double> YOLOV3_TINY_ANCHORS = {
    10, 14, 23,  27,  37,  58, // 16
    81, 82, 135, 169, 344, 319 // 32
};

static const std::vector<double> YOLOV4_ANCHORS = {
    12,  16,  19,  36,  40,  28,  // 8
    36,  75,  76,  55,  72,  146, // 16
    142, 110, 192, 243, 459, 401  // 32
};

static const std::vector<double> YOLOV5_ANCHORS_DEFAULT = {
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
    if (post_type.ends_with("seg")) {
      insertYolosegOp(builder);
    } else if (post_type.starts_with("yolo")) {
      insertYoloOp(builder);
    } else if (post_type == "ssd") {
      insertSsdOp(builder);
    } else if (post_type == "bnr") {
      insertDepackRawOp(builder);
    } else if (post_type == "mmap2rgbmap") {
      insertMmap2RgbmapOp(builder);
    }
    module::updateModuleTypes();
    module::setPostprocess(post_type);
  }

protected:
  void getYoloOperandsAndAnchors(std::vector<Value> &operands,
                                 std::vector<double> &anchors);
  void insertYoloOp(OpBuilder &builder);
  void insertYolosegOp(OpBuilder &builder);
  void insertSsdOp(OpBuilder &builder);
  void insertDepackRawOp(OpBuilder &builder);
  void insertMmap2RgbmapOp(OpBuilder &builder);

protected:
  std::vector<int64_t> in_shape;
  int64_t batch, height, width, topk;
  StringRef post_type;
  Operation *terminator;
  StringRef intermode, intercoordmode;
};

void AddPostprocessPass::getYoloOperandsAndAnchors(
    std::vector<Value> &operands, std::vector<double> &anchors) {
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
  // TODO: refine this mess
  if (post_type == "yolov5" && module::getShape(opds[0]).size() == 4) {
    for (auto opd : opds) {
      auto s = module::getShape(opd);
      if (s.size() != 4 || width % s[3] != 0) {
        terminator->dump();
        llvm_unreachable("outputs are not correct");
      }
      widths.push_back(s[3]);
    }
    std::vector<std::pair<int, int64_t>> result;
    topk_indices(result, widths.data(), widths.size(), widths.size(), true);
    anchors = YOLOV5_ANCHORS_DEFAULT;
    for (int i = 0; i < num_opds; i++) {
      auto idx = result[i].first;
      operands.push_back(opds[idx]);
    }
    return;
  }
  // yolov8
  if ((post_type == "yolov8" || post_type == "yolov11") && num_opds == 1) {
    operands.push_back(opds[0]);
    anchors = {0};
    return;
  }

  if ((post_type == "yolov8_seg" || post_type == "yolov11_seg") &&
      num_opds == 2) {
    operands.push_back(opds[0]);
    operands.push_back(opds[1]);
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
  std::vector<double> anchors;
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
      builder.getNamedAttr("anchors", builder.getF64ArrayAttr(anchors)));
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

// add yolov8_seg post
void AddPostprocessPass::insertYolosegOp(OpBuilder &builder) {

  std::vector<Value> operands;
  std::vector<NamedAttribute> attrs;
  std::vector<double> anchors;

  getYoloOperandsAndAnchors(operands, anchors);
  auto predication = operands[0];
  auto proto = operands[1];
  float conf_thres = 0.25;
  float max_wh = 7680.;
  float iou_thres = 0.7;
  auto p_shape = module::getShape(predication);
  assert(p_shape.size() == 3);
  auto proto_shape = module::getShape(proto);
  assert(proto_shape.size() == 4);
  int mh = proto_shape[2]; // mask height
  int mw = proto_shape[3]; // mask width
  int nm = 32;
  int nc = p_shape[1] - 4 - nm; // number of classes 80
  int mi = 4 + nc;              // mask start index 84
  int max_boxes = 100;

  operands.clear(); // xc = prediction[:, 4:mi].amax(1) > conf_thres

  auto predication_op = dyn_cast<top::ConcatOp>(predication.getDefiningOp());
  auto proto_op = dyn_cast<top::MulOp>(proto.getDefiningOp());
  auto none = module::getNoneOp(predication_op);
  operands.push_back(predication);
  operands.push_back(none);
  operands.push_back(none);
  operands.push_back(none);

  std::vector<int64_t> offsets(p_shape.size(), 0);
  offsets[1] = 4;
  std::vector<int64_t> ends(p_shape.size(),
                            std::numeric_limits<int64_t>::max());
  ends[1] = mi;
  std::vector<int64_t> steps(p_shape.size(), 1);

  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr(offsets)));
  attrs.push_back(builder.getNamedAttr("ends", builder.getI64ArrayAttr(ends)));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr(steps)));
  std::vector<int64_t> slice_shape = p_shape;
  slice_shape[1] = nc;
  auto new_type = module::getTypeLike(predication_op.getOutput(), slice_shape);

  auto new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_slice1"));
  auto slice_op1 = builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  std::vector<_Float32> weight_p4_plus(p_shape[2], 0);
  weight_p4_plus[p_shape[2] - 1] = conf_thres + 1e-3;
  auto coeff_type =
      RankedTensorType::get({1, 1, p_shape[2]}, builder.getF32Type());
  auto weight_p4_plus_op1 = WeightOp::create(slice_op1, "weight_p4_plus_op1",
                                             weight_p4_plus, coeff_type);

  attrs.clear();
  attrs.push_back(builder.getNamedAttr("axis", builder.getSI32IntegerAttr(1)));
  operands.clear();
  operands.push_back(slice_op1.getOutput());
  operands.push_back(weight_p4_plus_op1);
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_p4"));
  new_type = RankedTensorType::get(
      {p_shape[0], nc + 1, p_shape[2]},
      module::getElementType(predication_op)); // 1 8400 116
  auto concat_op_p4 =
      builder.create<ConcatOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  attrs.push_back(builder.getNamedAttr("axes", builder.getI64ArrayAttr({1})));
  attrs.push_back(builder.getNamedAttr("keepdims", builder.getBoolAttr(false)));
  attrs.push_back(
      builder.getNamedAttr("mode", builder.getStringAttr("ReduceMax")));
  new_type = RankedTensorType::get({slice_shape[0], slice_shape[2]},
                                   module::getElementType(slice_op1));
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_reduce_max1"));
  auto reduce_max_op1 = builder.create<ReduceOp>(
      new_loc, new_type, concat_op_p4.getOutput(), attrs);
  attrs.clear();

  attrs.push_back(
      builder.getNamedAttr("const_val", builder.getF64FloatAttr(conf_thres)));
  attrs.push_back(
      builder.getNamedAttr("mode", builder.getStringAttr("Greater")));
  attrs.push_back(builder.getNamedAttr("inversed", builder.getBoolAttr(false)));
  new_type = RankedTensorType::get({slice_shape[0], slice_shape[2]},
                                   module::getElementType(slice_op1));
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_compare_1"));
  auto compare_const_op1 = builder.create<CompareConstOp>(
      new_loc, new_type, reduce_max_op1.getOutput(), attrs);
  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("order", builder.getI64ArrayAttr({0, 2, 1})));
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_permute_1"));
  new_type = RankedTensorType::get({p_shape[0], p_shape[2], p_shape[1]},
                                   module::getElementType(predication_op));
  auto permute_op1 = builder.create<PermuteOp>(
      new_loc, new_type, predication_op.getOutput(), attrs);
  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 0, 0})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr(steps)));
  slice_shape = {p_shape[0], p_shape[2], 4};
  attrs.push_back(
      builder.getNamedAttr("ends", builder.getI64ArrayAttr({slice_shape})));
  new_type = module::getTypeLike(predication_op.getOutput(), slice_shape);

  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_slice2"));
  operands.clear();
  operands.push_back(permute_op1.getOutput());
  operands.push_back(none);
  operands.push_back(none);
  operands.push_back(none);
  auto slice_op2 = builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  slice_shape = {p_shape[0], p_shape[2], p_shape[1] - 4};
  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 0, 4})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr(steps)));
  attrs.push_back(builder.getNamedAttr(
      "ends", builder.getI64ArrayAttr({p_shape[0], p_shape[2], p_shape[1]})));
  new_type = module::getTypeLike(predication_op.getOutput(), slice_shape);
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_slice3"));
  auto slice_op3 = builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 0, 2})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr(steps)));
  slice_shape = {p_shape[0], p_shape[2], 1};
  attrs.push_back(builder.getNamedAttr(
      "ends", builder.getI64ArrayAttr({p_shape[0], p_shape[2], 3})));
  auto dim3_type =
      module::getTypeLike(predication_op.getOutput(), slice_shape); // 1,8400

  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_slice1"));
  std::vector<Value> xywh2xyxy_operands;
  xywh2xyxy_operands.push_back(slice_op2.getOutput());
  xywh2xyxy_operands.push_back(none);
  xywh2xyxy_operands.push_back(none);
  xywh2xyxy_operands.push_back(none);
  auto xywh2xyxy_slice1_op =
      builder.create<SliceOp>(new_loc, dim3_type, xywh2xyxy_operands, attrs);

  attrs.clear();
  auto single_dim_shape = {slice_shape[0], slice_shape[1]};
  auto dim2_type = module::getTypeLike(predication_op.getOutput(),
                                       single_dim_shape); // 1,8400
  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_squeeze1"));
  auto squeeze_op1 = builder.create<ReshapeOp>(
      new_loc, dim2_type, xywh2xyxy_slice1_op.getOutput(), attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("const_val", builder.getF64FloatAttr(0.5)));
  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_mulconst1"));
  auto mulconst_op1 = builder.create<MulConstOp>(
      new_loc, dim2_type, squeeze_op1.getOutput(), attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 0, 3})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr(steps)));
  slice_shape = {p_shape[0], p_shape[2], 1};
  attrs.push_back(builder.getNamedAttr(
      "ends", builder.getI64ArrayAttr({p_shape[0], p_shape[2], 4})));
  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_slice2"));
  auto xywh2xyxy_slice2_op =
      builder.create<SliceOp>(new_loc, dim3_type, xywh2xyxy_operands, attrs);

  attrs.clear();
  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_squeeze2"));
  auto squeeze_op2 = builder.create<ReshapeOp>(
      new_loc, dim2_type, xywh2xyxy_slice2_op.getOutput(), attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("const_val", builder.getF64FloatAttr(0.5)));
  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_mulconst2"));
  auto mulconst_op2 = builder.create<MulConstOp>(
      new_loc, dim2_type, squeeze_op2.getOutput(), attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 0, 0})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr(steps)));
  std::vector<int64_t> ends2(p_shape.size(),
                             std::numeric_limits<int64_t>::max());
  attrs.push_back(builder.getNamedAttr("ends", builder.getI64ArrayAttr(ends2)));

  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_slice3"));
  auto xywh2xyxy_slice3_op =
      builder.create<SliceOp>(new_loc, dim3_type, xywh2xyxy_operands, attrs);

  attrs.clear();
  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_squeeze3"));
  auto squeeze_op3 = builder.create<ReshapeOp>(
      new_loc, dim2_type, xywh2xyxy_slice3_op.getOutput(), attrs);

  attrs.clear();
  operands.clear();
  operands.push_back(squeeze_op3.getOutput());
  operands.push_back(mulconst_op1.getOutput());
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_sub1"));
  auto sub_op1 = builder.create<SubOp>(new_loc, dim2_type, operands, attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 0, 1})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr(steps)));
  attrs.push_back(builder.getNamedAttr(
      "ends", builder.getI64ArrayAttr({p_shape[0], p_shape[2], 2})));

  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_slice4"));

  auto xywh2xyxy_slice4_op =
      builder.create<SliceOp>(new_loc, dim3_type, xywh2xyxy_operands, attrs);

  attrs.clear();
  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_squeeze4"));
  auto squeeze_op4 = builder.create<ReshapeOp>(
      new_loc, dim2_type, xywh2xyxy_slice4_op.getOutput(), attrs);

  attrs.clear();
  operands.clear();
  operands.push_back(squeeze_op4.getOutput());
  operands.push_back(mulconst_op2.getOutput());
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_sub2"));
  auto sub_op2 = builder.create<SubOp>(new_loc, dim2_type, operands, attrs);

  operands.clear();
  operands.push_back(squeeze_op3.getOutput());
  operands.push_back(mulconst_op1.getOutput());
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_add1"));
  auto add_op1 = builder.create<AddOp>(new_loc, dim2_type, operands, attrs);

  operands.clear();
  operands.push_back(squeeze_op4.getOutput());
  operands.push_back(mulconst_op2.getOutput());
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_add2"));
  auto add_op2 = builder.create<AddOp>(new_loc, dim2_type, operands, attrs);

  attrs.push_back(builder.getNamedAttr("axes", builder.getI64ArrayAttr({-1})));
  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_unsqueeze1"));
  auto un_squeeze_op1 = builder.create<UnsqueezeOp>(new_loc, dim3_type,
                                                    sub_op1.getOutput(), attrs);
  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_unsqueeze2"));
  auto un_squeeze_op2 = builder.create<UnsqueezeOp>(new_loc, dim3_type,
                                                    sub_op2.getOutput(), attrs);
  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_unsqueeze3"));
  auto un_squeeze_op3 = builder.create<UnsqueezeOp>(new_loc, dim3_type,
                                                    add_op1.getOutput(), attrs);
  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_xywh2xyxy_unsqueeze4"));
  auto un_squeeze_op4 = builder.create<UnsqueezeOp>(new_loc, dim3_type,
                                                    add_op2.getOutput(), attrs);
  attrs.clear();
  attrs.push_back(builder.getNamedAttr("axis", builder.getSI32IntegerAttr(2)));
  operands.clear();
  operands.push_back(un_squeeze_op1.getOutput());
  operands.push_back(un_squeeze_op2.getOutput());
  operands.push_back(un_squeeze_op3.getOutput());
  operands.push_back(un_squeeze_op4.getOutput());
  operands.push_back(slice_op3.getOutput());
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_concat1"));
  new_type = RankedTensorType::get(
      {p_shape[0], p_shape[2], p_shape[1]},
      module::getElementType(predication_op)); // 1 8400 116
  auto concat_op1 =
      builder.create<ConcatOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_squeeze5"));
  auto new_type2 =
      RankedTensorType::get({p_shape[2], p_shape[1]},
                            module::getElementType(predication_op)); // 8400 116
  auto squeeze_op5 = builder.create<ReshapeOp>(
      new_loc, new_type2, concat_op1.getOutput(), attrs); // x

  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_squeeze6"));
  new_type =
      RankedTensorType::get({p_shape[2]},
                            module::getElementType(predication_op)); // 8400
  auto squeeze_op6 = builder.create<ReshapeOp>(
      new_loc, new_type, compare_const_op1.getOutput(), attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("order", builder.getStringAttr("ColMajor")));
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_nonzero1"));
  new_type =
      RankedTensorType::get({p_shape[2], 1},
                            module::getElementType(predication_op)); // 8400 1
  auto non_zero_op1 = builder.create<NonZeroOp>(new_loc, new_type,
                                                squeeze_op6.getOutput(), attrs);
  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("batch_dims", builder.getI64IntegerAttr(0)));
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_GatherND1"));
  operands.clear();
  operands.push_back(squeeze_op5.getOutput());
  operands.push_back(non_zero_op1.getOutput());
  new_type = RankedTensorType::get({p_shape[2], p_shape[1]},
                                   module::getElementType(predication_op));
  auto gatherND_op1 =
      builder.create<GatherNDOp>(new_loc, new_type, operands, attrs);
  operands.clear();
  operands.push_back(gatherND_op1.getOutput());
  operands.push_back(none);
  operands.push_back(none);
  operands.push_back(none);
  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 0})));
  attrs.push_back(
      builder.getNamedAttr("ends", builder.getI64ArrayAttr({p_shape[2], 4})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr({1, 1})));

  new_type =
      RankedTensorType::get({p_shape[2], 4},
                            module::getElementType(predication_op)); // 8400 4

  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_boxes_origin"));
  auto boxes_op = builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 4})));
  attrs.push_back(builder.getNamedAttr(
      "ends", builder.getI64ArrayAttr({p_shape[2], nc + 4})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr({1, 1})));

  new_type =
      RankedTensorType::get({p_shape[2], nc},
                            module::getElementType(predication_op)); // 8400 80

  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_clses_origin"));
  auto clses_op = builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, nc + 4})));
  attrs.push_back(builder.getNamedAttr(
      "ends", builder.getI64ArrayAttr({p_shape[2], p_shape[1]})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr({1, 1})));

  new_type =
      RankedTensorType::get({p_shape[2], nm},
                            module::getElementType(predication_op)); // 8400 32

  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_masks_origin"));
  auto masks_op = builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  attrs.push_back(builder.getNamedAttr("axis", builder.getI64IntegerAttr(1)));
  attrs.push_back(builder.getNamedAttr("keepdims", builder.getBoolAttr(true)));
  attrs.push_back(
      builder.getNamedAttr("mode", builder.getStringAttr("ArgMax")));
  attrs.push_back(
      builder.getNamedAttr("select_last_index", builder.getBoolAttr(false)));
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_cls_argmax"));

  std::vector<Type> argmax_type;
  argmax_type.push_back(RankedTensorType::get(
      {p_shape[2], 1}, module::getElementType(predication_op)));
  argmax_type.push_back(RankedTensorType::get(
      {p_shape[2], 1}, module::getElementType(predication_op)));

  std::vector<Location> locs_v = {new_loc};
  std::string out_values_name =
      module::getName(clses_op.getOutput()).str() + "_values";
  auto values_loc = NameLoc::get(builder.getStringAttr(out_values_name));
  locs_v.push_back(values_loc);

  auto fused_loc = FusedLoc::get(&getContext(), locs_v);

  auto argmax_op1 = builder.create<ArgOp>(fused_loc, argmax_type,
                                          clses_op.getOutput(), attrs);

  attrs.clear();
  attrs.push_back(builder.getNamedAttr("axis", builder.getSI32IntegerAttr(1)));
  operands.clear();
  operands.push_back(boxes_op.getOutput());
  operands.push_back(argmax_op1.getValues());
  operands.push_back(argmax_op1.getIndices());
  operands.push_back(masks_op.getOutput());
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_concat2"));
  new_type = RankedTensorType::get({p_shape[2], p_shape[1] - nc + 2},
                                   module::getElementType(predication_op));
  auto concat_op2 =
      builder.create<ConcatOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  auto dim1_type =
      module::getTypeLike(predication_op.getOutput(), {p_shape[2]}); // 8400
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_squeeze7"));
  auto squeeze_op7 = builder.create<ReshapeOp>(new_loc, dim1_type,
                                               argmax_op1.getValues(), attrs);

  operands.clear();
  operands.push_back(squeeze_op7.getOutput());
  operands.push_back(none);
  operands.push_back(none);
  operands.push_back(none);
  attrs.clear();
  attrs.push_back(builder.getNamedAttr("offset", builder.getI64ArrayAttr({0})));
  attrs.push_back(builder.getNamedAttr("ends", builder.getI64ArrayAttr({-1})));
  attrs.push_back(builder.getNamedAttr("steps", builder.getI64ArrayAttr({1})));

  new_type =
      RankedTensorType::get({p_shape[2] - 1},
                            module::getElementType(predication_op)); // 8400 4

  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_c1_slice"));
  auto c1_slice_op =
      builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  std::vector<_Float32> weight_c1_plus(1, conf_thres + 1e-3);
  coeff_type = RankedTensorType::get(1, builder.getF32Type());
  auto weight_c1_plus_op1 = WeightOp::create(slice_op1, "weight_c1_plus_op1",
                                             weight_c1_plus, coeff_type);

  attrs.clear();
  attrs.push_back(builder.getNamedAttr("axis", builder.getSI32IntegerAttr(0)));
  operands.clear();
  operands.push_back(c1_slice_op.getOutput());
  operands.push_back(weight_c1_plus_op1);
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_c1_plus_concat"));
  new_type = RankedTensorType::get(
      {p_shape[2]},
      module::getElementType(predication_op)); // 1 8400 116
  auto c1_plus_concat_op =
      builder.create<ConcatOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("const_val", builder.getF64FloatAttr(conf_thres)));
  attrs.push_back(
      builder.getNamedAttr("mode", builder.getStringAttr("Greater")));
  attrs.push_back(builder.getNamedAttr("inversed", builder.getBoolAttr(false)));

  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_compare_2"));
  auto compare_const_op2 = builder.create<CompareConstOp>(
      new_loc, dim1_type, c1_plus_concat_op.getOutput(), attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("order", builder.getStringAttr("ColMajor")));
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_nonzero2"));
  new_type = RankedTensorType::get({p_shape[2], 1},
                                   module::getElementType(predication_op));
  auto non_zero_op2 = builder.create<NonZeroOp>(
      new_loc, new_type, compare_const_op2.getOutput(), attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("batch_dims", builder.getI64IntegerAttr(0)));
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_GatherND2"));
  operands.clear();
  operands.push_back(concat_op2.getOutput());
  operands.push_back(non_zero_op2.getOutput());
  new_type = RankedTensorType::get({p_shape[2], p_shape[1] - nc + 2},
                                   module::getElementType(predication_op));
  auto gatherND_op2 =
      builder.create<GatherNDOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 5})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr({1, 1})));
  attrs.push_back(
      builder.getNamedAttr("ends", builder.getI64ArrayAttr({p_shape[2], 6})));
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_slice4"));
  operands.clear();
  operands.push_back(gatherND_op2.getOutput());
  operands.push_back(none);
  operands.push_back(none);
  operands.push_back(none);
  new_type = RankedTensorType::get({p_shape[2], 1},
                                   module::getElementType(predication_op));
  auto slice_op4 = builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_slice5"));
  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 4})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr({1, 1})));
  attrs.push_back(
      builder.getNamedAttr("ends", builder.getI64ArrayAttr({p_shape[2], 5})));
  auto slice_op5 = builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_slice6"));
  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 0})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr({1, 1})));
  attrs.push_back(
      builder.getNamedAttr("ends", builder.getI64ArrayAttr({p_shape[2], 4})));
  new_type =
      RankedTensorType::get({p_shape[2], 4},
                            module::getElementType(predication_op)); // 8400 4
  auto slice_op6 = builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("const_val", builder.getF64FloatAttr(max_wh)));
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_mulconst3"));
  new_type = RankedTensorType::get({p_shape[2], 1},
                                   module::getElementType(predication_op));
  auto mulconst_op3 = builder.create<MulConstOp>(new_loc, new_type,
                                                 slice_op4.getOutput(), attrs);

  attrs.clear();
  operands.clear();
  operands.push_back(slice_op6.getOutput());
  operands.push_back(mulconst_op3.getOutput());
  new_type = RankedTensorType::get({p_shape[2], 4},
                                   module::getElementType(predication_op));
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_add3"));
  auto add_op3 = builder.create<AddOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  attrs.push_back(builder.getNamedAttr("axes", builder.getI64ArrayAttr({0})));

  new_type =
      module::getTypeLike(predication_op.getOutput(), {1, p_shape[2], 4});
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_unsqueeze6"));
  auto unsqueeze_op6 = builder.create<UnsqueezeOp>(new_loc, new_type,
                                                   add_op3.getOutput(), attrs);

  attrs.clear();
  attrs.push_back(builder.getNamedAttr("axes", builder.getI64ArrayAttr({1})));

  new_type = module::getTypeLike(predication_op.getOutput(), {p_shape[2]});
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_conf_squeeze"));
  auto conf_squeeze = builder.create<SqueezeOp>(new_loc, new_type,
                                                slice_op5.getOutput(), attrs);

  attrs.clear();
  attrs.push_back(builder.getNamedAttr("axes", builder.getI64ArrayAttr({0})));

  new_type = module::getTypeLike(predication_op.getOutput(), {1, p_shape[2]});
  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_conf_unsqueeze1"));
  auto conf_unsqueeze1_op = builder.create<UnsqueezeOp>(
      new_loc, new_type, conf_squeeze.getOutput(), attrs);

  attrs.clear();
  attrs.push_back(builder.getNamedAttr("axes", builder.getI64ArrayAttr({0})));
  new_type =
      module::getTypeLike(predication_op.getOutput(), {1, 1, p_shape[2]});
  new_loc =
      NameLoc::get(builder.getStringAttr("yolo_seg_post_conf_unsqueeze2"));
  auto unsqueeze_op7 = builder.create<UnsqueezeOp>(
      new_loc, new_type, conf_unsqueeze1_op.getOutput(), attrs);

  std::vector<_Float32> weight1(1, std::numeric_limits<_Float32>::max());
  coeff_type = RankedTensorType::get(1, builder.getF32Type());
  auto weight_op1 =
      WeightOp::create(unsqueeze_op7, "weight1", weight1, coeff_type);

  std::vector<_Float32> weight2(1, iou_thres);
  coeff_type = RankedTensorType::get(1, builder.getF32Type());
  auto weight_op2 =
      WeightOp::create(unsqueeze_op7, "weight2", weight2, coeff_type);

  std::vector<_Float32> weight3(1, 0.0);
  coeff_type = RankedTensorType::get(1, builder.getF32Type());
  auto weight_op3 =
      WeightOp::create(unsqueeze_op7, "weight3", weight3, coeff_type);

  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("center_point_box", builder.getI64IntegerAttr(0)));
  attrs.push_back(builder.getNamedAttr("max_output_size",
                                       builder.getI64IntegerAttr(10000)));
  operands.clear();
  operands.push_back(unsqueeze_op6.getOutput());
  operands.push_back(unsqueeze_op7.getOutput());
  operands.push_back(weight_op1);
  operands.push_back(weight_op2);
  operands.push_back(weight_op3);

  new_type = module::getTypeLike(predication_op.getOutput(), {p_shape[2], 3});
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_nms"));

  auto nms_op = builder.create<NmsOp>(new_loc, new_type, operands, attrs);

  operands.clear();
  operands.push_back(nms_op.getOutput());
  operands.push_back(none);
  operands.push_back(none);
  operands.push_back(none);
  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 2})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr({1, 1})));
  attrs.push_back(
      builder.getNamedAttr("ends", builder.getI64ArrayAttr({p_shape[2], 3})));
  new_type =
      RankedTensorType::get({p_shape[2], 1},
                            module::getElementType(predication_op)); // 8400 1
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_slice7"));
  auto slice_op7 = builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  new_type =
      module::getTypeLike(predication_op.getOutput(), {p_shape[2]}); // 8400
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_squeeze8"));
  auto squeeze_op8 = builder.create<ReshapeOp>(new_loc, new_type,
                                               slice_op7.getOutput(), attrs);

  attrs.clear();
  new_type = module::getTypeLike(predication_op.getOutput(),
                                 {proto_shape[1], mh, mw}); // 32 160 160
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_squeeze9"));
  auto squeeze_op9 =
      builder.create<ReshapeOp>(new_loc, new_type, proto_op.getOutput(), attrs);

  attrs.clear();
  attrs.push_back(builder.getNamedAttr("axis", builder.getSI32IntegerAttr(0)));
  operands.clear();
  operands.push_back(gatherND_op2.getOutput());
  operands.push_back(squeeze_op8.getOutput());
  new_type = RankedTensorType::get({p_shape[2], p_shape[1] - nc + 2},
                                   module::getElementType(predication_op));
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_gather1"));
  auto gather_op1 =
      builder.create<GatherOp>(new_loc, new_type, operands, attrs);

  operands.clear();
  operands.push_back(gather_op1.getOutput());
  operands.push_back(none);
  operands.push_back(none);
  operands.push_back(none);
  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 6})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr({1, 1})));
  attrs.push_back(builder.getNamedAttr(
      "ends", builder.getI64ArrayAttr({p_shape[2], p_shape[1] - nc + 2})));
  new_type =
      RankedTensorType::get({p_shape[2], nm},
                            module::getElementType(predication_op)); // 8400 1
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_slice8"));
  auto slice_op8 = builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  new_type = module::getTypeLike(predication_op.getOutput(), {nm, mh * mw});
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_reshape1"));
  auto reshape_op1 = builder.create<ReshapeOp>(new_loc, new_type,
                                               squeeze_op9.getOutput(), attrs);

  operands.clear();
  operands.push_back(slice_op8.getOutput());
  operands.push_back(none);
  operands.push_back(none);
  operands.push_back(none);
  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 0, 0})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr({1, 1, 1})));
  attrs.push_back(
      builder.getNamedAttr("ends", builder.getI64ArrayAttr({max_boxes, nm})));
  new_type = RankedTensorType::get({max_boxes, nm},
                                   module::getElementType(predication_op));
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_slice9"));
  auto yolo_seg_post_slice9 =
      builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  operands.clear();
  operands.push_back(yolo_seg_post_slice9.getOutput());
  operands.push_back(reshape_op1.getOutput());
  operands.push_back(none);
  new_type =
      module::getTypeLike(predication_op.getOutput(), {max_boxes, mh * mw});
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_matmul1"));
  auto matmul_op1 =
      builder.create<MatMulOp>(new_loc, new_type, operands, attrs);

  attrs.clear();
  new_loc = NameLoc::get(builder.getStringAttr("yolo_seg_post_sigmoid1"));
  auto sigmoid_op1 = builder.create<SigmoidOp>(new_loc, new_type,
                                               matmul_op1.getOutput(), attrs);

  new_type =
      module::getTypeLike(predication_op.getOutput(), {max_boxes, mh, mw});
  new_loc = NameLoc::get(builder.getStringAttr("masks_uncrop_uncompare"));
  auto masks_uncrop_uncompare_op = builder.create<ReshapeOp>(
      new_loc, new_type, sigmoid_op1.getOutput(), attrs);

  operands.clear();
  operands.push_back(gather_op1.getOutput());
  operands.push_back(none);
  operands.push_back(none);
  operands.push_back(none);
  attrs.clear();
  attrs.push_back(
      builder.getNamedAttr("offset", builder.getI64ArrayAttr({0, 0})));
  attrs.push_back(
      builder.getNamedAttr("steps", builder.getI64ArrayAttr({1, 1})));
  attrs.push_back(
      builder.getNamedAttr("ends", builder.getI64ArrayAttr({max_boxes, 6})));
  new_type =
      RankedTensorType::get({max_boxes, 6},
                            module::getElementType(predication_op)); // 8400 1
  new_loc = NameLoc::get(builder.getStringAttr("seg_out"));
  auto seg_out_op = builder.create<SliceOp>(new_loc, new_type, operands, attrs);

  //   attrs.clear();
  //   attrs.push_back(
  //       builder.getNamedAttr("const_val", builder.getF64FloatAttr(0.5)));
  //   attrs.push_back(
  //       builder.getNamedAttr("mode", builder.getStringAttr("Greater")));
  //   attrs.push_back(
  //       builder.getNamedAttr("inversed", builder.getBoolAttr(false)));
  //   new_type = RankedTensorType::get({max_boxes, mh, mw},
  //                                       module::getElementType(predication_op));
  //   new_loc = NameLoc::get(builder.getStringAttr("masks_uncrop"));
  //   auto masks_uncrop_op =
  //           builder.create<CompareConstOp>(new_loc, new_type,
  //           masks_uncrop_uncompare_op.getOutput(), attrs);

  terminator->setOperands(
      {masks_uncrop_uncompare_op.getOutput(), seg_out_op.getOutput()});
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
  float white_level = 4095.;
  float black_level = 112.;
  std::string pixel_format;
  func.walk([&](top::InputOp inputOp) {
    if (inputOp.getDoPreprocess()) {
      pixel_format = inputOp.getPixelFormat()->str();
      return;
    }
  });
  auto operands = terminator->getOperands();
  auto opd = operands[0];
  auto shape = module::getShape(opd);
  int padding_h = 0;
  int padding_w = 0;
  int oh = (shape[2] - padding_h) * 2;
  int ow = (shape[3] - padding_w) * 3;
  std::vector<int64_t> channel_order;
  // RGBG->(3, 2, 0, 1)->GBRG
  // RGBG->(1, 0, 2, 3)->GRBG
  // RGBG->(0, 1, 3, 2)->RGGB
  // RGBG->(2, 3, 1, 0)->BGGR
  if (pixel_format == "gbrg")
    channel_order = {3, 2, 0, 1};
  else if (pixel_format == "grbg")
    channel_order = {1, 0, 2, 3};
  else if (pixel_format == "rggb")
    channel_order = {0, 1, 3, 2};
  else if (pixel_format == "bggr")
    channel_order = {2, 3, 1, 0};
  else
    llvm_unreachable("raw format not support current type");
  std::vector<NamedAttribute> attrs;
  attrs.emplace_back(
      builder.getNamedAttr("padding_h", builder.getI64IntegerAttr(padding_h)));
  attrs.emplace_back(
      builder.getNamedAttr("padding_w", builder.getI64IntegerAttr(padding_w)));
  attrs.emplace_back(builder.getNamedAttr(
      "white_level", builder.getF64FloatAttr(white_level)));
  attrs.emplace_back(builder.getNamedAttr(
      "black_level", builder.getF64FloatAttr(black_level)));
  attrs.emplace_back(builder.getNamedAttr(
      "channel_order", builder.getI64ArrayAttr(channel_order)));
  auto loc = NameLoc::get(builder.getStringAttr("depack_raw"));
  auto new_type = RankedTensorType::get({batch, 1, oh, ow},
                                        builder.getIntegerType(8, false));
  auto post_op = builder.create<top::DepackRawOp>(loc, new_type, opd, attrs);
  terminator->setOperand(0, post_op.getOutput());
}

void AddPostprocessPass::insertMmap2RgbmapOp(OpBuilder &builder) {
  auto operands = terminator->getOperands();
  auto opd = operands[0];
  auto shape = module::getShape(opd);
  int oh = shape[2];
  int ow = shape[3] * 6;

  auto loc = NameLoc::get(builder.getStringAttr("mmap2rgbmap"));
  auto new_type = RankedTensorType::get({batch, 1, oh, ow},
                                        builder.getIntegerType(8, false));
  auto post_op = builder.create<top::Mmap2RgbmapOp>(loc, new_type, opd);
  terminator->setOperand(0, post_op.getOutput());
}

std::unique_ptr<OperationPass<ModuleOp>> createAddPostprocessPass() {
  return std::make_unique<AddPostprocessPass>();
}
} // namespace top
} // namespace tpu_mlir
