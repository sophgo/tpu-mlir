//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::GenericCpuOp::init(InferenceParameter &p) {
  return success();
}
void tpu::GenericCpuOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::GenericCpuOp::inference(InferenceParameter &p) {
  std::string func_name = getCpuOpName().str();
  if (func_name == "quant") {
    assert(getInputs().size() == 1);
    auto num_elem = module::getNumElements(getOutputs()[0]);
    auto in_type = module::getStorageType(getInputs()[0]);
    auto out_type = module::getStorageType(getOutputs()[0]);
    if (in_type.isF32() && out_type.isSignedInteger()) {
      mlir::DictionaryAttr param = this->getParam().value();
      float scale = param.get("scale").cast<FloatAttr>().getValueAsDouble();
      quantizeToInt8(p.inputs[0], p.outputs[0], num_elem, scale);
    } else {
      llvm_unreachable("not supported!\n");
    }
  } else if (func_name == "interp") {
    InterpParam interp_param;
    mlir::DictionaryAttr param = this->getParam().value();
    interp_param.height = param.get("height").cast<IntegerAttr>().getInt();
    interp_param.width = param.get("width").cast<IntegerAttr>().getInt();
    interp_param.pad_beg = param.get("pad_beg").cast<IntegerAttr>().getInt();
    interp_param.pad_end = param.get("pad_end").cast<IntegerAttr>().getInt();
    interp_param.shrink_factor =
        param.get("shrink_factor").cast<IntegerAttr>().getInt();
    interp_param.zoom_factor =
        param.get("zoom_factor").cast<IntegerAttr>().getInt();
    interp_param.coordinate_transformation_mode =
        param.get("coordinate_transformation_mode")
            .cast<StringAttr>()
            .getValue();
    for (size_t i = 0; i < getInputs().size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = p.inputs[i];
      tensor_list.size = module::getNumElements(getInputs()[i]);
      tensor_list.shape = module::getShape(getInputs()[i]);
      interp_param.inputs.emplace_back(std::move(tensor_list));
    }
    interp_param.output.ptr = p.outputs[0];
    interp_param.output.size = module::getNumElements(getOutputs()[0]);
    interp_param.output.shape = module::getShape(getOutputs()[0]);
    assert(interp_param.inputs[0].shape.size() == 4);
    assert(interp_param.output.shape.size() == 4);
    InterpFunc interp_func(interp_param);
    interp_func.invoke();
  } else if (func_name == "embedding") {
    EmbeddingParam embedding_param;
    for (size_t i = 0; i < getInputs().size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = p.inputs[i];
      tensor_list.size = module::getNumElements(getInputs()[i]);
      tensor_list.shape = module::getShape(getInputs()[i]);
      embedding_param.inputs.emplace_back(std::move(tensor_list));
    }
    embedding_param.output.ptr = p.outputs[0];
    embedding_param.output.size = module::getNumElements(getOutputs()[0]);
    embedding_param.output.shape = module::getShape(getOutputs()[0]);
    EmbeddingFunc embedding_func(embedding_param);
    embedding_func.invoke();
  } else if (func_name == "detectionoutput") {
    DetParam det_param;
    det_param.loc_shape = module::getShape(getInputs()[0]);
    det_param.conf_shape = module::getShape(getInputs()[1]);
    det_param.prior_shape = module::getShape(getInputs()[2]);
    det_param.onnx_nms = p.inputs.size() >= 3 ? 0 : 1;
    det_param.loc_data = p.inputs[0];
    det_param.conf_data = p.inputs[1];
    det_param.prior_data = det_param.onnx_nms ? nullptr : p.inputs[2];
    det_param.output_data = p.outputs[0];
    mlir::DictionaryAttr param = this->getParam().value();
    det_param.keep_top_k = param.get("keep_top_k").cast<IntegerAttr>().getInt();
    det_param.top_k = param.get("top_k").cast<IntegerAttr>().getInt();
    det_param.num_classes =
        param.get("num_classes").cast<IntegerAttr>().getInt();
    det_param.background_label_id =
        param.get("background_label_id").cast<IntegerAttr>().getInt();
    det_param.share_location =
        param.get("share_location").cast<BoolAttr>().getValue();
    det_param.confidence_threshold =
        param.get("confidence_threshold").cast<FloatAttr>().getValueAsDouble();
    det_param.nms_threshold =
        param.get("nms_threshold").cast<FloatAttr>().getValueAsDouble();
    std::string str_code_type =
        param.get("code_type").cast<StringAttr>().getValue().str();
    if (str_code_type == "CORNER") {
      det_param.code_type = PriorBoxParameter_CodeType_CORNER;
    } else if (str_code_type == "CENTER_SIZE") {
      det_param.code_type = PriorBoxParameter_CodeType_CENTER_SIZE;
    } else if (str_code_type == "CORNER_SIZE") {
      det_param.code_type = PriorBoxParameter_CodeType_CORNER_SIZE;
    } else {
      llvm_unreachable("code type wrong");
    }
    DetectionOutputFunc det_func(det_param);
    det_func.invoke();
  } else if (func_name == "yolo_detection") {
    YoloDetParam yolo_param;
    mlir::DictionaryAttr param = this->getParam().value();
    yolo_param.keep_topk = param.get("keep_topk").cast<IntegerAttr>().getInt();
    yolo_param.class_num = param.get("class_num").cast<IntegerAttr>().getInt();
    yolo_param.net_input_h =
        param.get("net_input_h").cast<IntegerAttr>().getInt();
    yolo_param.net_input_w =
        param.get("net_input_w").cast<IntegerAttr>().getInt();
    yolo_param.nms_threshold =
        param.get("nms_threshold").cast<FloatAttr>().getValueAsDouble();
    yolo_param.obj_threshold =
        param.get("obj_threshold").cast<FloatAttr>().getValueAsDouble();
    auto anchors = param.get("anchors").cast<StringAttr>().getValue().str();
    std::istringstream iss(anchors);
    std::string s;
    while (std::getline(iss, s, ',')) {
      yolo_param.anchors.push_back(atoi(s.c_str()));
    }
    for (size_t i = 0; i < getInputs().size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = p.inputs[i];
      tensor_list.size = module::getNumElements(getInputs()[i]);
      tensor_list.shape = module::getShape(getInputs()[i]);
      yolo_param.inputs.emplace_back(std::move(tensor_list));
    }
    yolo_param.output.ptr = p.outputs[0];
    yolo_param.output.size = module::getNumElements(getOutputs()[0]);
    yolo_param.output.shape = module::getShape(getOutputs()[0]);
    YoloDetectionFunc yolo_func(yolo_param);
    yolo_func.invoke();
  } else if (func_name == "proposal") {
    ProposalParam proposal_param;
    mlir::DictionaryAttr param = this->getParam().value();
    proposal_param.anchor_base_size =
        param.get("anchor_base_size").cast<IntegerAttr>().getInt();
    proposal_param.feat_stride =
        param.get("feat_stride").cast<IntegerAttr>().getInt();
    proposal_param.net_input_h =
        param.get("net_input_h").cast<IntegerAttr>().getInt();
    proposal_param.net_input_w =
        param.get("net_input_w").cast<IntegerAttr>().getInt();
    proposal_param.rpn_nms_post_top_n =
        param.get("rpn_nms_post_top_n").cast<IntegerAttr>().getInt();
    proposal_param.rpn_obj_threshold =
        param.get("rpn_obj_threshold").cast<FloatAttr>().getValueAsDouble();
    proposal_param.rpn_nms_threshold =
        param.get("rpn_nms_threshold").cast<FloatAttr>().getValueAsDouble();
    for (size_t i = 0; i < getInputs().size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = p.inputs[i];
      tensor_list.size = module::getNumElements(getInputs()[i]);
      tensor_list.shape = module::getShape(getInputs()[i]);
      proposal_param.inputs.emplace_back(std::move(tensor_list));
    }
    proposal_param.output.ptr = p.outputs[0];
    proposal_param.output.size = module::getNumElements(getOutputs()[0]);
    proposal_param.output.shape = module::getShape(getOutputs()[0]);
    ProposalFunc proposal_func(proposal_param);
    proposal_func.invoke();
  } else if (func_name == "roi_pooling") {
    ROIPoolingParam roip_param;
    mlir::DictionaryAttr param = this->getParam().value();
    roip_param.pooled_h = param.get("pooled_h").cast<IntegerAttr>().getInt();
    roip_param.pooled_w = param.get("pooled_w").cast<IntegerAttr>().getInt();
    roip_param.spatial_scale =
        param.get("spatial_scale").cast<FloatAttr>().getValueAsDouble();
    for (size_t i = 0; i < getInputs().size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = p.inputs[i];
      tensor_list.size = module::getNumElements(getInputs()[i]);
      tensor_list.shape = module::getShape(getInputs()[i]);
      roip_param.inputs.emplace_back(std::move(tensor_list));
    }
    roip_param.output.ptr = p.outputs[0];
    roip_param.output.size = module::getNumElements(getOutputs()[0]);
    roip_param.output.shape = module::getShape(getOutputs()[0]);
    ROIPoolingFunc roip_func(roip_param);
    roip_func.invoke();
  } else if (func_name == "roi_align") {
    RoiAlignParam roia_param;
    mlir::DictionaryAttr param = this->getParam().value();
    roia_param.pooled_h =
        param.get("output_height").cast<IntegerAttr>().getInt();
    roia_param.pooled_w =
        param.get("output_width").cast<IntegerAttr>().getInt();
    roia_param.sampling_ratio =
        param.get("sampling_ratio").cast<IntegerAttr>().getInt();
    roia_param.spatial_scale =
        param.get("spatial_scale").cast<FloatAttr>().getValueAsDouble();
    roia_param.aligned = param.get("align_corners").cast<BoolAttr>().getValue();
    std::string str_roia_type =
        param.get("mode").cast<StringAttr>().getValue().str();
    if (str_roia_type == "Avg") {
      roia_param.mode = RoiAlignAvgMode;
    } else {
      llvm_unreachable("code type wrong");
    }
    tensor_list_t input_list, roi_list;
    input_list.ptr = p.inputs[0];
    input_list.size = module::getNumElements(getInputs()[0]);
    input_list.shape = module::getShape(getInputs()[0]);
    roia_param.inputs.emplace_back(std::move(input_list));
    roi_list.ptr = p.inputs[0];
    roi_list.size = module::getNumElements(getInputs()[1]);
    roi_list.shape = module::getShape(getInputs()[1]);
    roia_param.inputs.emplace_back(std::move(roi_list));
    roia_param.output.ptr = p.outputs[0];
    roia_param.output.size = module::getNumElements(getOutputs()[0]);
    roia_param.output.shape = module::getShape(getOutputs()[0]);
    RoiAlignFunc roia_func(roia_param);
    roia_func.invoke();
  } else if (func_name == "frcn_detection") {
    FrcnDetParam frcn_param;
    mlir::DictionaryAttr param = this->getParam().value();
    frcn_param.class_num = param.get("class_num").cast<IntegerAttr>().getInt();
    frcn_param.keep_topk = param.get("keep_topk").cast<IntegerAttr>().getInt();
    frcn_param.nms_threshold =
        param.get("nms_threshold").cast<FloatAttr>().getValueAsDouble();
    frcn_param.obj_threshold =
        param.get("obj_threshold").cast<FloatAttr>().getValueAsDouble();
    for (size_t i = 0; i < getInputs().size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = p.inputs[i];
      tensor_list.size = module::getNumElements(getInputs()[i]);
      tensor_list.shape = module::getShape(getInputs()[i]);
      frcn_param.inputs.emplace_back(std::move(tensor_list));
    }
    frcn_param.output.ptr = p.outputs[0];
    frcn_param.output.size = module::getNumElements(getOutputs()[0]);
    frcn_param.output.shape = module::getShape(getOutputs()[0]);
    FrcnDetctionFunc frcn_func(frcn_param);
    frcn_func.invoke();
  } else if (func_name == "retinaface_detection") {
    RetinaFaceDetectionParam retina_param;
    mlir::DictionaryAttr param = this->getParam().value();
    retina_param.keep_topk =
        param.get("keep_topk").cast<IntegerAttr>().getInt();
    retina_param.confidence_threshold =
        param.get("confidence_threshold").cast<FloatAttr>().getValueAsDouble();
    retina_param.nms_threshold =
        param.get("nms_threshold").cast<FloatAttr>().getValueAsDouble();
    RetinaFaceDetectionFunc func;
    std::vector<tensor_list_t> inputs;
    for (size_t i = 0; i < getInputs().size(); i++) {
      tensor_list_t tensor;
      tensor.ptr = p.inputs[i];
      tensor.size = module::getNumElements(getInputs()[i]);
      tensor.shape = module::getShape(getInputs()[i]);
      inputs.emplace_back(std::move(tensor));
    }
    tensor_list_t output;
    output.ptr = p.outputs[0];
    output.size = module::getNumElements(getOutputs()[0]);
    output.shape = module::getShape(getOutputs()[0]);
    func.setup(inputs, output, retina_param);
    func.invoke();
  } else if (func_name == "instance_norm") {
    InstanceNormParam inst_param;
    mlir::DictionaryAttr param = this->getParam().value();
    inst_param.eps =
        param.get("variance_epsilon").cast<FloatAttr>().getValueAsDouble();
    for (size_t i = 0; i < getInputs().size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = p.inputs[i];
      tensor_list.size = module::getNumElements(getInputs()[i]);
      tensor_list.shape = module::getShape(getInputs()[i]);
      inst_param.inputs.emplace_back(std::move(tensor_list));
    }
    inst_param.output.ptr = p.outputs[0];
    inst_param.output.size = module::getNumElements(getOutputs()[0]);
    inst_param.output.shape = module::getShape(getOutputs()[0]);
    InstanceNormFunc inst_func(inst_param);
    inst_func.invoke();
  } else if (func_name == "argmax_v3") {
    ArgMaxParam argmax_param;
    mlir::DictionaryAttr param = this->getParam().value();
    argmax_param.axis = param.get("axis").cast<IntegerAttr>().getInt();
    argmax_param.fmt_i8 = false;
    argmax_param.scale = 1.0;
    auto in_type = module::getStorageType(getInputs()[0]);
    if (in_type.isSignedInteger()) {
      argmax_param.fmt_i8 = true;
    }
    argmax_param.scale =
        param.get("scale").cast<FloatAttr>().getValueAsDouble();
    for (size_t i = 0; i < getInputs().size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = p.inputs[i];
      tensor_list.size = module::getNumElements(getInputs()[i]);
      tensor_list.shape = module::getShape(getInputs()[i]);
      argmax_param.inputs.emplace_back(std::move(tensor_list));
    }
    for (size_t i = 0; i < getOutputs().size(); ++i) {
      tensor_list_t tensor_list;
      tensor_list.ptr = p.outputs[i];
      tensor_list.size = module::getNumElements(getOutputs()[i]);
      tensor_list.shape = module::getShape(getOutputs()[i]);
      argmax_param.outputs.emplace_back(std::move(tensor_list));
    }
    ArgMaxFunc argmax_func(argmax_param);
    argmax_func.invoke();
  } else if (func_name == "onnx_nms") {
    mlir::DictionaryAttr dic_param = this->getParam().value();
    NmsParam param;
    param.max_output_boxes_per_class =
        dic_param.get("max_output_size").cast<IntegerAttr>().getInt();
    param.center_point_box = 0;
    int input_size = getInputs().size();
    std::vector<tensor_list_t> input_list(input_size);
    for (int i = 0; i < getInputs().size(); ++i) {
      tensor_list_t input;
      input.ptr = p.inputs[0];
      input.size = module::getNumElements(getInputs()[i]);
      input.shape = module::getShape(getInputs()[i]);
      input_list[i] = input;
    }
    param.box = p.inputs[0];
    param.score = p.inputs[1];
    int output_size = module::getNumElements(getOutputs()[0]);
    std::vector<float> output_tensor_data(output_size, 0);
    param.inputs = input_list;
    param.output = output_tensor_data.data();
    param.iou_threshold = p.inputs[3][0];
    param.score_threshold = p.inputs[4][0];
    NmsFunc func(param);
    auto true_num = func.invoke();
    auto tmp = (int *)output_tensor_data.data();
    for (int64_t j = 0; j < true_num; ++j) {
      p.outputs[0][j] = (float)tmp[j];
    }
  } else if (func_name == "topk") {
    mlir::DictionaryAttr dic_param = this->getParam().value();
    int axis = dic_param.get("axis").cast<IntegerAttr>().getInt();
    int K = dic_param.get("K").cast<IntegerAttr>().getInt();
    int is_sorted = dic_param.get("sorted").cast<IntegerAttr>().getInt();
    if (is_sorted == false) {
      llvm_unreachable("Not supported");
    }
    int is_largest = dic_param.get("largest").cast<IntegerAttr>().getInt();
    auto input_shape = module::getShape(getInputs()[0]);
    if (axis != input_shape.size() - 1) {
      llvm_unreachable("Not supported");
    }
    int axis_dim = input_shape[axis];
    int outer_dim = 1;
    for (int i = 0; i < axis; i++) {
      outer_dim *= input_shape[i];
    }
#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
    for (int i = 0; i < outer_dim; i++) {
      auto *ptr = p.inputs[0] + i * axis_dim;
      std::vector<std::pair<int, float>> result;
      topk_indices(result, ptr, axis_dim, K, is_largest);
      for (int k = 0; k < K; k++) {
        if (p.outputs.size() > 1) {
          auto indices_ptr = p.outputs[1] + i * K + k;
          *indices_ptr = (float)result[k].first;
        }
        auto values_ptr = p.outputs[0] + i * K + k;
        *values_ptr = result[k].second;
      }
    }
  } else if (func_name == "gathernd_tf") {
    mlir::DictionaryAttr dic_param = this->getParam().value();
    GatherNDParam param;
    param.batch_dims = dic_param.get("batch_dims").cast<IntegerAttr>().getInt();
    for (int i = 0; i < getInputs().size(); ++i) {
      tensor_list_t input;
      input.ptr = p.inputs[i];
      input.size = module::getNumElements(getInputs()[i]);
      input.shape = module::getShape(getInputs()[i]);
      param.inputs.push_back(input);
    }
    tensor_list_t output;
    output.ptr = p.outputs[0];
    output.size = module::getNumElements(getOutputs()[0]);
    output.shape = module::getShape(getOutputs()[0]);
    param.output = output;
    GatherndFunc func(param);
    func.invoke();
  } else if (func_name == "gatherelements_pt") {
    mlir::DictionaryAttr dic_param = this->getParam().value();
    GatherElementsParam param;
    param.axis = dic_param.get("axis").cast<IntegerAttr>().getInt();
    for (int i = 0; i < getInputs().size(); ++i) {
      tensor_list_t input;
      input.ptr = p.inputs[i];
      input.size = module::getNumElements(getInputs()[i]);
      input.shape = module::getShape(getInputs()[i]);
      param.inputs.push_back(input);
    }
    tensor_list_t output;
    output.ptr = p.outputs[0];
    output.size = module::getNumElements(getOutputs()[0]);
    output.shape = module::getShape(getOutputs()[0]);
    param.output = output;
    GatherElementsFunc func(param);
    func.invoke();
  } else if (func_name == "tensor_scatter") {
    ScatterNDParam param;
    mlir::DictionaryAttr dic_param = this->getParam().value();
    param.op_code = (CPU_SCATTER_OP_T)dic_param.get("reduction")
                        .cast<IntegerAttr>()
                        .getInt();
    for (int i = 0; i < getInputs().size(); ++i) {
      tensor_list_t input;
      input.ptr = p.inputs[i];
      input.size = module::getNumElements(getInputs()[i]);
      input.shape = module::getShape(getInputs()[i]);
      param.inputs.push_back(input);
    }
    tensor_list_t output;
    output.ptr = p.outputs[0];
    output.size = module::getNumElements(getOutputs()[0]);
    output.shape = module::getShape(getOutputs()[0]);
    param.output = output;
    ScatterNDFunc func(param);
    func.invoke();
  } else if (func_name == "grid_sampler") {
    mlir::DictionaryAttr dic_param = this->getParam().value();
    GridSamplerParam param;
    param.mode = dic_param.get("mode").cast<IntegerAttr>().getInt();
    param.padding_mode =
        dic_param.get("padding_mode").cast<IntegerAttr>().getInt();
    param.align_corners =
        dic_param.get("align_corners").cast<BoolAttr>().getValue();
    tensor_list_t input;
    tensor_list_t grid;
    input.ptr = p.inputs[0];
    input.size = module::getNumElements(getInputs()[0]);
    input.shape = module::getShape(getInputs()[0]);
    grid.ptr = p.inputs[1];
    grid.size = module::getNumElements(getInputs()[1]);
    grid.shape = module::getShape(getInputs()[1]);
    param.inputs.push_back(input);
    param.inputs.push_back(grid);

    tensor_list_t output;
    output.size = module::getNumElements(getOutputs()[0]);
    output.shape = module::getShape(getOutputs()[0]);
    output.ptr = p.outputs[0];
    param.output = output;

    GridSamplerFunc func(param);
    func.invoke();
  } else if (func_name == "deform_gather") {
    mlir::DictionaryAttr dic_param = this->getParam().value();
    DeformGatherParam param;
    param.mode = DEFORM_TORCHVISION_MODE;
    param.modulated = dic_param.get("use_mask").cast<BoolAttr>().getValue();
    param.deform_groups =
        dic_param.get("deform_group").cast<IntegerAttr>().getInt();
    param.kh = dic_param.get("kh").cast<IntegerAttr>().getInt();
    param.kw = dic_param.get("kw").cast<IntegerAttr>().getInt();
    param.pad_t = dic_param.get("pad_t").cast<IntegerAttr>().getInt();
    param.pad_b = dic_param.get("pad_b").cast<IntegerAttr>().getInt();
    param.pad_l = dic_param.get("pad_l").cast<IntegerAttr>().getInt();
    param.pad_r = dic_param.get("pad_r").cast<IntegerAttr>().getInt();
    param.stride_h = dic_param.get("stride_h").cast<IntegerAttr>().getInt();
    param.stride_w = dic_param.get("stride_w").cast<IntegerAttr>().getInt();
    param.dilation_h = dic_param.get("dilation_h").cast<IntegerAttr>().getInt();
    param.dilation_w = dic_param.get("dilation_w").cast<IntegerAttr>().getInt();
    tensor_list_t input;
    input.ptr = p.inputs[0];
    input.size = module::getNumElements(getInputs()[0]);
    input.shape = module::getShape(getInputs()[0]);
    tensor_list_t offset;
    offset.ptr = p.inputs[1];
    offset.size = module::getNumElements(getInputs()[1]);
    offset.shape = module::getShape(getInputs()[1]);
    param.inputs.push_back(input);
    param.inputs.push_back(offset);
    if (param.modulated) {
      tensor_list_t mask;
      mask.ptr = p.inputs[2];
      mask.size = module::getNumElements(getInputs()[2]);
      mask.shape = module::getShape(getInputs()[2]);
      param.inputs.push_back(mask);
    }
    tensor_list_t output;
    output.size = module::getNumElements(getOutputs()[0]);
    output.shape = module::getShape(getOutputs()[0]);
    output.ptr = p.outputs[0];
    param.output = output;
    DeformGatherFunc func(param);
    func.invoke();
  } else if (func_name == "cumsum") {
    mlir::DictionaryAttr dic_param = this->getParam().value();
    CumSumParam param;
    param.axis = dic_param.get("axis").cast<IntegerAttr>().getInt();
    tensor_list_t input;
    input.ptr = p.inputs[0];
    input.shape = module::getShape(getInputs()[0]);
    input.size = module::getNumElements(getInputs()[0]);
    param.inputs.push_back(input);
    tensor_list_t output;
    output.ptr = p.outputs[0];
    output.shape = module::getShape(getOutputs()[0]);
    output.size = module::getNumElements(getOutputs()[0]);
    param.output = output;
    CumSumFunc func(param);
    func.invoke();
  } else {
    llvm_unreachable("generic cpu func not supported!\n");
  }
  return success();
}

mlir::Type tpu::GenericCpuOp::type_verify(uint64_t opd_idx,
                                          TypeCastMode &mode) {
  std::string func_name = getCpuOpName().str();
  auto op = getOperation();
  if (func_name == "embedding") {
    if (opd_idx == 0) {
      return type_verify_case_type(op, opd_idx,
                                   Builder(op).getIntegerType(16, false), mode);
    }
    return type_verify_case_same(op, opd_idx, mode);
  }
  if (func_name == "argmax_v3") {
    if (opd_idx == 0) {
      return type_verify_case_type(op, opd_idx, op->getOperand(1).getType(),
                                   mode);
    }
    if (opd_idx == 1) {
      return do_nothing(mode);
    }
  }
  if (func_name == "onnx_nms") {
    return do_nothing(mode);
  }
  if (func_name == "tensor_scatter") {
    if (opd_idx == 1) {
      return type_verify_case_type(op, opd_idx,
                                   Builder(op).getIntegerType(32, true), mode);
    }
  }
  if (func_name == "gathernd_tf") {
    if (opd_idx == 1) {
      return type_verify_case_type(op, opd_idx,
                                   Builder(op).getIntegerType(32, true), mode);
    }
  }
  if (func_name == "gatherelements_pt") {
    if (opd_idx == 1) {
      return type_verify_case_type(op, opd_idx,
                                   Builder(op).getIntegerType(32, true), mode);
    }
  }
  return type_verify_case_same(op, opd_idx, mode);
}

bool tpu::GenericCpuOp::support_multi_core() { return false; }
