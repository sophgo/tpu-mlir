//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Module.h"

#include "bmcpu_common.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Support/MathUtils.h"
using namespace tpu_mlir::backend;

LogicalResult bmcpu_inference(tpu::GenericCpuOp &op, InferenceParameter &p) {
  std::vector<std::vector<int>> input_shapes_v;
  std::vector<std::vector<int>> output_shapes_v;
  std::vector<float *> input_tensor_data_v;
  std::vector<float *> output_tensor_data_v;
  for (int i = 0; i < op.getInputs().size(); ++i) {
    int shape[MAX_SHAPE_DIMS] = {1};
    module::getGlobalShape(op.getInputs()[i], shape);
    std::vector<int> shapev(shape,
                            shape + module::getShape(op.getInputs()[i]).size());
    input_shapes_v.push_back(shapev);
    input_tensor_data_v.push_back(p.inputs[i]);
  }
  for (int i = 0; i < op.getOutputs().size(); ++i) {
    int shape[MAX_SHAPE_DIMS] = {1};
    module::getGlobalShape(op.getOutputs()[i], shape);
    std::vector<int> shapev(
        shape, shape + module::getShape(op.getOutputs()[i]).size());
    output_shapes_v.push_back(shapev);
    output_tensor_data_v.push_back(p.outputs[i]);
  }
  BMCpuOp cpuOp(op);
  void *param = malloc(cpuOp.param_size);
  memcpy(param, cpuOp.param, cpuOp.param_size);
  BM1684::instance().dl_bmcpu_reshape(BM1684::instance().bmcpu_handle,
                                      cpuOp.op_type, param, cpuOp.param_size,
                                      input_shapes_v, output_shapes_v);
  BM1684::instance().dl_bmcpu_process(BM1684::instance().bmcpu_handle,
                                      cpuOp.op_type, param, cpuOp.param_size,
                                      input_tensor_data_v, input_shapes_v,
                                      output_tensor_data_v, output_shapes_v);
  for (int i = 0; i < op.getOutputs().size(); ++i) {
    auto dtype = BM168x::getDataType(op.getOutputs()[i]);
    if (dtype == DTYPE_FP32) {
      p.outputs[i] = output_tensor_data_v[i];
    } else if (dtype == DTYPE_INT32) {
      // The output of the CPU layer is a float pointer
      // when the actual output data type is int32
      // we need to first convert the float pointer to an int pointer in order
      // to obtain the correct output value.
      auto tmp = (int *)output_tensor_data_v[i];
      int64_t num_element = module::getNumElements(op.getOutputs()[i]);
#pragma omp parallel for schedule(static, omp_schedule(num_element))
      for (int64_t j = 0; j < num_element; ++j) {
        // p.output always need float
        p.outputs[i][j] = (float)tmp[j];
      }
    } else {
      llvm_unreachable("not support dtype");
    }
  }
  free(param);
  return success();
}

LogicalResult tpu::GenericCpuOp::init(InferenceParameter &p) {
  return success();
}
void tpu::GenericCpuOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::GenericCpuOp::inference(InferenceParameter &p) {
  std::string func_name = getCpuOpName().str();
  if (module::isBM1684Family()) {
    return bmcpu_inference(*this, p);
  }
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
    yolo_param.tiny = param.get("tiny").cast<BoolAttr>().getValue();
    yolo_param.yolo_v4 = param.get("yolo_v4").cast<BoolAttr>().getValue();
    yolo_param.spp_net = param.get("spp_net").cast<BoolAttr>().getValue();
    yolo_param.anchors = param.get("anchors").cast<StringAttr>().getValue();

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
  } else {
    llvm_unreachable("generic cpu func not supported!\n");
  }
  return success();
}

mlir::Type tpu::GenericCpuOp::type_verify(uint64_t opd_idx,
                                          TypeCastMode &mode) {
  std::string func_name = getCpuOpName().str();
  auto op = getOperation();
  if (func_name == "embedding" && opd_idx == 0) {
    return do_nothing(mode);
  }
  return type_verify_case_same(op, opd_idx, mode);
}
