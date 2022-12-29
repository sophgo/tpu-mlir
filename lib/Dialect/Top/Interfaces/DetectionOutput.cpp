//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "detection-output"



int64_t top::DetectionOutputOp::getFLOPs() {
  return module::getNumElements(output());
}

LogicalResult top::DetectionOutputOp::init(InferenceParameter &p) {
  return success();
}
void top::DetectionOutputOp::deinit(InferenceParameter &p) {}

LogicalResult top::DetectionOutputOp::inference(InferenceParameter &p) {
  DetParam param;
  param.keep_top_k = this->keep_top_k();
  param.confidence_threshold = this->confidence_threshold().convertToDouble();
  param.nms_threshold = this->nms_threshold().convertToDouble();
  param.top_k = this->top_k();
  param.num_classes = this->num_classes();
  param.share_location = this->share_location();
  param.background_label_id = this->background_label_id();
  std::vector<int64_t> loc_shape;
  std::vector<int64_t> conf_shape;
  std::vector<int64_t> prior_shape;
  module::getShapeVec(this->inputs()[0], param.loc_shape);
  module::getShapeVec(this->inputs()[1], param.conf_shape);
  module::getShapeVec(this->inputs()[2], param.prior_shape);

  std::string str_code_type = this->code_type().str();
  if (str_code_type == "CORNER") {
    param.code_type = PriorBoxParameter_CodeType_CORNER;
  } else if (str_code_type == "CENTER_SIZE") {
    param.code_type = PriorBoxParameter_CodeType_CENTER_SIZE;
  } else if (str_code_type == "CORNER_SIZE") {
    param.code_type = PriorBoxParameter_CodeType_CORNER_SIZE;
  } else {
    llvm_unreachable("code type wrong");
  }

  param.loc_data = p.inputs[0];
  param.conf_data = p.inputs[1];
  param.prior_data = p.inputs[2];
  param.output_data = p.outputs[0];

  DetectionOutputFunc det_func(param);
  det_func.invoke();
  return success();
}
