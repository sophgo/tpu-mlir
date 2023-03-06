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
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"


int64_t top::ProposalOp::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::ProposalOp::init(InferenceParameter &p) { return success(); }

void top::ProposalOp::deinit(InferenceParameter &p) {}

LogicalResult top::ProposalOp::inference(InferenceParameter &p) {
  ProposalParam param;
  param.net_input_h = getNetInputH();
  param.net_input_w = getNetInputW();
  param.feat_stride = getFeatStride();
  param.anchor_base_size = getAnchorBaseSize();
  param.rpn_obj_threshold = getRpnObjThreshold().convertToDouble();
  param.rpn_nms_threshold = getRpnNmsThreshold().convertToDouble();
  param.rpn_nms_post_top_n = getRpnNmsPostTopN();
  for (size_t i = 0; i < getInputs().size(); ++i) {
    tensor_list_t tensor_list;
    tensor_list.ptr = p.inputs[i];
    tensor_list.size = module::getNumElements(getInputs()[i]);
    tensor_list.shape = module::getShape(getInputs()[i]);
    param.inputs.emplace_back(std::move(tensor_list));
  }
  param.output.ptr = p.outputs[0];
  param.output.size = module::getNumElements(getOutput());
  param.output.shape = module::getShape(getOutput());
  ProposalFunc proposal_func(param);
  proposal_func.invoke();
  return success();
}

void top::ProposalOp::shape_inference() {}

