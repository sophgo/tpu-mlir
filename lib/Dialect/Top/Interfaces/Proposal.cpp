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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/GenericCpuFunc.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;
int64_t top::ProposalOp::getFLOPs() { return Module::getNumElements(output()); }

LogicalResult top::ProposalOp::init(InferenceParameter &p) { return success(); }

void top::ProposalOp::deinit(InferenceParameter &p) {}

LogicalResult top::ProposalOp::inference(InferenceParameter &p) {
  ProposalParam param;
  param.net_input_h = net_input_h();
  param.net_input_w = net_input_w();
  param.feat_stride = feat_stride();
  param.anchor_base_size = anchor_base_size();
  param.rpn_obj_threshold = rpn_obj_threshold().convertToDouble();
  param.rpn_nms_threshold = rpn_nms_threshold().convertToDouble();
  param.rpn_nms_post_top_n = rpn_nms_post_top_n();
  for (size_t i = 0; i < inputs().size(); ++i) {
    tensor_list_t tensor_list;
    tensor_list.ptr = p.inputs[i];
    tensor_list.size = Module::getNumElements(inputs()[i]);
    Module::getShapeVec(inputs()[i], tensor_list.shape);
    param.inputs.emplace_back(std::move(tensor_list));
  }
  param.output.ptr = p.outputs[0];
  param.output.size = Module::getNumElements(output());
  Module::getShapeVec(output(), param.output.shape);
  ProposalFunc proposal_func(param);
  proposal_func.invoke();
  return success();
}
