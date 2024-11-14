//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::MaskRCNNMaskPoolerOp::getFLOPs() {
  return module::getNumElements(getResultRes());
}

LogicalResult top::MaskRCNNMaskPoolerOp::init(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}
void top::MaskRCNNMaskPoolerOp::deinit(InferenceParameter &p) {}

LogicalResult top::MaskRCNNMaskPoolerOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::MaskRCNNMaskPoolerOp::shape_inference() {
  // Set res as [roi_num,CHANNEL_ROI,PH,PW]
  std::vector<int64_t> output_shape(4);
  std::vector<int64_t> input_shape = module::getShape(getX_0());
  const int batch_size = input_shape[0];
  const int output_batch = getROI_SLICE() * batch_size;
  // const int C = getCHANNEL_ROI();
  // assert(C==256);
  output_shape[0] = output_batch;
  output_shape[1] = getCHANNEL_ROI();
  output_shape[2] = getROI_PH();
  output_shape[3] = getROI_PW();
  module::setShapeOrVerify(getResultRes(), output_shape);
}
