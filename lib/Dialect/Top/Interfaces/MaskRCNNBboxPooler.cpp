//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::MaskRCNNBboxPoolerOp::getFLOPs() {
  return module::getNumElements(getResultRes());
}

LogicalResult top::MaskRCNNBboxPoolerOp::init(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}
void top::MaskRCNNBboxPoolerOp::deinit(InferenceParameter &p) {}

LogicalResult top::MaskRCNNBboxPoolerOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::MaskRCNNBboxPoolerOp::shape_inference() {
  // Set res as [roi_num,CHANNEL_ROI,PH,PW]
  std::vector<int64_t> shape_res(4);
  std::vector<int64_t> shape_rois(4);
  std::vector<int64_t> input_shape = module::getShape(getPtrFeat0());
  const int batch_size = input_shape[0];
  const int roi_num = getROI_SLICE() * batch_size;
  shape_res[0] = roi_num;
  shape_res[1] = getCHANNEL_ROI();
  shape_res[2] = getROI_PH();
  shape_res[3] = getROI_PW();

  shape_rois[0] = batch_size;
  shape_rois[1] = getROI_SLICE();
  shape_rois[2] = 1;
  shape_rois[3] = getROI_LEN();

  module::setShapeOrVerify(getResultRes(), shape_res);
  module::setShapeOrVerify(getResultRois(), shape_rois);
}
