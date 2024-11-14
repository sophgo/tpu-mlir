//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::MaskRCNNGetBboxBOp::getFLOPs() { return 0; }

LogicalResult top::MaskRCNNGetBboxBOp::init(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}
void top::MaskRCNNGetBboxBOp::deinit(InferenceParameter &p) {}

LogicalResult top::MaskRCNNGetBboxBOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::MaskRCNNGetBboxBOp::shape_inference() {
  std::vector<int64_t> input_shape = module::getShape(getPtrRois());
  assert(input_shape.size() == 4);
  const int num_classes = getNUM_CLASSES();
  const int num_indexes = getNUM_INDEXES();
  const int roi_len = num_classes + num_indexes;
  assert(roi_len == 5);

  const int batch_size = input_shape[0] * input_shape[1] * input_shape[2] *
                         input_shape[3] / roi_len / getMAX_PER_IMG();
  const int topk_onnx_nms_2nd = getTOPK_ONNX_NMS();
  assert(topk_onnx_nms_2nd == 250);

  // auto bbox_shape_in = module::getShape(getBbox());
  // ASSERT_WITH_DUMP(bbox_shape_in.size() == 2);
  std::vector<int64_t> bbox_shape(4);
  std::vector<int64_t> label_shape(4);
  bbox_shape[0] = batch_size;
  bbox_shape[1] = 1;
  bbox_shape[2] = getMAX_PER_IMG_GetBboxB();
  bbox_shape[3] = num_classes + num_indexes;
  label_shape[0] = batch_size;
  label_shape[1] = 1;
  label_shape[2] = getMAX_PER_IMG_GetBboxB();
  label_shape[3] = 1;
  module::setShapeOrVerify(getResultDetBboxes(), bbox_shape);
  module::setShapeOrVerify(getResultDetLabels(), label_shape);
  return;
}
