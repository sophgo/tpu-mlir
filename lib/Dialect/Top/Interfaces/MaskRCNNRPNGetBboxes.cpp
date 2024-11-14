//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::MaskRCNNRPNGetBboxesOp::getFLOPs() {
  return module::getNumElements(getResultList());
}

LogicalResult top::MaskRCNNRPNGetBboxesOp::init(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}
void top::MaskRCNNRPNGetBboxesOp::deinit(InferenceParameter &p) {}

LogicalResult top::MaskRCNNRPNGetBboxesOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::MaskRCNNRPNGetBboxesOp::shape_inference() {
  // Set result_list as [GLOBAL_BATCH_SIZE,1, MAX_LENGTH_STATIC_STRECHED,
  // NUM_INDEX+ NUM_CLASSES]
  std::vector<int64_t> input_shape = module::getShape(getClsScores_0());
  std::vector<int64_t> output_shape(4);
  output_shape[0] = input_shape[0];
  output_shape[1] = 1;
  output_shape[2] = getMAX_PER_IMG();
  output_shape[3] = getNUM_INDEXES() + getNUM_CLASSES();
  module::setShapeOrVerify(getResultList(), output_shape);
}
