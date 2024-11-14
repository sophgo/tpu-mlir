#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::MaskRCNNGetBboxBOp::init(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void tpu::MaskRCNNGetBboxBOp::deinit(InferenceParameter &p) { return; }

LogicalResult tpu::MaskRCNNGetBboxBOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

mlir::Type tpu::MaskRCNNGetBboxBOp::type_verify(uint64_t opd_idx,
                                                TypeCastMode &mode) {
  return type_verify_case_same(getOperation(), opd_idx, mode);
}

bool tpu::MaskRCNNGetBboxBOp::support_multi_core() { return false; }
