//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/GenericCpuFunc.h"

LogicalResult tpu::RoiExtractorOp::init(InferenceParameter &p) {
  return success();
}

void tpu::RoiExtractorOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RoiExtractorOp::inference(InferenceParameter &p) {
  auto batch = module::getShape(getRois())[0];
  auto shape = module::getShape(getInputs()[0]).vec();
  uint32_t ndims = shape.size();
  shape[0] = batch;
  shape[ndims - 1] = getOutputWidth();
  shape[ndims - 2] = getOutputHeight();
  module::setShape(getOutput(), shape);
  RoiExtractorParam param;
  param.num_levels = getNumLevels();
  param.mode = getMode().str() == "Avg" ? RoiAlignAvgMode : RoiAlignMaxMode;
  param.pooled_h = getOutputHeight();
  param.pooled_w = getOutputWidth();
  param.sampling_ratio = getSamplingRatio();
  auto spatial_scales = module::getF64Array(getSpatialScales());
  for (int i = 0; i < param.num_levels; i++) {
    param.spatial_scales.push_back(spatial_scales->at(i));
  }
  param.aligned = getAlignCorners();

  tensor_list_t rois, target_lvls;
  rois.ptr = p.inputs[0];
  rois.size = module::getNumElements(getRois());
  rois.shape = module::getShape(getRois());
  param.inputs.emplace_back(std::move(rois));

  target_lvls.ptr = p.inputs[1];
  target_lvls.size = module::getNumElements(getTargetLvls());
  target_lvls.shape = module::getShape(getTargetLvls());
  param.inputs.emplace_back(std::move(target_lvls));
  int offset = 2;
  for (int id_feature = 0; id_feature < param.num_levels; id_feature++) {
    tensor_list_t feature;
    feature.ptr = p.inputs[offset + id_feature];
    feature.size = module::getNumElements(getInputs()[id_feature]);
    feature.shape = module::getShape(getInputs()[id_feature]);
    param.inputs.emplace_back(std::move(feature));
  }

  // output
  param.output.ptr = p.outputs[0];
  param.output.size = module::getNumElements(getOutput());
  param.output.shape = module::getShape(getOutput());
  RoiExtractorFunc func(param);
  func.invoke();
  return success();
}

bool tpu::RoiExtractorOp::support_multi_core() { return false; }

mlir::Type tpu::RoiExtractorOp::type_verify(uint64_t opd_idx,
                                            TypeCastMode &mode) {
  auto op = getOperation();
  if (opd_idx == 1) {
    // target_lvls must be int32
    auto opd = op->getOperand(1);
    auto stype = module::getStorageType(opd);
    auto bitwidth = stype.getIntOrFloatBitWidth();
    if (module::isBM1684XFamily() || module::isBM1690Family()) {
      // indices should be int32 in BM1684x
      bitwidth = 32;
    }
    return Builder(op).getIntegerType(bitwidth);
  }
  return type_verify_case_same(op, opd_idx, mode);
}
