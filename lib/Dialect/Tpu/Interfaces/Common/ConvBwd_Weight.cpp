//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

using namespace tpu_mlir::backend;

convbwd_weight_attr_t tpu::ConvBwdWeightOp::parseParam() {
  convbwd_weight_attr_t p = {0};
  auto input_s = getInput().getType().cast<RankedTensorType>().getShape();
  auto gradout_s = getGradout().getType().cast<RankedTensorType>().getShape();
  p.has_bias = getGradBiasEnable();
  p.n = input_s[0];
  p.ic = input_s[1];
  p.ih = input_s.size() > 2 ? input_s[2] : 1;
  p.iw = input_s.size() > 3 ? input_s[3] : 1;
  p.oc = gradout_s[1];
  p.oh = gradout_s.size() > 2 ? gradout_s[2] : 1;
  p.ow = gradout_s.size() > 3 ? gradout_s[3] : 1;
  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(0);
  p.kw = kernel->at(1);
  auto pads_v = module::getI64Array(getPadding());
  p.pht = pads_v->at(0);
  p.pwl = pads_v->at(1);
  p.phb = pads_v->at(2);
  p.pwr = pads_v->at(3);
  auto strides_v = module::getI64Array(getStride());
  p.sh = strides_v->at(0);
  p.sw = strides_v->at(1);
  auto dhdw = module::getI64Array(getDilations(), 2, 1);
  p.dh = dhdw->at(0);
  p.dw = dhdw->at(1);
  p.groups = getGroups();
  return p;
}

LogicalResult tpu::ConvBwdWeightOp::init(InferenceParameter &p) {

  return success();
}

void tpu::ConvBwdWeightOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::ConvBwdWeightOp::inference(InferenceParameter &p) {
  return success();
}

LogicalResult tpu::ConvBwdWeightOp::BackwardH(int64_t &in_idx,
                                              int64_t &in_slice,
                                              int64_t out_idx,
                                              int64_t out_slice) {

  return success();
}

LogicalResult tpu::ConvBwdWeightOp::BackwardW(int64_t &in_idx,
                                              int64_t &in_slice,
                                              int64_t out_idx,
                                              int64_t out_slice) {

  return success();
}

void tpu::ConvBwdWeightOp::assign_sec_info(int64_t n_step, int64_t c_step,
                                           int64_t h_step, int64_t d_step,
                                           int64_t w_step,
                                           group_type_t group_type,
                                           local_sec_info_t &sec_info) {}

mlir::Type tpu::ConvBwdWeightOp::type_verify(uint64_t opd_idx,
                                             TypeCastMode &mode) {
  return type_verify_case_i32(getOperation(), opd_idx, mode);
}

LogicalResult tpu::ConvBwdWeightOp::DynBackwardH(int64_t &in_idx,
                                                 int64_t &in_slice,
                                                 int64_t out_idx,
                                                 int64_t out_slice) {

  return success();
}

LogicalResult tpu::ConvBwdWeightOp::DynBackwardKh(int64_t &in_kh,
                                                  int64_t out_kh) {

  return success();
}

LogicalResult tpu::ConvBwdWeightOp::DynBackwardStrideH(int64_t &in_stride_h,
                                                       int64_t out_stride_h) {

  return success();
}

LogicalResult tpu::ConvBwdWeightOp::DynBackwardUpPadH(int64_t &in_up_pad_h,
                                                      int64_t out_up_pad_h) {

  return success();
}

LogicalResult
tpu::ConvBwdWeightOp::DynBackwardDownPadH(int64_t &in_down_pad_h,
                                          int64_t out_down_pad_h) {

  return success();
}

LogicalResult tpu::ConvBwdWeightOp::LocalGenSupport() {
  if (!module::isBM1690Family()) {
    return failure();
  }
  return success();
}

int64_t tpu::ConvBwdWeightOp::DynForwardHeight(int64_t in_height) { return -1; }

ArrayAttr tpu::ConvBwdWeightOp::getIndexingMaps() {
  MLIRContext *context = getContext();
  // TODO: split OC
  AffineMap identity1Map = AffineMap::getMultiDimIdentityMap(1, context);
  AffineMap emptyMap = AffineMap::get(1, 0, context);

  SmallVector<AffineMap> indexingMaps{identity1Map, emptyMap};

  for (int i = 2, n = getNumOperands(); i < n; ++i) {
    indexingMaps.push_back(emptyMap);
  }
  indexingMaps.push_back(identity1Map);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

bool tpu::ConvBwdWeightOp::support_multi_core() { return false; }
