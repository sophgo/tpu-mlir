//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Support/MathUtils.h"

pool_attr_t tpu::Pool3DOp::parseParam() {
  pool_attr_t p = {0};
  assert(getKernelShape().size() == 3);
  auto ishape = getInput().getType().dyn_cast<RankedTensorType>().getShape();
  auto oshape = getOutput().getType().dyn_cast<RankedTensorType>().getShape();
  auto kernel = module::getI64Array(getKernelShape());
  auto stride = module::getI64Array(getStrides());
  auto pad = module::getI64Array(getPads());

  p.n = ishape[0];
  p.c = ishape[1];
  p.id = ishape[2];
  p.ih = ishape[3];
  p.iw = ishape[4];
  p.od = oshape[2];
  p.oh = oshape[3];
  p.ow = oshape[4];
  p.kd = kernel->at(0);
  p.kh = kernel->at(1);
  p.kw = kernel->at(2);
  p.sd = stride->at(0);
  p.sh = stride->at(1);
  p.sw = stride->at(2);
  p.pad_d = pad->at(0);
  p.pad_h = pad->at(1);
  p.pad_w = pad->at(2);
  p.pad_d_after = pad->at(3);
  p.pad_h_after = pad->at(4);
  p.pad_w_after = pad->at(5);
  p.pad_value = getPadValue();
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.is_global = p.id == p.kd && p.ih == p.kh && p.iw == p.kw && p.od == 1 &&
                p.oh == 1 && p.ow == 1;
  p.count_include_pad = getCountIncludePad();
  return p;
}

LogicalResult tpu::Pool3DOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  auto attr = parseParam();

  int izp = 0;
  auto dtype = getInput().getType().cast<RankedTensorType>().getElementType();
  bool is_avg_pooling = getPoolMode() == tpu::PoolMode::Avg;
  if (dtype.isa<quant::UniformQuantizedType>() && is_avg_pooling) {
    izp = dtype.cast<quant::UniformQuantizedType>().getZeroPoint();
  }
  pooling->setup(p.inputs[0], p.outputs[0], attr, is_avg_pooling, izp);
  p.handle = (void *)pooling;
  return success();
}

void tpu::Pool3DOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::Pool3DOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto pooling = (Pooling *)p.handle;
  pooling->run();

  if (getPoolMode() == tpu::PoolMode::Max) {
    if (getDoRelu()) {
      auto limit = getReluLimit().convertToDouble();
      function_relu(p.outputs[0], p.outputs[0],
                    module::getNumElements(getOutput()), limit,
                    module::getStorageType(getOutput()));
    }
    return success();
  }

  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());
  if (out_type.isInteger(8)) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (int64_t i = 0; i < num_elem; ++i) {
      p.outputs[0][i] = p.outputs[0][i] * pooling->kd * pooling->kh *
                            pooling->kw * getScale().value().convertToDouble() +
                        getOffset().value().convertToDouble();
      p.outputs[0][i] = saturate(p.outputs[0][i], out_type);
    }
  } else if (out_type.isa<FloatType>()) {
    if (getDoRelu()) {
      auto limit = getReluLimit().convertToDouble();
      function_relu(p.outputs[0], p.outputs[0], num_elem, limit);
    }
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  }

  return success();
}

LogicalResult tpu::Pool3DOp::LocalGenSupport() {
  auto attr = parseParam();
  if (attr.sd > 15 || attr.sh > 15 || attr.sw > 15) {
    return failure();
  }
  // do not support 3D local layer now
  return failure();
}

LogicalResult tpu::Pool3DOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto attr = parseParam();
  in_slice = (out_slice - 1) * attr.sh + attr.kh;
  in_idx = out_idx * attr.sh - attr.pad_h;
  bool is_last = (out_idx + out_slice == attr.oh);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.ih, is_last);
  return success();
}

LogicalResult tpu::Pool3DOp::DynBackwardH(int64_t &in_idx, int64_t &in_slice,
                                          int64_t out_idx, int64_t out_slice) {
  auto attr = parseParam();
  in_slice = (out_slice - 1) * attr.sh + attr.kh;
  in_idx = out_idx * attr.sh - attr.pad_h;
  return success();
}

LogicalResult tpu::Pool3DOp::DynBackwardKh(int64_t &in_kh, int64_t out_kh) {
  auto attr = parseParam();
  in_kh = (out_kh - 1) * attr.sh + attr.kh;
  return success();
}

LogicalResult tpu::Pool3DOp::DynBackwardStrideH(int64_t &in_stride_h,
                                                int64_t out_stride_h) {
  auto attr = parseParam();
  in_stride_h = out_stride_h * attr.sh;
  return success();
}

LogicalResult tpu::Pool3DOp::DynBackwardUpPadH(int64_t &in_up_pad_h,
                                               int64_t out_up_pad_h) {
  auto attr = parseParam();
  in_up_pad_h = out_up_pad_h * attr.sh + attr.pad_h;
  return success();
}

LogicalResult tpu::Pool3DOp::DynBackwardDownPadH(int64_t &in_down_pad_h,
                                                 int64_t out_down_pad_h) {
  auto attr = parseParam();
  in_down_pad_h = out_down_pad_h * attr.sh + attr.pad_h_after;
  return success();
}
