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

pool_attr_t tpu::Pool2DOp::parseParam() {
  pool_attr_t p = {0};
  p.id = 1;
  p.od = 1;
  p.kd = 1;
  p.sd = 1;
  auto ishape = getInput().getType().dyn_cast<RankedTensorType>().getShape();
  auto oshape = getOutput().getType().dyn_cast<RankedTensorType>().getShape();
  module::getNCHW(ishape, p.n, p.c, p.ih, p.iw);
  module::getNCHW(oshape, p.n, p.c, p.oh, p.ow);

  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(0);
  p.kw = kernel->at(1);
  auto stride = module::getI64Array(getStrides());
  p.sh = stride->at(0);
  p.sw = stride->at(1);
  auto pad = module::getI64Array(getPads());
  p.pad_h = pad->at(0);
  p.pad_w = pad->at(1);
  p.pad_h_after = pad->at(2);
  p.pad_w_after = pad->at(3);
  p.pad_value = getPadValue();
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.is_global = p.ih == p.kh && p.iw == p.kw && p.oh == 1 && p.ow == 1;
  p.count_include_pad = getCountIncludePad();
  return p;
}

LogicalResult tpu::Pool2DOp::init(InferenceParameter &p) {
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

void tpu::Pool2DOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::Pool2DOp::inference(InferenceParameter &p) {
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
  // average pooling
  bool is_cv18xx = module::isCV18xx();
  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());
  if (out_type.isInteger(8)) {
    auto i_qtype = module::getUniformQuantizedType(getInput());
    auto o_qtype = module::getUniformQuantizedType(getOutput());

    if (module::isAsymmetric() == false) {
      auto multi = getMultiplier().value();
      auto rs = getRshift().value();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; ++i) {
        int64_t v = 0;
        if (is_cv18xx) {
          v = to_int(p.outputs[0][i] * pooling->kh * pooling->kw,
                     ROUNDING_HALF_UP);
        } else {
          v = std::round(p.outputs[0][i] * pooling->kh * pooling->kw);
        }
        p.outputs[0][i] = applyMultiplierAndRShift(v, multi, rs);
        p.outputs[0][i] = out_type.isUnsignedInteger(8)
                              ? to_uint8(p.outputs[0][i])
                              : to_int8(p.outputs[0][i]);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; ++i) {
        p.outputs[0][i] = p.outputs[0][i] * pooling->kh * pooling->kw *
                              getScale().value().convertToDouble() +
                          getOffset().value().convertToDouble();
        p.outputs[0][i] = out_type.isUnsignedInteger(8)
                              ? to_uint8(p.outputs[0][i])
                              : to_int8(p.outputs[0][i]);
      }
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

LogicalResult tpu::Pool2DOp::LocalGenSupport() {
  auto attr = parseParam();
  auto stride = module::getI64Array(getStrides());
  if ((stride->at(0) > 15 || stride->at(1) > 15)) {
    return failure();
  }
  // if (attr.is_global) {
  //   // TODO: bug, need to be fixed
  //   auto in_stype = module::getStorageType(getInput());
  //   if (in_stype.isF16() || in_stype.isBF16()) {
  //     return failure();
  //   }
  // }
  return success();
}

LogicalResult tpu::Pool2DOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto &attr = getPool2DParam(*this);
  if (attr.is_global) {
    if (out_idx != 0 || out_slice != attr.oh) {
      return failure();
    }
    in_idx = 0;
    in_slice = attr.ih;
    return success();
  }
  in_slice = (out_slice - 1) * attr.sh + attr.kh;
  in_idx = out_idx * attr.sh - attr.pad_h;
  bool is_last = (out_idx + out_slice == attr.oh);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.ih, is_last);
  return success();
}

LogicalResult tpu::Pool2DOp::DynBackwardH(int64_t &in_idx, int64_t &in_slice,
                                          int64_t out_idx, int64_t out_slice) {
  auto &attr = getPool2DParam(*this);
  if (attr.is_global) {
    if (out_idx != 0 || out_slice != attr.oh) {
      return failure();
    }
    in_idx = 0;
    in_slice = attr.ih;
    return success();
  }
  in_slice = (out_slice - 1) * attr.sh + attr.kh;
  in_idx = out_idx * attr.sh - attr.pad_h;
  return success();
}

LogicalResult tpu::Pool2DOp::DynBackwardKh(int64_t &in_kh, int64_t out_kh) {
  auto &attr = getPool2DParam(*this);
  in_kh = (out_kh - 1) * attr.sh + attr.kh;
  return success();
}

LogicalResult tpu::Pool2DOp::DynBackwardStrideH(int64_t &in_stride_h,
                                                int64_t out_stride_h) {
  auto &attr = getPool2DParam(*this);
  in_stride_h = out_stride_h * attr.sh;
  return success();
}

LogicalResult tpu::Pool2DOp::DynBackwardUpPadH(int64_t &in_up_pad_h,
                                               int64_t out_up_pad_h) {
  auto &attr = getPool2DParam(*this);
  in_up_pad_h = out_up_pad_h * attr.sh + attr.pad_h;
  return success();
}

LogicalResult tpu::Pool2DOp::DynBackwardDownPadH(int64_t &in_down_pad_h,
                                                 int64_t out_down_pad_h) {
  auto &attr = getPool2DParam(*this);
  in_down_pad_h = out_down_pad_h * attr.sh + attr.pad_h_after;
  return success();
}
