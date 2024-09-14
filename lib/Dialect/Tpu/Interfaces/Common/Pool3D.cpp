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
  auto dtype = module::getElementType(getInput());
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
    auto round_mode = round_mode_convert(getRoundMode());
    if (getScale().has_value() || getOffset().has_value()) {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; ++i) {
        p.outputs[0][i] = p.outputs[0][i] * pooling->kd * pooling->kh *
                              pooling->kw *
                              getScale().value().convertToDouble() +
                          getOffset().value().convertToDouble();
        p.outputs[0][i] = saturate(p.outputs[0][i], out_type, round_mode);
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
    } else if (out_type.isFloat8E5M2()) {
      F8E5M2(p.outputs[0], p.outputs[0], num_elem, 1., true);
    } else if (out_type.isFloat8E4M3FN()) {
      auto scale = getFp8OutScale()->convertToDouble();
      F8E4M3(p.outputs[0], p.outputs[0], num_elem, 1 / scale, true);
    }
  }

  return success();
}

LogicalResult tpu::Pool3DOp::LocalGenSupport() {
  auto attr = parseParam();
  if (attr.sd > 15 || attr.sh > 15 || attr.sw > 15) {
    return failure();
  }
  if (module::isBM1684XFamily() || module::isBM1690Family()) {
    return success();
  } else if (module::isBM1684Family() &&
             !module::isUniformQuantized(getInput())) {
    return success();
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

LogicalResult tpu::Pool3DOp::BackwardW(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto attr = parseParam();
  in_slice = (out_slice - 1) * attr.sw + attr.kw;
  in_idx = out_idx * attr.sw - attr.pad_w;
  bool is_last = (out_idx + out_slice == attr.ow);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.iw, is_last);
  return success();
}

LogicalResult tpu::Pool3DOp::BackwardD(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto attr = parseParam();
  in_slice = (out_slice - 1) * attr.sd + attr.kd;
  in_idx = out_idx * attr.sd - attr.pad_d;
  bool is_last = (out_idx + out_slice == attr.od);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.id, is_last);
  return success();
}

void tpu::Pool3DOp::assign_sec_info(int64_t n_step, int64_t c_step,
                                    int64_t h_step, int64_t d_step,
                                    int64_t w_step, group_type_t group_type,
                                    local_sec_info_t &sec_info) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;

  auto attr = parseParam();
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step,
                                               d_step, w_step, c_step);
  sec_info.n_slice = in_gi.n_slice;
  sec_info.d_slice = in_gi.d_slice;
  sec_info.h_slice = in_gi.h_slice;
  sec_info.w_slice = in_gi.w_slice;
  sec_info.n_idx = in_gi.n_idx;
  sec_info.c_idx = in_gi.c_idx;
  sec_info.d_idx = in_gi.d_idx;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == attr.ih);
  sec_info.w_idx = in_gi.w_idx;
  sec_info.is_w_split = !(in_gi.w_idx == 0 && in_gi.w_slice == attr.iw);
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_idx = gi.w_idx;
  sec_info.out_w_slice = gi.w_slice;
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

int64_t tpu::Pool3DOp::DynForwardHeight(int64_t in_height) {
  // Todo
  return in_height;
}

void tpu::Pool3DOp::assign_fw_param(void *param) {
  llvm_unreachable("not implement");
}

ArrayAttr tpu::Pool3DOp::getIndexingMaps() {
  MLIRContext *ctx = getContext();
  AffineMap map = AffineMap::getMultiDimIdentityMap(2, ctx);
  AffineMap empty = AffineMap::get(2, 0, ctx);
  SmallVector<AffineMap> indexingMaps{map, empty, map};
  return Builder(ctx).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::Pool3DOp::support_multi_core() { return false; }
