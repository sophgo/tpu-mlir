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

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

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
  auto dtype = module::getElementType(getInput());
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
  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());
  if (out_type.isInteger(8)) {

    if (module::isAsymmetric() == false) {
      auto rmode =
          module::isCV18xx() ? ROUNDING_HALF_UP : ROUNDING_HALF_AWAY_FROM_ZERO;
      if (getMultiplier().has_value() || getRshift().has_value()) {
        auto multi = getMultiplier().value_or(1);
        auto rs = getRshift().value_or(0);
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
        for (int64_t i = 0; i < num_elem; ++i) {
          int64_t v =
              to_int(p.outputs[0][i] * pooling->kh * pooling->kw, rmode);
          p.outputs[0][i] = applyMultiplierAndRShift(v, multi, rs);
          p.outputs[0][i] = saturate(p.outputs[0][i], out_type);
        }
      } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
        for (int64_t i = 0; i < num_elem; ++i) {
          p.outputs[0][i] = saturate(p.outputs[0][i], out_type);
        }
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; ++i) {
        p.outputs[0][i] = p.outputs[0][i] * pooling->kh * pooling->kw *
                              getScale().value().convertToDouble() +
                          getOffset().value().convertToDouble();
        p.outputs[0][i] = saturate(p.outputs[0][i], out_type);
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
  if (module::isCV18xx()) {
    return success();
  }
  auto stride = module::getI64Array(getStrides());
  if ((stride->at(0) > 15 || stride->at(1) > 15)) {
    return failure();
  }
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

LogicalResult tpu::Pool2DOp::BackwardW(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto &attr = getPool2DParam(*this);
  if (attr.is_global) {
    if (out_idx != 0 || out_slice != attr.ow) {
      return failure();
    }
    in_idx = 0;
    in_slice = attr.iw;
    return success();
  }
  in_slice = (out_slice - 1) * attr.sw + attr.kw;
  in_idx = out_idx * attr.sw - attr.pad_w;
  bool is_last = (out_idx + out_slice == attr.ow);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.iw, is_last);
  return success();
}

void tpu::Pool2DOp::assign_sec_info(int64_t n_step, int64_t c_step,
                                    int64_t h_step, int64_t d_step,
                                    int64_t w_step, group_type_t group_type,
                                    local_sec_info_t &sec_info) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;

  auto attr = parseParam();
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step,
                                               h_step, d_step, w_step, c_step);
  sec_info.n_slice = in_gi.n_slice;
  sec_info.d_slice = in_gi.d_slice;
  sec_info.h_slice = in_gi.h_slice;
  sec_info.w_slice = in_gi.w_slice;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.w_idx = in_gi.w_idx;
  sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == attr.ih);
  sec_info.is_w_split = !(in_gi.w_idx == 0 && in_gi.w_slice == attr.iw);
  if (!module::isCV18xx()) {
    int64_t pad_h_b =
        (in_gi.h_idx + in_gi.h_slice == attr.ih ? attr.pad_h_after : 0);
    int64_t pad_w_r =
        (in_gi.w_idx + in_gi.w_slice == attr.iw ? attr.pad_w_after : 0);
    // to be compatible with nntoolchain
    if (sec_info.is_h_split) {
      sec_info.h_idx = in_gi.h_idx == 0 ? -attr.pad_h : in_gi.h_idx;
      sec_info.h_slice = sec_info.h_idx < 0 ? sec_info.h_slice - sec_info.h_idx
                                            : sec_info.h_slice;
      sec_info.h_slice = sec_info.h_slice + pad_h_b;
    }
    if (sec_info.is_w_split) {
      sec_info.w_idx = in_gi.w_idx == 0 ? -attr.pad_w : in_gi.w_idx;
      sec_info.w_slice = sec_info.w_idx < 0 ? sec_info.w_slice - sec_info.w_idx
                                            : sec_info.w_slice;
      sec_info.w_slice = sec_info.w_slice + pad_w_r;
    }
  }
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_idx = gi.w_idx;
  sec_info.out_w_slice = gi.w_slice;
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

int64_t tpu::Pool2DOp::DynForwardHeight(int64_t in_height) {
  auto &attr = getPool2DParam(*this);
  int out_height = 0;
  if ((in_height + attr.pad_h + attr.pad_h_after) >= attr.kh) {
    out_height =
        (in_height + attr.pad_h + attr.pad_h_after - attr.kh) / attr.sh + 1;
  } else {
    out_height = 0;
  }
  if ((in_height + attr.pad_h + attr.pad_h_after) >= attr.kh &&
      ((in_height + attr.pad_h + attr.pad_h_after - attr.kh) % attr.sh != 0) &&
      (out_height * attr.sh < (in_height + attr.pad_h))) {
    out_height++;
  }
  return out_height;
}

void tpu::Pool2DOp::assign_fw_param(void *param) {
  fw_pool_layer_param_t *fw_pool_layer_param = (fw_pool_layer_param_t *)param;
  pool_attr_t attr = parseParam();
  fw_pool_layer_param->ic = attr.c;
  fw_pool_layer_param->kh_kw =
      ((uint32_t)attr.kh << 16) | ((uint32_t)attr.kw & 0xffff);
  fw_pool_layer_param->pad_h_top = attr.pad_h;
  fw_pool_layer_param->pad_h_bottom = attr.pad_h_after;
  fw_pool_layer_param->pad_w_left = attr.pad_w;
  fw_pool_layer_param->pad_w_right = attr.pad_w_after;
  fw_pool_layer_param->stride_h = attr.sh;
  fw_pool_layer_param->stride_w = attr.sw;
  fw_pool_layer_param->if_relu = attr.do_relu;
  fw_pool_layer_param->relu_upper_limit = attr.relu_limit;
  bool is_avg_pooling = getPoolMode() == tpu::PoolMode::Avg;
  // is_avg_pool = 0: max, 1: avg, 2:max_with_mask
  fw_pool_layer_param->is_avg_pool = is_avg_pooling; // max_with_mask not
                                                     // support
  fw_pool_layer_param->avg_pooling_mode = attr.count_include_pad ? 0 : 1;
  fw_pool_layer_param->opd0_sign = module::isSign(getInput());
  fw_pool_layer_param->res_sign = module::isSign(getOutput());
  fw_pool_layer_param->is_global_pool = attr.is_global;
  fw_pool_layer_param->out_ceil_mode =
      0; // only suport 0:RoundMode_FLOOR, TODO(1:RoundMode_CEIL
         // 2:RoundMode_CFDFT 3:RoundMode_TF_SAME_PAD)
}
