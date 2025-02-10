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

pool_attr_t tpu::Pool1DOp::parseParam() {
  pool_attr_t p = {0};
  p.id = 1;
  p.od = 1;
  p.kd = 1;
  p.kw = 1;
  p.sd = 1;
  p.sw = 1;
  auto ishape = getInput().getType().dyn_cast<RankedTensorType>().getShape();
  auto oshape = getOutput().getType().dyn_cast<RankedTensorType>().getShape();
  module::getNCHW(ishape, p.n, p.c, p.ih, p.iw);
  module::getNCHW(oshape, p.n, p.c, p.oh, p.ow);
  assert(p.iw == 1 && p.ow == 1);

  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(0);
  auto stride = module::getI64Array(getStrides());
  p.sh = stride->at(0);
  auto pad = module::getI64Array(getPads());
  p.pad_h = pad->at(0);
  p.pad_h_after = pad->at(1);
  p.pad_value = getPadValue();
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.is_global = p.ih == p.kh && p.iw == p.kw && p.oh == 1 && p.ow == 1;
  p.count_include_pad = getCountIncludePad();
  return p;
}

LogicalResult tpu::Pool1DOp::init(InferenceParameter &p) {
  auto pooling = new Pooling();
  auto attr = parseParam();
  // for dynamic tpu-inference.
  std::vector<int64_t> new_pads{attr.pad_h, attr.pad_h_after};
  if (getCeilMode().has_value() && getCeilMode().value()) {
    auto ishape = getInput().getType().dyn_cast<RankedTensorType>().getShape();
    auto kernel_shape = module::getI64Array(getKernelShape());
    auto kernel_len = kernel_shape->size();
    auto stride = module::getI64Array(getStrides());
    for (uint32_t i = 0; i < kernel_len; i++) {
      auto remain_pixel =
          (ishape[i + 2] + 2 * new_pads[i] - kernel_shape->at(i)) %
          stride->at(i);
      if (remain_pixel > 0) {
        new_pads[i + kernel_len] += (stride->at(i) - remain_pixel);
      }
    }
  }
  attr.pad_h = new_pads[0];
  attr.pad_h_after = new_pads[1];
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

void tpu::Pool1DOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto pooling = (Pooling *)p.handle;
    delete pooling;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::Pool1DOp::inference(InferenceParameter &p) {
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
  auto round_mode = round_mode_convert(getRoundMode());
  if (out_type.isInteger(8)) {

    if (module::isAsymmetric() == false) {
      auto multi = getMultiplier().value();
      auto rs = getRshift().value();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; ++i) {
        p.outputs[0][i] = applyMultiplierAndRShift(
            std::round(p.outputs[0][i] * pooling->kh), multi, rs);
        p.outputs[0][i] = saturate(p.outputs[0][i], out_type);
      }
    } else {
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
      for (int64_t i = 0; i < num_elem; ++i) {
        p.outputs[0][i] = p.outputs[0][i] * pooling->kh *
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
    }
  }

  return success();
}

LogicalResult tpu::Pool1DOp::LocalGenSupport() {
  auto stride = module::getI64Array(getStrides());
  if (stride->at(0) > 15) {
    return failure();
  }
  return success();
}

LogicalResult tpu::Pool1DOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto attr = parseParam();
  in_slice = (out_slice - 1) * attr.sh + attr.kh;
  in_idx = out_idx * attr.sh - attr.pad_h;
  bool is_last = (out_idx + out_slice == attr.oh);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.ih, is_last);
  return success();
}

LogicalResult tpu::Pool1DOp::BackwardW(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto attr = parseParam();
  in_slice = (out_slice - 1) * attr.sw + attr.kw;
  in_idx = out_idx * attr.sw - attr.pad_w;
  bool is_last = (out_idx + out_slice == attr.ow);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.iw, is_last);
  return success();
}

void tpu::Pool1DOp::assign_sec_info(int64_t n_step, int64_t c_step,
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

LogicalResult tpu::Pool1DOp::DynBackwardH(int64_t &in_idx, int64_t &in_slice,
                                          int64_t out_idx, int64_t out_slice) {
  auto attr = parseParam();
  in_slice = (out_slice - 1) * attr.sh + attr.kh;
  in_idx = out_idx * attr.sh - attr.pad_h;
  return success();
}

LogicalResult tpu::Pool1DOp::DynBackwardKh(int64_t &in_kh, int64_t out_kh) {
  auto attr = parseParam();
  in_kh = (out_kh - 1) * attr.sh + attr.kh;
  return success();
}

LogicalResult tpu::Pool1DOp::DynBackwardStrideH(int64_t &in_stride_h,
                                                int64_t out_stride_h) {
  auto attr = parseParam();
  in_stride_h = out_stride_h * attr.sh;
  return success();
}

LogicalResult tpu::Pool1DOp::DynBackwardUpPadH(int64_t &in_up_pad_h,
                                               int64_t out_up_pad_h) {
  auto attr = parseParam();
  in_up_pad_h = out_up_pad_h * attr.sh + attr.pad_h;
  return success();
}

LogicalResult tpu::Pool1DOp::DynBackwardDownPadH(int64_t &in_down_pad_h,
                                                 int64_t out_down_pad_h) {
  auto attr = parseParam();
  in_down_pad_h = out_down_pad_h * attr.sh + attr.pad_h_after;
  return success();
}

int64_t tpu::Pool1DOp::DynForwardHeight(int64_t in_height) {
  // Todo
  return in_height;
}

void tpu::Pool1DOp::assign_fw_param(void *param) {
  llvm_unreachable("not implement");
}

ArrayAttr tpu::Pool1DOp::getIndexingMaps() {
  MLIRContext *ctx = getContext();
  AffineMap map = AffineMap::getMultiDimIdentityMap(2, ctx);
  SmallVector<AffineMap> indexingMaps{map, map};
  return Builder(ctx).getAffineMapArrayAttr(indexingMaps);
};

bool tpu::Pool1DOp::support_multi_core() { return false; }
