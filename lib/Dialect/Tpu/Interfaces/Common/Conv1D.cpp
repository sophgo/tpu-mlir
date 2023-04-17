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

conv_attr_t tpu::Conv1DOp::parseParam() {
  conv_attr_t p = {0};
  p.id = p.od = p.kd = p.sd = p.dd = 1;
  p.iw = p.ow = p.kw = p.sw = p.dw = 1;
  auto i_s = getInput().getType().cast<RankedTensorType>().getShape();
  auto o_s = getOutput().getType().cast<RankedTensorType>().getShape();
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.has_bias = getWithBias();
  p.n = i_s[0];
  p.ic = i_s[1];
  p.ih = i_s[2];
  p.oc = o_s[1];
  p.oh = o_s[2];
  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(0);
  auto pads_v = module::getI64Array(getPads());
  p.pht = pads_v->at(0);
  p.phb = pads_v->at(1);
  if (module::isUniformQuantized(getInput())) {
    p.pad_value = module::getUniformQuantizedType(getInput()).getZeroPoint();
  }
  if (module::isUniformQuantized(getFilter())) {
    p.kernel_zp = module::getUniformQuantizedType(getFilter()).getZeroPoint();
  }
  auto strides_v = module::getI64Array(getStrides());
  p.sh = strides_v->at(0);
  auto dilation = module::getI64Array(getDilations(), 1, 1);
  p.dh = dilation->at(0);
  auto ins = module::getI64Array(getInserts(), 1, 0);
  p.ins_h = ins->at(0);
  assert(p.ins_h == 0 && p.ins_w == 0);
  p.groups = getGroup();
  p.is_dw = (p.oc == p.ic && p.oc == p.groups && p.groups > 1);
  return p;
}

LogicalResult tpu::Conv1DOp::init(InferenceParameter &p) {
  auto conv = new Conv();
  auto attr = parseParam();

  conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], attr);
  p.handle = (void *)conv;
  return success();
}

void tpu::Conv1DOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto conv = (Conv *)p.handle;
    delete conv;
    p.handle = nullptr;
  }
}

LogicalResult tpu::Conv1DOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto conv = (Conv *)p.handle;
  conv->run();
  // requant
  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (module::isUniformQuantized(getOutput())) {
    int64_t n, c, h, w;
    auto sType = module::getStorageType(getOutput());
    module::getNCHW(getOutput(), n, c, h, w);
    auto o_qtype = module::getUniformQuantizedType(getOutput());
    auto rshift_v = module::getI64Array(getRshift().value());
    auto multiplier_v =
        module::getI64Array(getMultiplier(), rshift_v->size(), 1);
    bool per_axis = rshift_v->size() == c;
    auto mode = getQuantMode();
    bool is_tf = mode == tpu::RequantMode::QDM ||
                 mode == tpu::RequantMode::TFLite ||
                 mode == tpu::RequantMode::TFLite_LShift;
    auto rmode = is_tf ? ROUNDING_HALF_AWAY_FROM_ZERO : ROUNDING_HALF_UP;
#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int ic = 0; ic < c; ic++) {
      int64_t shift = per_axis ? rshift_v->at(ic) : rshift_v->at(0);
      int64_t multi = per_axis ? multiplier_v->at(ic) : multiplier_v->at(0);
      for (int in = 0; in < n; in++) {
        for (int hw = 0; hw < h * w; hw++) {
          int offset = (in * c + ic) * h * w + hw;
          auto v = applyMultiplierAndRShift(p.outputs[0][offset], multi, shift,
                                            mode, rmode) +
                   o_qtype.getZeroPoint();
          p.outputs[0][offset] = saturate(v, sType);
        }
      }
    }
  }

  return success();
}

LogicalResult tpu::Conv1DOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto attr = parseParam();
  int kh_with_dh = (attr.kh - 1) * attr.dh + 1;
  in_slice = (out_slice - 1) * attr.sh +
             (kh_with_dh >= attr.sh ? kh_with_dh : attr.sh);
  in_idx = out_idx * attr.sh - attr.pht;
  bool is_last = (out_idx + out_slice == attr.oh);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.ih, is_last);
  return success();
}

LogicalResult tpu::Conv1DOp::BackwardW(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto attr = parseParam();
  int kw_with_dw = (attr.kw - 1) * attr.dw + 1;
  in_slice = (out_slice - 1) * attr.sw +
             (kw_with_dw >= attr.sw ? kw_with_dw : attr.sw);
  in_idx = out_idx * attr.sw - attr.pwl;
  bool is_last = (out_idx + out_slice == attr.ow);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.iw, is_last);
  return success();
}

void tpu::Conv1DOp::assign_sec_info(int64_t n_step, int64_t h_step, int64_t d_step, int64_t w_step,
                                    group_type_t group_type,
                                    local_sec_info_t &sec_info) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;

  auto attr = parseParam();
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step, d_step, w_step);
  int64_t pad_h_b = (in_gi.h_idx + in_gi.h_slice == attr.ih ? attr.phb : 0);
  int64_t pad_w_r = (in_gi.w_idx + in_gi.w_slice == attr.iw ? attr.pwr : 0);
  sec_info.n_slice = in_gi.n_slice;
  sec_info.d_slice = in_gi.d_slice;
  sec_info.h_slice = in_gi.h_slice;
  sec_info.w_slice = in_gi.w_slice;
  sec_info.h_idx = in_gi.h_idx;
  sec_info.w_idx = in_gi.w_idx;
  sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == attr.ih);
  sec_info.is_w_split = !(in_gi.w_idx == 0 && in_gi.w_slice == attr.iw);
  // to be compatible with nntoolchain
  if (sec_info.is_h_split) {
    sec_info.h_idx = h_step == 0 ? -attr.pht : in_gi.h_idx;
    sec_info.h_slice = sec_info.h_idx < 0 ? sec_info.h_slice - sec_info.h_idx
                                          : sec_info.h_slice;
    sec_info.h_slice = sec_info.h_slice + pad_h_b;
  }
  if (sec_info.is_w_split) {
    sec_info.w_idx = w_step == 0 ? -attr.pwl : in_gi.w_idx;
    sec_info.w_slice = sec_info.w_idx < 0 ? sec_info.w_slice - sec_info.w_idx
                                          : sec_info.w_slice;
    sec_info.w_slice = sec_info.w_slice + pad_w_r;
  }
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_idx = gi.w_idx;
  sec_info.out_w_slice = gi.w_slice;
}

mlir::Type tpu::Conv1DOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  return type_verify_case_i32(getOperation(), opd_idx, mode);
}

LogicalResult tpu::Conv1DOp::DynBackwardH(int64_t &in_idx, int64_t &in_slice,
                                          int64_t out_idx, int64_t out_slice) {
  auto attr = parseParam();
  int kh_with_dh = (attr.kh - 1) * attr.dh + 1;
  in_slice = (out_slice - 1) * attr.sh +
             (kh_with_dh >= attr.sh ? kh_with_dh : attr.sh);
  in_idx = out_idx * attr.sh - attr.pht;
  return success();
}

LogicalResult tpu::Conv1DOp::DynBackwardKh(int64_t &in_kh, int64_t out_kh) {
  auto attr = parseParam();
  int kh_with_dh = (attr.kh - 1) * attr.dh + 1;
  in_kh =
      (out_kh - 1) * attr.sh + (kh_with_dh >= attr.sh ? kh_with_dh : attr.sh);
  return success();
}

LogicalResult tpu::Conv1DOp::DynBackwardStrideH(int64_t &in_stride_h,
                                                int64_t out_stride_h) {
  auto attr = parseParam();
  in_stride_h = out_stride_h * attr.sh;
  return success();
}

LogicalResult tpu::Conv1DOp::DynBackwardUpPadH(int64_t &in_up_pad_h,
                                               int64_t out_up_pad_h) {
  auto attr = parseParam();
  in_up_pad_h = out_up_pad_h * attr.sh + attr.pht;
  return success();
}

LogicalResult tpu::Conv1DOp::DynBackwardDownPadH(int64_t &in_down_pad_h,
                                                 int64_t out_down_pad_h) {
  auto attr = parseParam();
  in_down_pad_h = out_down_pad_h * attr.sh + attr.phb;
  return success();
}

LogicalResult tpu::Conv1DOp::LocalGenSupport() {
  return success();
}

int64_t tpu::Conv1DOp::DynForwardHeight(int64_t in_height) {
  //Todo
  return in_height;
}
