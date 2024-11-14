//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "float.h"
#include "tpu_mlir/Support/MathUtils.h"

pool_attr_t tpu::MaxPoolWithMaskOp::parseParam() {
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
  p.pad_value = 0;
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.is_global = p.ih == p.kh && p.iw == p.kw && p.oh == 1 && p.ow == 1;
  p.count_include_pad = 0;
  return p;
}

LogicalResult tpu::MaxPoolWithMaskOp::init(InferenceParameter &p) {
  return success();
}

void tpu::MaxPoolWithMaskOp::deinit(InferenceParameter &p) { return; }

LogicalResult tpu::MaxPoolWithMaskOp::inference(InferenceParameter &p) {
  auto attr = parseParam();
  int64_t nc = attr.n * attr.c;
  auto num_elem = module::getNumElements(getOutput());
  std::fill_n(p.outputs[0], num_elem, (float)(-FLT_MAX));
#pragma omp parallel for schedule(static, omp_schedule(nc))
  for (int64_t idx = 0; idx < nc; ++idx) {
    auto bottom_data = p.inputs[0] + idx * attr.ih * attr.iw;
    auto top_data = p.outputs[0] + idx * attr.oh * attr.ow;
    auto top_mask = p.outputs[1] + idx * attr.oh * attr.ow;
    for (int64_t ph = 0; ph < attr.oh; ++ph) {
      for (int64_t pw = 0; pw < attr.ow; ++pw) {
        auto hstart = ph * attr.sh - attr.pad_h;
        auto wstart = pw * attr.sw - attr.pad_w;
        auto hend = std::min(hstart + attr.kh, attr.ih);
        auto wend = std::min(wstart + attr.kw, attr.iw);
        if (hstart < 0) {
          hstart = 0;
        }
        if (wstart < 0) {
          wstart = 0;
        }
        auto pool_index = ph * attr.ow + pw;
        for (int64_t h = hstart; h < hend; ++h) {
          for (int64_t w = wstart; w < wend; ++w) {
            auto index = h * attr.iw + w;
            if (bottom_data[index] > top_data[pool_index]) {
              top_data[pool_index] = bottom_data[index];
              top_mask[pool_index] = index;
            }
          }
        }
      }
    }
  }

  return success();
}

LogicalResult tpu::MaxPoolWithMaskOp::LocalGenSupport() {
  // auto stride = module::getI64Array(getStrides());
  // if ((stride->at(0) > 15 || stride->at(1) > 15)) {
  //   return failure();
  // }
  return failure();
}

LogicalResult tpu::MaxPoolWithMaskOp::BackwardH(int64_t &in_idx,
                                                int64_t &in_slice,
                                                int64_t out_idx,
                                                int64_t out_slice) {
  auto attr = parseParam();
  in_slice = (out_slice - 1) * attr.sh + attr.kh;
  in_idx = out_idx * attr.sh - attr.pad_h;
  bool is_last = (out_idx + out_slice == attr.oh);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.ih, is_last);
  return success();
}

LogicalResult tpu::MaxPoolWithMaskOp::BackwardW(int64_t &in_idx,
                                                int64_t &in_slice,
                                                int64_t out_idx,
                                                int64_t out_slice) {
  auto attr = parseParam();
  in_slice = (out_slice - 1) * attr.sw + attr.kw;
  in_idx = out_idx * attr.sw - attr.pad_w;
  bool is_last = (out_idx + out_slice == attr.ow);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.iw, is_last);
  return success();
}

void tpu::MaxPoolWithMaskOp::assign_sec_info(int64_t n_step, int64_t c_step,
                                             int64_t h_step, int64_t d_step,
                                             int64_t w_step,
                                             group_type_t group_type,
                                             local_sec_info_t &sec_info) {
  memset(&sec_info, 0, sizeof(local_sec_info_t));
  sec_info.group_type = group_type;

  auto attr = parseParam();
  auto gi = getGroupInfo(n_step, h_step, d_step, w_step, c_step);
  auto in_gi = LocalGenInterface::getGroupInfo(getInput(), n_step, h_step,
                                               d_step, w_step, c_step);
  int64_t pad_h_b =
      (in_gi.h_idx + in_gi.h_slice == attr.ih ? attr.pad_h_after : 0);
  int64_t pad_w_r =
      (in_gi.w_idx + in_gi.w_slice == attr.iw ? attr.pad_w_after : 0);
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
  sec_info.out_n_slice = gi.n_slice;
  sec_info.out_h_idx = gi.h_idx;
  sec_info.out_h_slice = gi.h_slice;
  sec_info.out_w_idx = gi.w_idx;
  sec_info.out_w_slice = gi.w_slice;
}

bool tpu::MaxPoolWithMaskOp::support_multi_core() { return true; }
