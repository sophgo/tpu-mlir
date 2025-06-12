//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Float8.h"

#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"

using namespace tpu_mlir::backend;

conv_attr_t tpu::Conv2DOp::parseParam() {
  conv_attr_t p = {0};
  p.id = p.od = p.kd = p.sd = p.dd = 1;
  auto i_s = getInput().getType().cast<RankedTensorType>().getShape();
  auto f_s = getFilter().getType().cast<RankedTensorType>().getShape();
  auto o_s = getOutput().getType().cast<RankedTensorType>().getShape();
  p.fn = f_s[0];
  p.fc = f_s[1];
  p.fh = f_s.size() > 2 ? f_s[2] : 1;
  p.fw = f_s.size() > 3 ? f_s[3] : 1;
  p.do_relu = getDoRelu();
  p.relu_limit = getReluLimit().convertToDouble();
  p.has_bias = getWithBias();
  p.weight_is_coeff = getWeightIsCoeff();
  p.dims = i_s.size() - 2;
  p.n = i_s[0];
  p.ic = i_s[1];
  p.ih = i_s.size() > 2 ? i_s[2] : 1;
  p.iw = i_s.size() > 3 ? i_s[3] : 1;
  p.oc = o_s[1];
  p.oh = o_s.size() > 2 ? o_s[2] : 1;
  p.ow = o_s.size() > 3 ? o_s[3] : 1;
  auto kernel = module::getI64Array(getKernelShape());
  p.kh = kernel->at(0);
  p.kw = kernel->at(1);
  auto pads_v = module::getI64Array(getPads());
  p.pht = pads_v->at(0);
  p.pwl = pads_v->at(1);
  p.phb = pads_v->at(2);
  p.pwr = pads_v->at(3);
  if (module::isUniformQuantized(getInput())) {
    p.pad_value = module::getUniformQuantizedType(getInput()).getZeroPoint();
  }
  p.kernel_zp = getKernelZp();
  auto strides_v = module::getI64Array(getStrides());
  p.sh = strides_v->at(0);
  p.sw = strides_v->at(1);
  auto dhdw = module::getI64Array(getDilations(), 2, 1);
  p.dh = dhdw->at(0);
  p.dw = dhdw->at(1);
  auto ins = module::getI64Array(getInserts(), 2, 0);
  p.ins_h = ins->at(0);
  p.ins_w = ins->at(1);
  p.groups = getGroup();
  p.is_dw = (p.oc == p.ic && p.oc == p.groups && p.groups > 1);

  if (getUseWinograd().value_or(0) != 0) {
    p.kh = 3;
    p.kw = 3;
  }
  p.use_winograd = getUseWinograd().value_or(0);
  p.use_3ic_optimize = getUse_3icOptimize();
  return p;
}

LogicalResult tpu::Conv2DOp::init(InferenceParameter &p) {
  auto conv = new Conv();
  p.handle = (void *)conv;
  return success();
}

void tpu::Conv2DOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto conv = (Conv *)p.handle;
    delete conv;
    p.handle = nullptr;
  }
}

LogicalResult tpu::Conv2DOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto conv = (Conv *)p.handle;
  auto attr = parseParam();
  if (module::isUniformQuantized(getOutput()) && attr.has_bias) {
    attr.do_relu = false;
    for (int i = 0; i < attr.oc; i++) {
      p.inputs[2][i] = 0.f;
    }
  }

  int use_winograd = getUseWinograd().value_or(0);
  if (use_winograd) {
    // AT @ P @ A -> AE @ P
    float AEQ[4][16] = {
        {1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0.},
        {0., 1., -1., 1., 0., 1., -1., 1., 0., 1., -1., 1., 0., 0., 0., 0.},
        {0., 0., 0., 0., 1., 1., 1., 0., -1., -1., -1., 0., 1., 1., 1., 0.},
        {0., 0., 0., 0., 0., 1., -1., 1., 0., -1., 1., -1., 0., 1., -1., 1.}};

    //
    float BEQ[16][16] = {
        {1., 0., -1., 0., 0., 0., 0., 0., -1., 0., 1., 0., 0., 0., 0., 0.},
        {0., 1., 1., 0., 0., 0., 0., 0., 0., -1., -1., 0., 0., 0., 0., 0.},
        {0., -1., 1., 0., 0., 0., 0., 0., 0., 1., -1., 0., 0., 0., 0., 0.},
        {0., -1., 0., 1., 0., 0., 0., 0., 0., 1., 0., -1., 0., 0., 0., 0.},
        {0., 0., 0., 0., 1., 0., -1., 0., 1., 0., -1., 0., 0., 0., 0., 0.},
        {0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0.},
        {0., 0., 0., 0., 0., -1., 1., 0., 0., -1., 1., 0., 0., 0., 0., 0.},
        {0., 0., 0., 0., 0., -1., 0., 1., 0., -1., 0., 1., 0., 0., 0., 0.},
        {0., 0., 0., 0., -1., 0., 1., 0., 1., 0., -1., 0., 0., 0., 0., 0.},
        {0., 0., 0., 0., 0., -1., -1., 0., 0., 1., 1., 0., 0., 0., 0., 0.},
        {0., 0., 0., 0., 0., 1., -1., 0., 0., -1., 1., 0., 0., 0., 0., 0.},
        {0., 0., 0., 0., 0., 1., 0., -1., 0., -1., 0., 1., 0., 0., 0., 0.},
        {0., 0., 0., 0., -1., 0., 1., 0., 0., 0., 0., 0., 1., 0., -1., 0.},
        {0., 0., 0., 0., 0., -1., -1., 0., 0., 0., 0., 0., 0., 1., 1., 0.},
        {0., 0., 0., 0., 0., 1., -1., 0., 0., 0., 0., 0., 0., -1., 1., 0.},
        {0., 0., 0., 0., 0., 1., 0., -1., 0., 0., 0., 0., 0., -1., 0., 1.}};

    int n = attr.n;
    int ic = attr.ic;
    int ih = attr.ih;
    int iw = attr.iw;
    int pih = attr.ih + attr.phb + attr.pht;
    int piw = attr.iw + attr.pwl + attr.pwr;

    int window_h = (pih - 4) / 2 + 1;
    int window_w = (piw - 4) / 2 + 1;
    int row_num = window_w * window_h;

    int oc = attr.oc;
    int ow = attr.ow;
    int oh = attr.oh;
    auto gt = p.inputs[1] + (ic * oc * 3 * 3); // b, ic, iw, ih

    float *input_unfolded = new float[row_num * attr.ic * 16];
    float *P = new float[row_num * attr.oc * 16];
    float *IA_wino = new float[row_num * attr.ic * 16];

    bool need_pad = (attr.phb + attr.pht + attr.pwl + attr.pwr) > 0;

    float *inputs;
    // for each batch
    for (int bs = 0; bs < n; bs++) {
      inputs = need_pad ? new float[ic * pih * piw]
                        : p.inputs[0] + (bs * ic * ih * iw);
      if (need_pad) {
        memset(inputs, attr.pad_value,
               sizeof(float) * ic * (ih + attr.pht + attr.phb) *
                   (iw + attr.pwl + attr.pwr));
        for (int i = 0; i < ic; i++) {
          for (int j = 0; j < ih; j++) {
            for (int k = 0; k < iw; k++) {
              int input_index =
                  (bs * ic * ih * iw) + (i * ih * iw) + (j * iw) + (k);
              int output_index =
                  (i * pih * piw) + ((j + attr.pht) * piw) + (k + attr.pwl);

              inputs[output_index] = p.inputs[0][input_index];
            }
          }
        }
      }

      /**
       * 1. unfold image (im2col)
       * input_unfolded = (
       *     input_image.unfold(2, 4, 2)
       *     .unfold(3, 4, 2)
       *     .permute(0, 2, 3, 1, 4, 5)
       *     .reshape(row_num * ic, 16)
       * )
       */
      int output_index = 0;
      for (int i = 0; i < window_h * 2; i += 2) {   // unfold(3, 4, 2)
        for (int j = 0; j < window_w * 2; j += 2) { // unfold(2, 4, 2)
          for (int ci = 0; ci < attr.ic; ci++) {    // permute(0, 2, 3, 1, 4, 5)
            for (int hi = 0; hi < 4; hi++) {
              for (int wi = 0; wi < 4; wi++) {
                int input_index =
                    (ci * pih * piw) + ((i + hi) * piw) + (j + wi);
                input_unfolded[output_index] = inputs[input_index];
                output_index++;
              }
            }
          }
        }
      }

      // 2.
      // input_unfolded @ BEQ.transpose(0, 1) -> IA_wino
      auto matmul = new MatMul();
      matmul->setup(input_unfolded, (float *)BEQ, 0, IA_wino, 1, 1,
                    row_num * ic, 16, 16, false, 0, 0, 0, true, false, false,
                    false);
      matmul->run();
      delete matmul;

      // 3.
      memset(P, 0, sizeof(float) * row_num * attr.oc * 16);
      // (gt * IA_wino with broadcast and reduce sum) -> P

#pragma omp parallel for schedule(static, omp_schedule(row_num))
      for (int i = 0; i < row_num; i++) {
        for (int oi = 0; oi < attr.oc; oi++) { // for each kernel
          for (int k = 0; k < 16; k++) {       // for each window
            for (int j = 0; j < ic; j++) {     // for each input channel
              int input_index = (i * ic * 16) + (j * 16) + (k);
              // IA_wino[input_index]
              int gt_index = (oi * ic * 16) + (j * 16) + (k);
              int output_index = (i * attr.oc * 16) + (oi * 16) + (k);
              P[output_index] += IA_wino[input_index] * gt[gt_index];
            }
          }
        }
      }

      // 4.
      // P @ AEQ.transpose(0, 1) -> wino_res
      float *wino_res = new float[row_num * attr.oc * 4];
      matmul = new MatMul();
      matmul->setup(P, (float *)AEQ, 0, wino_res, 1, 1, row_num * attr.oc, 16,
                    4, false, 0, 0, 0, true, false, false, 0);
      matmul->run();
      delete matmul;

      // 5.
#pragma omp parallel for schedule(static, omp_schedule(row_num))
      for (int r = 0; r < row_num; r++) {
        for (int c = 0; c < oc; c++) {
          for (int h = 0; h < 2; h++) {
            for (int w = 0; w < 2; w++) {
              int wino_res_idx = (r * oc * 4) + (c * 4) + h * 2 + w;
              int h_idx = ((r * 2) / ow) * 2 + h;
              int w_idx = (r * 2) % ow + w;
              int unfolded_idx =
                  (bs * oc * oh * ow) + (c * oh * ow) + (h_idx * ow) + (w_idx);
              p.outputs[0][unfolded_idx] = wino_res[wino_res_idx];
            }
          }
        }
      }

      // fold wino_res -> p.outputs[0]
      delete[] wino_res;
    }
    if (need_pad) {
      delete[] inputs;
    }
    delete[] P;
    delete[] IA_wino;
    delete[] input_unfolded;

  } else {
    conv->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], attr);
    conv->run();
  }

  // requant
  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());
  if (out_type.isa<FloatType>()) {
    if (out_type.isBF16()) {
      BF16(p.outputs[0], p.outputs[0], num_elem);
    } else if (out_type.isFloat8E4M3FN()) {
      if (!getOutF8Scales().has_value())
        llvm_unreachable("should have out scale for conv2d in f8 mode");
      f64_array_t quant_scale_v = module::getF64Array(getOutF8Scales().value());

      for (int i = 0; i < quant_scale_v.get()->size(); i++) {
        size_t n = module::getShape(getOutput())[0];
        size_t out_c_num = (num_elem / n) / quant_scale_v.get()->size();
        for (int n_ = 0; n_ < n; n_++) {
#pragma omp parallel for schedule(static, omp_schedule(out_c_num))
          for (int j = 0; j < out_c_num; j++) {
            p.outputs[0][n_ * num_elem / n + i * out_c_num + j] =
                F8E4M3(p.outputs[0][n_ * num_elem / n + i * out_c_num + j],
                       1.0 / quant_scale_v.get()->at(i), true);
          }
        }
      }
    } else if (out_type.isFloat8E5M2()) {
      F8E5M2(p.outputs[0], p.outputs[0], num_elem, 1.0, true);
    } else if (out_type.isF16()) {
      F16(p.outputs[0], p.outputs[0], num_elem);
    }
  } else if (module::isUniformQuantized(getOutput())) {
    int64_t n, c, h, w;
    module::getNCHW(getOutput(), n, c, h, w);
    auto o_qtype = module::getUniformQuantizedType(getOutput());
    auto rshift_v = module::getI64Array(getRshift().value());
    auto multiplier_v =
        module::getI64Array(getMultiplier(), rshift_v->size(), 1);
    bool per_axis = rshift_v->size() == c;
    bool use_winograd = attr.use_winograd;
    // do bias after conv prevent precision issue
    auto bias_i32 = std::make_shared<std::vector<int32_t>>(c, 0);
    bool do_relu = getDoRelu();
    if (getWithBias()) {
      auto biasOp = cast<top::WeightOp>(getBias().getDefiningOp());
      bias_i32 = biasOp.read_as_int32();
    }
    auto qmode = getQuantMode();
    bool is_tf = qmode == tpu::RequantMode::QDM ||
                 qmode == tpu::RequantMode::TFLite ||
                 qmode == tpu::RequantMode::TFLite_LShift;
    auto rmode = is_tf ? ROUNDING_HALF_AWAY_FROM_ZERO : ROUNDING_HALF_UP;

#pragma omp parallel for schedule(static, omp_schedule(c))
    for (int ic = 0; ic < c; ic++) {
      int64_t shift = per_axis
                          ? rshift_v->at(ic)
                          : use_winograd ? rshift_v->at(1) : rshift_v->at(0);
      int64_t multi = 1;
      if (qmode != tpu::RequantMode::OnlyShift) {
        multi = per_axis ? multiplier_v->at(ic) : multiplier_v->at(0);
      }
      int32_t bias = bias_i32->at(ic + (use_winograd ? c : 0));
      for (int in = 0; in < n; in++) {
        for (int hw = 0; hw < h * w; hw++) {
          int offset = (in * c + ic) * h * w + hw;
          int64_t v = 0;
          int64_t tmp = p.outputs[0][offset] + bias;
          v = applyMultiplierAndRShift(tmp, multi, shift, qmode, rmode) +
              o_qtype.getZeroPoint();
          if (do_relu && (v < o_qtype.getZeroPoint())) {
            v = o_qtype.getZeroPoint();
          }
          p.outputs[0][offset] = saturate(v, out_type);
        }
      }
    }
  }

  return success();
}

LogicalResult tpu::Conv2DOp::BackwardH(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto &attr = getConv2DParam(*this);
  int kh_with_dh = (attr.kh - 1) * attr.dh + 1;
  in_slice = (out_slice - 1) * attr.sh +
             (kh_with_dh >= attr.sh ? kh_with_dh : attr.sh);
  in_idx = out_idx * attr.sh - attr.pht;
  bool is_last = (out_idx + out_slice == attr.oh);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.ih, is_last);
  return success();
}

LogicalResult tpu::Conv2DOp::BackwardW(int64_t &in_idx, int64_t &in_slice,
                                       int64_t out_idx, int64_t out_slice) {
  auto &attr = getConv2DParam(*this);
  if (attr.dims == 1) {
    if (out_idx != 0 || out_slice != 1) {
      return failure();
    }
    in_idx = 0;
    in_slice = 1;
    return success();
  }
  int kw_with_dw = (attr.kw - 1) * attr.dw + 1;
  in_slice = (out_slice - 1) * attr.sw +
             (kw_with_dw >= attr.sw ? kw_with_dw : attr.sw);
  in_idx = out_idx * attr.sw - attr.pwl;
  bool is_last = (out_idx + out_slice == attr.ow);
  LocalGenInterface::fixSlice(in_idx, in_slice, attr.iw, is_last);
  return success();
}

void tpu::Conv2DOp::assign_sec_info(int64_t n_step, int64_t c_step,
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
  sec_info.h_idx = in_gi.h_idx;
  sec_info.w_idx = in_gi.w_idx;
  sec_info.is_h_split = !(in_gi.h_idx == 0 && in_gi.h_slice == attr.ih);
  sec_info.is_w_split = !(in_gi.w_idx == 0 && in_gi.w_slice == attr.iw);
  // to be compatible with nntoolchain
  if (!module::isCV18xx()) {
    int64_t pad_h_b = (in_gi.h_idx + in_gi.h_slice == attr.ih ? attr.phb : 0);
    int64_t pad_w_r = (in_gi.w_idx + in_gi.w_slice == attr.iw ? attr.pwr : 0);
    if (sec_info.is_h_split) {
      sec_info.h_idx = in_gi.h_idx == 0 ? -attr.pht : in_gi.h_idx;
      sec_info.h_slice = sec_info.h_idx < 0 ? sec_info.h_slice - sec_info.h_idx
                                            : sec_info.h_slice;
      sec_info.h_slice = sec_info.h_slice + pad_h_b;
    }
    if (sec_info.is_w_split) {
      sec_info.w_idx = in_gi.w_idx == 0 ? -attr.pwl : in_gi.w_idx;
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

mlir::Type tpu::Conv2DOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  if (opd_idx == 2 && !module::isBM1690Family()) {
    return module::getElementType(getOperand(opd_idx));
  }
  return type_verify_case_i16_or_i32(getOperation(), opd_idx, mode);
}

LogicalResult tpu::Conv2DOp::DynBackwardH(int64_t &in_idx, int64_t &in_slice,
                                          int64_t out_idx, int64_t out_slice) {
  auto &attr = getConv2DParam(*this);
  int kh_with_dh = (attr.kh - 1) * attr.dh + 1;
  in_slice = (out_slice - 1) * attr.sh +
             (kh_with_dh >= attr.sh ? kh_with_dh : attr.sh);
  in_idx = out_idx * attr.sh - attr.pht;
  return success();
}

LogicalResult tpu::Conv2DOp::DynBackwardKh(int64_t &in_kh, int64_t out_kh) {
  auto &attr = getConv2DParam(*this);
  int kh_with_dh = (attr.kh - 1) * attr.dh + 1;
  in_kh =
      (out_kh - 1) * attr.sh + (kh_with_dh >= attr.sh ? kh_with_dh : attr.sh);
  return success();
}

LogicalResult tpu::Conv2DOp::DynBackwardStrideH(int64_t &in_stride_h,
                                                int64_t out_stride_h) {
  auto &attr = getConv2DParam(*this);
  in_stride_h = out_stride_h * attr.sh;
  return success();
}

LogicalResult tpu::Conv2DOp::DynBackwardUpPadH(int64_t &in_up_pad_h,
                                               int64_t out_up_pad_h) {
  auto &attr = getConv2DParam(*this);
  in_up_pad_h = out_up_pad_h * attr.sh + attr.pht;
  return success();
}

LogicalResult tpu::Conv2DOp::DynBackwardDownPadH(int64_t &in_down_pad_h,
                                                 int64_t out_down_pad_h) {
  auto &attr = getConv2DParam(*this);
  in_down_pad_h = out_down_pad_h * attr.sh + attr.phb;
  return success();
}

LogicalResult tpu::Conv2DOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    auto attr = parseParam();
    if (attr.ic > MAX_TIU_CHL || attr.oc > MAX_TIU_CHL ||
        attr.iw > MAX_TIU_CHL || attr.ow > MAX_TIU_CHL ||
        !attr.weight_is_coeff) {
      return failure();
    }
    if (attr.groups > 1 && false == attr.is_dw) {
      // for group conv
      // if oc / g > 32, then we will have two bias at one lane without
      // EU_NUM align,
      // so we can only specify the align type to bias memory layout
      // but skip the oc/g>32 cases.
      auto chunk_size = attr.oc / attr.groups;
      if (chunk_size > CV18xx::NPU_NUM || chunk_size % 2) {
        return failure();
      }
    }
    if (attr.ins_h > 0 || attr.ins_w > 0) {
      // ins mode cant slice h/w
      return failure();
    }
  } else if (module::isBM1684Family()) {
    auto attr = parseParam();
    if (attr.fn == attr.ic && attr.fc == attr.n)
      return failure();
    if (attr.sh > 15 || attr.sw > 15 || attr.dh > 15 || attr.dw > 15 ||
        attr.ic >= (1 << 12) || attr.oc >= (1 << 12)) {
      return failure();
    }
  } else {
    auto attr = parseParam();
    if (module::isMARS3() &&
        (attr.sh > 15 || attr.sw > 15 || attr.dh > 15 || attr.dw > 15)) {
      return failure();
    }
  }
  return success();
}

int64_t tpu::Conv2DOp::DynForwardHeight(int64_t in_height) {
  auto &attr = getConv2DParam(*this);
  int out_height = 0;
  int kh_with_dh = (attr.kh - 1) * attr.dh + 1;
  if ((in_height + attr.pht + attr.phb) >= kh_with_dh) {
    out_height = (in_height + attr.pht + attr.phb - kh_with_dh) / attr.sh + 1;
  } else {
    out_height = 0;
  }
  return out_height;
}

void tpu::Conv2DOp::assign_fw_param(void *param) {
  fw_conv_layer_param_t fw_conv_layer_param = {0};
  auto attr = parseParam();
  fw_conv_layer_param.ic_oc =
      ((uint32_t)attr.ic << 16) | ((uint32_t)attr.oc & 0xffff);
  fw_conv_layer_param.groups = attr.groups;
  fw_conv_layer_param.kh_kw =
      ((uint32_t)attr.kh << 16) | ((uint32_t)attr.kw & 0xffff);
  fw_conv_layer_param.dh = attr.dh;
  fw_conv_layer_param.dw = attr.dw;
  fw_conv_layer_param.pad_h = attr.pht;
  fw_conv_layer_param.pad_h_after = attr.phb;
  fw_conv_layer_param.pad_w = attr.pwl;
  fw_conv_layer_param.pad_w_after = attr.pwr;
  fw_conv_layer_param.stride_h = attr.sh;
  fw_conv_layer_param.stride_w = attr.sw;
  fw_conv_layer_param.using_bias = getWithBias();
  fw_conv_layer_param.if_relu = attr.do_relu;
  fw_conv_layer_param.relu_upper_limit = attr.relu_limit;
  fw_conv_layer_param.use_winograd = 0; // not support now
  uint8_t rshift = 0;
  if (module::isUniformQuantized(getInput())) {
    auto shift_v = module::getI64Array(getRshift(), 1, 0);
    rshift = shift_v->at(0);
  }
  fw_conv_layer_param.rshiftbits = rshift;
  fw_conv_layer_param.opd0_sign = module::isSign(getInput());
  fw_conv_layer_param.opd1_sign = module::isSign(getFilter());
  fw_conv_layer_param.opd2_sign = getWithBias() && module::isSign(getBias());
  fw_conv_layer_param.res_sign = module::isSign(getOutput());
  fw_conv_layer_param.weight_is_tensor = !module::isWeight(getFilter());
  memcpy(param, &fw_conv_layer_param, sizeof(fw_conv_layer_param_t));
}

ArrayAttr tpu::Conv2DOp::getIndexingMaps() {
  auto &attr = getConv2DParam(*this);
  bool is_depthwise =
      attr.ic == attr.oc && attr.ic == attr.groups && attr.groups > 1;
  auto in_etype = module::getStorageType(getInput());
  if (is_depthwise == false && in_etype.isIntOrIndex() == false) {
    return {};
  }
  MLIRContext *context = getContext();
  AffineMap inputMap, filterMap, outputMap, empty;

  if (getGroup() != 1 || getCoeffMerged()) {
    // TODO: group conv, int8 conv
    inputMap = AffineMap::getMultiDimIdentityMap(1, context);
    outputMap = AffineMap::getMultiDimIdentityMap(1, context);
    filterMap = AffineMap::get(1, 0, context);
    empty = AffineMap::get(1, 0, context);
  } else {
    AffineExpr d0, d1;
    bindDims(context, d0, d1);
    auto c0 = mlir::getAffineConstantExpr(0, context);
    inputMap = AffineMap::get(2, 0, d0);
    // n = 1 after reordered.
    // c0 is a placeholder, signaling the parallelPass to leave this dimension
    // unchanged.
    filterMap = AffineMap::get(2, 0, {c0, d1}, context);
    outputMap = AffineMap::getMultiDimIdentityMap(2, context);
    empty = AffineMap::get(2, 0, context);
  }
  SmallVector<AffineMap> indexingMaps{inputMap, filterMap};

  for (int i = 2, n = getNumOperands(); i < n; ++i) {
    if (isa_and_nonnull<top::NoneOp>(getOperand(i).getDefiningOp()))
      indexingMaps.push_back(empty);
    else
      indexingMaps.push_back(filterMap);
  }
  indexingMaps.push_back(outputMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

bool tpu::Conv2DOp::support_multi_core() {
  auto &attr = getConv2DParam(*this);
  bool is_depthwise =
      attr.ic == attr.oc && attr.ic == attr.groups && attr.groups > 1;
  auto in_etype = module::getStorageType(getInput());
  if (is_depthwise == false && in_etype.isIntOrIndex() == false) {
    return true;
  }
  return false;
}
