//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022  Technologies Inc.  All rights reserved.
//
//===----------------------------------------------------------------------===//


#pragma once
#include "ppl_tiu_func.h"

namespace ppl {
namespace tiu {

// ====================================================================
// Deprecation API
// ====================================================================

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void conv(tensor<DataType0> &dst, tensor<DataType1> &src,
          tensor<DataType2> &weight, DataType3 &bias, int oc, dim2 *k_shape,
          padding_t *pad, dim2 *stride, dim2 *dilation, uint8 rshift,
          bool result_relu) {
  conv(dst, src, weight, bias, oc, k_shape, pad, stride, dilation,
       0,                  /*pad_val*/
       result_relu, false, /*result_add*/
       true,               /*sym*/
       rshift,             /*quant*/
       false,              /*rq*/
       0,                  /*requant*/
       0,                  /*rq_shift*/
       0,                  /*out_zp*/
       false, RM_HALF_UP, false);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void conv(tensor<DataType0> &dst, tensor<DataType1> &src,
          tensor<DataType2> &weight, DataType3 &bias, int oc, dim2 *k_shape,
          padding_t *pad, dim2 *stride, dim2 *dilation) {
  conv(dst, src, weight, bias, oc, k_shape, pad, stride, dilation,
       0,            /*pad_val*/
       false, false, /*result_add*/
       true,         /*sym*/
       0,            /*quant*/
       false,        /*rq*/
       0,            /*requant*/
       0,            /*rq_shift*/
       0,            /*out_zp*/
       false, RM_HALF_UP, false);
}

template <typename DataType0, typename DataType1, typename DataType2>
void conv(tensor<DataType0> &dst, tensor<DataType1> &src,
          tensor<DataType2> &weight, int oc, dim2 *k_shape, padding_t *pad,
          dim2 *stride, dim2 *dilation, uint8 rshift, bool result_relu) {
  tensor<DataType0> *bias = nullptr;
  conv(dst, src, weight, bias, oc, k_shape, pad, stride, dilation,
       0,                  /*pad_val*/
       result_relu, false, /*result_add*/
       true,               /*sym*/
       rshift,             /*quant*/
       false,              /*rq*/
       0,                  /*requant*/
       0,                  /*rq_shift*/
       0,                  /*out_zp*/
       false, RM_HALF_UP, false);
}

template <typename DataType0, typename DataType1, typename DataType2>
void conv(tensor<DataType0> &dst, tensor<DataType1> &src,
          tensor<DataType2> &weight, int oc, dim2 *k_shape, padding_t *pad,
          dim2 *stride, dim2 *dilation) {
  tensor<DataType0> *bias = nullptr;
  conv(dst, src, weight, bias, oc, k_shape, pad, stride, dilation,
       0,            /*pad_val*/
       false, false, /*result_add*/
       true,         /*sym*/
       0,            /*quant*/
       false,        /*rq*/
       0,            /*requant*/
       0,            /*rq_shift*/
       0,            /*out_zp*/
       false, RM_HALF_UP, false);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4>
void conv(tensor<DataType0> &dst, tensor<DataType1> &src,
          tensor<DataType2> &weight, DataType3 kzp_val, DataType4 pad_val,
          int oc, dim2 *k_shape, padding_t *pad, dim2 *stride, dim2 *dilation,
          bool result_add, bool kzp_is_unsigned = IS_UNSIGNED(DataType3)) {
  tensor<DataType0> *bias = nullptr;
  conv(dst, src, weight, bias, oc, k_shape, pad, stride, dilation,
       pad_val,           /*pad_val*/
       false, result_add, /*result_add*/
       false,             /*sym*/
       kzp_val,           /*quant*/
       false,             /*rq*/
       0,                 /*requant*/
       0,                 /*rq_shift*/
       0,                 /*out_zp*/
       false, RM_HALF_UP, false);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4>
void conv(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 weight_c,
          DataType3 kzp_val, DataType4 pad_val, int oc, dim2 *k_shape,
          padding_t *pad, dim2 *stride, dim2 *dilation, bool result_add,
          bool c_is_unsigned = IS_UNSIGNED(DataType2),
          bool kzp_is_unsigned = IS_UNSIGNED(DataType3)) {
  tensor<DataType0> *bias = nullptr;
  conv_asym(dst, src, weight_c, oc, k_shape, pad, stride, dilation, pad_val,
            kzp_val, false, result_add, RM_HALF_UP, false);
  conv(dst, src, weight_c, bias, oc, k_shape, pad, stride, dilation,
       pad_val,           /*pad_val*/
       false, result_add, /*result_add*/
       false,             /*sym*/
       kzp_val,           /*quant*/
       false,             /*rq*/
       0,                 /*requant*/
       0,                 /*rq_shift*/
       0,                 /*out_zp*/
       false, RM_HALF_UP, false);
}

template <typename DataType0, typename DataType1, typename DataType2>
void fmm2(tensor<DataType0> &dst, tensor<DataType1> &left,
          tensor<DataType1> &right, DataType2 &bias, bool ltrans, bool rtrans,
          bool do_relu, bool result_add, bool out_dtype,
          bool do_saturate = false) {
  fmm2(dst, left, right, bias, ltrans, rtrans, false, do_relu, result_add,
       static_cast<data_type_t>(out_dtype), true, do_saturate);
}

template <typename DataType0, typename DataType1>
void fmm2(tensor<DataType0> &dst, tensor<DataType1> &left,
          tensor<DataType1> &right, bool ltrans, bool rtrans, bool do_relu,
          bool result_add, bool out_dtype, bool do_saturate = false) {
  tensor<DataType0> *bias = nullptr;
  fmm2(dst, left, right, bias, ltrans, rtrans, false, do_relu, result_add,
       static_cast<data_type_t>(out_dtype), false, do_saturate);
}

template <typename DataType0, typename DataType1, typename DataType2>
void fmm2(tensor<DataType0> &dst, DataType1 &left, DataType2 &right,
          bool ltrans, bool rtrans, bool rst_trans, bool result_add,
          data_type_t out_dtype, bool saturate = false) {
  tensor<DataType0> *bias = nullptr;
  fmm2(dst, left, right, bias, ltrans, rtrans, rst_trans, false, result_add,
       out_dtype, false, saturate);
}
// template <typename DataType0, typename DataType1, typename DataType2>
// void mm2(tensor<int> &dst, tensor<DataType0> &left, tensor<DataType1> &right,
//          DataType2 &bias, int zp_val, bool ltrans, bool rtrans,
//          bool result_add);
// template <typename DataType0, typename DataType1>
// void mm2(tensor<int> &dst, tensor<DataType0> &left, tensor<DataType1> &right,
//          int zp_val, bool ltrans, bool rtrans, bool result_add) {
//   tensor<DataType1> *bias = nullptr;
//   mm2(dst, left, right, bias, zp_val, ltrans, rtrans, result_add);
// }

// template <typename DataType0, typename DataType1, typename DataType2>
// void mm2(tensor<int> &dst, tensor<DataType0> &left, tensor<DataType1> &right,
//          DataType2 &bias, tensor<DataType1> &zp_val, bool ltrans, bool
//          rtrans, bool result_add);
// template <typename DataType0, typename DataType1>
// void mm2(tensor<int> &dst, tensor<DataType0> &left, tensor<DataType1> &right,
//          tensor<DataType1> &zp_val, bool ltrans, bool rtrans, bool
//          result_add) {
//   tensor<DataType1> *bias = nullptr;
//   mm2(dst, left, right, bias, zp_val, ltrans, rtrans, result_add);
// }

// template <typename DataType0, typename DataType1, typename DataType2,
//           typename DataType3, typename DataType4, typename DataType5>
// void mm2(tensor<DataType0> &dst, DataType1 &left, tensor<DataType2> &right,
//          DataType3 &bias, DataType4 &r_zp, DataType5 &requant, int
//          multiplier, int8 rshift, int16 y_zp, bool ltrans, bool rtrans, bool
//          result_add, bool do_relu, bool do_rq, bool is_perchannel, bool
//          saturate, rounding_mode_t round_mode);

//=================== conv sym without requant ===================
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void conv_sym(tensor<DataType0> &dst, tensor<DataType1> &src,
              tensor<DataType2> &weight, DataType3 &bias, int oc, dim2 *k_shape,
              dim2 *stride, dim2 *dilation, padding_t *pad, bool result_relu,
              bool has_bias, int8 rshift) {
  int pad_val = 0;
  int requant = 0;
  int8 rq_shift = 0;
  short out_zp = 0;
#if defined(__bm1684x__)
  dim4 bias_shape = {1, oc, 1, 1};
  if constexpr (std::is_same<DataType3, tensor<int32>>::value) {
    auto bias_cast = tensor<int16>(bias_shape, TPU_COMPACT);
    cast(bias_cast, bias);
    conv(dst, src, weight, bias_cast, oc, k_shape, stride, dilation, pad,
         (dim2 *)nullptr, pad_val, result_relu, false, DT_NONE, has_bias, true,
         rshift, false, requant, rq_shift, out_zp, false, RM_HALF_UP, false);
  } else if (std::is_same<DataType3, tensor<uint32>>::value) {
    auto bias_cast = tensor<uint16>(bias_shape, TPU_COMPACT);
    cast(bias_cast, bias);
    conv(dst, src, weight, bias_cast, oc, k_shape, stride, dilation, pad,
         (dim2 *)nullptr, pad_val, result_relu, false, DT_NONE, has_bias, true,
         rshift, false, requant, rq_shift, out_zp, false, RM_HALF_UP, false);
  } else {
    conv(dst, src, weight, bias, oc, k_shape, stride, dilation, pad,
         (dim2 *)nullptr, pad_val, result_relu, false, DT_NONE, has_bias, true,
         rshift, false, requant, rq_shift, out_zp, false, RM_HALF_UP, false);
  }
#else
  conv(dst, src, weight, bias, oc, k_shape, stride, dilation, pad,
       (dim2 *)nullptr, pad_val, /*pad_val*/
       result_relu, false,       /*result_add*/
       DT_NONE,                  /*out_dtype*/
       has_bias,                 /*has_bias*/
       true,                     /*sym*/
       rshift,                   /*quant*/
       false,                    /*rq*/
       requant,                  /*requant*/
       rq_shift,                 /*rq_shift*/
       out_zp,                   /*out_zp*/
       false,                    /*sym_range*/
       RM_HALF_UP,               /*round_mode*/
       false /*kernel rotate*/);
#endif
}

// tpu_bdc_int8_sym_quant_conv2d
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void conv_sym(tensor<DataType0> &dst, tensor<DataType1> &src,
              tensor<DataType2> &weight, DataType3 &bias, int oc, dim2 *k_shape,
              dim2 *stride, dim2 *dilation, padding_t *pad, bool result_relu,
              int8 rshift) {
  conv_sym(dst, src, weight, bias, oc, k_shape, stride, dilation, pad,
           result_relu, true, rshift);
}
template <typename DataType0, typename DataType1, typename DataType2>
void conv_sym(tensor<DataType0> &dst, tensor<DataType1> &src,
              tensor<DataType2> &weight, int oc, dim2 *k_shape, dim2 *stride,
              dim2 *dilation, padding_t *pad, bool result_relu, int8 rshift) {
  tensor<fp32> *bias = nullptr;
  conv_sym(dst, src, weight, *bias, oc, k_shape, stride, dilation, pad,
           result_relu, false, rshift);
}

//=================== conv asym without requant ===================
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4>
void conv_asym(tensor<DataType0> &dst, tensor<DataType1> &src,
               DataType2 &weight, int oc, dim2 *k_shape, dim2 *stride,
               dim2 *dilation, padding_t *pad, DataType3 pad_val,
               bool result_add, data_type_t out_dtype, DataType4 &kzp) {
  tensor<fp32> *bias = nullptr;
  int requant = 0;
  int rq_shift = 0;
  int out_zp = 0;
  conv(dst, src, weight, bias, oc, k_shape, stride, dilation, pad,
       (dim2 *)nullptr, pad_val, /*pad_val*/
       false, result_add,        /*result_add*/
       out_dtype,                /*out_dtype*/
       false,                    /*has_bias*/
       false,                    /*sym*/
       kzp,                      /*quant*/
       false,                    /*rq*/
       requant,                  /*requant*/
       rq_shift,                 /*rq_shift*/
       out_zp,                   /*out_zp*/
       false, RM_HALF_UP, false);
}

// tpu_bdc_int8_asym_quant_conv2d (kzp and pad_val are const)
template <typename DataType0, typename DataType1, typename DataType2>
void conv_asym(tensor<DataType0> &dst, tensor<DataType1> &src,
               tensor<DataType2> &weight, int oc, dim2 *k_shape, dim2 *stride,
               dim2 *dilation, padding_t *pad, bool result_add,
               data_type_t out_dtype, int kzp) {
  int pad_val = 0;
  conv_asym(dst, src, weight, oc, k_shape, stride, dilation, pad, pad_val,
            result_add, out_dtype, kzp);
}

// tpu_bdc_int8_asym_quant_conv2d_kernel_const (weight, kzp and pad_val are
// const)
template <typename DataType0, typename DataType1>
void conv_asym(tensor<DataType0> &dst, tensor<DataType1> &src, int8 weight,
               int oc, dim2 *k_shape, dim2 *stride, dim2 *dilation,
               padding_t *pad, bool result_add, data_type_t out_dtype,
               int kzp) {
  int pad_val = 0;
  conv_asym(dst, src, weight, oc, k_shape, stride, dilation, pad, pad_val,
            result_add, out_dtype, kzp);
}

// tpu_bdc_int8_asym_pc_quant_conv2d
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4>
void conv_asym(tensor<DataType0> &dst, tensor<DataType1> &src,
               tensor<DataType2> &weight, int oc, dim2 *k_shape, dim2 *stride,
               dim2 *dilation, padding_t *pad, tensor<DataType3> &pad_val,
               bool result_add, data_type_t out_dtype, tensor<DataType4> &kzp) {
  tensor<fp32> *bias = nullptr;
  int requant = 0;
  int rq_shift = 0;
  int out_zp = 0;
  conv(dst, src, weight, bias, oc, k_shape, stride, dilation, pad,
       (dim2 *)nullptr, pad_val, /*pad_val*/
       false, result_add,        /*result_add*/
       out_dtype,                /*out_dtype*/
       false,                    /*has_bias*/
       false,                    /*sym*/
       kzp,                      /*quant*/
       false,                    /*rq*/
       requant,                  /*requant*/
       rq_shift,                 /*rq_shift*/
       out_zp,                   /*out_zp*/
       false, RM_HALF_UP, false);
}
//=================== conv sym with requant ===================
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4>
void conv_sym_rq(tensor<DataType0> &dst, tensor<DataType1> &src,
                 tensor<DataType2> &weight, DataType3 &bias, int oc,
                 dim2 *k_shape, dim2 *stride, dim2 *dilation, padding_t *pad,
                 bool result_relu, bool has_bias, DataType4 &requant,
                 int8 rshift, short out_zp, bool sym_range) {
  int pad_val = 0;
  int kzp = 0;
#if defined(__bm1684x__)
  tensor<fp32> *bias_null = nullptr;
  int requant_ = 0;
  int rshift_ = 0;
  int out_zp_ = 0;
  bool has_bias_ = false;
  bool result_relu_ = false;
  conv(dst, src, weight, bias_null, oc, k_shape, stride, dilation, pad,
       (dim2 *)nullptr, pad_val, result_relu_, false, DT_NONE, has_bias_, true,
       kzp, false, requant_, rshift_, out_zp_, sym_range, RM_HALF_UP, false);
  if (has_bias) {
    add(dst, dst, bias);
  }

  if constexpr (!std::is_same<DataType0, int16>::value) {
    tensor<int32> dst_i32;
    if constexpr (std::is_fundamental<DataType4>::value) {
      mul(dst_i32, dst, requant, rshift, RM_HALF_UP, true);
    } else {
      rq1(dst_i32, dst, requant, RM_HALF_UP);
    }
    if (result_relu) {
      max(dst_i32, dst_i32, 0);
    }
    add(dst_i32, dst_i32, out_zp);
    cast(dst, dst_i32);
  } else {
    if constexpr (std::is_fundamental<DataType4>::value) {
      mul(dst, dst, requant, rshift, RM_HALF_UP, true);
    } else {
      rq1(dst, dst, requant, RM_HALF_UP);
    }
    if (result_relu) {
      max(dst, dst, 0);
    }
    add(dst, dst, out_zp);
  }

#else
  conv(dst, src, weight, bias, oc, k_shape, stride, dilation, pad,
       (dim2 *)nullptr, pad_val, /*pad_val*/
       result_relu,              /*result_relu*/
       false,                    /*result_add*/
       DT_NONE, has_bias, true,  /*sym*/
       kzp,                      /*quant*/
       true,                     /*rq*/
       requant,                  /*requant or multiplier*/
       rshift,                   /*rq_shift*/
       out_zp,                   /*out_zp*/
       sym_range, RM_HALF_UP, false);
#endif
}

// tpu_bdc_conv2d_requant_C (only in bm1690)
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void conv_sym_rq(tensor<DataType0> &dst, tensor<DataType1> &src,
                 tensor<DataType2> &weight, DataType3 &bias, int oc,
                 dim2 *k_shape, dim2 *stride, dim2 *dilation, padding_t *pad,
                 bool result_relu, int multiplier, int8 rshift, short out_zp,
                 bool sym_range) {
  conv_sym_rq(dst, src, weight, bias, oc, k_shape, stride, dilation, pad,
              result_relu, true, multiplier, rshift, out_zp, sym_range);
}
template <typename DataType0, typename DataType1, typename DataType2>
void conv_sym_rq(tensor<DataType0> &dst, tensor<DataType1> &src,
                 tensor<DataType2> &weight, int oc, dim2 *k_shape, dim2 *stride,
                 dim2 *dilation, padding_t *pad, bool result_relu,
                 int multiplier, int8 rshift, short out_zp, bool sym_range) {
  tensor<fp32> *bias = nullptr;
  conv_sym_rq(dst, src, weight, bias, oc, k_shape, stride, dilation, pad,
              result_relu, false, multiplier, rshift, out_zp, sym_range);
}

// tpu_bdc_conv2d_requant_pc (only in bm1690)
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4>
void conv_sym_rq(tensor<DataType0> &dst, tensor<DataType1> &src,
                 tensor<DataType2> &weight, DataType3 &bias, int oc,
                 dim2 *k_shape, dim2 *stride, dim2 *dilation, padding_t *pad,
                 bool result_relu, DataType4 &requant, bool sym_range) {
  int8 rshift = 0;
  short out_zp = 0;
  conv_sym_rq(dst, src, weight, bias, oc, k_shape, stride, dilation, pad,
              result_relu, true, requant, rshift, out_zp, sym_range);
}
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void conv_sym_rq(tensor<DataType0> &dst, tensor<DataType1> &src,
                 tensor<DataType2> &weight, int oc, dim2 *k_shape, dim2 *stride,
                 dim2 *dilation, padding_t *pad, bool result_relu,
                 DataType3 &requant, bool sym_range) {
  tensor<fp32> *bias = nullptr;
  int8 rshift = 0;
  short out_zp = 0;
  conv_sym_rq(dst, src, weight, bias, oc, k_shape, stride, dilation, pad,
              result_relu, false, requant, rshift, out_zp, sym_range);
}
// conv2d fp
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void fconv(tensor<DataType0> &dst, tensor<DataType1> &src, DataType3 &weight,
           DataType2 &bias, int oc, dim2 *k_shape, dim2 *stride, dim2 *dilation,
           padding_t *pad, dim2 *ins, bool result_relu, bool result_add,
           data_type_t out_dtype, bool has_bias, bool saturate,
           bool kernel_rotate); // Conv2DFpOp

template <typename DataType0, typename DataType1, typename DataType2>
void fconv(tensor<DataType0> &dst, tensor<DataType1> &src,
           tensor<DataType1> &weight, DataType2 &bias, int oc, dim2 *k_shape,
           dim2 *stride, dim2 *dilation, padding_t *pad, bool result_add,
           data_type_t out_dtype, bool has_bias = true) {
  fconv(dst, src, weight, bias, oc, k_shape, stride, dilation, pad,
        (dim2 *)nullptr, false, result_add, out_dtype, has_bias, false, false);
} // tpu_bdc_fp_conv2d with bias

template <typename DataType0, typename DataType1>
void fconv(tensor<DataType0> &dst, tensor<DataType1> &src,
           tensor<DataType1> &weight, int oc, dim2 *k_shape, dim2 *stride,
           dim2 *dilation, padding_t *pad, bool result_add,
           data_type_t out_dtype) {
  tensor<fp32> *bias = nullptr;
  fconv(dst, src, weight, bias, oc, k_shape, stride, dilation, pad,
        (dim2 *)nullptr, false, result_add, out_dtype, false, false, false);
} // tpu_bdc_fp_conv2d without bias

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void fconv(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 &weight_C,
           DataType3 &bias, int oc, dim2 *k_shape, dim2 *stride, dim2 *dilation,
           padding_t *pad, bool result_add, data_type_t out_dtype) {

  fconv(dst, src, weight_C, bias, oc, k_shape, stride, dilation, pad,
        (dim2 *)nullptr, false, result_add, out_dtype, true, false, false);
} // tpu_bdc_fp_conv2d_kernel_const

template <typename DataType0, typename DataType1, typename DataType2>
void fconv(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 &weight_C,
           int oc, dim2 *k_shape, dim2 *stride, dim2 *dilation, padding_t *pad,
           bool result_add, data_type_t out_dtype) {
  tensor<fp32> *bias = nullptr;
  fconv(dst, src, weight_C, bias, oc, k_shape, stride, dilation, pad,
        (dim2 *)nullptr, false, result_add, out_dtype, false, false, false);
} // tpu_bdc_fp_conv2d_kernel_const

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void fconv(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 &weight,
           DataType3 &bias, dim2 *k_shape, dim2 *stride, dim2 *dilation,
           padding_t *pad, dim2 *ins, bool result_add, bool saturate) {
  int oc = 0;
  fconv(dst, src, weight, bias, oc, k_shape, stride, dilation, pad, ins, false,
        result_add, DT_NONE, true, saturate, false);
} // sg.fconv or sg.fconva

} // namespace tiu

namespace hau {}
} // namespace ppl
