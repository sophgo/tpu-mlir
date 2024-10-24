#pragma once
#include "ppl_defs.h"
#include "ppl_tpu.h"
#include "ppl_types.h"
#include "ppl_utils.h"

namespace ppl {
namespace tiu {

template <typename DataType>
void arange_broadcast(tensor<DataType> &dst, int start, int step, int num);

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void mm(tensor<DataType0> &dst, DataType1 &left, tensor<DataType2> &right,
        DataType3 &bias, bool ltrans, bool rtrans, bool result_add, int lshift,
        int rshift, bool do_relu, rounding_mode_t round_mode); // MMOp

//=================== mm int ===================
// tpu_bdc_int_mm
template <typename DataType0, typename DataType1, typename DataType2>
void mm(tensor<DataType0> &dst, tensor<DataType1> &left,
        tensor<DataType2> &right, int rshift, rounding_mode_t round_mode) {
  tensor<int> *bias = nullptr;
  bool ltrans = false;
  bool rtrans = false;
  bool result_add = false;
  int lshift = 0;
  bool do_relu = false;
  mm(dst, left, right, bias, ltrans, rtrans, result_add, lshift, rshift,
     do_relu, round_mode);
}
template <typename DataType0, typename DataType1, typename DataType2>
void mm(tensor<DataType0> &dst, tensor<DataType1> &left,
        tensor<DataType2> &right) {
  int rshift = 0;
  rounding_mode_t round_mode = RM_HALF_UP;
  mm(dst, left, right, rshift, round_mode);
}

// tpu_bdc_int_mm_L_trans
template <typename DataType0, typename DataType1, typename DataType2>
void mm(tensor<DataType0> &dst, tensor<DataType1> &left,
        tensor<DataType2> &right, bool ltrans, int rshift,
        rounding_mode_t round_mode) {
  tensor<int> *bias = nullptr;
  mm(dst, left, right, bias, ltrans, false, false, 0, rshift, false,
     round_mode);
}
template <typename DataType0, typename DataType1, typename DataType2>
void mm(tensor<DataType0> &dst, tensor<DataType1> &left,
        tensor<DataType2> &right, bool ltrans) {
  int rshift = 0;
  rounding_mode_t round_mode = RM_HALF_UP;
  mm(dst, left, right, ltrans, rshift, round_mode);
}

// tpu_bdc_int_mm_L_const
template <typename DataType0, typename DataType1, typename DataType2>
void mm(tensor<DataType0> &dst, DataType1 left, tensor<DataType2> &right,
        int rshift, rounding_mode_t round_mode) {
  tensor<int> *bias = nullptr;
  mm(dst, left, right, bias, false, false, false, 0, rshift, false, round_mode);
}
template <typename DataType0, typename DataType1, typename DataType2>
void mm(tensor<DataType0> &dst, DataType1 left, tensor<DataType2> &right) {
  int rshift = 0;
  rounding_mode_t round_mode = RM_HALF_UP;
  mm(dst, left, right, rshift, round_mode);
}

//=================== mm int8 ===================
// tpu_bdc_int8_mm
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void mm(tensor<DataType0> &dst, tensor<DataType1> &left,
        tensor<DataType2> &right, DataType3 &bias, bool result_add, int lshift,
        int rshift, bool do_relu) {
  bool ltrans = false;
  bool rtrans = false;
  rounding_mode_t round_mode = RM_HALF_UP;
  mm(dst, left, right, bias, ltrans, rtrans, result_add, lshift, rshift,
     do_relu, round_mode);
}

// tpu_bdc_int8_mm_L_trans
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void mm(tensor<DataType0> &dst, tensor<DataType1> &left,
        tensor<DataType2> &right, DataType3 &bias, bool ltrans, bool result_add,
        int lshift, int rshift, bool do_relu) {
  mm(dst, left, right, bias, ltrans, false, result_add, lshift, rshift, do_relu,
     RM_HALF_UP);
}

// tpu_bdc_int8_mm_L_const
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void mm(tensor<DataType0> &dst, DataType1 left, tensor<DataType2> &right,
        DataType3 &bias, bool result_add, int lshift, int rshift,
        bool do_relu) {
  mm(dst, left, right, bias, false, false, result_add, lshift, rshift, do_relu,
     RM_HALF_UP);
}

//=================== mm fp32 ===================
// tpu_bdc_fp32_mm
void fmm(tensor<fp32> &dst, tensor<fp32> &left, tensor<fp32> &right, fp32 &bias,
         bool result_add) {
  mm(dst, left, right, bias, false, false, result_add, 0, 0, false, RM_HALF_UP);
}
void fmm(tensor<fp32> &dst, tensor<fp32> &left, tensor<fp32> &right,
         bool result_add) {
  tensor<fp32> *bias = nullptr;
  mm(dst, left, right, bias, false, false, result_add, 0, 0, false, RM_HALF_UP);
}

// tpu_bdc_fp32_mm_L_trans
void fmm(tensor<fp32> &dst, tensor<fp32> &left, tensor<fp32> &right, fp32 &bias,
         bool ltrans, bool result_add) {
  mm(dst, left, right, bias, ltrans, false, result_add, 0, 0, false,
     RM_HALF_UP);
}
void fmm(tensor<fp32> &dst, tensor<fp32> &left, tensor<fp32> &right,
         bool ltrans, bool result_add) {
  tensor<fp32> *bias = nullptr;
  mm(dst, left, right, bias, ltrans, false, result_add, 0, 0, false,
     RM_HALF_UP);
}

// tpu_bdc_fp32_mm_L_const
void fmm(tensor<fp32> &dst, fp32 &left, tensor<fp32> &right, fp32 &bias,
         bool result_add) {
  mm(dst, left, right, bias, false, false, result_add, 0, 0, false, RM_HALF_UP);
}
void fmm(tensor<fp32> &dst, fp32 &left, tensor<fp32> &right, bool result_add) {
  tensor<fp32> *bias = nullptr;
  mm(dst, left, right, bias, false, false, result_add, 0, 0, false, RM_HALF_UP);
}

//=================== mm2 int ===================
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4, typename DataType5>
void mm2(tensor<DataType0> &rst, DataType1 &left, DataType2 &right,
         DataType3 &bias, DataType4 &r_zp, DataType5 &requant, int multiplier,
         int8 rshift, int16 y_zp, bool ltrans, bool rtrans, bool rst_trans,
         bool result_add, data_type_t out_dtype, bool has_bias, bool do_relu,
         bool do_rq, bool saturate, rounding_mode_t round_mode); // MM2Op

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4>
void mm2(tensor<DataType0> &rst, DataType1 &left, DataType2 &right,
         DataType3 &bias, DataType4 &r_zp, bool ltrans, bool rtrans,
         bool rst_trans, bool result_add, data_type_t out_dtype,
         bool has_bias) {
  tensor<int32> *requant = nullptr;
  int multiplier = 0;
  int8 rshift = 0;
  int16 y_zp = 0;
  bool do_relu = false;
  bool do_rq = false;
  bool saturate = false;
  rounding_mode_t round_mode = RM_HALF_AWAY_FROM_ZERO;
  mm2(rst, left, right, bias, r_zp, requant, multiplier, rshift, y_zp, ltrans,
      rtrans, rst_trans, result_add, out_dtype, has_bias, do_relu, do_rq,
      saturate, round_mode);
}

// tpu_bdc_int8_zp_mm_all_trans, tpu_bdc_int8_zp_mm_R_trans in BM1684x
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void mm2(tensor<DataType0> &rst, DataType1 &left, DataType2 &right,
         DataType3 &r_zp, bool ltrans, bool rtrans, bool rst_trans,
         bool result_add, data_type_t out_dtype) {
  tensor<int32> *bias = nullptr;
  bool has_bias = false;
  mm2(rst, left, right, bias, r_zp, ltrans, rtrans, rst_trans, result_add,
      out_dtype, has_bias);
}

template <typename DataType0, typename DataType1, typename DataType2>
void mm2(tensor<DataType0> &rst, DataType1 &left, DataType2 &right, bool ltrans,
         bool rtrans, bool rst_trans) {
  int8 r_zp = 0;
  mm2(rst, left, right, r_zp, ltrans, rtrans, rst_trans, false, DT_NONE);
}

template <typename DataType0, typename DataType1, typename DataType2>
void mm2(tensor<DataType0> &rst, DataType1 &left, DataType2 &right) {
  int8 r_zp = 0;
  mm2(rst, left, right, r_zp, false, false, false, false, DT_NONE);
}

// tpu_bdc_int8_zp_mm
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void mm2(tensor<DataType0> &rst, DataType1 &left, DataType2 &right,
         DataType3 r_zp, bool result_add, data_type_t out_dtype) {
  tensor<int32> *bias = nullptr;
  bool ltrans = false;
  bool rtrans = false;
  bool rst_trans = false;
  mm2(rst, left, right, bias, r_zp, ltrans, rtrans, rst_trans, result_add,
      out_dtype, false);
}

//=================== mm2 fp32 ===================
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void fmm2(tensor<DataType0> &dst, DataType1 &left, DataType2 &right,
          DataType3 &bias, bool ltrans, bool rtrans, bool rst_trans,
          bool do_relu, bool result_add, data_type_t out_dtype, bool has_bias,
          bool saturate = false); // MM2FpOp

// tpu_bdc_fp_mm_with_bias in BM1690
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void fmm2(tensor<DataType0> &dst, DataType1 &left, DataType2 &right,
          DataType3 &bias, bool result_add, data_type_t out_dtype,
          bool has_bias, bool saturate = false) {
  fmm2(dst, left, right, bias, false, false, false, false, result_add,
       out_dtype, has_bias, saturate);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void fmm2(tensor<DataType0> &dst, DataType1 &left, DataType2 &right,
          DataType3 &bias) {
  fmm2(dst, left, right, bias, false, false, false, false, false,
       convert_dtype<DataType0>(), true, false);
}

// tpu_bdc_fp_mm_all_trans, tpu_bdc_fp_mm_R_trans in BM1684x
template <typename DataType0, typename DataType1, typename DataType2>
void fmm2(tensor<DataType0> &dst, DataType1 &left, DataType2 &right,
          bool ltrans, bool rtrans, bool rst_trans, bool do_relu,
          bool result_add, data_type_t out_dtype, bool saturate = false) {
  tensor<DataType0> *bias = nullptr;
  fmm2(dst, left, right, bias, ltrans, rtrans, rst_trans, do_relu, result_add,
       out_dtype, false, saturate);
}

template <typename DataType0, typename DataType1, typename DataType2>
void fmm2(tensor<DataType0> &dst, DataType1 &left, DataType2 &right,
          bool ltrans, bool rtrans, bool rst_trans) {
  tensor<DataType0> *bias = nullptr;
  fmm2(dst, left, right, bias, ltrans, rtrans, rst_trans, false, false,
       convert_dtype<DataType0>(), false, false);
}
// tpu_bdc_fp_mm in BM1684x
template <typename DataType0, typename DataType1, typename DataType2>
void fmm2(tensor<DataType0> &dst, DataType1 &left, DataType2 &right,
          bool result_add, data_type_t out_dtype) {
  tensor<DataType0> *bias = nullptr;
  fmm2(dst, left, right, bias, false, false, false, false, result_add,
       out_dtype, false, false);
}
template <typename DataType0, typename DataType1, typename DataType2>
void fmm2(tensor<DataType0> &dst, DataType1 &left, DataType2 &right) {
  tensor<DataType0> *bias = nullptr;
  fmm2(dst, left, right, bias, false, false, false, false, false,
       convert_dtype<DataType0>(), false, false);
}

template <typename DataType0, typename DataType1>
void cast(tensor<DataType0> &dst, tensor<DataType1> &src,
          rounding_mode_t round_mode);

template <typename DataType0, typename DataType1>
void cast(tensor<DataType0> &dst, tensor<DataType1> &src) {
  return cast(dst, src, RM_HALF_TO_EVEN);
};

template <typename DataType>
void abs(tensor<DataType> &dst, tensor<DataType> &src);
template <typename DataType> void abs(tensor<DataType> &dst, int C);

template <typename DataType>
void fabs(tensor<DataType> &dst, tensor<DataType> &src);
template <typename DataType> void fabs(tensor<DataType> &dst, float C);

/*
 * Note:
 * 1. use when shape.c > shape.w
 */
template <typename DataType>
void transpose_cw(tensor<DataType> &dst, tensor<DataType> &src);

/*
 * Note:
 * 1. use when shape.w > shape.c
 */
template <typename DataType>
void transpose_wc(tensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType0, typename DataType1>
void rq0(tensor<DataType0> &dst, tensor<DataType1> &src, float scale,
         float offset, rounding_mode_t dst_round_mode,
         rounding_mode_t src_round_mode);

template <typename DataType0, typename DataType1>
void rq0(tensor<DataType0> &dst, tensor<DataType1> &src, tensor<float> &quant,
         rounding_mode_t dst_round_mode, rounding_mode_t src_round_mode);

template <typename DataType0, typename DataType1>
void rq1(tensor<DataType0> &dst, tensor<DataType1> &src, int multiplier,
         int shift, int offset, rounding_mode_t round_mode);

template <typename DataType0, typename DataType1>
void rq1(tensor<DataType0> &dst, tensor<DataType1> &src, tensor<float> &quant,
         rounding_mode_t round_mode);

template <typename DataType0, typename DataType1>
void dq0(tensor<DataType0> &dst, tensor<DataType1> &src, short offset,
         float scale, rounding_mode_t round_mode);

template <typename DataType0, typename DataType1>
void dq0(tensor<DataType0> &dst, tensor<DataType1> &src, tensor<int> &quant,
         rounding_mode_t round_mode);

template <typename DataType0, typename DataType1>
void dq1(tensor<DataType0> &dst, tensor<DataType1> &src, short offset,
         int multiplier, char shift, rounding_mode_t round_mode);

template <typename DataType0, typename DataType1>
void dq1(tensor<DataType0> &dst, tensor<DataType1> &src, tensor<int> &quant,
         rounding_mode_t round_mode);

template <typename DataType0, typename DataType1>
void dq2(tensor<DataType0> &dst, tensor<DataType1> &src,
         tensor<uint32> &offst_scale, int gsize);

// dst = src0 * src1 + src2
template <typename DataType>
void fmul_add(tensor<DataType> &dst, tensor<DataType> &src0,
              tensor<DataType> &src1, tensor<DataType> &src2);
template <typename DataType>
void fmul_add(tensor<DataType> &dst, float src0_c, tensor<DataType> &src1,
              tensor<DataType> &src2);
template <typename DataType>
void fmul_add(tensor<DataType> &dst, tensor<DataType> &src0, float src1_c,
              tensor<DataType> &src2);
template <typename DataType>
void fmul_add(tensor<DataType> &dst, tensor<DataType> &src0,
              tensor<DataType> &src1, float src2_c);
template <typename DataType>
void fmul_add(tensor<DataType> &dst, tensor<DataType> &src0, float src1_c,
              float src2_c);
template <typename DataType>
void fmul_add(tensor<DataType> &dst, float src0_c, float src1_c, float src2_c);

// dst = (src + bias)^2
template <typename DataType>
void fadd_sqr(tensor<DataType> &dst, tensor<DataType> &src,
              tensor<DataType> &bias);
template <typename DataType>
void fadd_sqr(tensor<DataType> &dst, float src_c, tensor<DataType> &bias);
template <typename DataType>
void fadd_sqr(tensor<DataType> &dst, tensor<DataType> &src, float bias_c);
template <typename DataType>
void fadd_sqr(tensor<DataType> &dst, float src_c, float bias_c);

// dst = (src - bias)^2
template <typename DataType>
void fsub_sqr(tensor<DataType> &dst, tensor<DataType> &src,
              tensor<DataType> &bias);
template <typename DataType>
void fsub_sqr(tensor<DataType> &dst, float src_c, tensor<DataType> &bias);
template <typename DataType>
void fsub_sqr(tensor<DataType> &dst, tensor<DataType> &src, float bias_c);
template <typename DataType>
void fsub_sqr(tensor<DataType> &dst, float src_c, float bias_c);

template <typename DataType, typename DataType2, typename DataType3>
void fscale(tensor<DataType> &dst, tensor<DataType> &src, DataType2 &scalar,
            DataType3 &bias);

template <typename DataType0, typename DataType1, typename DataType2>
void fscale(tensor<DataType0> &dst, tensor<DataType1> &src,
            tensor<DataType2> &scale) {
  void *bias = nullptr;
  fscale(dst, src, scale, bias);
}

template <typename DataType0, typename DataType1, typename DataType2>
void fbias(tensor<DataType0> &dst, tensor<DataType1> &src,
           tensor<DataType2> &bias) {
  void *scalar = nullptr;
  fscale(dst, src, scalar, bias);
}

template <typename DataType>
void fdiv(tensor<DataType> &dst, tensor<DataType> &src0, tensor<DataType> &src1,
          int num_iter = 3);
template <typename DataType>
void fdiv(tensor<DataType> &dst, tensor<DataType> &src, float C,
          int num_iter = 3);
template <typename DataType>
void fdiv(tensor<DataType> &dst, float C, tensor<DataType> &src,
          int num_iter = 3);

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

template <typename DataType0, typename DataType1, typename DataType2>
void fdeconv(tensor<DataType0> &dst, tensor<DataType1> &src,
             tensor<DataType1> &weight, DataType2 &bias, int oc, dim2 *k_shape,
             dim2 *dilation, padding_t *pad, dim2 *insert, bool result_add,
             data_type_t out_dtype, bool has_bias);

template <typename DataType0, typename DataType1, typename DataType2>
void fdeconv(tensor<DataType0> &dst, tensor<DataType1> &src,
             tensor<DataType1> &weight, int oc, dim2 *k_shape, dim2 *dilation,
             padding_t *pad, dim2 *insert, bool result_add,
             data_type_t out_dtype) {
  tensor<DataType0> *bias = nullptr;
  fdeconv(dst, src, weight, bias, oc, k_shape, dilation, pad, insert,
          result_add, out_dtype, false);
}

template <typename DataType>
void fdw_conv(tensor<DataType> &dst, tensor<DataType> &src,
              tensor<DataType> &weight, tensor<DataType> &bias, dim2 *k_shape,
              dim2 *stride, dim2 *dilation, padding_t *pad, bool has_bias);

template <typename DataType>
void fdw_conv(tensor<DataType> &dst, tensor<DataType> &src,
              tensor<DataType> &weight, dim2 *k_shape, dim2 *stride,
              dim2 *dilation, padding_t *pad) {
  tensor<DataType> *bias = nullptr;
  fdw_conv(dst, src, weight, bias, k_shape, stride, dilation, pad, false);
}

template <typename DataType0, typename DataType1>
void fdw_deconv(tensor<DataType0> &dst, tensor<DataType1> &src,
                tensor<DataType1> &weight, tensor<DataType1> &bias,
                dim2 *k_shape, dim2 *dilation, padding_t *pad, dim2 *insert,
                data_type_t out_dtype, bool has_bias);

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4, typename DataType5,
          typename DataType6>
void conv(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 &weight,
          DataType3 &bias, int oc, dim2 *k_shape, dim2 *stride, dim2 *dilation,
          padding_t *pad, dim2 *ins, DataType4 &pad_val, bool result_relu,
          bool result_add, data_type_t out_dtype, bool has_bias, bool sym,
          DataType5 &quant, bool rq, DataType6 &requant, int8 rq_shift,
          short out_zp, bool saturate, rounding_mode_t round,
          bool kernel_rotate); // Conv2DOp

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void conv(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 &weight,
          DataType3 &bias, int oc, dim2 *k_shape, dim2 *stride, dim2 *dilation,
          padding_t *pad, bool result_relu, bool result_add,
          data_type_t out_dtype, bool has_bias); // Conv2DOp

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
  conv_sym(dst, src, weight, bias, oc, k_shape, stride, dilation, pad,
           result_relu, false, rshift);
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

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4, typename DataType5,
          typename DataType6>
void deconv(tensor<DataType0> &dst, tensor<DataType1> &src,
            tensor<DataType2> &weight, DataType3 &bias, int oc, dim2 *k_shape,
            dim2 *dilation, padding_t *pad, dim2 *ins, DataType4 &pad_val,
            DataType5 insert_val, bool result_relu, bool result_add,
            data_type_t out_dtype, bool has_bias, bool sym, DataType6 &quant);

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void deconv_sym(tensor<DataType0> &dst, tensor<DataType1> &src,
                tensor<DataType2> &weight, DataType3 &bias, int oc,
                dim2 *k_shape, dim2 *dilation, padding_t *pad, dim2 *ins,
                bool result_relu, data_type_t out_dtype, bool has_bias,
                uint8 rshift) {
  deconv(dst, src, weight, bias, oc, k_shape, dilation, pad, ins, 0, /*pad val*/
         0,                         /*insert val*/
         result_relu, false,        /*result add*/
         out_dtype, has_bias, true, /*sym*/
         rshift);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void deconv_asym(tensor<DataType0> &dst, tensor<DataType1> &src,
                 tensor<DataType2> &weight, int oc, dim2 *k_shape,
                 dim2 *dilation, padding_t *pad, dim2 *ins, int pad_val,
                 int insert_val, bool result_add, data_type_t out_dtype,
                 DataType3 kzp_val) {
  tensor<DataType1> *bias = nullptr;
  deconv(dst, src, weight, bias, oc, k_shape, dilation, pad, ins, pad_val,
         insert_val, false, /*result_relu*/ result_add, out_dtype, false,
         /*has_bias*/ false, /*sym*/ kzp_val);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4, typename DataType5,
          typename DataType6>
void deconv_asym(tensor<DataType0> &dst, tensor<DataType1> &src,
                 tensor<DataType2> &weight, int oc, dim2 *k_shape,
                 dim2 *dilation, padding_t *pad, dim2 *ins,
                 tensor<DataType4> &pad_insert_val, bool result_add,
                 data_type_t out_dtype, tensor<DataType6> &kzp) {
  tensor<DataType1> *bias = nullptr;
  deconv(dst, src, weight, bias, oc, k_shape, dilation, pad, ins,
         pad_insert_val, 0, /*insert_val*/ false, /*result_relu*/ result_add,
         out_dtype, false,
         /*has_bias*/ false, /*sym*/ kzp);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4, typename DataType5>
void dw_conv(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 &weight,
             DataType3 &bias, dim2 *k_shape, dim2 *stride, dim2 *dilation,
             padding_t *pad, DataType4 &pad_val, bool result_relu,
             data_type_t out_dtype, bool has_bias, uint8 rshift, bool rq,
             DataType5 &requant, bool saturate, rounding_mode_t round);

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4, typename DataType6>
void dw_conv_rq(tensor<DataType0> &dst, tensor<DataType1> &src,
                tensor<DataType2> &weight, DataType3 &bias, dim2 *k_shape,
                dim2 *stride, dim2 *dilation, padding_t *pad,
                DataType4 &pad_val, bool result_relu, data_type_t out_dtype,
                bool has_bias, tensor<DataType6> &requant, bool saturate,
                rounding_mode_t round) {
  dw_conv(dst, src, weight, bias, k_shape, stride, dilation, pad, pad_val,
          result_relu, out_dtype, has_bias, 0, /*rshfit*/ true, /*rq*/ requant,
          saturate, round);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4>
void dw_conv(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 &weight,
             DataType3 &bias, dim2 *k_shape, dim2 *stride, dim2 *dilation,
             padding_t *pad, DataType4 &pad_val, bool result_relu,
             data_type_t out_dtype, bool has_bias, uint8 rshift,
             rounding_mode_t round) {
  tensor<DataType1> *requant = nullptr;
  dw_conv(dst, src, weight, bias, k_shape, stride, dilation, pad, pad_val,
          result_relu, out_dtype, has_bias, rshift, false, requant, false,
          round);
}

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4>
void dw_deconv(tensor<DataType0> &dst, tensor<DataType1> &src,
               tensor<DataType2> &weight, DataType3 &bias, dim2 *k_shape,
               dim2 *dilation, padding_t *pad, dim2 *ins, DataType4 &pad_val,
               bool result_relu, data_type_t out_dtype, bool has_bias,
               uint8 rshift, rounding_mode_t round);

template <typename DataType>
void bitwise_and(tensor<DataType> &dst, tensor<DataType> &src0,
                 tensor<DataType> &src1);
template <typename DataType, typename DataType2>
void bitwise_and(tensor<DataType> &dst, tensor<DataType> &src, DataType2 C);

template <typename DataType>
void bitwise_or(tensor<DataType> &dst, tensor<DataType> &src0,
                tensor<DataType> &src1);
template <typename DataType>
void bitwise_or(tensor<DataType> &dst, tensor<DataType> &src, DataType C);

template <typename DataType>
void bitwise_xor(tensor<DataType> &dst, tensor<DataType> &src0,
                 tensor<DataType> &src1);
template <typename DataType>
void bitwise_xor(tensor<DataType> &dst, tensor<DataType> &src, DataType C);

template <typename DataType>
void bitwise_not(tensor<DataType> &dst, tensor<DataType> &src);
template <typename DataType> void bitwise_not(tensor<DataType> &dst, int C);

template <typename DataType0, typename DataType1, typename DataType2>
void min(tensor<DataType0> &dst, tensor<DataType1> &src0,
         tensor<DataType2> &src1);
template <typename DataType0, typename DataType1, typename DataType2>
void min(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 C);
template <typename DataType0, typename DataType1, typename DataType2>
void fmin(tensor<DataType0> &dst, tensor<DataType1> &src0,
          tensor<DataType2> &src1);
template <typename DataType0, typename DataType1, typename DataType2>
void fmin(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 C);

template <typename DataType0>
void max(tensor<DataType0> &dst, tensor<DataType0> &src0,
         tensor<DataType0> &src1);
template <typename DataType0, typename DataType1>
void max(tensor<DataType0> &dst, tensor<DataType0> &src, DataType1 C);
template <typename DataType0>
void fmax(tensor<DataType0> &dst, tensor<DataType0> &src0,
          tensor<DataType0> &src1);
template <typename DataType0, typename DataType1>
void fmax(tensor<DataType0> &dst, tensor<DataType0> &src, DataType1 C);

template <typename DataType>
void fadd(tensor<DataType> &dst, tensor<DataType> &src0,
          tensor<DataType> &src1);
template <typename DataType>
void fadd(tensor<DataType> &dst, tensor<DataType> &src, float C);

template <typename DataType>
void fsub(tensor<DataType> &dst, tensor<DataType> &src0,
          tensor<DataType> &src1);
template <typename DataType>
void fsub(tensor<DataType> &dst, tensor<DataType> &src, float C);
template <typename DataType>
void fsub(tensor<DataType> &dst, float C, tensor<DataType> &src);

template <typename DataType>
void fmul(tensor<DataType> &dst, tensor<DataType> &src0,
          tensor<DataType> &src1);
template <typename DataType>
void fmul(tensor<DataType> &dst, tensor<DataType> &src, float C);

template <typename DataType>
void fsub_abs(tensor<DataType> &dst, tensor<DataType> &src0,
              tensor<DataType> &src1);
template <typename DataType>
void fsub_abs(tensor<DataType> &dst, tensor<DataType> &src, float C);

// dst += src0 * src1
void fmac(tensor<float> &dst, tensor<float> &src0, tensor<float> &src1);
void fmac(tensor<float> &dst, tensor<float> &src, float C);

template <typename DataType0, typename DataType1, typename DataType2>
void add(tensor<DataType0> &dst, tensor<DataType1> &src0,
         tensor<DataType2> &src1, char shift, rounding_mode_t round_mode,
         bool saturation);
template <typename DataType0, typename DataType1, typename DataType2>
void add(tensor<DataType0> &dst, tensor<DataType1> &src0,
         tensor<DataType2> &src1) {
  char shift = 0;
  rounding_mode_t round_mode = RM_HALF_UP;
  bool saturation = false;
  add(dst, src0, src1, shift, round_mode, saturation);
}
template <typename DataType0, typename DataType1, typename DataType2>
void add(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 C,
         char shift, rounding_mode_t round_mode, bool saturation);
template <typename DataType0, typename DataType1, typename DataType2>
void add(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 C) {
  char shift = 0;
  rounding_mode_t round_mode = RM_HALF_UP;
  bool saturation = false;
  add(dst, src, C, shift, round_mode, saturation);
}

template <typename DataType0, typename DataType1, typename DataType2>
void mul(tensor<DataType0> &dst, tensor<DataType1> &src0,
         tensor<DataType2> &src1, char shift, rounding_mode_t round_mode,
         bool saturation);
template <typename DataType0, typename DataType1, typename DataType2>
void mul(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 C,
         char shift, rounding_mode_t round_mode, bool saturation);

template <typename DataType0, typename DataType1, typename DataType2>
void sub(tensor<DataType0> &dst, tensor<DataType1> &src0,
         tensor<DataType2> &src1, char shift, rounding_mode_t round_mode,
         bool saturation);
template <typename DataType0, typename DataType1, typename DataType2>
void sub(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 C,
         char shift, rounding_mode_t round_mode, bool saturation);
template <typename DataType0, typename DataType1, typename DataType2>
void sub(tensor<DataType0> &dst, DataType1 C, tensor<DataType2> &src,
         char shift, rounding_mode_t round_mode, bool saturation);
/*
 * Note:
 * 1. the shape of shift must be [1, shape->c, 1, 1]
 * 2. the data type of shift must be DT_INT8
 */
template <typename DataType0, typename DataType1>
void mul(tensor<DataType0> &dst, tensor<DataType1> &src0,
         tensor<DataType1> &src1, tensor<int8> &shift,
         rounding_mode_t round_mode, bool saturation);
template <typename DataType0, typename DataType1, typename DataType2>
void mul(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 C,
         tensor<int8> &shift, rounding_mode_t round_mode, bool saturation);

template <typename DataType0, typename DataType1>
void sub(tensor<DataType0> &dst, tensor<DataType1> &src0,
         tensor<DataType1> &src1, tensor<int8> &shift,
         rounding_mode_t round_mode, bool saturation);
template <typename DataType0, typename DataType1, typename DataType2>
void sub(tensor<DataType0> &dst, DataType1 C, tensor<DataType2> &src0,
         tensor<int8> &shift, rounding_mode_t round_mode, bool saturation);
template <typename DataType0, typename DataType1, typename DataType2>
void sub(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 C,
         tensor<int8> &shift, rounding_mode_t round_mode, bool saturation);

template <typename DataType0, typename DataType1>
void add(tensor<DataType0> &dst, tensor<DataType1> &src0,
         tensor<DataType1> &src1, tensor<int8> &shift,
         rounding_mode_t round_mode, bool saturation);
template <typename DataType0, typename DataType1, typename DataType2>
void add(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 C,
         tensor<int8> &shift, rounding_mode_t round_mode, bool saturation);

template <typename DataType>
void vc_min(tensor<DataType> &dst, tensor<DataType> &src0,
            tensor<DataType> &src1);
template <typename DataType>
void fvc_min(tensor<DataType> &dst, tensor<DataType> &src0,
             tensor<DataType> &src1);

template <typename DataType>
void vc_max(tensor<DataType> &dst, tensor<DataType> &src0,
            tensor<DataType> &src1);
template <typename DataType>
void fvc_max(tensor<DataType> &dst, tensor<DataType> &src0,
             tensor<DataType> &src1);

template <typename DataType0, typename DataType1, typename DataType2>
void gt(tensor<DataType0> &dst, tensor<DataType1> &src0,
        tensor<DataType1> &src1, DataType2 true_val);
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void gt(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 C,
        DataType3 true_val);

template <typename DataType0, typename DataType1, typename DataType2>
void lt(tensor<DataType0> &dst, tensor<DataType1> &src0,
        tensor<DataType1> &src1, DataType2 true_val);
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void lt(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 C,
        DataType3 true_val);

template <typename DataType0, typename DataType1, typename DataType2>
void eq(tensor<DataType0> &dst, tensor<DataType1> &src0,
        tensor<DataType1> &src1, DataType2 true_val);
template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3>
void eq(tensor<DataType0> &dst, tensor<DataType1> &src, DataType2 C,
        DataType3 true_val);

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4>
void gt_select(tensor<DataType0> &dst, DataType1 &src0, DataType2 &src1,
               DataType3 &src2, DataType4 &src3);

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4>
void lt_select(tensor<DataType0> &dst, DataType1 &src0, DataType2 &src1,
               DataType3 &src2, DataType4 &src3);

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4>
void eq_select(tensor<DataType0> &dst, DataType1 &src0, DataType2 &src1,
               DataType3 &src2, DataType4 &src3);

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4, typename DataType5>
void max_gt_select(tensor<DataType0> &dst0, tensor<DataType1> &dst1,
                   DataType2 &src0, DataType3 &src1, DataType4 &src2,
                   DataType5 &src3);

template <typename DataType0, typename DataType1, typename DataType2,
          typename DataType3, typename DataType4, typename DataType5>
void min_lt_select(tensor<DataType0> &dst0, tensor<DataType1> &dst1,
                   DataType2 &src0, DataType3 &src1, DataType4 &src2,
                   DataType5 &src3);

/*
 * Note:
 * 1. l_shift ∈ [0, 31], r_shift ∈ [0, 31]
 * 2. left and right data type must be same
 * 3. if left and right data type is DT_INT8, dst data type must be DT_INT16
 *    same as                        DT_UINT8                       DT_UINT16
 */
template <typename DataType0, typename DataType1, typename DataType2>
void mac(tensor<DataType0> &dst, tensor<DataType1> &src0,
         tensor<DataType2> &src1, int l_shift, int r_shift,
         rounding_mode_t round_mode);
template <typename DataType0, typename DataType1, typename DataType2>
void mac(tensor<DataType0> &dst, tensor<DataType1> &src0,
         tensor<DataType2> &src1, rounding_mode_t round_mode) {
  mac(dst, src0, src1, 0, 0, round_mode);
}
template <typename DataType0, typename DataType1, typename DataType2>
void mac(tensor<DataType0> &dst, tensor<DataType1> &src0,
         tensor<DataType2> &src1) {
  mac(dst, src0, src1, 0, 0, RM_HALF_TO_EVEN);
}

template <typename DataType0, typename DataType1>
void mac(tensor<DataType0> &dst, tensor<DataType1> &src, int C, int l_shift,
         int r_shift, rounding_mode_t round_mode);
template <typename DataType0, typename DataType1>
void mac(tensor<DataType0> &dst, tensor<DataType1> &src, int C,
         rounding_mode_t round_mode) {
  mac(dst, src, C, 0, 0, round_mode);
}
template <typename DataType0, typename DataType1>
void mac(tensor<DataType0> &dst, tensor<DataType1> &src, int C) {
  mac(dst, src, C, 0, 0, RM_HALF_TO_EVEN);
}

template <typename DataType0, typename DataType1>
void mac(tensor<DataType0> &dst, int C, tensor<DataType1> &src, int l_shift,
         int r_shift, rounding_mode_t round_mode);
template <typename DataType0, typename DataType1>
void mac(tensor<DataType0> &dst, int C, tensor<DataType1> &src,
         rounding_mode_t round_mode) {
  mac(dst, C, src, 0, 0, round_mode);
}
template <typename DataType0, typename DataType1>
void mac(tensor<DataType0> &dst, int C, tensor<DataType1> &src) {
  mac(dst, C, src, 0, 0, RM_HALF_TO_EVEN);
}

template <typename DataType>
void round(tensor<DataType> &dst, tensor<DataType> &src,
           rounding_mode_t round_mode);

template <typename DataType>
void floor(tensor<DataType> &dst, tensor<DataType> &src) {
  return round(dst, src, RM_DOWN);
}

template <typename DataType>
void ceiling(tensor<DataType> &dst, tensor<DataType> &src) {
  return round(dst, src, RM_UP);
}

template <typename DataType>
void pool_max(tensor<DataType> &dst, tensor<DataType> &src, dim2 *kernel,
              padding_t *pad, dim2 *stride, dim2 *dilation);
template <typename DataType>
void fpool_max(tensor<DataType> &dst, tensor<DataType> &src, dim2 *kernel,
               padding_t *pad, dim2 *stride, dim2 *dilation);

template <typename DataType>
void pool_min(tensor<DataType> &dst, tensor<DataType> &, dim2 *kernel,
              padding_t *pad, dim2 *stride, dim2 *dilation);
template <typename DataType>
void fpool_min(tensor<DataType> &dst, tensor<DataType> &, dim2 *kernel,
               padding_t *pad, dim2 *stride, dim2 *dilation);

template <typename DataType>
void fpool_avg(tensor<DataType> &dst, tensor<DataType> &src, dim2 *kernel,
               padding_t *pad, dim2 *stride, dim2 *dilation, dim2 *ins,
               float scale);
template <typename DataType>
void fpool_avg(tensor<DataType> &dst, tensor<DataType> &src, dim2 *kernel,
               padding_t *pad, dim2 *stride, dim2 *dilation, float scale) {
  return fpool_avg(dst, src, kernel, pad, stride, dilation, (dim2 *)nullptr,
                   scale);
}
template <typename DataType>
void fpool_avg(tensor<DataType> &dst, tensor<DataType> &src, dim2 *kernel,
               padding_t *pad, dim2 *stride, dim2 *dilation, dim2 *ins) {
  return fpool_avg(dst, src, kernel, pad, stride, dilation, ins, 1.0f);
}
template <typename DataType>
void fpool_avg(tensor<DataType> &dst, tensor<DataType> &src, dim2 *kernel,
               padding_t *pad, dim2 *stride, dim2 *dilation) {
  return fpool_avg(dst, src, kernel, pad, stride, dilation, (dim2 *)nullptr,
                   1.0f);
}

template <typename DataType0, typename DataType1>
void pool_avg(tensor<DataType0> &dst, tensor<DataType1> &src, dim2 *kernel,
              padding_t *pad, dim2 *stride, dim2 *dilation, dim2 *ins,
              int scale, int rshift);
template <typename DataType0, typename DataType1>
void pool_avg(tensor<DataType0> &dst, tensor<DataType1> &src, dim2 *kernel,
              padding_t *pad, dim2 *stride, dim2 *dilation, int scale,
              int rshift) {
  return pool_avg(dst, src, kernel, pad, stride, dilation, (dim2 *)nullptr,
                  scale, rshift);
}
template <typename DataType0, typename DataType1>
void pool_avg(tensor<DataType0> &dst, tensor<DataType1> &src, dim2 *kernel,
              padding_t *pad, dim2 *stride, dim2 *dilation, int scale) {
  return pool_avg(dst, src, kernel, pad, stride, dilation, (dim2 *)nullptr,
                  scale, 0);
}

template <typename DataType>
void move(tensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType0, typename DataType1>
void fill(tensor<DataType0> &dst, DataType1 scalar);

template <typename DataType> void zero(tensor<DataType> &data) {
  fill(data, 0);
}

template <typename DataType0, typename DataType1, typename DataType2>
void shift(tensor<DataType0> &dst, tensor<DataType1> &src,
           tensor<DataType2> &shift, rounding_mode_t round_mode);
template <typename DataType0, typename DataType1>
void shift(tensor<DataType0> &dst, tensor<DataType1> &src, char shift_c,
           rounding_mode_t round_mode);

template <typename DataType0, typename DataType1, typename DataType2>
void logical_shift(tensor<DataType0> &dst, tensor<DataType1> &src,
                   tensor<DataType2> &shift, rounding_mode_t round_mode);
template <typename DataType0, typename DataType1>
void logical_shift(tensor<DataType0> &dst, tensor<DataType1> &src, char shift_c,
                   rounding_mode_t round_mode);

template <typename DataType0, typename DataType1>
void circular_shift(tensor<DataType0> &dst, tensor<DataType1> &src,
                    tensor<char> &shift, rounding_mode_t round_mode);
template <typename DataType0, typename DataType1>
void circular_shift(tensor<DataType0> &dst, tensor<DataType1> &src,
                    char shift_c, rounding_mode_t round_mode);

template <typename d> d scalar_cast(float scalar, int mode);

template <typename DataType>
void broadcast(tensor<DataType> &dst, tensor<DataType> &src,
               int num = LANE_NUM);

template <typename DataType>
void smem_bcast(tensor<DataType> &dst, coeff_table_mode_t static_mem_mode);

template <typename DataType>
void smem_dist(tensor<DataType> &dst, coeff_table_mode_t static_mem_mode);

// template <typename DataType>
// void npu_bcast_from_static(tensor<DataType> &dst, int npu_num, int len,
//                            coeff_table_mode_t mode);

// template <typename DataType>
// void load_fp_exp_coeff(tensor<DataType> &coeff, int len) {
//   npu_bcast_from_static(coeff, NPU_NUM, len, EXP);
// }

template <typename DataType>
void fsin_base(tensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType>
void fcos_base(tensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType>
void farcsin_base(tensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType>
void farccos_base(tensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType>
void ftan_base(tensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType>
void fcot_base(tensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType>
void fexp(tensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType>
void flog_base(tensor<DataType> &dst, tensor<DataType> &src);

template <typename DataType>
void taylor(tensor<DataType> &dst, tensor<DataType> &src,
            tensor<DataType> &coeff_addr, int len);
// int fp_get_coeff_len(coeff_table_mode_t mode);

template <typename DataType>
void fp_load_coeff(tensor<DataType> &coeff, coeff_table_mode_t mode);

template <typename DataType0, typename DataType1>
void fexp_part(tensor<DataType0> &dst, tensor<DataType1> &src);

template <typename DataType>
void fsqrt(tensor<DataType> &dst, tensor<DataType> &src, int num_iter = 3);

template <typename DataType>
void frsqrt(tensor<DataType> &dst, tensor<DataType> &src, int num_iter = 3);

// template <typename DataType>
// void flogx(tensor<DataType> &dst, tensor<DataType> &src,
//            tensor<DataType> &work0, tensor<DataType> &coeff, float x);

template <typename DataType0, typename DataType1>
void gather_hw(tensor<DataType0> &dst, tensor<DataType0> &param,
               tensor<DataType1> &index);
template <typename DataType0, typename DataType1, typename DataType2>
void gather_hw(tensor<DataType0> &dst, tensor<DataType0> &param,
               tensor<DataType1> &index, DataType2 C, bool fill_const);

template <typename DataType0, typename DataType1>
void gather_w(tensor<DataType0> &dst, tensor<DataType0> &param,
              tensor<DataType1> &index);
template <typename DataType0, typename DataType1>
void gather_w(tensor<DataType0> &dst, tensor<DataType0> &param,
              tensor<DataType1> &index, bool is_param_repeated);
template <typename DataType0, typename DataType1, typename DataType2>
void gather_w(tensor<DataType0> &dst, tensor<DataType0> &param,
              tensor<DataType1> &index, DataType2 C, bool fill_const);
template <typename DataType0, typename DataType1, typename DataType2>
void gather_w(tensor<DataType0> &dst, tensor<DataType0> &param,
              tensor<DataType1> &index, DataType2 C, bool is_param_repeated,
              bool fill_const);

template <typename DataType0, typename DataType1>
void gather_h(tensor<DataType0> &dst, tensor<DataType0> &param,
              tensor<DataType1> &index, bool is_param_repeated);
template <typename DataType0, typename DataType1, typename DataType2>
void gather_h(tensor<DataType0> &dst, tensor<DataType0> &param,
              tensor<DataType1> &index, DataType2 C, bool is_param_repeated,
              bool fill_const);

template <typename DataType0, typename DataType1>
void scatter_hw(tensor<DataType0> &dst, tensor<DataType0> &param,
                tensor<DataType1> &index);

template <typename DataType0, typename DataType1>
void scatter_w(tensor<DataType0> &dst, tensor<DataType0> &param,
               tensor<DataType1> &index);
template <typename DataType0, typename DataType1>
void scatter_w(tensor<DataType0> &dst, tensor<DataType0> &param,
               tensor<DataType1> &index, bool is_param_repeated);
template <typename DataType0, typename DataType1>
void scatter_h(tensor<DataType0> &dst, tensor<DataType0> &param,
               tensor<DataType1> &index, bool is_param_repeated);

template <typename DataType>
void nonzero(tensor<DataType> &dst_idx, tensor<uint16> &dst_cnt,
             tensor<DataType> &src);

template <typename DataType0, typename DataType1>
void norm(tensor<DataType0> &dst, tensor<DataType1> &src);

template <typename DataType0, typename DataType1>
void clz(tensor<DataType0> &dst, tensor<DataType1> &src);

template <typename DataType0, typename DataType1>
void clo(tensor<DataType0> &dst, tensor<DataType1> &src);

template <typename DataType0, typename DataType1>
void mask_select(tensor<DataType0> &dst, tensor<uint16> &dst_cnt,
                 tensor<DataType0> &src, tensor<DataType1> &mask);

} // namespace tiu
} // namespace ppl
