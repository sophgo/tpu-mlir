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

} // namespace tiu

namespace hau {}
} // namespace ppl
