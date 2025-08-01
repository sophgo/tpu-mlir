#ifndef CONV_UTIL_H
#define CONV_UTIL_H
#include "sg_fp16.h"

#include "bd_reg_value.h"
#if  defined (__sg2380__) || defined(__mars3__) || defined(__sgtpuv8__)
#include <typeinfo>
#endif
#ifdef __cplusplus
#include "similarity.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#define IS_NAN(x) ((((x >> 23) & 0xff) == 0xff) && ((x & 0x7fffff) != 0))

#ifndef BIAS_PSUM_INT_BITS
#define BIAS_PSUM_INT_BITS    32
#define BIAS_PSUM_FLOAT_BITS  32
#define CONV_8B_B_P           BIAS_PSUM_INT_BITS
#define CONV_BF16_B_P         BIAS_PSUM_FLOAT_BITS
#endif

static inline float ConvFloatCut20(float src, int bit_num) {
  if (bit_num == 32) return src;
  *((uint32_t*)&src) &= 0xFFFFF000;
  return src;
}

typedef enum {
  ORIGINAL = 0,
  MID_FP20 = 1,
  MID_INT26 = 2
} MID_PREC;

#ifdef __arm__
typedef unsigned int uint;
#endif

template <typename T>
static inline T ConvIntCut26(T src, int bit_num)  {
  if (bit_num == 32) return src;
  int src_ = (int)src;
  int low_bits = 0;
  if(src_ < 0) low_bits = bit_num - 1;
  else low_bits = bit_num;
  int base_one = 0x0;
  base_one = ((uint)1<<low_bits)-1;
  if(src_ < 0) {
    base_one |= 0x80000000;
    src_ -= 1;
    src_ = src_^(((uint)1<<31)-1);
    int res = src_ & base_one;
    T res_b = (res^(((uint)1<<31)-1)) + 1;
    return res_b;
  }
  T res = src_ & base_one;
  return res;
}

static void to_int(void* from, int* to, int sz, PREC prec, bool sign) {
  for (int i = 0; i < sz; i++) {
    if (prec == INT8 && sign)
      to[i] = (int)(reinterpret_cast<s8 *>(from)[i]);
    else if (prec == INT8 && !sign)
      to[i] = (int)(reinterpret_cast<u8 *>(from)[i]);
    else if (prec == INT16 && sign)
      to[i] = (int)(reinterpret_cast<s16 *>(from)[i]);
    else if (prec == INT16 && !sign)
      to[i] = (int)(reinterpret_cast<u16 *>(from)[i]);
    else if (prec == INT32 && sign)
      to[i] = (int)(reinterpret_cast<s32 *>(from)[i]);
    else if (prec == INT32 && !sign)
      to[i] = (int)(reinterpret_cast<u32 *>(from)[i]);
    else if (prec == INT4 && sign)
      to[i] = (int)(reinterpret_cast<int4_t *>(from)[i].val);
    else if (prec == INT4 && !sign)
      to[i] = (int)(reinterpret_cast<uint4_t *>(from)[i].val);
    else
        ASSERT(0 && "not support prec");
  }
}

static void to_int(void* from, int* to, int sz, sg_data_type_t dtype) {
  for (int i = 0; i < sz; i++) {
    if (dtype == SG_DTYPE_INT8)
      to[i] = (int)(reinterpret_cast<s8 *>(from)[i]);
    else if (dtype == SG_DTYPE_UINT8)
      to[i] = (int)(reinterpret_cast<u8 *>(from)[i]);
    else if (dtype == SG_DTYPE_INT16)
      to[i] = (int)(reinterpret_cast<s16 *>(from)[i]);
    else if (dtype == SG_DTYPE_UINT16)
      to[i] = (int)(reinterpret_cast<u16 *>(from)[i]);
    else if (dtype == SG_DTYPE_INT32)
      to[i] = (int)(reinterpret_cast<s32 *>(from)[i]);
    else if (dtype == SG_DTYPE_UINT32)
      to[i] = (int)(reinterpret_cast<u32 *>(from)[i]);
    else if (dtype == SG_DTYPE_INT4)
      to[i] = (int)(reinterpret_cast<int4_t *>(from)[i].val);
    else if (dtype == SG_DTYPE_UINT4)
      to[i] = (int)(reinterpret_cast<uint4_t *>(from)[i].val);
    else
        ASSERT(0 && "not support prec");
  }
}

// (oc, ic, kh, kw) -> (oc, ceil(ic/N), kh*kw, N)
template <typename T>
void convert_to_nIC(const T *weight, T *weight_nIC, int ic, int oc, int kh, int kw, int N) {
    for (int i = 0; i < oc; i++) {
        for (int j = 0; j < ceiling_func(ic, N); j++) {
            for (int k = 0; k < kh * kw; k++) {
                int inner = sg_min(ic - N * j, N);
                for (int l = 0; l < inner; l++) {
                    weight_nIC[i * ceiling_func(ic, N) * kh * kw * N +
                                j * kh * kw * N +
                                k * N +
                                l] =
                         weight[i * ic * kh * kw +
                                j * kh * kw * N +
                                l * kh * kw +
                                k];
                }
            }
        }
    }
}

inline static int calc_offset(const int *shape, const int *offset) {
    return ((offset[0] * shape[1] + offset[1]) * shape[2] + offset[2]) * shape[3] + offset[3];
}

inline static int mul(int a, int b) {
    return a * b;
}
inline static float mul(float a, float b) {
    return a * b;
}
inline static float mul(fp16 a, fp16 b) {
    return fp16_to_fp32(a).fval * fp16_to_fp32(b).fval ;
}
inline static float mul(bf16 a, bf16 b) {
    return bf16_to_fp32(a).fval * bf16_to_fp32(b).fval ;
}
inline static float mul(fp8e5m2 a, fp8e4m3 b) {
    return fp8_to_fp32(a.bits, true).fval * fp8_to_fp32(b.bits, false).fval;
}
inline static float mul(fp8e4m3 a, fp8e5m2 b) {
    return fp8_to_fp32(a.bits, false).fval * fp8_to_fp32(b.bits, true).fval;
}
inline static float mul(fp8e5m2 a, fp8e5m2 b) {
    return fp8_to_fp32(a.bits, true).fval * fp8_to_fp32(b.bits, true).fval;
}
inline static float mul(fp8e4m3 a, fp8e4m3 b) {
    return fp8_to_fp32(a.bits, false).fval * fp8_to_fp32(b.bits, false).fval;
}

extern "C" {
static int fp32_similarity(float* exp_f32, float* got_f32, int len, float threshold) {
  float cos_similarity = cos_dist(exp_f32, got_f32, len);
  int ret = -1;
  if (cos_similarity >= threshold) {
    ret = 0;
    printf("compare cos similarity success, cos_similarity:%f, threshold:%f \n", cos_similarity, threshold);
  } else {
     printf("compare cos similarity faild, cos_similarity:%f, threshold:%f \n", cos_similarity, threshold);
  }
  return ret;
}

static int similarity(void* p_exp, void* p_got, int len, PREC prec, float threshold) {
  float* exp_f32 = new float[len];
  float* got_f32 = new float[len];

  for (int i = 0; i < len; i++) {
    if (prec == FP16) {
      exp_f32[i] = fp16_to_fp32(((fp16*)p_exp)[i]).fval;
      got_f32[i] = fp16_to_fp32(((fp16*)p_got)[i]).fval;
    } else if (prec == BFP16) {
      exp_f32[i] = bf16_to_fp32(((bf16*)p_exp)[i]).fval;
      got_f32[i] = bf16_to_fp32(((bf16*)p_got)[i]).fval;
    }
  }

  int ret = fp32_similarity(exp_f32, got_f32, len, threshold);

  delete [] exp_f32;
  delete [] got_f32;
  return ret;
}

inline int array_cmp_cos_similarity(PREC prec, void* p_exp, void* p_got, int len, float threshhold) {
  int ret = 0;
  if (prec == FP16) {
    ret = similarity(p_exp, p_got, len, prec, threshhold);
  } else if (prec == BFP16) {
    ret = similarity(p_exp, p_got, len, prec, threshhold);
  } else if (prec == FP32 || prec == TF32) {
    ret = fp32_similarity((float*)p_exp, (float*)p_got, len, threshhold);
  } else {
    ASSERT_INFO(0, "unsupport fp prec\n");
  }
  return ret;
}
}

template <typename T1, typename T2, typename T3, typename T4>
void conv_native_core(
        T2 *ofmap, const T1 *ifmap, const T4 *weight,
        const T3 *bias, const T1 *pad_ins,
        int input_n, int input_c, int input_h, int input_w,
        int output_c, int groups,
        int kh, int kw, int dh, int dw, int ins_h, int ins_w,
        int pht, int phb, int pwl, int pwr,
        int stride_h, int stride_w, int kernel_rotate,
        bool with_bias, bool result_add, bool with_relu,
        PAD_MODE pad_mode, bool nc_trans=false, bool with_rescale=false,
        float* rescale = nullptr, bool with_saturate=false, MID_PREC use_middle_prec = ORIGINAL,
        int output_padding_h = 0,int output_padding_w = 0) {

    // printf("show conv info --------------------------------------\n");
    // printf("input_n=%d, input_c=%d, input_h=%d, input_w=%d \n", input_n, input_c, input_h, input_w);
    // printf("output_c=%d, groups=%d \n", output_c, groups);
    // printf("kh=%d, kw=%d, dh=%d, dw=%d, ins_h=%d, ins_w=%d \n", kh, kw, dh, dw, ins_h, ins_w);
    // printf("pht=%d, phb=%d, pwl=%d, pwr=%d, stride_h=%d, stride_w=%d \n", pht, phb, pwl, pwr, stride_h, stride_w);
    // printf("show conv info --------------------------------------\n");

    int kh_ext = dh * (kh - 1) + 1;
    int kw_ext = dw * (kw - 1) + 1;
    int ih_ext = (input_h - 1) * (ins_h + 1) + pht + phb + 1;
    int iw_ext = (input_w - 1) * (ins_w + 1) + pwr + pwl + 1;
    int oh = (ih_ext - kh_ext) / stride_h + 1 + output_padding_h;// output_padding for convtranspose
    int ow = (iw_ext - kw_ext) / stride_w + 1 + output_padding_w;

    int i_shape[4];
    i_shape[0] = input_n;
    i_shape[1] = input_c;
    i_shape[2] = input_h;
    i_shape[3] = input_w;
    int o_shape[4];
    o_shape[0] = input_n;
    o_shape[1] = output_c;
    o_shape[2] = oh;
    o_shape[3] = ow;
    int k_shape[4];
    k_shape[0] = output_c;
    k_shape[1] = input_c / groups;
    k_shape[2] = kh;
    k_shape[3] = kw;

    int o_g = output_c / groups;
    int k_g = input_c / groups;
    int o_head, k_head;

    if (!result_add) {
        memset(ofmap, 0, input_n * output_c * oh * ow * sizeof(T2));
    } else {
      if (use_middle_prec == MID_FP20) {
        for (int i = 0; i < input_n * output_c * oh * ow; ++i) {
          ofmap[i] = ConvFloatCut20(ofmap[i], CONV_BF16_B_P);
        }
      } else if (use_middle_prec == MID_INT26) {
         for (int i = 0; i < input_n * output_c * oh * ow; ++i) {
          ofmap[i] = ConvIntCut26(ofmap[i], CONV_8B_B_P);
        }
      }
    }
    for (int n = 0; n < input_n; n++) {
        for (int g = 0; g < groups; g++) {
            for (int o = 0; o < o_g; o++) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, (oh * ow + omp_get_num_threads() - 1) / omp_get_num_threads()) collapse(2)
#endif
                for (int y = 0; y < oh; y++) {
                    for (int x = 0; x < ow; x++) {
                        o_head = o_g * g;
                        k_head = k_g * g;
                        int weight_offset[4];
                        int in_offset[4];
                        int out_offset[4];
                        int out_idx = 0;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        out_offset[2] = y;
                        out_offset[3] = x;
                        for (int k = 0; k < k_g; k++) {
                            for (int p = 0; p < kh; p++) {
                                for (int q = 0; q < kw; q++) {
                                    int ih_pos = y * stride_h + p * dh - pht;
                                    int iw_pos = x * stride_w + q * dw - pwl;
                                    bool pos_ins = ih_pos % (ins_h + 1) > 0 ||
                                                   iw_pos % (ins_w + 1) > 0;
                                    bool pos_pad = ih_pos < 0 || ih_pos >= (ih_ext - phb - pht) ||
                                                   iw_pos < 0 || iw_pos >= (iw_ext - pwl - pwr);
                                    in_offset[0] = n;
                                    in_offset[1] = k + k_head;
                                    in_offset[2] = ih_pos / (ins_h + 1);
                                    in_offset[3] = iw_pos / (ins_w + 1);
                                    int src_idx = calc_offset(i_shape, in_offset);
                                    T1 ival;
                                    memset(&ival, 0, sizeof(T1));
                                    if (ih_pos >= 0 && ih_pos / (ins_h + 1) < input_h &&
                                        iw_pos >= 0 && iw_pos / (ins_w + 1) < input_w)
                                        ival = ifmap[src_idx];
                                    if (pos_ins) {
                                        ival = pad_ins[nc_trans ? (n * 2 + 1) : ((k_head + k) * 2 + 1)];
                                    }
                                    if (pos_pad) {
                                        if (pad_mode == PAD_CONSTANT) {
                                            ival = pad_ins[nc_trans ? n * 2 : (k_head + k) * 2];
                                        } else if (pad_mode == PAD_REFLECTION) {
                                            ih_pos = ih_pos < 0 ? -ih_pos : ih_pos;
                                            ih_pos = ih_pos >= (ih_ext - phb - pht) ?
                                                     2 * (ih_ext - pht - phb - 1) - ih_pos : ih_pos;
                                            iw_pos = iw_pos < 0 ? -iw_pos : iw_pos;
                                            iw_pos = iw_pos >= (iw_ext - pwl - pwr) ?
                                                     2 * (iw_ext - pwl - pwr - 1) - iw_pos : iw_pos;
                                            in_offset[2] = ih_pos / (ins_h + 1);
                                            in_offset[3] = iw_pos / (ins_w + 1);
                                            src_idx = calc_offset(i_shape, in_offset);
                                            ival = (ih_pos % (ins_h + 1) || iw_pos % (ins_w + 1)) ?
                                                   pad_ins[nc_trans ? (n * 2 + 1) : ((k_head + k) * 2 + 1)] : ifmap[src_idx];
                                        } else if (pad_mode == PAD_REPLICATION) {
                                            ih_pos = ih_pos < 0 ? 0 : ih_pos;
                                            ih_pos = ih_pos >= (ih_ext - phb - pht) ?
                                                     ih_ext - phb - pht - 1 : ih_pos;
                                            iw_pos = iw_pos < 0 ? 0 : iw_pos;
                                            iw_pos = iw_pos >= (iw_ext - pwl - pwr) ?
                                                     iw_ext - pwl - pwr - 1 : iw_pos;
                                            in_offset[2] = ih_pos / (ins_h + 1);
                                            in_offset[3] = iw_pos / (ins_w + 1);
                                            src_idx = calc_offset(i_shape, in_offset);
                                            ival = (ih_pos % (ins_h + 1) || iw_pos % (ins_w + 1)) ?
                                                   pad_ins[nc_trans ? (n * 2 + 1) : ((k_head + k) * 2 + 1)] : ifmap[src_idx];
                                        } else if (pad_mode == PAD_CIRCULAR) {
                                            ih_pos = ih_pos < 0 ?
                                                     ih_ext - pht - phb + ih_pos : ih_pos;
                                            ih_pos = ih_pos >= (ih_ext - phb - pht) ?
                                                     ih_pos - (ih_ext - pht - phb) : ih_pos;
                                            iw_pos = iw_pos < 0 ?
                                                     iw_ext - pwl - pwr + iw_pos : iw_pos;
                                            iw_pos = iw_pos >= (iw_ext - pwl - pwr) ?
                                                     iw_pos - (iw_ext - pwl - pwr) : iw_pos;
                                            in_offset[2] = ih_pos / (ins_h + 1);
                                            in_offset[3] = iw_pos / (ins_w + 1);
                                            src_idx = calc_offset(i_shape, in_offset);
                                            ival = (ih_pos % (ins_h + 1) || iw_pos % (ins_w + 1)) ?
                                                   pad_ins[nc_trans ? (n * 2 + 1) : (k_head + k) * 2 + 1] : ifmap[src_idx];
                                        } else {
                                            ASSERT(0);
                                        }
                                    }
                                    weight_offset[0] = o + o_head;
                                    weight_offset[1] = k;
                                    if (kernel_rotate) {
                                        weight_offset[2] = (kh - 1 - p);
                                        weight_offset[3] = (kw - 1 - q);
                                    } else {
                                        weight_offset[2] = p;
                                        weight_offset[3] = q;
                                    }
                                    out_idx = calc_offset(o_shape, out_offset);
                                    int widx = calc_offset(k_shape, weight_offset);
                                    if (use_middle_prec == MID_FP20) {
                                      ofmap[out_idx] = ConvFloatCut20(ofmap[out_idx] + ConvFloatCut20(mul(ival, weight[widx]), CONV_BF16_B_P), CONV_BF16_B_P);
                                    } else if (use_middle_prec == MID_INT26) {
                                      ofmap[out_idx] = ConvIntCut26(ofmap[out_idx] + ConvIntCut26(mul(ival, weight[widx]), CONV_8B_B_P), CONV_8B_B_P);
                                    } else {
                                      ofmap[out_idx] = ofmap[out_idx] + mul(ival, weight[widx]);
                                    }
//#define  DEBUG_CONV
#ifdef DEBUG_CONV
                                    if(out_idx == 722656) {
                                      printf(" %x , %x \n", *(unsigned char*)&ival,  *((unsigned char*)weight +widx));
                                      printf("  %f \n", *((float*)ofmap +out_idx));
                                    }
#endif
                                }
                            }
                        }
                        if (with_bias) {
                            if (use_middle_prec == MID_FP20) {
                              ofmap[out_idx] += ConvFloatCut20(bias[g * o_g + o], CONV_BF16_B_P);
                              ofmap[out_idx] = ConvFloatCut20(ofmap[out_idx], CONV_BF16_B_P);
                            } else if (use_middle_prec == MID_INT26) {
                              ofmap[out_idx] += ConvIntCut26(bias[g * o_g + o], CONV_8B_B_P);
                              ofmap[out_idx] = ConvIntCut26(ofmap[out_idx], CONV_8B_B_P);
                            } else {
                              ofmap[out_idx] += bias[g * o_g + o];
                            }
                        }
                        if (with_relu) {
                            ofmap[out_idx] =
                                sg_max(ofmap[out_idx], 0);
                        }
                        if (with_rescale) {
                            ofmap[out_idx] *= rescale[g * o_g + o];
                        }
#if  defined (__sg2380__) || defined(__mars3__) || defined(__sgtpuv8__) || defined(__sg2262__)
                        else if (with_saturate) {
                            if(typeid(T1) == typeid(fp16)) {
                                ASSERT(typeid(T2) == typeid(float));
                                fp32 out_f32 = {.fval = (float)ofmap[out_idx]};
                                fp16 out_f16 = fp32_to_fp16(out_f32, ROUND_HALF_TO_EVEN, with_saturate);
                                ofmap[out_idx] = fp16_to_fp32(out_f16).fval;
                            } else if(typeid(T1) == typeid(bf16)) {
                                ASSERT(typeid(T2) == typeid(float));
                                fp32 out_f32 = {.fval = (float)ofmap[out_idx]};
                                bf16 out_f16 = fp32_to_bf16(out_f32, ROUND_HALF_TO_EVEN, with_saturate);
                                ofmap[calc_offset(o_shape, out_offset)] = bf16_to_fp32(out_f16).fval;
                            }
                        }
#endif
                    }
                }
            }
        }
    }
}

#endif //end def c++

static void cal_kernel_align_shape(int *shape, PREC prec) {
  int oc = shape[0], ic = shape[1], kh = shape[2], kw = shape[3];
  shape[0] = 1;
  shape[1] = oc;
  if (prec == TF32) {
#if defined (__sg2262__)
    shape[2] = ceiling_func(ic, 8) * kh * kw;
    shape[3] = 8;
#else
    shape[2] = ceiling_func(ic, 16) * kh * kw;
    shape[3] = 16;
#endif
  } else if (prec == FP16 || prec == BFP16) {
    shape[2] = ceiling_func(ic, 32) * kh * kw;
    shape[3] = 32;
  } else if (prec == FP8) {
#if defined (__sg2262__)
    shape[2] = ceiling_func(ic, 32) * kh * kw;
    shape[3] = 32;
#else
    shape[2] = ceiling_func(ic, 64) * kh * kw;
    shape[3] = 64;
#endif
  } else if (prec == FP32) {
    shape[2] = ic * kh * kw;
    shape[3] = 1;
  } else {
    ASSERT_INFO(0, "dtype not implement");
  }
}

static int cal_conv2d_input_ext(int in, int insert, int pad_0, int pad_1) {
  return ((in - 1) * (insert + 1) + pad_0 + pad_1 + 1);
}

static int cal_conv2d_kernel_ext(int kernel, int dilation) {
  return (dilation * (kernel - 1) + 1);
}

static int cal_conv2d_out_size(int in, int kernel, int dilation, int stride, int pad_0, int pad_1, int insert) {
  int kernel_ext = cal_conv2d_kernel_ext(kernel, dilation);
  int in_ext = cal_conv2d_input_ext(in, insert, pad_0, pad_1);
  return (in_ext - kernel_ext) / stride + 1;
}

static u64 conv_bw_ops(
    int n,
    int ic,
    int oc,
    int oh,
    int ow,
    int kh,
    int kw)
{
  u64 total = (u64)oc * kh * kw * ic *
              (oh * ow * n * 2 - 1);
  return total;
}

#endif
