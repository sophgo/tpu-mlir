#include "atomic_tiu.h"

void atomic_conv_quant_check(
    u32 input_addr,
    u32 weight_addr, // or weight const value
    u32 bias_addr, // or bias const value
    u32 pad_ins_addr, // pad const value
    u32 kzp_addr, // kzp const value
    u32 requant_addr, // multipiler const value
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_c,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dilation_h,
    int dilation_w,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int kernel_is_const,
    int bias_is_const,
    int pad_ins_is_const,
    int kzp_is_const,
    int kernel_rotate,
    int result_add,
    u32 ins_const_val,
    int input_sign,
    int weight_sign,
    int bias_sign,
    int res_sign,
    int *input_stride,
    int do_relu,
    int sym_saturate,
    int do_requant,
    int requant_is_const,
    int shift_num, // s8
    int ozp, // s16
    ROUND_MODE rm_mode,
    PREC input_prec,
    PREC weight_prec,
    PREC output_prec,
    PAD_MODE pad_mode) {

#ifdef USING_CMODEL
#ifdef __sg2262__
    ASSERT_FS_INFO(0, "Not support");
#else
    int kh_ext = dilation_h * (kh - 1) + 1;
    int kw_ext = dilation_w * (kw - 1) + 1;
    int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
    int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_r + pad_w_l + 1;
    int output_h = (ih_ext - kh_ext) / stride_h + 1;
    int output_w = (iw_ext - kw_ext) / stride_w + 1;
    int input_short_str = input_stride == NULL ? 0 : 3;
    u32 str[4] = {0};
    if (input_stride != NULL) {
        memcpy(str, input_stride, 4 * sizeof(int));
        str[0] = input_n == 1 ? 0 : str[0];
        str[1] = input_c <= NPU_NUM ? 0 : str[1];
    }
    if (bias_is_const && bias_addr == 0) {
        bias_sign = 0;
    }
    ASSERT(input_addr < LOCAL_MEM_SIZE);
    ASSERT(pad_ins_is_const || (pad_ins_addr < LOCAL_MEM_SIZE));
    ASSERT(output_addr < LOCAL_MEM_SIZE * NPU_NUM);
    ASSERT(bias_is_const || get_npu_index(bias_addr) == get_npu_index(output_addr));
    ASSERT(kernel_is_const || (get_npu_index(weight_addr) == get_npu_index(output_addr)));
    ASSERT(kzp_is_const || get_npu_index(kzp_addr) == get_npu_index(output_addr));
    ASSERT(!do_requant || requant_is_const || get_npu_index(requant_addr) == get_npu_index(output_addr));
    ASSERT(input_stride || input_addr % ALIGN_BYTES == 0);
    ASSERT(kernel_is_const || weight_addr % ALIGN_BYTES == 0);
    ASSERT(bias_is_const || bias_addr % sizeof(int) == 0);
    ASSERT(pad_ins_is_const || pad_ins_addr % (sizeof(char) * 2) == 0);
    ASSERT(kzp_is_const || kzp_addr % sizeof(short) == 0);
    ASSERT(!do_requant || requant_is_const || requant_addr % (sizeof(int) * 2) == 0);
    ASSERT(output_addr % ALIGN_BYTES == 0);
    ASSERT(input_prec == INT8);
    ASSERT(weight_prec == INT8);
    ASSERT(is_fixed_prec(output_prec) && output_prec != INT4);
    ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
    ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
    ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
    ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
    ASSERT(ih_ext < (((int)1) << 16) && (ih_ext > 0));
    ASSERT(iw_ext < (((int)1) << 16) && (iw_ext > 0));
    ASSERT(output_c < (((int)1) << 16) && (output_c > 0));
    ASSERT(output_h > 0 && output_h < (((int)1) << 16));
    ASSERT(output_w > 0 && output_w < (((int)1) << 16));
    ASSERT(stride_h > 0 && stride_h < 16);
    ASSERT(stride_w > 0 && stride_w < 16);
    ASSERT(pad_h_t >= 0 && pad_h_t < 16);
    ASSERT(pad_h_b >= 0 && pad_h_b < 16);
    ASSERT(pad_w_r >= 0 && pad_w_r < 16);
    ASSERT(pad_w_l >= 0 && pad_w_l < 16);
    ASSERT(dilation_h > 0 && dilation_h < 16);
    ASSERT(dilation_w > 0 && dilation_w < 16);
    ASSERT(ins_h >= 0 && ins_h < 15);
    ASSERT(ins_w >= 0 && ins_w < 15);
    ASSERT(kh > 0 && kh < 65536 && kw > 0 && kw < 65536);
    ASSERT(kernel_is_const >= 0 && kernel_is_const < 2);
    ASSERT(bias_is_const >= 0 && bias_is_const < 2);
    ASSERT(pad_ins_is_const >= 0 && pad_ins_is_const < 2);
    ASSERT(kzp_is_const >= 0 && kzp_is_const < 2);
    ASSERT(kernel_rotate >= 0 && kernel_rotate < 2);
    ASSERT(input_sign >= 0 && input_sign < 2);
    ASSERT(weight_sign >= 0 && weight_sign < 2);
    ASSERT(bias_sign >= 0 && bias_is_const < 2);
    ASSERT(res_sign >= 0 && res_sign < 2);
    ASSERT(do_relu >= 0 && do_relu < 2);
    ASSERT(sym_saturate >= 0 && sym_saturate < 2);
    ASSERT(do_requant >= 0 && do_requant < 2);
    ASSERT((requant_is_const >= 0 && requant_is_const < 2));
    ASSERT(shift_num >= -128 && shift_num < 128);
    ASSERT(ozp >= -32768 && ozp < 32768);
    ASSERT(str[0] < (1 << 16) && str[1] < (1 << 16) && str[2] < (1 << 16) && str[3] < (1 << 16));
    ASSERT(input_prec == INT8);
#endif
#endif
}

void atomic_conv_check(
    u32 input_addr,
    u32 weight_addr,
    u32 bias_addr,
    u32 pad_ins_addr,
    u32 rescale_addr,
    u32 output_addr,
    int input_n,
    int input_c,
    int input_h,
    int input_w,
    int output_c,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int ins_h,
    int ins_w,
    int dilation_h,
    int dilation_w,
    int pad_h_t,
    int pad_h_b,
    int pad_w_l,
    int pad_w_r,
    int kernel_is_const,
    int bias_is_const,
    int pad_ins_is_const,
    int kernel_rotate,
    int result_add,
    u32 ins_const_val,
    int *input_stride,
    int do_relu,
    int saturate,
    PREC input_prec,
    PREC output_prec,
    PREC bias_prec,
    int input_sign,
    int weight_sign,
    int res_sign,
    int bias_sign,
    int do_rescale,
    int rescale_is_const,
    PAD_MODE pad_mode) {

#ifdef USING_CMODEL
    int kh_ext = dilation_h * (kh - 1) + 1;
    int kw_ext = dilation_w * (kw - 1) + 1;
    int ih_ext = (input_h - 1) * (ins_h + 1) + pad_h_t + pad_h_b + 1;
    int iw_ext = (input_w - 1) * (ins_w + 1) + pad_w_r + pad_w_l + 1;
    int output_h = (ih_ext - kh_ext) / stride_h + 1;
    int output_w = (iw_ext - kw_ext) / stride_w + 1;

    ASSERT(input_addr < LOCAL_MEM_SIZE * (input_prec == FP32 ? NPU_NUM : 1));
  #ifdef __sg2262__
    ASSERT(pad_ins_is_const == 1);
  #else
    ASSERT(pad_ins_is_const ||get_npu_index(pad_ins_addr) == get_npu_index(input_addr));
    ASSERT(pad_ins_is_const || pad_ins_addr % (get_bytesize(input_prec) * 2) == 0);
  #endif
    ASSERT(output_addr < LOCAL_MEM_SIZE * NPU_NUM);
    ASSERT(kernel_is_const || get_npu_index(weight_addr) == get_npu_index(output_addr));
    ASSERT(bias_is_const || get_npu_index(bias_addr) == get_npu_index(output_addr));
    ASSERT((input_stride && (input_addr % get_bytesize(input_prec) == 0)) || (input_addr % ALIGN_BYTES == 0));
    ASSERT((input_prec == FP16 && (output_prec == FP32 || output_prec == FP16)) ||
           (input_prec == BFP16 && (output_prec == FP32 || output_prec == BFP16)) ||
  #ifdef __sg2262__
           (input_prec == FP8 && (output_prec == FP32 || output_prec == FP16 || output_prec == FP8 || output_prec == BFP16)) ||
  #else
           (input_prec == FP8 && (output_prec == FP32 || output_prec == FP16 || output_prec == FP8) ) ||
  #endif
           (input_prec == FP32 && output_prec == FP32) ||
           (input_prec == TF32 && output_prec == FP32));
    ASSERT((input_prec == FP16 && (bias_prec == FP16 || bias_prec == FP32)) ||
           (input_prec == BFP16 && (bias_prec == BFP16 || bias_prec == FP32)) ||
  #ifdef __sg2262__
           (input_prec == FP8 && (bias_prec == FP16 || bias_prec == FP32 || bias_prec == BFP16)) ||
  #else
           (input_prec == FP8 && (bias_prec == FP16 || bias_prec == FP32)) ||
  #endif
           (input_prec == FP32 && bias_prec == FP32) ||
           (input_prec == TF32 && bias_prec == FP32));
    ASSERT(kernel_is_const || (weight_addr % (input_prec == FP32 ? sizeof(float) : ALIGN_BYTES) == 0));
    ASSERT(bias_is_const || (bias_addr % sizeof(float) == 0));
    ASSERT(output_addr % ALIGN_BYTES == 0);
    ASSERT(input_n < (((int)1) << 16) && (input_n > 0));
    ASSERT(input_c < (((int)1) << 16) && (input_c > 0));
    ASSERT(input_h < (((int)1) << 16) && (input_h > 0));
    ASSERT(input_w < (((int)1) << 16) && (input_w > 0));
    ASSERT(ih_ext < (((int)1) << 16) && (ih_ext > 0));
    ASSERT(iw_ext < (((int)1) << 16) && (iw_ext > 0));
    ASSERT(output_c < (((int)1) << 16) && (output_c > 0));
    ASSERT(output_h > 0 && output_h < (((int)1) << 16));
    ASSERT(output_w > 0 && output_w < (((int)1) << 16));
    ASSERT(stride_h > 0 && stride_h < 16);
    ASSERT(stride_w > 0 && stride_w < 16);
    ASSERT(pad_h_t >= 0 && pad_h_t < 16);
    ASSERT(pad_h_b >= 0 && pad_h_b < 16);
    ASSERT(pad_w_r >= 0 && pad_w_r < 16);
    ASSERT(pad_w_l >= 0 && pad_w_l < 16);
    ASSERT(dilation_h > 0 && dilation_h < 16);
    ASSERT(dilation_w > 0 && dilation_w < 16);
    ASSERT(ins_h >= 0 && ins_h < 15);
    ASSERT(ins_w >= 0 && ins_w < 15);
    ASSERT(kh > 0 && kh < 65536 && kw > 0 && kw < 65536);
    ASSERT(kernel_is_const >= 0 && kernel_is_const < 2);
    ASSERT(bias_is_const >= 0 && bias_is_const < 2);
  #ifdef __sg2262__
    ASSERT(pad_ins_is_const == 1);
  #else
    ASSERT(pad_ins_is_const >= 0 && pad_ins_is_const < 2);
  #endif
    ASSERT(kernel_rotate >= 0 && kernel_rotate < 2);
    ASSERT(do_relu >= 0 && do_relu < 2);
  #ifdef __sg2262__
       ASSERT(saturate >= 0 && saturate < 2);
  #endif
    ASSERT(input_sign >= 0 && input_sign < 2);
    ASSERT(weight_sign >= 0 && weight_sign < 2);
    ASSERT(bias_sign >= 0 && bias_sign < 2);
    ASSERT(res_sign >= 0 && res_sign < 2);
  #ifdef __sg2262__
    ASSERT((do_rescale == 0 && rescale_is_const == 1) || do_rescale == 1);
  #else
    ASSERT(do_rescale == 0 || do_rescale == 1);
  #endif
#endif
}
