#ifndef ATOMIC_CHECK
#define ATOMIC_CHECK

#include "atomic_conv_bw.h"
#include "atomic_gdma.h"
#include "atomic_pooling_depthwise.h"
#include "atomic_random_gen.h"
#include "atomic_sdma.h"
#include "atomic_tensor_arithmetic.h"
#include "atomic_tiu.h"

// tiu
void arithmetic_check(ppl_tensor_t *rst, ppl_variable_t *lhs,
                      ppl_variable_t *rhs, int satu, int arith_mode);

void cmp_check(ppl_tensor_t *dst, ppl_variable_t *src0, ppl_variable_t *src1,
               uint32_t true_val, AR_OP op);
// void arithmetic_check(ppl_tensor_t *rst, ppl_tensor_t *lhs, ppl_tensor_t
// *rhs,
//                       int satu, int arith_mode);

// void cmp_check(ppl_tensor_t *dst, ppl_tensor_t *src0, ppl_tensor_t *src1,
//                uint32_t true_val, AR_OP op);

void arithmetic_lhs_c_check(ppl_tensor_t *rst, uint32_t lhs, ppl_tensor_t *rhs,
                            int satu, int arith_mode);

void arithmetic_rhs_c_check(ppl_tensor_t *rst, ppl_tensor_t *lhs, uint32_t rhs,
                            int satu, int arith_mode);

void cmp_lhs_c_check(ppl_tensor_t *dst, uint32_t src0, ppl_tensor_t *src1,
                     uint32_t true_val, AR_OP op);

void cmp_rhs_c_check(ppl_tensor_t *dst, ppl_tensor_t *src0, uint32_t src1,
                     uint32_t true_val, AR_OP op);

void arithmetic_shift_check(ppl_tensor_t *rst, ppl_variable_t *lhs,
                            ppl_variable_t *rhs, unsigned int shift, int satu,
                            int arith_mode, int round);

void arithmetic_single_opd_check(ppl_tensor_t *dst, ppl_tensor_t *src,
                                 AR_OP op);

void cvt_check(ppl_tensor_t *dst, ppl_tensor_t *src,
               rounding_mode_t round_mode);

void sfu_check(ppl_tensor_t *dst, ppl_tensor_t *src, ppl_tensor_t *coeff,
               int num, SFU_OP sfu_op);

void int_mac_check(ppl_tensor_t *rst, ppl_variable_t *src0,
                   ppl_variable_t *src1, unsigned char lshift,
                   unsigned char rshift, int round);

void fused_linear_check(ppl_tensor_t *dst, ppl_tensor_t *src,
                        ppl_variable_t *scale, ppl_variable_t *bias,
                        int B_is_const, int C_is_const, LIN_OP op_lin,
                        int satu);

void arithmetic_div_check(ppl_tensor_t *rst, ppl_variable_t *lhs,
                          ppl_variable_t *rhs, int satu, int num_iter);

void arithmetic_c_div_check(ppl_tensor_t *rst, uint32_t lhs, ppl_tensor_t *rhs,
                            int satu, int num_iter);

void arithmetic_div_c_check(ppl_tensor_t *rst, ppl_tensor_t *lhs, uint32_t rhs,
                            int satu, int num_iter);

void fused_cmp_check(ppl_tensor_t *R0, ppl_tensor_t *R1, ppl_variable_t *A,
                     ppl_variable_t *B, ppl_variable_t *C, ppl_variable_t *D,
                     int side, int bin_w, CMP_OP op);

void conv_quant_check(ppl_tensor_t *input, ppl_tensor_t *output,
                      ppl_variable_t *weight, ppl_variable_t *bias,
                      ppl_variable_t *pad_ins, ppl_variable_t *kzp,
                      ppl_variable_t *requant, int kh, int kw, int stride_h,
                      int stride_w, int ins_h, int ins_w, int dilation_h,
                      int dilation_w, int pad_h_t, int pad_h_b, int pad_w_l,
                      int pad_w_r, int kernel_rotate, int result_add,
                      u32 ins_const_val, int do_relu, int sym_saturate,
                      int do_requant, int shift_num, int ozp,
                      ROUND_MODE rm_mode, PAD_MODE pad_mode);

void conv_fp_check(ppl_tensor_t *input, ppl_tensor_t *output,
                   ppl_variable_t *weight, ppl_variable_t *bias,
                   ppl_variable_t *pad_ins, ppl_variable_t *rescale, int kh,
                   int kw, int stride_h, int stride_w, int ins_h, int ins_w,
                   int dilation_h, int dilation_w, int pad_h_t, int pad_h_b,
                   int pad_w_l, int pad_w_r, int kernel_rotate, int result_add,
                   u32 ins_const_val, int do_relu, int saturate,
                   PAD_MODE pad_mode);

void cpy_cross_npu_check(ppl_tensor_t *src, ppl_tensor_t *dst);

void dq0_check(ppl_tensor_t *input, ppl_variable_t *B_tensor,
               ppl_tensor_t *output, int offset, float scale, int round_mode);

void dq1_check(ppl_tensor_t *input, ppl_variable_t *B_tensor,
               ppl_tensor_t *output, int zp_value, int scale_factor,
               int shift_num, int round_mode);

void rq0_check(ppl_tensor_t *input, ppl_variable_t *scale, ppl_tensor_t *output,
               float scale_value, float offset, int output_round_mode,
               int input_round_mode);

void rq1_check(ppl_tensor_t *input, ppl_variable_t *quant, ppl_tensor_t *output,
               int scale_val, char shift_val, short zp_val, int round_mode);

void depthwise_quant_check(ppl_tensor_t *input, ppl_tensor_t *output,
                           ppl_variable_t *weight, ppl_variable_t *bias,
                           ppl_variable_t *pad_ins, ppl_variable_t *requant,
                           int kh, int kw, int stride_h, int stride_w,
                           int ins_h, int ins_w, int dh, int dw, int pad_h_t,
                           int pad_h_b, int pad_w_l, int pad_w_r,
                           int kernel_rotate, int do_relu, int sym_saturate,
                           int do_requant, int shift_num, int ozp,
                           ROUND_MODE rm_mode, PAD_MODE pad_mode);

void fp_exponent_part_check(ppl_tensor_t *dst, ppl_tensor_t *src);

void sfu_taylor_check(ppl_tensor_t *dst, ppl_tensor_t *src, ppl_tensor_t *table,
                      int num);

void sgl_hgather_check(ppl_tensor_t *tensorR, ppl_tensor_t *tensorA,
                       ppl_tensor_t *tensorB, int A_cstride_is0,
                       int if_fill_const, u32 fill_const_val, int limit_enable,
                       SG_OP op);

void pes_sg_d1hzd_check(ppl_tensor_t *tensorR, ppl_tensor_t *tensorA,
                        ppl_tensor_t *tensorB, int A_cstride_is0,
                        int if_fill_const, u32 fill_const_val,
                        int limit_enable);

void pl_sgd2_check(ppl_tensor_t *tensorR, ppl_tensor_t *tensorA,
                   ppl_tensor_t *tensorB, int if_fill_const, u32 fill_const_val,
                   int limit_enable, SG_OP op);

void pl_sgd1_check(ppl_tensor_t *tensorR, ppl_tensor_t *tensorA,
                   ppl_tensor_t *tensorB, int if_fill_const, u32 fill_const_val,
                   int limit_enable, SG_OP op);

void static_broad_txp_check(ppl_tensor_t *dst, u32 src_addr);
void static_broad_check(ppl_tensor_t *dst, u32 src_addr);

void static_distribute_txp_check(ppl_tensor_t *dst, u32 src_addr);

// dma
void dma_stride_move_check(ppl_tensor_t *dst, ppl_tensor_t *src, int direction,
                           int trans);

void dma_compact_move_check(ppl_tensor_t *dst, ppl_tensor_t *src, int direction,
                            int trans);

void broadcast_move_check(ppl_tensor_t *dst, ppl_tensor_t *src, int direction);

void dma_nonzero_move_check(ppl_tensor_t *dst, ppl_tensor_t *src, u32 base_idx,
                            int direction);

void dma_reverse_gen_cmd_check(ppl_tensor_t *dst, ppl_tensor_t *src,
                               int32_t reverse_axis, int32_t direction);

void hgather_gdma_check(ppl_tensor_t *dst, ppl_tensor_t *src,
                        ppl_tensor_t *index, u64 const_val, int direction,
                        int index_in_lmem, int start_pos);

void fill_constant_check_tensor(const ppl_tensor_t *dst, const void *const_val);

void dma_general_move_check(ppl_tensor_t *dst, ppl_tensor_t *src, int direction,
                            int trans);

//mask select
//MaskSelect
void mask_select_check(ppl_tensor_t *dst,
                       ppl_tensor_t *src,
                       ppl_tensor_t *mask,
                       int direction);
//MaskSelectBatchBcast
void mask_select_batch_bcast_check(ppl_tensor_t *dst,
                                   ppl_tensor_t *count,
                                   ppl_tensor_t *src,
                                   ppl_tensor_t *mask);

//nonzero
void nonzero_check(ppl_tensor_t *src,
                   ppl_tensor_t *dst_idx,
                   ppl_tensor_t *dst_cnt,
                   int support_hzd);

//normal
void normal_check(ppl_tensor_t *dst, ppl_tensor_t *src);

//NPU bcast
void npu_bcast_check(ppl_tensor_t *dst, ppl_tensor_t *src);

//scatter
void batch_bcast_hscatter_check(ppl_tensor_t *output, ppl_tensor_t *param, ppl_tensor_t *index);
void batch_bcast_wscatter_check(ppl_tensor_t *output, ppl_tensor_t *param, ppl_tensor_t *index);
void hscatter_check(ppl_tensor_t *output, ppl_tensor_t *param, ppl_tensor_t *index);
void hwscatter_check(ppl_tensor_t *output, ppl_tensor_t *param, ppl_tensor_t *index);
void wscatter_check(ppl_tensor_t *output, ppl_tensor_t *param, ppl_tensor_t *index);


//sdma
void sdma_general_move_check(ppl_tensor_t* dst, ppl_tensor_t* src, int transpose);
void sdma_setc_check(ppl_tensor_t* dst, void* scalar);
void sdma_hgather_check(ppl_tensor_t* output, ppl_tensor_t* param, ppl_tensor_t* index, void* c);
void sdma_hscatter_check(ppl_tensor_t* output, ppl_tensor_t* param, ppl_tensor_t* index);

//shift
void shift_check(
    ppl_tensor_t *dst,
    ppl_tensor_t *src,
    ppl_variable_t *sft,
    int op,
    int round_mode);

//tiu l2l
void l2l_copy_check(ppl_tensor_t *dst,
                    ppl_tensor_t *src);

//tiu mask select
void tiu_mask_select_check(ppl_tensor_t *src, ppl_tensor_t *dst,
                       ppl_tensor_t *dst_cnt, ppl_tensor_t *mask_num,
                       int A_cstride_is0);

//tiu setC
void setc_check(ppl_tensor_t *dst, void *scalar, int scalar_dtype);

//tiu transpose
void cw_transpose_check(ppl_tensor_t *dst, ppl_tensor_t *src);
void wc_transpose_check(ppl_tensor_t *dst, ppl_tensor_t *src);

//vc arith
void vc_arith_check(ppl_tensor_t *rst,
                    ppl_tensor_t *lhs,
                    ppl_tensor_t *rhs,
                    AR_OP         op,
                    int           round_mode);

void fmm2_check(ppl_tensor_t *rst, ppl_variable_t *lhs, ppl_tensor_t *rhs,
                ppl_variable_t *bias, ppl_variable_t *requant, int add_result,
                int do_relu, int satu, bool is_L_trans, bool is_R_trans);

void mm2_check(ppl_tensor_t *rst, ppl_variable_t *lhs, ppl_tensor_t *rhs,
               ppl_variable_t *bias, ppl_variable_t *requant,
               ppl_variable_t *rzp, int rshift_rd, int add_result, int do_relu,
               int satu, bool is_L_trans, bool is_R_trans);

void fmm_check(ppl_tensor_t *rst, ppl_variable_t *lhs, ppl_tensor_t *rhs,
               ppl_variable_t *bias, int add_result, int do_relu,
               bool is_L_trans);

void avg_pool_check(ppl_tensor_t *input, ppl_tensor_t *output, dim2 kernel,
                    dim2 stride, dim2 ins, dim2 dilation, int pt, int pb,
                    int pl, int pr, u32 scale);

void avg_pool_fixed_check(ppl_tensor_t *input, ppl_tensor_t *output,
                          dim2 kernel, dim2 stride, dim2 ins, dim2 dilation,
                          int pt, int pb, int pl, int pr, u32 scale);

void max_min_pool_check(ppl_tensor_t *input, ppl_tensor_t *output, dim2 kernel,
                        dim2 stride, dim2 ins, dim2 dilation, int pt, int pb,
                        int pl, int pr, int pool_op);

void roi_avg_pool_check(ppl_tensor_t *input, ppl_tensor_t *roi,
                        ppl_tensor_t *output, dim2 kernel, u32 scale,
                        int do_relu, int round_mode, int sym_range);

void roi_max_min_pool_check(ppl_tensor_t *input, ppl_tensor_t *roi,
                            ppl_tensor_t *output, dim2 kernel, int do_relu,
                            int pool_op);

void check_match_2(ppl_tensor_t *input, ppl_tensor_t *output, const dim4* perm_opt, const dim4* bmask_opt);

void check_match_3(ppl_tensor_t* a, ppl_tensor_t* b, ppl_tensor_t* c);

void check_match_4(ppl_tensor_t* a, ppl_tensor_t* b, ppl_tensor_t* c, ppl_tensor_t* d);

void check_match_5(ppl_tensor_t* a, ppl_tensor_t* b, ppl_tensor_t* c, ppl_tensor_t* d, ppl_tensor_t* e);

void check_matmul(ppl_tensor_t* lhs,
                  ppl_tensor_t* rhs,
                  ppl_tensor_t* output,
                  int a, int b,
                  int t /* 0:NN, 1:NT, 2:TT */);

void check_mm(ppl_tensor_t* left, ppl_tensor_t* right, ppl_tensor_t* dst);

void check_size(ppl_tensor_t* a, ppl_tensor_t* b);

#endif /* ATOMIC_CHECK */
