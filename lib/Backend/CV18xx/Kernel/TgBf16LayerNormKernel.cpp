//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/Kernel/TgConcatKernel.hpp"

#define DEBUG_TYPE "cvi_backend_layernorm_kernel"

#define ASSERT(x) assert(x)

namespace tpu_mlir {
namespace backend {
void cvi_backend_tg_bf16_layernorm_kernel(uint32_t layer_id, gaddr_t ga_input,
                                          gaddr_t ga_table,
                                          gaddr_t ga_mantissa_table,
                                          gaddr_t ga_scale, gaddr_t ga_bias,
                                          gaddr_t ga_output, int batch_size,
                                          int normalized_size, float eps,
                                          bool has_scale, bool has_bias) {
  int h, w;
  bool ret = CV18xx::size_to_hw(normalized_size, h, w);
  if (ret == false) {
    llvm::errs() << llvm::format("LayerNorm normalized size[%d] is too large\n",
                                 normalized_size);
    assert(0);
  }
  uint32_t lmem_used = 0;
  cvk_tl_shape_t table_shape = CV18xx::lut_table_shape(CVK_FMT_BF16);
  cvk_tl_t *tl_lut = CV18xx::lmem_alloc_tensor(table_shape, CVK_FMT_BF16, 1);
  cvk_tl_t *tl_lut_mantissa =
      CV18xx::lmem_alloc_tensor(table_shape, CVK_FMT_BF16, 1);
  CV18xx::tdma_load_table(tl_lut, ga_table);
  CV18xx::tdma_load_table(tl_lut_mantissa, ga_mantissa_table);
  lmem_used += 2 * CV18xx::lmem_tensor_to_size(table_shape, CVK_FMT_BF16, 1);
  cvk_tl_t *tl_scale = nullptr;
  cvk_tl_t *tl_bias = nullptr;
  auto affine_shape = CV18xx::tl_shape_t4(1, CV18xx::NPU_NUM, h, w);
  // load scale and bias
  cvk_tg_stride_t wb_gstride =
      CV18xx::tg_default_stride(CV18xx::NPU_NUM, h, w, CVK_FMT_BF16);
  wb_gstride.c = 0;

  if (has_scale) {
    tl_scale = CV18xx::lmem_alloc_tensor(affine_shape, CVK_FMT_BF16, 1);
    CV18xx::tdma_load_stride(tl_scale, ga_scale, wb_gstride);
    lmem_used += CV18xx::lmem_tensor_to_size(affine_shape, CVK_FMT_BF16, 1);
    tl_scale->stride.c = 0;
    tl_scale->stride.n = 0;
  }
  if (has_bias) {
    tl_bias = CV18xx::lmem_alloc_tensor(affine_shape, CVK_FMT_BF16, 1);
    CV18xx::tdma_load_stride(tl_bias, ga_bias, wb_gstride);
    lmem_used += CV18xx::lmem_tensor_to_size(affine_shape, CVK_FMT_BF16, 1);
    tl_bias->stride.c = 0;
    tl_bias->stride.n = 0;
  }

  int batch_step = std::min(batch_size, MAX_CHANNEL);
  while (batch_step > 0) {
    // for input and square
    uint32_t mem_need =
        2 * CV18xx::lmem_tensor_to_size(1, batch_step, h, w, CVK_FMT_BF16, 1);
    // for mean and var
    mem_need +=
        3 * CV18xx::lmem_tensor_to_size(1, batch_step, 1, 1, CVK_FMT_BF16, 1);
    if (lmem_used + mem_need <= (uint32_t)CV18xx::LMEM_BYTES) {
      break;
    }
    if (batch_step % CV18xx::NPU_NUM != 0) {
      batch_step -= batch_step % CV18xx::NPU_NUM;
    } else {
      batch_step -= CV18xx::NPU_NUM;
    }
  }
  if (batch_step == 0) {
    llvm::errs() << llvm::format(
        "Tilling LayerNorm failed, src shape:[1,%d,%d,%d]\n", batch_size, h, w);
    assert(0);
  }
  CV18xx::parallel_disable();
  for (int batch_pos = 0; batch_pos < batch_size; batch_pos += batch_step) {
    int batch = std::min(batch_step, batch_size - batch_pos);
    auto input_shape = CV18xx::tl_shape_t4(1, batch, h, w);
    auto *tl_input = CV18xx::lmem_alloc_tensor(input_shape, CVK_FMT_BF16, 1);
    uint64_t offset = batch_pos * h * w * CV18xx::bytesize_of_fmt(CVK_FMT_BF16);
    CV18xx::tdma_load(tl_input, ga_input + offset);
    auto mean_shape = CV18xx::tl_shape_t4(1, batch, 1, 1);
    cvk_tl_t *tl_mean = CV18xx::lmem_alloc_tensor(mean_shape, CVK_FMT_BF16, 1);
    cvk_tiu_average_pooling_param_t p1 = {0};
    p1.ofmap = tl_mean;
    p1.ifmap = tl_input;
    p1.kh = h;
    p1.kw = w;
    p1.ins_h = 0;
    p1.ins_last_h = 0;
    p1.ins_w = 0;
    p1.ins_last_w = 0;
    p1.stride_h = 1;
    p1.stride_w = 1;
    p1.avg_pooling_const = CV18xx::convert_fp32_to_bf16(1.0);
    p1.ins_val = 0;
    p1.ins_fp = CV18xx::convert_fp32_to_bf16(0.0);
    p1.layer_id = layer_id;
    CV18xx::tiu_average_pooling(&p1);
    // expand [1,batch,1,1] =>[1,batch,h,w]
    tl_mean->shape = input_shape;
    tl_mean->stride.w = 0;
    tl_mean->stride.h = 0;
    cvk_tiu_sub_param_t p3 = {0};
    p3.res_high = 0;
    p3.res_low = tl_input;
    p3.a_high = 0;
    p3.a_low = tl_input;
    p3.b_high = 0;
    p3.b_low = tl_mean;
    p3.rshift_bits = 0;
    p3.layer_id = layer_id;
    CV18xx::tiu_sub(&p3);
    cvk_tl_t *tl_square =
        CV18xx::lmem_alloc_tensor(input_shape, CVK_FMT_BF16, 1);
    cvk_tiu_mul_param_t p4 = {0};
    p4.res_high = nullptr;
    p4.res_low = tl_square;
    p4.a = tl_input;
    p4.b = tl_input;
    p4.b_is_const = 0;
    p4.rshift_bits = 0;
    p4.layer_id = layer_id;
    p4.relu_enable = 0;
    CV18xx::tiu_mul(&p4);
    tl_mean->shape = mean_shape;
    tl_mean->stride = CV18xx::tl_default_stride(mean_shape, CVK_FMT_BF16, 1);
    cvk_tiu_average_pooling_param_t p5 = {0};
    p5.ofmap = tl_mean;
    p5.ifmap = tl_square;
    p5.kh = h;
    p5.kw = w;
    p5.ins_h = 0;
    p5.ins_last_h = 0;
    p5.ins_w = 0;
    p5.ins_last_w = 0;
    p5.stride_h = 1;
    p5.stride_w = 1;
    p5.avg_pooling_const = CV18xx::convert_fp32_to_bf16(1.0);
    p5.ins_val = 0;
    p5.ins_fp = CV18xx::convert_fp32_to_bf16(0.0);
    p5.layer_id = layer_id;
    CV18xx::tiu_average_pooling(&p5);
    cvk_tiu_add_param_t p6 = {0};
    p6.res_high = nullptr;
    p6.res_low = tl_mean;
    p6.a_high = nullptr;
    p6.a_low = tl_mean;
    p6.b_is_const = true;
    p6.b_const.val = CV18xx::convert_fp32_to_bf16(eps);
    p6.rshift_bits = 0;
    p6.layer_id = layer_id;
    p6.relu_enable = 0;
    CV18xx::tiu_add(&p6);
    cvk_tl_t *tl_var = CV18xx::lmem_alloc_tensor(mean_shape, CVK_FMT_BF16, 1);
    cvk_tl_t *tl_buf = CV18xx::lmem_alloc_tensor(mean_shape, CVK_FMT_BF16, 1);

    cvk_tiu_bf16_lookup_interp_table_param_t p7 = {0};
    p7.ifmap = tl_mean;
    p7.buf = tl_buf;
    p7.tbl_answer = tl_lut;
    p7.tbl_answer_mantissa = tl_lut_mantissa;
    p7.ofmap = tl_var;
    p7.is_scientific = 1;
    CV18xx::tiu_bf16_lookup_interp_table(&p7);
    // mul
    tl_var->shape = input_shape;
    tl_var->stride.w = 0;
    tl_var->stride.h = 0;
    cvk_tiu_mul_param_t p9 = {0};
    p9.res_high = nullptr;
    p9.res_low = tl_input;
    p9.a = tl_input;
    p9.b = tl_var;
    p9.b_is_const = 0;
    p9.rshift_bits = 0;
    p9.layer_id = layer_id;
    p9.relu_enable = 0;
    CV18xx::tiu_mul(&p9);
    if (has_scale) {
      tl_scale->shape.c = batch;
      cvk_tiu_mul_param_t p10 = {0};
      p10.res_high = nullptr;
      p10.res_low = tl_input;
      p10.a = tl_input;
      p10.b = tl_scale;
      p10.layer_id = layer_id;
      p10.relu_enable = 0;
      CV18xx::tiu_mul(&p10);
    }
    if (has_bias) {
      tl_bias->shape.c = batch;
      cvk_tiu_add_param_t p11 = {0};
      p11.res_high = nullptr;
      p11.res_low = tl_input;
      p11.a_high = nullptr;
      p11.a_low = tl_input;
      p11.b.high = nullptr;
      p11.b.low = tl_bias;
      p11.rshift_bits = 0;
      p11.layer_id = layer_id;
      p11.relu_enable = 0;
      CV18xx::tiu_add(&p11);
    }
    CV18xx::tdma_store(tl_input, ga_output + offset);
    CV18xx::lmem_free_tensor(tl_buf);
    CV18xx::lmem_free_tensor(tl_var);
    CV18xx::lmem_free_tensor(tl_square);
    CV18xx::lmem_free_tensor(tl_mean);
    CV18xx::lmem_free_tensor(tl_input);
  }
  if (has_bias) {
    CV18xx::lmem_free_tensor(tl_bias);
  }
  if (has_scale) {
    CV18xx::lmem_free_tensor(tl_scale);
  }
  CV18xx::lmem_free_tensor(tl_lut_mantissa);
  CV18xx::lmem_free_tensor(tl_lut);
}
} // namespace backend
} // namespace tpu_mlir
