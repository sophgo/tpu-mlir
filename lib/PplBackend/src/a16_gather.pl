// A16Gather: Quantized embedding lookup
// input: weight[1,1,vocab,dim_packed] (uint8), indices[1,1,N,1] (uint32)
// output: output[1,1,N,dim] (bf16/f16)

#include "ppl.h"
#include "ppl_wrapper_func.h"
using namespace ppl;

__KERNEL__ void a16_gather_bf16(
    bf16 *output_ptr,
    uint8 *weight_ptr,
    uint32 *indices_ptr,
    bf16 *scale_ptr,
    uint8 *zp_ptr,
    const int vocab_size,
    const int dim,
    const int N,
    const int weight_bits,
    const int q_group_size) {

  // Constants
  const int block_h = 32;
  const int dim_packed = (weight_bits == 4) ? (dim / 2) : dim;
  const int n_groups = dim / q_group_size;

  // Global memory
  dim4 weight_g_shape = {1, 1, vocab_size, dim_packed};
  dim4 indices_g_shape = {1, 1, N, 1};
  dim4 output_g_shape = {1, 1, N, dim};
  dim4 scale_g_shape = {1, 1, vocab_size, n_groups};
  dim4 zp_g_shape = {1, 1, vocab_size, n_groups};

  auto weight_g = gtensor<uint8>(weight_g_shape, GLOBAL, weight_ptr);
  auto indices_g = gtensor<uint32>(indices_g_shape, GLOBAL, indices_ptr);
  auto output_g = gtensor<bf16>(output_g_shape, GLOBAL, output_ptr);
  auto scale_g = gtensor<bf16>(scale_g_shape, GLOBAL, scale_ptr);
  auto zp_g = gtensor<uint8>(zp_g_shape, GLOBAL, zp_ptr);

  // Local memory
  dim4 weight_block_shape = {1, 1, block_h, dim_packed};
  dim4 indices_block_shape = {1, 1, block_h, 1};
  dim4 output_block_shape = {1, 1, block_h, dim};
  dim4 scale_block_shape = {1, 1, block_h, n_groups};
  dim4 zp_block_shape = {1, 1, block_h, n_groups};

  auto weight_l = tensor<uint8>(weight_block_shape);
  auto indices_l = tensor<uint32>(indices_block_shape);
  auto output_l = tensor<bf16>(output_block_shape);
  auto scale_l = tensor<bf16>(scale_block_shape);
  auto zp_l = tensor<uint8>(zp_block_shape);

  // For 4-bit unpack (only used in 4-bit path)
  auto low4 = tensor<uint8>(weight_block_shape);
  auto high4 = tensor<uint8>(weight_block_shape);
  auto tmp_shift = tensor<uint8>(weight_block_shape);

  // Pre-allocated buffer for zp cast
  dim4 max_scalar_shape = {1, 1, block_h, 1};
  auto zp_buf = tensor<bf16>(max_scalar_shape);
  dim4 zero_offset = {0, 0, 0, 0};
  dim4 scalar_shape = {1, 1, block_h, 1};

  // Process in blocks
  for (int n_idx = 0; n_idx < N; n_idx += block_h) {
    int cur_h = (block_h < (N - n_idx)) ? block_h : (N - n_idx);
    dim4 cur_indices_block = {1, 1, cur_h, 1};
    dim4 cur_output_block = {1, 1, cur_h, dim};
    dim4 cur_weight_block = {1, 1, cur_h, dim_packed};
    dim4 cur_scale_block = {1, 1, cur_h, n_groups};
    dim4 cur_zp_block = {1, 1, cur_h, n_groups};
    dim4 cur_scalar_shape = {1, 1, cur_h, 1};

    dim4 indices_offset = {0, 0, n_idx, 0};

    // Load indices for this block
    dma::load(indices_l.view(cur_indices_block),
              indices_g.sub_view(cur_indices_block, indices_offset));

    // Gather packed weights, scale and zp
    auto weight_view = weight_l.view(cur_weight_block);
    auto indices_view = indices_l.view(cur_indices_block);
    dma::gather_h(weight_view, weight_g, indices_view, 0);
    dma::gather_h(scale_l.view(cur_scale_block), scale_g, indices_view, 0);
    dma::gather_h(zp_l.view(cur_zp_block), zp_g, indices_view, 0);

    if (weight_bits == 8) {
      // 8-bit: cast weight to bf16, then dequantize group by group
      auto output_view = output_l.view(cur_output_block);
      tiu::cast(output_view, weight_view);  // uint8 -> bf16

      for (int g = 0; g < n_groups; g++) {
        dim4 group_slice = {1, 1, cur_h, q_group_size};
        dim4 group_offset = {0, 0, 0, g * q_group_size};
        auto out_slice = output_view.sub_view(group_slice, group_offset);

        dim4 scalar_offset = {0, 0, 0, g};
        auto zp_scalar_u8 = zp_l.sub_view(cur_scalar_shape, scalar_offset);
        auto scale_scalar = scale_l.sub_view(cur_scalar_shape, scalar_offset);

        auto zp_scalar_fp = zp_buf.sub_view(cur_scalar_shape, zero_offset);
        tiu::cast(zp_scalar_fp, zp_scalar_u8);

        tiu::fsub(out_slice, out_slice, zp_scalar_fp);
        tiu::fmul(out_slice, out_slice, scale_scalar);
      }
    } else {
      // 4-bit: unpack then dequantize column by column
      tiu::bitwise_and(low4, weight_view, 0x0F);
      tiu::logical_shift(tmp_shift, weight_view, -4, RM_DOWN);
      tiu::bitwise_and(high4, tmp_shift, 0x0F);

      auto output_view = output_l.view(cur_output_block);

      int prev_g_low = -1, prev_g_high = -1;
      for (int i = 0; i < dim_packed; i++) {
        dim4 single_shape = {1, 1, cur_h, 1};
        dim4 src_offset = {0, 0, 0, i};

        // Dequantize low4[i] -> output[2*i]
        int g_low = (2 * i) / q_group_size;
        if (g_low != prev_g_low) {
          dim4 scalar_offset = {0, 0, 0, g_low};
          auto zp_u8 = zp_l.sub_view(cur_scalar_shape, scalar_offset);
          auto zp_fp = zp_buf.sub_view(cur_scalar_shape, zero_offset);
          tiu::cast(zp_fp, zp_u8);
          prev_g_low = g_low;
        }

        dim4 out_low_offset = {0, 0, 0, 2 * i};
        auto out_low = output_view.sub_view(single_shape, out_low_offset);
        auto low_val = low4.sub_view(single_shape, src_offset);
        dim4 scale_offset = {0, 0, 0, g_low};
        auto scale_low = scale_l.sub_view(cur_scalar_shape, scale_offset);
        auto zp_fp = zp_buf.sub_view(cur_scalar_shape, zero_offset);

        tiu::cast(out_low, low_val);
        tiu::fsub(out_low, out_low, zp_fp);
        tiu::fmul(out_low, out_low, scale_low);

        // Dequantize high4[i] -> output[2*i+1]
        int g_high = (2 * i + 1) / q_group_size;
        if (g_high != prev_g_high) {
          dim4 scalar_offset2 = {0, 0, 0, g_high};
          auto zp_u8_2 = zp_l.sub_view(cur_scalar_shape, scalar_offset2);
          auto zp_fp_2 = zp_buf.sub_view(cur_scalar_shape, zero_offset);
          tiu::cast(zp_fp_2, zp_u8_2);
          prev_g_high = g_high;
        }

        dim4 out_high_offset = {0, 0, 0, 2 * i + 1};
        auto out_high = output_view.sub_view(single_shape, out_high_offset);
        auto high_val = high4.sub_view(single_shape, src_offset);
        dim4 scale_offset2 = {0, 0, 0, g_high};
        auto scale_high = scale_l.sub_view(cur_scalar_shape, scale_offset2);
        auto zp_fp_2 = zp_buf.sub_view(cur_scalar_shape, zero_offset);

        tiu::cast(out_high, high_val);
        tiu::fsub(out_high, out_high, zp_fp_2);
        tiu::fmul(out_high, out_high, scale_high);
      }
    }

    // Store output
    dim4 output_offset = {0, 0, n_idx, 0};
    dma::store(output_g.sub_view(cur_output_block, output_offset),
              output_l.view(cur_output_block));
  }
}

// FP16 version
__KERNEL__ void a16_gather_f16(
    fp16 *output_ptr,
    uint8 *weight_ptr,
    uint32 *indices_ptr,
    fp16 *scale_ptr,
    uint8 *zp_ptr,
    const int vocab_size,
    const int dim,
    const int N,
    const int weight_bits,
    const int q_group_size) {

  // Constants
  const int block_h = 32;
  const int dim_packed = (weight_bits == 4) ? (dim / 2) : dim;
  const int n_groups = dim / q_group_size;

  // Global memory
  dim4 weight_g_shape = {1, 1, vocab_size, dim_packed};
  dim4 indices_g_shape = {1, 1, N, 1};
  dim4 output_g_shape = {1, 1, N, dim};
  dim4 scale_g_shape = {1, 1, vocab_size, n_groups};
  dim4 zp_g_shape = {1, 1, vocab_size, n_groups};

  auto weight_g = gtensor<uint8>(weight_g_shape, GLOBAL, weight_ptr);
  auto indices_g = gtensor<uint32>(indices_g_shape, GLOBAL, indices_ptr);
  auto output_g = gtensor<fp16>(output_g_shape, GLOBAL, output_ptr);
  auto scale_g = gtensor<fp16>(scale_g_shape, GLOBAL, scale_ptr);
  auto zp_g = gtensor<uint8>(zp_g_shape, GLOBAL, zp_ptr);

  // Local memory
  dim4 weight_block_shape = {1, 1, block_h, dim_packed};
  dim4 indices_block_shape = {1, 1, block_h, 1};
  dim4 output_block_shape = {1, 1, block_h, dim};
  dim4 scale_block_shape = {1, 1, block_h, n_groups};
  dim4 zp_block_shape = {1, 1, block_h, n_groups};

  auto weight_l = tensor<uint8>(weight_block_shape);
  auto indices_l = tensor<uint32>(indices_block_shape);
  auto output_l = tensor<fp16>(output_block_shape);
  auto scale_l = tensor<fp16>(scale_block_shape);
  auto zp_l = tensor<uint8>(zp_block_shape);

  // For 4-bit unpack (only used in 4-bit path)
  auto low4 = tensor<uint8>(weight_block_shape);
  auto high4 = tensor<uint8>(weight_block_shape);
  auto tmp_shift = tensor<uint8>(weight_block_shape);

  // Pre-allocated buffer for zp cast
  dim4 max_scalar_shape = {1, 1, block_h, 1};
  auto zp_buf = tensor<fp16>(max_scalar_shape);
  dim4 zero_offset = {0, 0, 0, 0};
  dim4 scalar_shape = {1, 1, block_h, 1};

  // Process in blocks
  for (int n_idx = 0; n_idx < N; n_idx += block_h) {
    int cur_h = (block_h < (N - n_idx)) ? block_h : (N - n_idx);
    dim4 cur_indices_block = {1, 1, cur_h, 1};
    dim4 cur_output_block = {1, 1, cur_h, dim};
    dim4 cur_weight_block = {1, 1, cur_h, dim_packed};
    dim4 cur_scale_block = {1, 1, cur_h, n_groups};
    dim4 cur_zp_block = {1, 1, cur_h, n_groups};
    dim4 cur_scalar_shape = {1, 1, cur_h, 1};

    dim4 indices_offset = {0, 0, n_idx, 0};

    // Load indices for this block
    dma::load(indices_l.view(cur_indices_block),
              indices_g.sub_view(cur_indices_block, indices_offset));

    // Gather packed weights, scale and zp
    auto weight_view = weight_l.view(cur_weight_block);
    auto indices_view = indices_l.view(cur_indices_block);
    dma::gather_h(weight_view, weight_g, indices_view, 0);
    dma::gather_h(scale_l.view(cur_scale_block), scale_g, indices_view, 0);
    dma::gather_h(zp_l.view(cur_zp_block), zp_g, indices_view, 0);

    if (weight_bits == 8) {
      // 8-bit: cast weight to fp16, then dequantize group by group
      auto output_view = output_l.view(cur_output_block);
      tiu::cast(output_view, weight_view);  // uint8 -> fp16

      for (int g = 0; g < n_groups; g++) {
        dim4 group_slice = {1, 1, cur_h, q_group_size};
        dim4 group_offset = {0, 0, 0, g * q_group_size};
        auto out_slice = output_view.sub_view(group_slice, group_offset);

        dim4 scalar_offset = {0, 0, 0, g};
        auto zp_scalar_u8 = zp_l.sub_view(cur_scalar_shape, scalar_offset);
        auto scale_scalar = scale_l.sub_view(cur_scalar_shape, scalar_offset);

        auto zp_scalar_fp = zp_buf.sub_view(cur_scalar_shape, zero_offset);
        tiu::cast(zp_scalar_fp, zp_scalar_u8);

        tiu::fsub(out_slice, out_slice, zp_scalar_fp);
        tiu::fmul(out_slice, out_slice, scale_scalar);
      }
    } else {
      // 4-bit: unpack then dequantize column by column
      tiu::bitwise_and(low4, weight_view, 0x0F);
      tiu::logical_shift(tmp_shift, weight_view, -4, RM_DOWN);
      tiu::bitwise_and(high4, tmp_shift, 0x0F);

      auto output_view = output_l.view(cur_output_block);

      int prev_g_low = -1, prev_g_high = -1;
      for (int i = 0; i < dim_packed; i++) {
        dim4 single_shape = {1, 1, cur_h, 1};
        dim4 src_offset = {0, 0, 0, i};

        // Dequantize low4[i] -> output[2*i]
        int g_low = (2 * i) / q_group_size;
        if (g_low != prev_g_low) {
          dim4 scalar_offset = {0, 0, 0, g_low};
          auto zp_u8 = zp_l.sub_view(cur_scalar_shape, scalar_offset);
          auto zp_fp = zp_buf.sub_view(cur_scalar_shape, zero_offset);
          tiu::cast(zp_fp, zp_u8);
          prev_g_low = g_low;
        }

        dim4 out_low_offset = {0, 0, 0, 2 * i};
        auto out_low = output_view.sub_view(single_shape, out_low_offset);
        auto low_val = low4.sub_view(single_shape, src_offset);
        dim4 scale_offset = {0, 0, 0, g_low};
        auto scale_low = scale_l.sub_view(cur_scalar_shape, scale_offset);
        auto zp_fp = zp_buf.sub_view(cur_scalar_shape, zero_offset);

        tiu::cast(out_low, low_val);
        tiu::fsub(out_low, out_low, zp_fp);
        tiu::fmul(out_low, out_low, scale_low);

        // Dequantize high4[i] -> output[2*i+1]
        int g_high = (2 * i + 1) / q_group_size;
        if (g_high != prev_g_high) {
          dim4 scalar_offset2 = {0, 0, 0, g_high};
          auto zp_u8_2 = zp_l.sub_view(cur_scalar_shape, scalar_offset2);
          auto zp_fp_2 = zp_buf.sub_view(cur_scalar_shape, zero_offset);
          tiu::cast(zp_fp_2, zp_u8_2);
          prev_g_high = g_high;
        }

        dim4 out_high_offset = {0, 0, 0, 2 * i + 1};
        auto out_high = output_view.sub_view(single_shape, out_high_offset);
        auto high_val = high4.sub_view(single_shape, src_offset);
        dim4 scale_offset2 = {0, 0, 0, g_high};
        auto scale_high = scale_l.sub_view(cur_scalar_shape, scale_offset2);
        auto zp_fp_2 = zp_buf.sub_view(cur_scalar_shape, zero_offset);

        tiu::cast(out_high, high_val);
        tiu::fsub(out_high, out_high, zp_fp_2);
        tiu::fmul(out_high, out_high, scale_high);
      }
    }

    // Store output
    dim4 output_offset = {0, 0, n_idx, 0};
    dma::store(output_g.sub_view(cur_output_block, output_offset),
              output_l.view(cur_output_block));
  }
}
