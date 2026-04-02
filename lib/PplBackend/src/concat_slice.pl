//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "ppl.h"
#include "ppl_wrapper_func.h"
using namespace ppl;

/*
ConcatSlice kernel.

Concatenates in0[c,h,w0] and in1[c,h,w1] along the w dimension, then slices
off the front portion so the output shape matches in0[c,h,w0].

Equivalent to:
  concat_result = concatenate(in0, in1, axis=w)  // shape [c, h, w0+w1]
  output = concat_result[:, :, w1:]               // shape [c, h, w0]

This is useful for KV cache updates: shift old data left and append new data.

block_h controls tiling along the h dimension.
*/
template <typename T>
void concat_slice_kernel(T *ptr_out, T *ptr_in0, T *ptr_in1, const int c,
                         const int h, const int w0, const int w1,
                         const int block_h, const int core_num) {
  int core_index = get_core_index();
  if (core_index >= core_num) {
    return;
  }

  // Global shapes: in0[c,h,w0], in1[c,h,w1], out[c,h,w0]
  dim4 in0_g_shape = {1, c, h, w0};
  dim4 in1_g_shape = {1, c, h, w1};
  dim4 out_g_shape = {1, c, h, w0};

  auto g_in0 = gtensor<T>(in0_g_shape, GLOBAL, ptr_in0);
  auto g_in1 = gtensor<T>(in1_g_shape, GLOBAL, ptr_in1);
  auto g_out = gtensor<T>(out_g_shape, GLOBAL, ptr_out);

  // Distribute h across cores
  int per_core = (h + core_num - 1) / core_num;
  int h_start = core_index * per_core;
  int h_end = min(h_start + per_core, h);

  dim4 block0_shape = {1, c, block_h, w0};
  dim4 block1_shape = {1, c, block_h, w1};

  // Process blocks along the h dimension
  for (int hi = h_start; hi < h_end; hi += block_h) {
    enable_pipeline();
    int cur_block = min(block_h, h_end - hi);

    dim4 in0_real_shape = {1, c, cur_block, w0};
    dim4 in1_real_shape = {1, c, cur_block, w1};
    auto dst = make_tensor<T>(block0_shape, in0_real_shape);
    auto in0 = make_tensor<T>(block0_shape, in0_real_shape);
    auto in1 = make_tensor<T>(block1_shape, in1_real_shape);

    dim4 h_offset = {0, 0, hi, 0};
    dma::load(in0, g_in0.sub_view(in0_real_shape, h_offset));
    dma::load(in1, g_in1.sub_view(in1_real_shape, h_offset));

    // Copy tail of in0 (skip first w1 elements) -> head of output
    dim4 dst0_shape = {1, c, cur_block, w0 - w1};
    dim4 in0_offset = {0, 0, 0, w1};
    auto in0_sub = in0.sub_view(dst0_shape, in0_offset);
    dim4 dst0_offset = {0, 0, 0, 0};
    auto dst_in0 = dst.sub_view(dst0_shape, dst0_offset);
    tiu::move(dst_in0, in0_sub);

    // Copy in1 -> tail of output
    dim4 dst1_offset = {0, 0, 0, w0 - w1};
    auto dst_in1 = dst.sub_view(in1_real_shape, dst1_offset);
    tiu::move(dst_in1, in1);

    dma::store(g_out.sub_view(block0_shape, h_offset), dst);
  }
}

__KERNEL__ void concat_slice_bf16(bf16 *ptr_out, bf16 *ptr_in0, bf16 *ptr_in1,
                                  const int c, const int h, const int w0,
                                  const int w1, const int block_h,
                                  const int core_num) {
  concat_slice_kernel<bf16>(ptr_out, ptr_in0, ptr_in1, c, h, w0, w1,
                            block_h, core_num);
}

__KERNEL__ void concat_slice_f16(fp16 *ptr_out, fp16 *ptr_in0, fp16 *ptr_in1,
                                 const int c, const int h, const int w0,
                                 const int w1, const int block_h,
                                 const int core_num) {
  concat_slice_kernel<fp16>(ptr_out, ptr_in0, ptr_in1, c, h, w0, w1,
                            block_h, core_num);
}

__TEST__ void concat_slice_test() {
  const int c = 2, h = 8, w0 = 4, w1 = 2;
  const int core_num = 1;
  const int block_h = 2;

  dim4 in0_shape = {1, c, h, w0};
  dim4 in1_shape = {1, c, h, w1};
  dim4 out_shape = {1, c, h, w0};

  auto ptr_in0 = malloc<bf16>(&in0_shape);
  auto ptr_in1 = malloc<bf16>(&in1_shape);
  auto ptr_out = malloc<bf16>(&out_shape);

  ppl::read_npz(ptr_in0, "concat_slice_input.npz", "in0");
  ppl::read_npz(ptr_in1, "concat_slice_input.npz", "in1");

  concat_slice_bf16(ptr_out, ptr_in0, ptr_in1, c, h, w0, w1, block_h,
                    core_num);
}
