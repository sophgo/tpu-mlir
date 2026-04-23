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
#define INF 3.40282e38


template<typename DataType>
void argSort(gtensor<DataType> &input_g, gtensor<int32_t> &seq_index_g,
             gtensor<DataType> &output_g, gtensor<int32_t> &out_index_g,
             gtensor<uint32_t> &scatter_index_g,
             int n, const int tile_n) {
  float eps = 1e-9; // eps must be small enough
  dim4 input_lshape = {1, 1, 1, tile_n};
  dim4 mask_shape = {1, 1, tile_n, tile_n};
  auto input1_local = tensor<DataType>(input_lshape);
  auto index1_local = tensor<int32_t>(input_lshape);
  auto seq1_local = tensor<int32_t>(input_lshape);
  auto bias1_fp32 = tensor<fp32>(input_lshape);
  auto input1_fp32 = tensor<fp32>(input_lshape);
  auto input2_local = tensor<DataType>(input_lshape);
  auto index2_local = tensor<int32_t>(input_lshape);
  auto seq2_local = tensor<int32_t>(input_lshape);
  auto bias2_fp32 = tensor<fp32>(input_lshape);
  auto input2_fp32 = tensor<fp32>(input_lshape);
  auto output_local = tensor<DataType>(input_lshape);
  auto out_index_local = tensor<int32_t>(input_lshape);

  auto mask_local = tensor<uint8_t>(mask_shape);
  dim4 temp_stride = get_stride<DataType>(input_lshape, TPU_ALIGN);
  dim4 bw_stride = {temp_stride.n, temp_stride.c, 1, 0};
  dim4 bh_stride = {temp_stride.n, temp_stride.c, 0, 1};
  dim4 index_shape = {1, 1, tile_n, 1};
  assert(n <= 65535);
  auto scatter_index = tensor<uint16_t>(index_shape);
  auto scatter_buffer = tensor<uint16_t>(index_shape);
  auto scatter_index_u32 = tensor<uint32_t>(index_shape);
  const int pool_eu_num = 32;
  pool_param kernel_param = pool::param::kernel(1, tile_n, 1, 1); // kh,kw,stride_h,stride_w
  pool_param pad_param = pool::param::padding(0, 0, 0, 0, 0); // pad_value,up_pad,down_pad,left_pad,right_pad
  for (int n1_idx = 0; n1_idx < n; n1_idx += tile_n) {
    int cur1_n = min(tile_n, n - n1_idx);
    dim4 cur1_shape = {1, 1, 1, cur1_n};
    dim4 offset1 = {0, 0, 0, n1_idx};
    auto cur1_input = input1_local.view(cur1_shape);
    auto cur1_seq = seq1_local.view(cur1_shape);
    auto cur1_bias_fp32 = bias1_fp32.view(cur1_shape);
    auto cur1_input_fp32 = input1_fp32.view(cur1_shape);
    dma::load(cur1_input, input_g.sub_view(cur1_shape, offset1));
    tiu::cast(cur1_input_fp32, cur1_input);
    arange_broadcast(cur1_seq, 1, n1_idx, 1, cur1_n);
    tiu::cast(cur1_bias_fp32, cur1_seq);
    tiu::fmul(cur1_bias_fp32, cur1_bias_fp32, eps);
    tiu::fadd(cur1_input_fp32, cur1_input_fp32, cur1_bias_fp32);
    if (cur1_n < tile_n) {
      dim4 pad_shape = {1, 1, 1, tile_n - cur1_n};
      dim4 pad_offset = {0, 0, 0, cur1_n};
      tiu::fill(input1_fp32.sub_view(pad_shape, pad_offset), INF);
    }
    tiu::fill(scatter_index, 0);
    dma::store(seq_index_g.sub_view(cur1_shape, offset1), cur1_seq);
    for (int n2_idx = 0; n2_idx < n; n2_idx += tile_n) {
      ppl::enable_pipeline();
      int cur2_n = min(tile_n, n - n2_idx);
      dim4 cur2_shape = {1, 1, 1, cur2_n};
      dim4 offset2 = {0, 0, 0, n2_idx};
      auto cur2_input = input2_local.view(cur2_shape);
      auto cur2_seq = seq2_local.view(cur2_shape);
      auto cur2_bias_fp32 = bias2_fp32.view(cur2_shape);
      auto cur2_input_fp32 = input2_fp32.view(cur2_shape);
      dma::load(cur2_input, input_g.sub_view(cur2_shape, offset2));
      tiu::cast(cur2_input_fp32, cur2_input);
      arange_broadcast(cur2_seq, 1, n2_idx, 1, cur2_n);
      tiu::cast(cur2_bias_fp32, cur2_seq);
      tiu::fmul(cur2_bias_fp32, cur2_bias_fp32, eps);
      tiu::fadd(cur2_input_fp32, cur2_input_fp32, cur2_bias_fp32);
      if (cur2_n < tile_n) {
        dim4 pad_shape = {1, 1, 1, tile_n - cur2_n};
        dim4 pad_offset = {0, 0, 0, cur2_n};
        tiu::fill(input2_fp32.sub_view(pad_shape, pad_offset), -INF);
      }
      auto left_lhs = input1_fp32.view(input_lshape, bw_stride);
      auto right_lhs = input2_fp32.view(input_lshape, bh_stride);
      tiu::lt(mask_local, left_lhs, right_lhs, 1);
      // sum to get the sort order index and accumulate
      if (tile_n % pool_eu_num == 0 && tile_n / pool_eu_num < 15) {
        // two-step pool optimization: first compress mask from tile_n columns to 32 columns (uint8->uint8), then pool 32 columns (uint8->uint16)
        dim4 compressed_mask_shape = {1, 1, tile_n, pool_eu_num};
        pool_param step1_kernel = pool::param::kernel(1, tile_n / pool_eu_num, 1, tile_n / pool_eu_num);
        tiu::pool_avg(mask_local.view(compressed_mask_shape), mask_local, step1_kernel, pad_param, 1, 0);
        pool_param step2_kernel = pool::param::kernel(1, pool_eu_num, 1, 1);
        tiu::pool_avg(scatter_buffer, mask_local.view(compressed_mask_shape), step2_kernel, pad_param, 1, 0);
      } else {
        tiu::pool_avg(scatter_buffer, mask_local, kernel_param, pad_param, 1, 0);
      }
      tiu::add(scatter_index, scatter_index, scatter_buffer);
    }
    tiu::cast(scatter_index_u32.view(cur1_shape), scatter_index.view(cur1_shape));
    dma::store(scatter_index_g.sub_view(cur1_shape, offset1), scatter_index_u32.view(cur1_shape));
  }
  // scatter on global memory using scatter_h
  dim4 view_gshape = {1, 1, n, 1};
  dma::scatter_h(output_g.view(view_gshape), input_g.view(view_gshape), scatter_index_g.view(view_gshape));
  dma::scatter_h(out_index_g.view(view_gshape), seq_index_g.view(view_gshape), scatter_index_g.view(view_gshape));
}

/*
Slice batch across cores, slice select within each core.
Full-sort topk: sort all elements and take the top k.
*/
template<typename DataType>
void allsort_topk_slice_batch(DataType *ptr_input, DataType *value_output, int32_t *index_output,
                               DataType *value_buffer, int32_t *index_buffer,
                               uint32_t *scatter_index_buffer, int32_t *seq_index_buffer,
                               int batch, int select, const int tile_select, const int k,
                               const int core_num) {
  int core_idx = get_core_index();
  if (core_idx >= core_num) {
    return;
  }
  int sliceB = div_up(batch, core_num);
  int coreB = min(sliceB, batch - core_idx * sliceB);
  if (sliceB <= 0) {
    return;
  }
  int startB = core_idx * sliceB;
  int endB = startB + coreB;

  // global tensors
  // assert(select % NPU_NUM == 0 && tile_select % NPU_NUM == 0);
  dim4 input_gshape = {batch, 1, 1, select};
  dim4 output_gshape = {batch, 1, 1, k};
  dim4 buffer_gshape = {core_num, 1, 1, select};
  auto input_gtensor = gtensor<DataType>(input_gshape, GLOBAL, ptr_input);
  auto value_gtensor = gtensor<DataType>(output_gshape, GLOBAL, value_output);
  auto index_gtensor = gtensor<int32_t>(output_gshape, GLOBAL, index_output);
  auto value_buffer_gtensor = gtensor<DataType>(buffer_gshape, GLOBAL, value_buffer);
  auto index_buffer_gtensor = gtensor<int32_t>(buffer_gshape, GLOBAL, index_buffer);
  auto scatter_gtensor = gtensor<uint32_t>(buffer_gshape, GLOBAL, scatter_index_buffer);
  auto seq_gtensor = gtensor<int32_t>(buffer_gshape, GLOBAL, seq_index_buffer);

  const int tile_n = 256;
  for (int bi = startB; bi < endB; bi++) {
    // 1. sort all select elements
    dim4 sort_shape = {1, 1, 1, select};
    dim4 src_goffset = {bi, 0, 0, 0};
    dim4 sort_goffset = {core_idx, 0, 0, 0};
    auto input_gt = input_gtensor.sub_view(sort_shape, src_goffset);
    auto out_gt = value_buffer_gtensor.sub_view(sort_shape, sort_goffset);
    auto out_index_gt = index_buffer_gtensor.sub_view(sort_shape, sort_goffset);
    auto seq_gt = seq_gtensor.sub_view(sort_shape, sort_goffset);
    auto scatter_gt = scatter_gtensor.sub_view(sort_shape, sort_goffset);
    argSort(input_gt, seq_gt, out_gt, out_index_gt, scatter_gt, select, tile_n);

    // 2. take top k from sorted result and write to output
    dim4 save_shape = {1, 1, 1, k};
    dim4 dst_offset = {bi, 0, 0, 0};
    dma::move(value_gtensor.sub_view(save_shape, dst_offset), value_buffer_gtensor.sub_view(save_shape, sort_goffset));
    dma::move(index_gtensor.sub_view(save_shape, dst_offset), index_buffer_gtensor.sub_view(save_shape, sort_goffset));
  }
}

__KERNEL__ void allsort_topk_fp32(fp32 *ptr_input, fp32 *value_output, int32_t *index_output,
                                 fp32 *value_buffer, int32_t *index_buffer,
                                 uint32_t *scatter_index_buffer, int32_t *seq_index_buffer,
                                 int batch, int select, const int tile_select, const int k,
                                 const int core_num) {
  allsort_topk_slice_batch<fp32>(ptr_input, value_output, index_output, value_buffer, index_buffer,
                           scatter_index_buffer, seq_index_buffer,
                           batch, select, tile_select, k, core_num);
}

__KERNEL__ void allsort_topk_fp16(fp16 *ptr_input, fp16 *value_output, int32_t *index_output,
                                  fp16 *value_buffer, int32_t *index_buffer,
                                  uint32_t *scatter_index_buffer, int32_t *seq_index_buffer,
                                  int batch, int select, const int tile_select, const int k,
                                  const int core_num) {
  allsort_topk_slice_batch<fp16>(ptr_input, value_output, index_output, value_buffer, index_buffer,
                           scatter_index_buffer, seq_index_buffer,
                           batch, select, tile_select, k, core_num);
}

__KERNEL__ void allsort_topk_bf16(bf16 *ptr_input, bf16 *value_output, int32_t *index_output,
                                  bf16 *value_buffer, int32_t *index_buffer,
                                  uint32_t *scatter_index_buffer, int32_t *seq_index_buffer,
                                  int batch, int select, const int tile_select, const int k,
                                  const int core_num) {
  allsort_topk_slice_batch<bf16>(ptr_input, value_output, index_output, value_buffer, index_buffer,
                           scatter_index_buffer, seq_index_buffer,
                           batch, select, tile_select, k, core_num);
}

__TEST__ void allsort_topk_main() {
  int batch = 16;
  int select = 256;
  const int tile_select = 256;
  const int k = 8;
  const int core_num = 1;
  dim4 input_shape = {batch, 1, 1, select};
  dim4 out_shape = {batch, 1, 1, k};
  dim4 buffer_shape = {core_num, 1, 1, select};
  auto ptr_in = ppl::malloc<fp32>(&input_shape);
  auto ptr_out = ppl::malloc<fp32>(&out_shape);
  auto ptr_out_index = ppl::malloc<int32_t>(&out_shape);
  auto ptr_value_buffer = ppl::malloc<fp32>(&buffer_shape);
  auto ptr_index_buffer = ppl::malloc<int32_t>(&buffer_shape);
  auto ptr_scatter_index_buffer = ppl::malloc<uint32_t>(&buffer_shape);
  auto ptr_seq_index_buffer = ppl::malloc<int32_t>(&buffer_shape);
  ppl::rand(ptr_in, &input_shape, -1000.f, 1000.f);
  allsort_topk_fp32(ptr_in, ptr_out, ptr_out_index, ptr_value_buffer, ptr_index_buffer,
                    ptr_scatter_index_buffer, ptr_seq_index_buffer,
                    batch, select, tile_select, k, core_num);
}
