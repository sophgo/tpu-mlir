#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;
#ifdef __cv184x__
#define DTYPE bf16
#else
#define DTYPE fp32
#endif

__KERNEL__ void softmax_h_dim(DTYPE *ptr_output, DTYPE *ptr_input, const int N,
                              const int C, const int H, const int W,
                              const int block_c, const int block_h,
                              const int block_w) {
  const int max_c = N * C;
  const int max_h = H;
  const int max_w = W;
  const int block_h_num = max(max_h / block_h, 1);

  dim4 global_shape = {1, max_c, max_h, max_w};
  auto in_gtensor = gtensor<DTYPE>(global_shape, GLOBAL, ptr_input);
  auto out_gtensor = gtensor<DTYPE>(global_shape, GLOBAL, ptr_output);

  dim2 stride = {1, 1};
  padding_t pad = {0, 0, 0, 0};
  dim2 dilation = {1, 1};
  for (auto w_idx = 0; w_idx < max_w; w_idx += block_w) {
    int real_block_w = min(block_w, max_w - w_idx);

    for (auto c_idx = 0; c_idx < max_c; c_idx += block_c) {
      enable_pipeline();
      int real_block_c = min(block_c, max_c - c_idx);

      dim4 pool_buffer_block_shape = {1, block_c, 1, block_w};
      dim4 pool_buffer_shape = {1, real_block_c, 1, real_block_w};
      auto pool_buffer =
          make_tensor<DTYPE>(pool_buffer_block_shape, pool_buffer_shape);
      auto max_pool_buffer =
          make_tensor<DTYPE>(pool_buffer_block_shape, pool_buffer_shape);
      auto avg_pool_buffer =
          make_tensor<DTYPE>(pool_buffer_block_shape, pool_buffer_shape);
      tiu::fill(avg_pool_buffer, 0.0f);

      dim4 in_block_shape = {1, block_c, block_h, block_w};
      for (auto h_idx = 0; h_idx < max_h; h_idx += block_h) {
        int real_block_h = min(block_h, max_h - h_idx);
        dim4 global_offset = {0, c_idx, h_idx, w_idx};

        dim4 real_in_shape = {1, real_block_c, real_block_h, real_block_w};
        auto in_sub = make_tensor<DTYPE>(in_block_shape, real_in_shape);
        dma::load(in_sub, in_gtensor.sub_view(real_in_shape, global_offset));

        dim2 kernel = {real_in_shape.h, 1};
        tiu::fpool_max(pool_buffer, in_sub, &kernel, &pad, &stride, &dilation);
        tiu::fmax(max_pool_buffer, max_pool_buffer, pool_buffer);

        // (src - max_val)
        auto sub = make_tensor<DTYPE>(in_block_shape, real_in_shape);
        tiu::fsub(sub, in_sub, max_pool_buffer);

        // exp(src - max_val) -- in
        exp_no_overflow(in_sub, sub, &in_block_shape, &real_in_shape);

        tiu::fpool_avg(pool_buffer, in_sub, &kernel, &pad, &stride, &dilation,
                       1.f);
        tiu::fadd(avg_pool_buffer, avg_pool_buffer, pool_buffer);

        // exp(src - max_val) / sum(exp(src - max_val))
        auto out = make_tensor<DTYPE>(in_block_shape, real_in_shape);
        tiu::fdiv(avg_pool_buffer, 1, avg_pool_buffer, 3);
        tiu::fmul(out, in_sub, avg_pool_buffer);
        dma::store(out_gtensor.sub_view(real_in_shape, global_offset), out);
      }
    }
  }
}

__KERNEL__ void softmax_w_dim(fp32 *ptr_output, fp32 *ptr_input, const int N,
                              const int C, const int H, const int W,
                              const int block_c, const int block_w) {
  const int max_w = H * W;
  const int max_c = N * C;

  dim4 global_shape = {1, max_c, 1, max_w};
  auto in_gtensor = gtensor<fp32>(global_shape, GLOBAL, ptr_input);
  auto out_gtensor = gtensor<fp32>(global_shape, GLOBAL, ptr_output);

  dim4 pool_buffer_block_shape = {1, block_c, 1, 1};
  dim4 local_in_block_shape = {1, block_c, 1, block_w};
  dim2 stride = {1, 1};
  padding_t pad = {0, 0, 0, 0};
  dim2 dilation = {1, 1};
  for (auto c_idx = 0; c_idx < max_c; c_idx += block_c) {
    enable_pipeline();
    int real_block_c = min(block_c, max_c - c_idx);
    dim4 pool_buffer_real_shape = {1, real_block_c, 1, 1};
    auto local_pool_buffer =
        make_tensor<fp32>(pool_buffer_block_shape, pool_buffer_real_shape);
    auto local_max_pool =
        make_tensor<fp32>(pool_buffer_block_shape, pool_buffer_real_shape);
    auto local_avg_pool =
        make_tensor<fp32>(pool_buffer_block_shape, pool_buffer_real_shape);
    tiu::fill(local_avg_pool, 0.0f);

    for (auto w_idx = 0; w_idx < max_w; w_idx += block_w) {
      int real_block_w = min(block_w, max_w - w_idx);
      dim4 real_local_in_shape = {1, real_block_c, 1, real_block_w};
      dim4 global_offset = {0, c_idx, 0, w_idx};

      auto local_in_sub =
          make_tensor<fp32>(local_in_block_shape, real_local_in_shape);
      dma::load(local_in_sub,
                in_gtensor.sub_view(real_local_in_shape, global_offset));

      dim2 kernel = {real_local_in_shape.h, real_local_in_shape.w};
      tiu::fpool_max(local_pool_buffer, local_in_sub, &kernel, &pad, &stride,
                     &dilation);
      tiu::fmax(local_max_pool, local_max_pool, local_pool_buffer);

      // exp(src - max_val) -- local_in
      auto sub = make_tensor<fp32>(local_in_block_shape, real_local_in_shape);
      tiu::fsub(sub, local_in_sub, local_max_pool);
      exp_no_overflow(local_in_sub, sub, &local_in_block_shape,
                      &real_local_in_shape);

      // exp(src - max_val) / sum(exp(src - max_val)) -- out
      auto out = make_tensor<fp32>(local_in_block_shape, real_local_in_shape);
      tiu::fpool_avg(local_pool_buffer, local_in_sub, &kernel, &pad, &stride,
                     &dilation, 1.f);
      tiu::fadd(local_avg_pool, local_avg_pool, local_pool_buffer);
      tiu::fdiv(local_avg_pool, 1, local_avg_pool, 3);
      tiu::fmul(out, local_in_sub, local_avg_pool);
      dma::store(out_gtensor.sub_view(real_local_in_shape, global_offset), out);
    }
  }
}
