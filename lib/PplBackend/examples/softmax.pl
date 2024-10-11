#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

__KERNEL__ void softmax_h_dim(fp32 *ptr_output, fp32 *ptr_input, const int N,
                              const int C, const int H, const int W,
                              const int block_c, const int block_h,
                              const int block_w) {
  const int max_c = N * C;
  const int max_h = H;
  const int max_w = W;
  const int block_h_num = max(max_h / block_h, 1);

  dim4 global_shape = {1, max_c, max_h, max_w};
  auto in_gtensor = gtensor<fp32>(global_shape, GLOBAL, ptr_input);
  auto out_gtensor = gtensor<fp32>(global_shape, GLOBAL, ptr_output);

  dim2 stride = {1, 1};
  padding_t pad = {0, 0, 0, 0};
  dim2 dilation = {1, 1};
  for (auto w_idx = 0; w_idx < max_w; w_idx += block_w) {
    int real_block_w = min(block_w, max_w - w_idx);

    for (auto c_idx = 0; c_idx < max_c; c_idx += block_c) {
      int real_block_c = min(block_c, max_c - c_idx);

      dim4 max_pool_buffer_shape = {1, block_c, block_h_num, block_w};
      dim4 pool_buffer_shape = {1, real_block_c, block_h_num, block_w};
      auto pool_buffer =
          make_tensor<fp32>(max_pool_buffer_shape, pool_buffer_shape);

      dim4 max_in_shape = {1, block_c, max_h, block_w};
      dim4 in_shape = {1, real_block_c, max_h, real_block_w};
      auto in_buffer = make_tensor<fp32>(max_in_shape, in_shape);

      for (auto h_idx = 0; h_idx < max_h; h_idx += block_h) {
        int real_block_h = min(block_h, max_h - h_idx);

        dim4 real_in_shape = {1, real_block_c, real_block_h, real_block_w};
        dim4 global_offset = {0, c_idx, h_idx, w_idx};
        dim4 in_offset = {0, 0, h_idx, 0};
        auto in_sub = in_buffer.sub_view(real_in_shape, in_offset);
        dma::load(in_sub, in_gtensor.sub_view(real_in_shape, global_offset));

        dim2 kernel1 = {real_in_shape.h, 1};
        dim4 max_pool_shape = {1, real_block_c, 1, block_w};
        dim4 max_pool_offset = {0, 0, h_idx / block_h, 0};
        auto max_pool_sub =
            pool_buffer.sub_view(max_pool_shape, max_pool_offset);
        // print("max_pool_sub : %s\n", to_string(max_pool_sub));
        tiu::fpool_max(max_pool_sub, in_sub, &kernel1, &pad, &stride,
                       &dilation);
        // dim4 test_shape = {1, real_block_c, 1, 1};
        // dim4 test_offset = {0, 0, 0, 0};
        // dma::store(out_gtensor.sub_view(test_shape, test_offset),
        //            max_pool_sub.sub_view(test_shape, test_offset));
      }

      dim4 all_pool_max_shape = {1, block_c, 1, block_w};
      dim4 all_pool_real_shape = {1, real_block_c, 1, block_w};
      dim2 kernel2 = {block_h_num, 1};
      auto all_pool_val =
          make_tensor<fp32>(all_pool_max_shape, all_pool_real_shape);
      if (block_h_num > 1) {
        tiu::fpool_max(all_pool_val, pool_buffer, &kernel2, &pad, &stride,
                       &dilation);
      }
      // dim4 test_shape = {1, real_block_c, 1, 1};
      // dim4 test_offset = {0, 0, 0, 0};
      // dma::store(out_gtensor.sub_view(test_shape, test_offset),
      //            all_pool_val.sub_view(test_shape, test_offset));

      for (auto h_idx = 0; h_idx < max_h; h_idx += block_h) {
        enable_pipeline();
        int real_block_h = min(block_h, max_h - h_idx);

        dim4 max_in_shape = {1, block_c, block_h, block_w};
        dim4 real_in_shape = {1, real_block_c, real_block_h, real_block_w};
        dim4 in_offset = {0, 0, h_idx, 0};
        auto in_sub = in_buffer.sub_view(real_in_shape, in_offset);
        auto sub = make_tensor<fp32>(max_in_shape, real_in_shape);
        // (src - max_val)
        dim4 real_pool_shape = {1, real_block_c, 1, real_block_w};
        if (block_h_num > 1) {
          tiu::fsub(sub, in_sub, all_pool_val.view(real_pool_shape));
        } else {
          tiu::fsub(sub, in_sub, pool_buffer.view(real_pool_shape));
        }
        // dim4 test_offset = {0, 0, h_idx, w_idx};
        // dma::store(out_gtensor.sub_view(real_in_shape, test_offset),
        //            sub);

        // exp(src - max_val) -- in
        exp_no_overflow(in_sub, sub, &max_in_shape, &real_in_shape);
        // dim4 test_offset = {0, 0, h_idx, w_idx};
        // dma::store(out_gtensor.sub_view(real_in_shape, test_offset),
        //            in_sub);

        dim2 kernel3 = {real_in_shape.h, 1};
        dim4 mean_pool_shape = {1, real_block_c, 1, block_w};
        dim4 mean_pool_offset = {0, 0, h_idx / block_h, 0};
        auto mean_pool_sub =
            pool_buffer.sub_view(mean_pool_shape, mean_pool_offset);
        tiu::fpool_avg(mean_pool_sub, in_sub, &kernel3, &pad, &stride,
                       &dilation, 1.f);
        // dim4 test_offset = {0, 0, h_idx / block_h, w_idx};
        // dma::store(out_gtensor.sub_view(mean_pool_shape, test_offset),
        //            mean_pool_sub);
      }

      // sum(exp(src - max_val))
      if (block_h_num > 1) {
        tiu::fpool_avg(all_pool_val, pool_buffer, &kernel2, &pad, &stride,
                       &dilation);
      }
      // dim4 test_offset = {0, 0, 0, w_idx};
      // dma::store(out_gtensor.sub_view(all_pool_real_shape, test_offset),
      //            all_pool_val);

      for (auto h_idx = 0; h_idx < max_h; h_idx += block_h) {
        enable_pipeline();
        int real_block_h = min(block_h, max_h - h_idx);

        dim4 global_offset = {0, c_idx, h_idx, w_idx};
        dim4 max_in_shape = {1, block_c, block_h, block_w};
        dim4 real_in_shape = {1, real_block_c, real_block_h, real_block_w};
        dim4 in_offset = {0, 0, h_idx, 0};
        auto in_sub = in_buffer.sub_view(real_in_shape, in_offset);
        auto out = make_tensor<fp32>(max_in_shape, real_in_shape);
        // exp(src - max_val) / sum(exp(src - max_val))
        dim4 real_pool_shape = {1, real_block_c, 1, real_block_w};
        if (block_h_num > 1) {
          auto real_pool_val = all_pool_val.view(real_pool_shape);
          tiu::fdiv(real_pool_val, 1, real_pool_val, 3);
          tiu::fmul(out, in_sub, real_pool_val);
        } else {
          auto real_pool_val = pool_buffer.view(real_pool_shape);
          tiu::fdiv(real_pool_val, 1, real_pool_val, 3);
          tiu::fmul(out, in_sub, real_pool_val);
        }
        dma::store(out_gtensor.sub_view(real_in_shape, global_offset), out);
      }
    }
  }
}

__KERNEL__ void softmax_w_dim(fp32 *ptr_output, fp32 *ptr_input, const int N,
                              const int C, const int H, const int W,
                              const int block_c, const int block_w) {
  const int alignment = EU_BYTES;
  const int max_w = H * W;
  const int max_c = N * C;
  const int block_w_num = max(1, max_w / block_w);

  dim4 global_shape = {1, max_c, 1, max_w};
  auto in_gtensor = gtensor<fp32>(global_shape, GLOBAL, ptr_input);
  auto out_gtensor = gtensor<fp32>(global_shape, GLOBAL, ptr_output);

  dim2 stride = {1, 1};
  padding_t pad = {0, 0, 0, 0};
  dim2 dilation = {1, 1};
  for (auto c_idx = 0; c_idx < max_c; c_idx += block_c) {
    int real_block_c = min(block_c, max_c - c_idx);
    dim4 pool_buffer_shape = {1, block_c, block_w_num, alignment};
    dim4 pool_buffer_real_shape = {1, real_block_c, block_w_num, alignment};
    auto local_pool_buffer =
        make_tensor<fp32>(pool_buffer_shape, pool_buffer_real_shape);

    dim4 local_in_all_shape = {1, block_c, 1, max_w};
    dim4 local_in_real_shape = {1, block_c, 1, max_w};
    auto local_in = make_tensor<fp32>(local_in_all_shape, local_in_real_shape);

    for (auto w_idx = 0; w_idx < max_w; w_idx += block_w) {
      // enable_pipeline();
      int real_block_w = min(block_w, max_w - w_idx);
      dim4 real_local_in_shape = {1, real_block_c, 1, real_block_w};
      dim4 input_global_offset = {0, c_idx, 0, w_idx};

      dim4 local_in_offset = {0, 0, 0, w_idx};
      auto local_in_sub =
          local_in.sub_view(real_local_in_shape, local_in_offset);
      dma::load(local_in_sub,
                in_gtensor.sub_view(real_local_in_shape, input_global_offset));

      dim2 kernel = {real_local_in_shape.h, real_local_in_shape.w};
      dim4 real_pool_shape = {1, real_block_c, 1, alignment};
      dim4 real_pool_offset = {0, 0, w_idx / block_w, 0};
      tiu::fpool_max(
          local_pool_buffer.sub_view(real_pool_shape, real_pool_offset),
          local_in_sub, &kernel, &pad, &stride, &dilation);

      // dim4 test_shape = {1, real_block_c, 1, 1};
      // dim4 test_offset = {0, c_idx, 0, w_idx / block_w};
      // dma::store(out_gtensor.sub_view(test_shape, test_offset),
      //            local_pool_buffer.sub_view(test_shape, real_pool_offset));
    }

    dim4 mu_shape = {1, block_c, 1, alignment};
    dim4 real_mu_shape = {1, real_block_c, 1, alignment};
    dim2 kernel2 = {block_w_num, alignment};
    auto local_mu = make_tensor<fp32>(mu_shape, real_mu_shape);
    if (block_w_num > 1) {
      tiu::fpool_max(local_mu, local_pool_buffer, &kernel2, &pad, &stride,
                     &dilation);
    }

    // dim4 test_shape = {1, real_block_c, 1, 1};
    // dim4 test_offset = {0, c_idx, 0, 0};
    // dim4 pool_offset = {0, 0, 0, 0};
    // dma::store(out_gtensor.sub_view(test_shape, test_offset),
    //            local_mu.sub_view(test_shape, pool_offset));

    for (auto w_idx = 0; w_idx < max_w; w_idx += block_w) {
      enable_pipeline();
      dim4 global_offset = {0, c_idx, 0, w_idx};
      int real_block_w = min(block_w, max_w - w_idx);
      dim4 local_in_shape = {1, block_c, 1, block_w};
      dim4 real_local_in_shape = {1, real_block_c, 1, real_block_w};
      dim4 local_in_offset = {0, 0, 0, w_idx};
      auto local_in_sub =
          local_in.sub_view(real_local_in_shape, local_in_offset);

      auto sub = make_tensor<fp32>(local_in_shape, real_local_in_shape);
      // (src - max_val)
      dim4 real_mu_shape = {1, real_block_c, 1, 1};
      if (block_w_num > 1) {
        tiu::fsub(sub, local_in_sub, local_mu.view(real_mu_shape));
      } else {
        tiu::fsub(sub, local_in_sub, local_pool_buffer.view(real_mu_shape));
      }
      // exp(src - max_val) -- local_in
      exp_no_overflow(local_in_sub, sub, &local_in_shape, &real_local_in_shape);

      dim2 kernel = {real_local_in_shape.h, real_local_in_shape.w};
      dim4 real_pool_shape = {1, real_block_c, 1, alignment};
      dim4 real_pool_offset = {0, 0, w_idx / block_w, 0};
      tiu::fpool_avg(
          local_pool_buffer.sub_view(real_pool_shape, real_pool_offset),
          local_in_sub, &kernel, &pad, &stride, &dilation, 1.f);
    }
    // sum(exp(src - max_val))  -- local_mu
    if (block_w_num > 1) {
      tiu::fpool_avg(local_mu, local_pool_buffer, &kernel2, &pad, &stride,
                     &dilation, 1.f);
    }

    for (auto w_idx = 0; w_idx < max_w; w_idx += block_w) {
      enable_pipeline();
      dim4 global_offset = {0, c_idx, 0, w_idx};
      int real_block_w = min(block_w, max_w - w_idx);
      dim4 local_in_shape = {1, block_c, 1, block_w};
      dim4 real_local_in_shape = {1, real_block_c, 1, real_block_w};
      dim4 local_in_offset = {0, 0, 0, w_idx};
      auto in_sub = local_in.sub_view(real_local_in_shape, local_in_offset);
      auto out = make_tensor<fp32>(local_in_shape, real_local_in_shape);
      // exp(src - max_val) / sum(exp(src - max_val)) -- out
      dim4 real_mu_shape = {1, real_block_c, 1, 1};
      if (block_w_num > 1) {
        auto real_pool_val = local_mu.view(real_mu_shape);
        tiu::fdiv(real_pool_val, 1, real_pool_val, 3);
        tiu::fmul(out, in_sub, real_pool_val);
      } else {
        auto real_pool_val = local_pool_buffer.view(real_mu_shape);
        tiu::fdiv(real_pool_val, 1, real_pool_val, 3);
        tiu::fmul(out, in_sub, real_pool_val);
      }
      dma::store(out_gtensor.sub_view(real_local_in_shape, global_offset), out);
    }
  }
}
