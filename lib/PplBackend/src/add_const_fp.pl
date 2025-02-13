#include "ppl.h"
#include "ppl_wrapper_func.h"
using namespace ppl;

int get_max_common_div(int v, int max_v) {
  for (int i = max_v; i > 0; i--) {
    if (v % i == 0) {
      return i;
    }
  }
  return 1;
}

template <typename T>
void add_const_fp(T *ptr_dst, T *ptr_src, float rhs, int N, int C, int H, int W,
                  const int block_w, bool relu) {
  // reshape src [N, C, H, W] -> [1, c_slice, 1, w_slice]
  // c_slice <= LANE_NUM
  int total_length = N * C * H * W;
  int c_slice = get_max_common_div(total_length, LANE_NUM);
  int w_slice = total_length / c_slice;

  int block_c = LANE_NUM;
  // global tensor shape
  dim4 src_shape = {1, c_slice, 1, w_slice};
  // local tensor block shape
  dim4 src_block_shape = {1, block_c, 1, block_w};

  auto dst_gtensor = gtensor<T>(src_shape, GLOBAL, ptr_dst);
  auto src_gtensor = gtensor<T>(src_shape, GLOBAL, ptr_src);

  for (int idx_c = 0; idx_c < c_slice; idx_c += block_c) {
    int cur_c = min(block_c, c_slice - idx_c);
    for (int idx_w = 0; idx_w < w_slice; idx_w += block_w) {
      ppl::enable_pipeline();
      int cur_w = min(block_w, w_slice - idx_w);
      // local tensor real shape
      dim4 src_real_shape = {1, cur_c, 1, cur_w};
      auto in_tensor = make_tensor<T>(src_block_shape, src_real_shape);
      auto out_tensor = make_tensor<T>(src_block_shape, src_real_shape);
      dim4 offset = {0, idx_c, 0, idx_w};
      dma::load(in_tensor, src_gtensor.sub_view(src_real_shape, offset));
      tiu::fadd(out_tensor, in_tensor, rhs);
      if (relu) {
        tiu::max(out_tensor, out_tensor, 0.0f);
      }
      dma::store(dst_gtensor.sub_view(src_real_shape, offset), out_tensor);
    }
  }
}
__KERNEL__ void add_const_f32(float *ptr_dst, float *ptr_src, float rhs, int N, int C,
                   int H, int W, const int block_w, bool relu) {
  add_const_fp(ptr_dst, ptr_src, rhs, N, C, H, W, block_w, relu);
}
__KERNEL__ void add_const_f16(fp16 *ptr_dst, fp16 *ptr_src, float rhs, int N, int C, int H,
                   int W, const int block_w, bool relu) {
  add_const_fp(ptr_dst, ptr_src, rhs, N, C, H, W, block_w, relu);
}
__KERNEL__ void add_const_bf16(bf16 *ptr_dst, bf16 *ptr_src, float rhs, int N, int C,
                    int H, int W, const int block_w, bool relu) {
  add_const_fp(ptr_dst, ptr_src, rhs, N, C, H, W, block_w, relu);
}

template <typename T>
void add_const_mc_fp(T *ptr_dst, T *ptr_src, float rhs, int N, int C, int H,
                     int W, const int block_w, const int core_num, bool relu) {
  set_group_num(1);
  set_block_num(core_num);

  int cur_block_num = get_block_num();
  int block_idx = get_block_index();

  // reshape src [N, C, H, W] -> [1, c_slice, 1, w_slice]
  // c_slice <= LANE_NUM
  int total_length = N * C * H * W;
  int c_slice = get_max_common_div(total_length, LANE_NUM);
  int w_slice = total_length / c_slice;

  // The w size to be calculated on each core
  int w_slice_core = div_up(w_slice, cur_block_num);
  // The w size to be calculated on current core
  int cur_w_slice_core = min(w_slice_core, w_slice - block_idx * w_slice_core);

  if (cur_w_slice_core > 0) {
    int block_c = LANE_NUM;
    // global tensor shape
    dim4 src_shape = {1, c_slice, 1, w_slice};
    // local tensor block shape
    dim4 src_block_shape = {1, block_c, 1, block_w};

    auto dst_gtensor = gtensor<T>(src_shape, GLOBAL, ptr_dst);
    auto src_gtensor = gtensor<T>(src_shape, GLOBAL, ptr_src);

    int w_start = w_slice_core * block_idx;
    int w_end = cur_w_slice_core + w_start;

    for (int idx_c = 0; idx_c < c_slice; idx_c += block_c) {
      int cur_c = min(block_c, c_slice - idx_c);
      for (int idx_w = w_start; idx_w < w_end; idx_w += block_w) {
        ppl::enable_pipeline();
        int cur_w = min(block_w, w_end - idx_w);
        // local tensor real shape
        dim4 src_real_shape = {1, cur_c, 1, cur_w};
        auto in_tensor = make_tensor<T>(src_block_shape, src_real_shape);
        auto out_tensor = make_tensor<T>(src_block_shape, src_real_shape);
        dim4 offset = {0, idx_c, 0, idx_w};
        dma::load(in_tensor, src_gtensor.sub_view(src_real_shape, offset));
        tiu::fadd(out_tensor, in_tensor, rhs);
        if (relu) {
          tiu::max(out_tensor, out_tensor, 0.0f);
        }
        dma::store(dst_gtensor.sub_view(src_real_shape, offset), out_tensor);
      }
    }
  }
}
__KERNEL__ void add_const_mc_f32(float *ptr_dst, float *ptr_src, float rhs, int N, int C,
                       int H, int W, const int block_w, const int core_num,
                       bool relu) {
  add_const_mc_fp(ptr_dst, ptr_src, rhs, N, C, H, W, block_w, core_num, relu);
}
__KERNEL__ void add_const_mc_fp16(fp16 *ptr_dst, fp16 *ptr_src, float rhs, int N, int C,
                       int H, int W, const int block_w, const int core_num,
                       bool relu) {
  add_const_mc_fp(ptr_dst, ptr_src, rhs, N, C, H, W, block_w, core_num, relu);
}
__KERNEL__ void add_const_mc_bf16(bf16 *ptr_dst, bf16 *ptr_src, float rhs, int N, int C,
                       int H, int W, const int block_w, const int core_num,
                       bool relu) {
  add_const_mc_fp(ptr_dst, ptr_src, rhs, N, C, H, W, block_w, core_num, relu);
}

__TEST__ void topk_test_main() {

  const int N = 4;
  const int C = 2;
  const int H = 10;
  const int W = 1024;

  dim4 src_shape = {N, C, H, W};

  auto dst = rand<float>(&src_shape, 0, 0);
  auto src = rand<float>(&src_shape, 0.0, 1000.0);
  float scalar = 1.0f;
  int block_w = 1024;

  add_const_mc_f32(dst, src, scalar, N, C, H, W, block_w, 8, true);
  // add_const_f32(dst, src, scalar, N, C, H, W, block_w, true);
}
