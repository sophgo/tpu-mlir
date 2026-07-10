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

// Multi-input (A, B) -> multi-output (A*scale, A*scale+B).
// Both inputs are assumed to share the same NCHW shape.
template <typename T>
void scale_add_fp(T *ptr_out0, T *ptr_out1, T *ptr_in0, T *ptr_in1, float scale,
                  int N, int C, int H, int W, const int block_w) {
  // reshape [N, C, H, W] -> [1, c_slice, 1, w_slice], c_slice <= LANE_NUM
  int total_length = N * C * H * W;
  int c_slice = get_max_common_div(total_length, LANE_NUM);
  int w_slice = total_length / c_slice;

  int block_c = LANE_NUM;
  dim4 gshape = {1, c_slice, 1, w_slice};
  dim4 block_shape = {1, block_c, 1, block_w};

  auto out0_gt = gtensor<T>(gshape, GLOBAL, ptr_out0);
  auto out1_gt = gtensor<T>(gshape, GLOBAL, ptr_out1);
  auto in0_gt = gtensor<T>(gshape, GLOBAL, ptr_in0);
  auto in1_gt = gtensor<T>(gshape, GLOBAL, ptr_in1);

  for (int idx_c = 0; idx_c < c_slice; idx_c += block_c) {
    int cur_c = min(block_c, c_slice - idx_c);
    for (int idx_w = 0; idx_w < w_slice; idx_w += block_w) {
      ppl::enable_pipeline();
      int cur_w = min(block_w, w_slice - idx_w);
      dim4 real_shape = {1, cur_c, 1, cur_w};
      auto in0_t = make_tensor<T>(block_shape, real_shape);
      auto in1_t = make_tensor<T>(block_shape, real_shape);
      auto out0_t = make_tensor<T>(block_shape, real_shape);
      auto out1_t = make_tensor<T>(block_shape, real_shape);
      dim4 offset = {0, idx_c, 0, idx_w};
      dma::load(in0_t, in0_gt.sub_view(real_shape, offset));
      dma::load(in1_t, in1_gt.sub_view(real_shape, offset));
      tiu::fmul(out0_t, in0_t, scale);   // out0 = in0 * scale
      tiu::fadd(out1_t, out0_t, in1_t);  // out1 = in0 * scale + in1
      dma::store(out0_gt.sub_view(real_shape, offset), out0_t);
      dma::store(out1_gt.sub_view(real_shape, offset), out1_t);
    }
  }
}

__KERNEL__ void scale_add_f32(float *ptr_out0, float *ptr_out1, float *ptr_in0,
                              float *ptr_in1, float scale, int N, int C, int H,
                              int W, const int block_w) {
  scale_add_fp(ptr_out0, ptr_out1, ptr_in0, ptr_in1, scale, N, C, H, W,
               block_w);
}
__KERNEL__ void scale_add_f16(fp16 *ptr_out0, fp16 *ptr_out1, fp16 *ptr_in0,
                              fp16 *ptr_in1, float scale, int N, int C, int H,
                              int W, const int block_w) {
  scale_add_fp(ptr_out0, ptr_out1, ptr_in0, ptr_in1, scale, N, C, H, W,
               block_w);
}
__KERNEL__ void scale_add_bf16(bf16 *ptr_out0, bf16 *ptr_out1, bf16 *ptr_in0,
                               bf16 *ptr_in1, float scale, int N, int C, int H,
                               int W, const int block_w) {
  scale_add_fp(ptr_out0, ptr_out1, ptr_in0, ptr_in1, scale, N, C, H, W,
               block_w);
}
