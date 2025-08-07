#include "ppl.h"
#include "ppl_wrapper_func.h"
using namespace ppl;

template <typename T>
void add_const_fp_local(uint32 o_local_ddr, uint32 i_local_ddr, float rhs, int N, int C, int H, int W, bool relu) {
  dim4 src_shape = {N, C, H, W};
  auto in_tensor = tensor<T>(src_shape, TPU_ALIGN, i_local_ddr);
  auto out_tensor = tensor<T>(src_shape, TPU_ALIGN, o_local_ddr);
  tiu::fadd(out_tensor, in_tensor, rhs);
  if (relu) {
    tiu::max(out_tensor, out_tensor, 0.0f);
  }
}
__KERNEL__ void add_const_f32_local(uint32 o_local_ddr, uint32 i_local_ddr, float rhs, int N, int C,
                              int H, int W, bool relu) {
  add_const_fp_local<float>(o_local_ddr, i_local_ddr, rhs, N, C, H, W, relu);
}
__KERNEL__ void add_const_f16_local(uint32 o_local_ddr, uint32 i_local_ddr, float rhs, int N, int C, int H,
                              int W, bool relu) {
  add_const_fp_local<fp16>(o_local_ddr, i_local_ddr, rhs, N, C, H, W, relu);
}
__KERNEL__ void add_const_bf16_local(uint32 o_local_ddr, uint32 i_local_ddr, float rhs, int N, int C,
                               int H, int W, bool relu) {
  add_const_fp_local<bf16>(o_local_ddr, i_local_ddr, rhs, N, C, H, W, relu);
}

// __KERNEL__ void add_const_f32_test(float *ptr_dst, float *ptr_src, uint32 o_local_ddr, uint32 i_local_ddr, float rhs, int N, int C,
//                    int H, int W, bool relu) {
//   dim4 src_shape = {N, C, H, W};
//   auto dst_gtensor = gtensor<float>(src_shape, GLOBAL, ptr_dst);
//   auto src_gtensor = gtensor<float>(src_shape, GLOBAL, ptr_src);
//   auto in_tensor = tensor<float>(src_shape, TPU_ALIGN, i_local_ddr);
//   auto out_tensor = tensor<float>(src_shape, TPU_ALIGN, o_local_ddr);
//   dma::load(in_tensor, src_gtensor);
//   add_const_fp_local<float>(o_local_ddr, i_local_ddr, rhs, N, C, H, W, relu);
//   dma::store(dst_gtensor, out_tensor);
// }

// __TEST__ void addconst_test_main() {

//   const int N = 1;
//   const int C = 32;
//   const int H = 1;
//   const int W = 16;

//   dim4 src_shape = {N, C, H, W};

//   auto dst = rand<float>(&src_shape, 0, 0);
//   auto src = rand<float>(&src_shape, 0.0, 1000.0);
//   float scalar = 1.0f;

//   add_const_f32_test(dst, src, 0, 4096, scalar, N, C, H, W, true);
// }
