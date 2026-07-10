#include "scale_add_fp.h"  // generated from scale_add_fp.pl by ppl-compile
#include "tpu_utils.h"

extern "C" {
using KernelFunc = int (*)(global_addr_t ptr_out0, global_addr_t ptr_out1,
                           global_addr_t ptr_in0, global_addr_t ptr_in1,
                           float scale, int N, int C, int H, int W, int block_w);

int scale_add_tiling(global_addr_t ptr_out0, global_addr_t ptr_out1,
                     global_addr_t ptr_in0, global_addr_t ptr_in1, float scale,
                     int N, int C, int H, int W, int dtype) {
  KernelFunc func;
  if (dtype == SG_DTYPE_FP32) {
    func = scale_add_f32;
  } else if (dtype == SG_DTYPE_FP16) {
    func = scale_add_f16;
  } else if (dtype == SG_DTYPE_BFP16) {
    func = scale_add_bf16;
  } else {
    assert(0 && "unsupported dtype");
  }
  int block_w = align_up(N * C * H * W, 32);
  int ret = -1;
  while (block_w > 1) {
    ret = func(ptr_out0, ptr_out1, ptr_in0, ptr_in1, scale, N, C, H, W, block_w);
    if (ret == 0) {
      return 0;
    } else if (ret == PplLocalAddrAssignErr) {
      block_w = block_w / 2;
      continue;
    } else if (ret == PplL2AddrAssignErr) {
      assert(0);
    } else {
      assert(0);
      return ret;
    }
  }
  return ret;
}
}
