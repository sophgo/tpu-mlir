#include "add_const_fp.h"
#include "tpu_utils.h"

extern "C" {
using KernelFunc = int (*)(global_addr_t, global_addr_t, float, int, int, int, int, int, bool);

int add_tiling(global_addr_t ptr_dst, global_addr_t ptr_src, float rhs, int N, int C, int H,
               int W, bool relu, int dtype) {
  KernelFunc func;
  if (dtype == SG_DTYPE_FP32) {
    func = add_const_f32;
  } else if (dtype == SG_DTYPE_FP16) {
    func = add_const_f16;
  } else if (dtype == SG_DTYPE_BFP16) {
    func = add_const_bf16;
  } else {
    assert(0 && "unsupported dtype");
  }
  auto chip = get_chip();
  (void)chip;
  const int nof_core = get_core_num();
  (void)nof_core;
  int block_w = align_up(N * C * H * W, 32);
  int ret = -1;
  while (block_w > 1) {
    ret = func(ptr_dst, ptr_src, rhs, N, C, H, W, block_w, relu);
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
