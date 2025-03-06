#include "add_const_fp_local.h"
#include "tpu_utils.h"
#include "helper.h"

extern "C" {
using KernelFunc = int (*)(local_addr_t, local_addr_t, float, int, int, int, int, bool);
int addconst_local(local_addr_t ptr_dst, local_addr_t ptr_src, float rhs, int N, int C,
                   int H, int W, bool relu, int dtype) {
  KernelFunc func;
  if (dtype == SG_DTYPE_FP32) {
    func = add_const_f32_local;
  } else if (dtype == SG_DTYPE_FP16) {
    func = add_const_f16_local;
  } else if (dtype == SG_DTYPE_BFP16) {
    func = add_const_bf16_local;
  } else {
    assert(0 && "unsupported dtype");
  }
  int ret = func(ptr_dst, ptr_src, rhs, N, C, H, W, relu);
  if (ret == 0) {
    return 0;
  } else if (ret == PplLocalAddrAssignErr) {
    assert(0);
  } else if (ret == PplL2AddrAssignErr) {
    assert(0);
  } else {
    CHECK_PPL_RET(ret);
  }
  return ret;
}
}

