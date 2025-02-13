#include "add_const_fp.h"
#include "helper.h"
#include "tpu_mlir/Backend/BM168x/Param.h"
#include <assert.h>
#include <cstdio>
#include <functional>
#include <stddef.h>
#include <stdint.h>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif
int tiling_and_call(int dtype, const char *chip, void *cmdid, uint64_t ptr_dst,
                    uint64_t ptr_src, float rhs, int N, int C, int H, int W,
                    int block_w, bool relu) {
  block_w = align_up(N * C * H * W, 32);
  while (block_w > 0) {
    int ret = -1;
    switch (dtype) {
    case DTYPE_FP32:
      ret = add_const_f32(chip, cmdid, ptr_dst, ptr_src, rhs, N, C, H, W,
                          block_w, relu);
      break;
    case DTYPE_FP16:
      ret = add_const_f16(chip, cmdid, ptr_dst, ptr_src, rhs, N, C, H, W,
                          block_w, relu);
      break;
    case DTYPE_BFP16:
      ret = add_const_bf16(chip, cmdid, ptr_dst, ptr_src, rhs, N, C, H, W,
                           block_w, relu);
      break;
    default:
      assert(0 && "dtype not supported\n");
    }
    if (ret == 0) {
      // success
      return 0;
    } else if (ret == PplLocalAddrAssignErr) {
      // local memory not enough, reduce block_w size
      block_w = align_up(block_w / 2, 32);
      continue;
    } else if (ret == PplL2AddrAssignErr) {
      // L2 memory not enough, This is impossible for the current operator to
      // occur
      assert(0);
    } else {
      // Critical error encountered
      assert(0);
      return ret;
    }
  }
  return 1;
}

void api_add_const_fp_global(void *param, size_t param_size, void *input_spec,
                             void *output_spec, const int core_num,
                             const char *chip, void *cmdid) {

  constbinary_global_spec_t *_param = (constbinary_global_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  auto rhs = _param->common.B_const_val;
  bool do_relu = _param->common.if_relu;

  tiling_and_call(in_spec->dtype, chip, cmdid, out_spec->addr, in_spec->addr,
                  rhs, in_spec->shape[0], in_spec->shape[1], in_spec->shape[2],
                  in_spec->shape[3], 0, do_relu);
}

#ifdef __cplusplus
}
#endif
