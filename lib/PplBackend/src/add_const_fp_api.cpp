#include "add_const_fp.h"
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
extern int add_tiling(gaddr_t ptr_dst, gaddr_t ptr_src, float rhs, int N, int C,
                      int H, int W, bool relu, int dtype, int &block_w);
// static interface
void api_add_const_fp_global(void *param, size_t param_size, void *input_spec,
                             void *output_spec) {
  constbinary_global_spec_t *_param = (constbinary_global_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  auto rhs = _param->common.B_const_val;
  bool do_relu = _param->common.if_relu;
  int block_w;
  add_tiling(out_spec->addr, in_spec->addr, rhs, in_spec->shape[0],
             in_spec->shape[1], in_spec->shape[2], in_spec->shape[3], do_relu,
             in_spec->dtype, block_w);
}

// dynamic interface
using fill_buffer_func = int (*)(gaddr_t, gaddr_t, float, int, int, int, int,
                                 int, bool, void *buffer);
int api_dyn_add_const_fp_global(void *param, void *input_spec,
                                void *output_spec, void *buffer) {
  constbinary_global_spec_t *_param = (constbinary_global_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  auto dtype = in_spec[0].dtype;
  auto rhs = _param->common.B_const_val;
  bool do_relu = _param->common.if_relu;
  int block_w;
  if (buffer)
    add_tiling(out_spec->addr, in_spec->addr, rhs, in_spec->shape[0],
               in_spec->shape[1], in_spec->shape[2], in_spec->shape[3], do_relu,
               in_spec->dtype, block_w);
  fill_buffer_func func;
  if (dtype == DTYPE_FP32) {
    func = fill_add_const_f32_struct;
  } else if (dtype == DTYPE_FP16) {
    func = fill_add_const_f16_struct;
  } else if (dtype == DTYPE_BFP16) {
    func = fill_add_const_bf16_struct;
  } else {
    assert(0 && "unsupported dtype");
  }
  return func(out_spec->addr, in_spec->addr, rhs, in_spec->shape[0],
              in_spec->shape[1], in_spec->shape[2], in_spec->shape[3], block_w,
              do_relu, buffer);
}
#ifdef __cplusplus
}
#endif
