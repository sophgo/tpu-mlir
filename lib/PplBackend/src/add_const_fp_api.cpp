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
                      int H, int W, bool relu, int dtype);
void api_add_const_fp_global(void *param, size_t param_size, void *input_spec,
                             void *output_spec) {
  std::string chip_str = get_chip_str();
  constbinary_global_spec_t *_param = (constbinary_global_spec_t *)param;
  tensor_spec_t *in_spec = (tensor_spec_t *)input_spec;
  tensor_spec_t *out_spec = (tensor_spec_t *)output_spec;
  auto rhs = _param->common.B_const_val;
  bool do_relu = _param->common.if_relu;
  add_tiling(out_spec->addr, in_spec->addr, rhs, in_spec->shape[0],
             in_spec->shape[1], in_spec->shape[2], in_spec->shape[3], do_relu,
             in_spec->dtype);
}

#ifdef __cplusplus
}
#endif
