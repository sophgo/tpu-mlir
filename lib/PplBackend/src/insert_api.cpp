#include "insert_api.h"
#include "ppl_static_host.h"
#include <assert.h>
#include <cstdio>
#include <functional>
#include <stddef.h>
#include <stdint.h>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif
// ======================================
// Global GenInterface
// ======================================

// static interface
using local_func = int (*)(gaddr_t, gaddr_t, int, int, int, int, int, int, int);
int api_insert_global(void *param, size_t param_size, void *input,
                      void *output) {
  insert_spec_t *_param = (insert_spec_t *)param;
  tensor_spec_t *src_spec = &((tensor_spec_t *)input)[1];
  tensor_spec_t *dst_spec = (tensor_spec_t *)output;
  int offset = _param->offset;
  int axis = _param->axis;
  auto dtype = src_spec->dtype;
  int dbytes = get_dtype_bytes(src_spec->dtype);

  return insert_tensor(dst_spec->addr, src_spec->addr, axis, offset,
                       dst_spec->shape[0], dst_spec->shape[1],
                       dst_spec->shape[2], dst_spec->shape[3],
                       src_spec->shape[axis], dbytes);
}

#ifdef __cplusplus
}
#endif
