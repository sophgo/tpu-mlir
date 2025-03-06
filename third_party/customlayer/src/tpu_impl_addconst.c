#include "tpu_impl_custom_ops.h"

int get_addconst_local_bfsz(const int *shape, data_type_t dtype) {
  dim4 _shape = {.n = shape[0], .c = shape[1], .h = shape[2], .w = shape[3]};
  return tpu_get_local_size(&_shape, dtype, 0, true);
}
