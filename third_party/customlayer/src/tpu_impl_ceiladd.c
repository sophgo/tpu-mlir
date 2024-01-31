#include "tpu_impl_custom_ops.h"

void tpu_impl_ceiladd_global(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    const int *shape,
    float b_val,
    data_type_t dtype)
{
  local_addr_t local_in_addr = 0;
  local_addr_t local_mid_addr = BANK_SIZE * 4;
  local_addr_t local_out_addr = BANK_SIZE * 8;
  dim4 _shape = {.n = shape[0], .c = shape[1], .h = shape[2], .w = shape[3]};
  tpu_gdma_cpy_S2L(local_in_addr, input_global_addr, &_shape, NULL, NULL, dtype);
  tpu_bdc_fp_ceil(local_mid_addr, local_in_addr, &_shape, NULL, NULL, dtype);
  scalar_t C = {.f32 = b_val};
  tpu_bdc_fp_add_C(local_out_addr, local_mid_addr, tpu_fp_cast(C, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO), &_shape, NULL, NULL, dtype);
  tpu_gdma_cpy_L2S(output_global_addr, local_out_addr, &_shape, NULL, NULL, dtype);
}

int get_ceiladd_local_bfsz(
    const int *shape,
    data_type_t dtype)
{
  dim4 _shape = {.n = shape[0], .c = shape[1], .h = shape[2], .w = shape[3]};
  return tpu_get_local_size(&_shape, dtype, 0, true);
}

void tpu_impl_ceiladd_local(
    local_addr_t input_addr,
    local_addr_t output_addr,
    local_addr_t buffer_addr,
    const int *shape,
    float b_val,
    data_type_t dtype)
{
  dim4 _shape = {.n = shape[0], .c = shape[1], .h = shape[2], .w = shape[3]};
  tpu_bdc_fp_ceil(buffer_addr, input_addr, &_shape, NULL, NULL, dtype);
  scalar_t C = {.f32 = b_val};
  tpu_bdc_fp_add_C(output_addr, buffer_addr, tpu_fp_cast(C, dtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO), &_shape, NULL, NULL, dtype);
}
