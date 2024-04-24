#include "tpu_impl_custom_ops.h"

void tpu_impl_preprocess_global(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    const int *shape,
    float scale,
    float mean,
    data_type_t idtype,
    data_type_t odtype)
{
  local_addr_t local_in_addr = 0;
  local_addr_t local_mid_addr = BANK_SIZE * 4;
  local_addr_t local_out_addr = BANK_SIZE * 8;
  local_addr_t local_res_addr = BANK_SIZE * 12;
  dim4 _shape = {.n = shape[0], .c = shape[1], .h = shape[2], .w = shape[3]};
  tpu_gdma_cpy_S2L(local_in_addr, input_global_addr, &_shape, NULL, NULL, idtype);
  scalar_t mean_scalar = {.f32 = mean};
  if(idtype == DT_UINT8 || idtype == DT_INT8) {
      // (in - mean)
      tpu_bdc_int_sub_C(
        local_mid_addr,
        local_in_addr,
        tpu_cast(mean_scalar, idtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
        &_shape,
        NULL,
        NULL,
        DT_INT16,
        idtype,
        idtype,
        0,
        RM_HALF_AWAY_FROM_ZERO,
        true);
      // (in - mean) * scale
      scalar_t scale_scalar = {.f32 = scale};
      tpu_bdc_int_mul_C(local_out_addr,
                        local_mid_addr,
                        tpu_cast(scale_scalar, DT_INT16, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                        &_shape,
                        NULL,
                        NULL,
                        DT_INT16,
                        DT_INT16,
                        DT_INT16,
                        0,
                        RM_HALF_AWAY_FROM_ZERO,
                        true);
  } else {
      tpu_bdc_fp_sub_C(
      local_mid_addr,//round even val - 1
      local_in_addr,
      tpu_cast(mean_scalar, idtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
      &_shape,
      NULL,
      NULL,
      idtype);

      // (in - mean) * scale
      scalar_t scale_scalar = {.f32 = scale};
      tpu_bdc_fp_mul_C(local_out_addr,
                        local_mid_addr,
                        tpu_cast(scale_scalar, idtype, DT_FP32, RM_HALF_AWAY_FROM_ZERO),
                        &_shape,
                        NULL,
                        NULL,
                        idtype);
  }
  if (odtype != idtype) {
    tpu_bdc_cast(
        local_res_addr,
        local_out_addr,
        &_shape,
        NULL,
        NULL,
        odtype,
        idtype == DT_UINT8 ? DT_INT16 : idtype,
        RM_HALF_AWAY_FROM_ZERO);
  }
  tpu_gdma_cpy_L2S(output_global_addr, odtype != idtype ? local_res_addr : local_out_addr , &_shape, NULL, NULL, odtype);
}