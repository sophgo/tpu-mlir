#include "tpu_impl_custom_ops.h"

void tpu_impl_crop_global(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    const int *shape,
    int hoffset,
    int woffset,
    int hnew,
    int wnew,
    data_type_t dtype)
{
    TPUKERNEL_ASSERT(0 < hnew);
    TPUKERNEL_ASSERT(0 < wnew);
    TPUKERNEL_ASSERT(0 <= hoffset && hoffset + hnew <= shape[2]);
    TPUKERNEL_ASSERT(0 <= woffset && woffset + wnew <= shape[3]);
    int data_size = tpu_data_type_size(dtype);
    dim4 old_shape = {.n=shape[0], .c=shape[1], .h=shape[2], .w=shape[3]};
    dim4 old_stride = {0};
    tpu_continuous_stride(&old_stride, &old_shape);
    dim4 new_shape = {.n=shape[0], .c=shape[1], .h=hnew, .w=wnew};
    tpu_gdma_cpy_S2S(
        output_global_addr,
        input_global_addr + (hoffset * old_stride.h + woffset) * data_size,
        &new_shape,
        NULL,
        &old_stride,
        dtype);
}
