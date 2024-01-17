#include "tpu_impl_custom_ops.h"

void tpu_impl_swapchannel_global(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    const int *shape,
    const int *order,
    data_type_t dtype)
{
    dim4 channel_shape = {.n = shape[0], .c = shape[1], .h = shape[2], .w = shape[3]};
    dim4 stride = {0};
    stride.w = 1, stride.h = channel_shape.w;
    stride.c = stride.h * channel_shape.h;
    stride.n = stride.c * channel_shape.c;
    channel_shape.c = 1;
    int data_size = tpu_data_type_size(dtype);
    int offset = channel_shape.w * channel_shape.h * data_size;
    for (int i = 0; i < 3; i++) {
        tpu_gdma_cpy_S2S(
            output_global_addr + i * offset,
            input_global_addr + order[i] * offset,
            &channel_shape,
            &stride,
            &stride,
            dtype);
    }
}
