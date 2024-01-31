#pragma once
#include "api_common.h"

static inline bool get_is_true3d(int group_type, int dims) {
    return group_type == GROUP_3D && dims > 4;
}

static inline void parse_NCDHW(int group_type, const int *shape, int dims, dim5* out_shape) {
    out_shape->n = dims > 0 ? shape[0] : 1;
    out_shape->c = dims > 1 ? shape[1] : 1;
    const int is_true3d = get_is_true3d(group_type, dims);
    out_shape->d = is_true3d ? shape[2] : 1;
    out_shape->h = is_true3d ? shape[3] : (dims>2 ? shape[2] : 1);
    out_shape->w = 1;
    for(int i = is_true3d ? 4 : 3; i < dims; i++)
        out_shape->w *= shape[i];
}

static inline void parse_NCHW(const int *shape, int dims, dim4* out_shape) {
    dim5 out_shape_5d;
    parse_NCDHW(GROUP_NORMAL, shape, dims, &out_shape_5d);
    tpu_local_shape_5d_to_4d(&out_shape_5d, out_shape);
}

// parse as NCDHW
static inline void parse_input_slice_shape_with_d(
    const local_sec_info_t *sec_info,
    const local_tensor_spec_t *spec,
    int slice_shape[5])
{
    dim5 shape_5d;
    parse_NCDHW(sec_info->group_type, spec->shape, spec->dims, &shape_5d);
    slice_shape[0] = shape_5d.n != 1 ? sec_info->n_slice : 1;
    slice_shape[1] = shape_5d.c != 1 ? sec_info->c_slice : 1;
    slice_shape[2] = shape_5d.d != 1 ? sec_info->d_slice : 1;
    slice_shape[3] = shape_5d.h != 1 ? sec_info->h_slice : 1;
    slice_shape[4] = shape_5d.w != 1 ? sec_info->w_slice : 1;
}

// parse as NCDHW
static inline void parse_output_slice_shape_with_d(
    const local_sec_info_t *sec_info,
    const local_tensor_spec_t *spec,
    int slice_shape[5])
{
    dim5 shape_5d;
    parse_NCDHW(sec_info->group_type, spec->shape, spec->dims, &shape_5d);
    slice_shape[0] = shape_5d.n != 1 ? sec_info->out_n_slice : 1;
    slice_shape[1] = shape_5d.c != 1 ? sec_info->c_slice : 1;
    slice_shape[2] = shape_5d.d != 1 ? sec_info->d_slice : 1;
    slice_shape[3] = shape_5d.h != 1 ? sec_info->out_h_slice : 1;
    slice_shape[4] = shape_5d.w != 1 ? sec_info->out_w_slice : 1;
}

// parse as NCHW
static inline void parse_input_slice_shape(
    const local_sec_info_t *sec_info,
    const local_tensor_spec_t *spec,
    int slice_shape[4])
{
    int slice_shape_5d[5];
    parse_input_slice_shape_with_d(sec_info, spec, slice_shape_5d);
    tpu_local_shape_5d_to_4d((dim5*)slice_shape_5d, (dim4*)slice_shape);
}

// parse as NCHW
static inline void parse_output_slice_shape(
    const local_sec_info_t *sec_info,
    const local_tensor_spec_t *spec,
    int slice_shape[4])
{
    int slice_shape_5d[5];
    parse_output_slice_shape_with_d(sec_info, spec, slice_shape_5d);
    tpu_local_shape_5d_to_4d((dim5*)slice_shape_5d, (dim4*)slice_shape);
}
