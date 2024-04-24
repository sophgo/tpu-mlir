#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "tpu_defs.h"

#define MAX_SHAPE_DIMS  8
typedef struct local_tensor_spec {
    uint64_t addr;
    int32_t dtype;
    int32_t dims;
    int32_t shape[MAX_SHAPE_DIMS];
    uint8_t consume_num;
    int *host_data;
    int elem_num;
} tensor_spec_t;

typedef tensor_spec_t local_tensor_spec_t;
typedef tensor_spec_t global_tensor_spec_t;

typedef struct local_sec_info {
    int32_t group_type;

    int32_t n_slice;
    int32_t out_n_slice;

    int32_t d_slice;

    int32_t is_h_split;
    int32_t h_idx;
    int32_t h_slice;
    int32_t out_h_idx;
    int32_t out_h_slice;

    int32_t is_w_split;
    int32_t w_idx;
    int32_t w_slice;
    int32_t out_w_idx;
    int32_t out_w_slice;

    int32_t is_c_split;
    int32_t c_idx;
    int32_t c_slice;
} local_sec_info_t;

typedef struct {
    int h_idx;
    int h_slice;
} tensor_slice_t;

typedef union {
    int int_t;
    float float_t;
    // max size of int and float array is set as 16
    int int_arr_t[16];
    float float_arr_t[16];
} custom_param_t;

typedef enum {
    /* 3D group if this group has CONV3D/DECONV3D/POOL3D
     * for 1684X, data in local memory storage as {d * n, c, h, w}
     * data in global memory always storage as {n, c, d, h, w}
     */
    GROUP_NORMAL = 0,
    GROUP_3D = 1,
} group_type_t;

static inline void* get_real_param_ptr(const void* param) {
    return (void*)((custom_param_t*)param + 1);
}

#define PARSE_PARAM(op_name, real_param, raw_param) \
    op_name##_param_t real_param = op_name##_parse_param(get_real_param_ptr(raw_param));

static inline int get_local_buffer_addr(const void* param) {
    return ((int*)param)[0];
}

static inline int64_t get_global_buffer_addr(const void* param) {
    return ((int64_t*)param)[0];
}
