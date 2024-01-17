#pragma once

#include <stdint.h>

#define MAX_SHAPE_DIMS 8

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
} local_sec_info_t;

typedef union {
    int int_t;
    float float_t;
    // max size of int and float array is set as 16
    int int_arr_t[16];
    float float_arr_t[16];
} custom_param_t;
