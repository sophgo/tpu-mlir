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

typedef union {
    int int_t;
    float float_t;
    // max size of int and float array is set as 16
    int int_arr_t[16];
    float float_arr_t[16];
} custom_param_t;

static inline void* get_real_param_ptr(const void* param) {
    return (void*)((custom_param_t*)param + 1);
}

#define PARSE_PARAM(op_name, real_param, raw_param) \
    op_name##_param_t real_param = op_name##_parse_param(get_real_param_ptr(raw_param));
