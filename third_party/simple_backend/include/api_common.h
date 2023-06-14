#pragma once

#include <stdint.h>

#ifndef MAX_SHAPE_DIMS
#define MAX_SHAPE_DIMS 8
#endif

enum DynamicTensorType
{
    DYNAMIC_NEURON = 0,
    DYNAMIC_COEFF = 1,
    DYNAMIC_SHAPE = 2
};

static inline int dynamic_is_neuron(int type)
{
    return type == DYNAMIC_NEURON;
}

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

static unsigned long long global_tensor_length(global_tensor_spec_t* tensor){
  unsigned long long length = 1;
  for(int i=0; i<tensor->dims; i++){
    length *= tensor->shape[i];
  }
  return length;
}
