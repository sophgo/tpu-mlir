/*****************************************************************************
 *
 *    Copyright (c) 2016-2026 by Sophgo Technologies Inc. All rights reserved.
 *
 *    The material in this file is confidential and contains trade secrets
 *    of Sophgo Technologies Inc. This is proprietary information owned by
 *    Sophgo Technologies Inc. No part of this work may be disclosed,
 *    reproduced, copied, transmitted, or used in any way for any purpose,
 *    without the express written permission of Sophgo Technologies Inc.
 *
 *****************************************************************************/

#ifndef __BMRUNTIME_DEFINE_H__
#define __BMRUNTIME_DEFINE_H__

#include "bmlib_runtime.h"
#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

/* --------------------------------------------------------------------------*/
/* basic definitions */

/* bm_data_type_t holds the type for a scalar value */
typedef enum bm_data_type_e {
  BM_FLOAT32 = 0,
  BM_FLOAT16 = 1,
  BM_INT8 = 2,
  BM_UINT8 = 3,
  BM_INT16 = 4,
  BM_UINT16 = 5,
  BM_INT32 = 6,
  BM_UINT32 = 7,
  BM_BFLOAT16 = 8,
  BM_INT4 = 9,
  BM_UINT4 = 10,
} bm_data_type_t;

/* store mode definitions */
typedef enum bm_store_mode_e {
  BM_STORE_1N = 0, /* default, if not sure, use 0 */
  BM_STORE_2N = 1,
  BM_STORE_4N = 2,
} bm_store_mode_t;

/* bm_shape_t holds the shape info */
#define BM_MAX_DIMS_NUM 8
typedef struct bm_shape_s {
  int num_dims;
  int dims[BM_MAX_DIMS_NUM];
} bm_shape_t;

typedef struct bm_shape_ex_s {
  bm_shape_t shape;
  int        elem_num;
} bm_shape_ex_t;

/*
bm_tensor_t holds a multi-dimensional array of elements of a single data type
and tensor are in device memory */
typedef struct bm_tensor_s {
  bm_data_type_t dtype;
  bm_shape_t shape;
  bm_device_mem_t device_mem;
  bm_store_mode_t st_mode; /* user can set 0 as default store mode */
} bm_tensor_t;


typedef struct sg_tensor_s {
  bm_data_type_t dtype;
  bm_shape_t shape;
  sg_device_mem_t device_mem;
  bm_store_mode_t st_mode; /* user can set 0 as default store mode */
} sg_tensor_t;

/* --------------------------------------------------------------------------*/
/* network information structure */

/* bm_stage_info_t holds input shapes and output shapes; every network can contain one or more
 * stages */
typedef struct bm_stage_info_s {
  bm_shape_t* input_shapes;  /* input_shapes[0] / [1] / ... / [input_num-1] */
  bm_shape_t* output_shapes; /* output_shapes[0] / [1] / ... / [output_num-1] */
} bm_stage_info_t;

/* bm_tensor_info_t holds all information of one net.
 * scale for float type is 1.0 as default */
typedef struct bm_net_info_s {
  const char* name;              /* net name */
  bool is_dynamic;               /* dynamic or static */
  int input_num;                 /* number of inputs */
  char const** input_names;      /* input_names[0] / [1] / .../ [input_num-1] */
  bm_data_type_t* input_dtypes;  /* input_dtypes[0] / [1] / .../ [input_num-1] */
  float* input_scales;           /* input_scales[0] / [1] / .../ [input_num-1] */
  int output_num;                /* number of outputs */
  char const** output_names;     /* output_names[0] / [1] / .../ [output_num-1] */
  bm_data_type_t* output_dtypes; /* output_dtypes[0] / [1] / .../ [output_num-1] */
  float* output_scales;          /* output_scales[0] / [1] / .../ [output_num-1] */
  int stage_num;                 /* number of stages */
  bm_stage_info_t* stages;       /* stages[0] / [1] / ... / [stage_num-1] */
  size_t* max_input_bytes;       /* max_input_bytes[0]/ [1] / ... / [input_num-1] */
  size_t* max_output_bytes;      /* max_output_bytes[0] / [1] / ... / [output_num-1] */
  int* input_zero_point;         /* input_zero_point[0] / [1] / .../ [input_num-1] */
  int* output_zero_point;        /* output_zero_point[0] / [1] / .../ [input_num-1] */
} bm_net_info_t;

#if defined(__cplusplus)
}
#endif

#endif /* __BM_NET_H__ */
