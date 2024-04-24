#pragma once
#include "common.h"
#include "tpu_kernel.h"
#include "api_utils.h"
#include "interface_custom_ops.h"

void tpu_impl_absadd_global(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    const int *shape,
    float b_val,
    data_type_t dtype);

void tpu_impl_preprocess_global(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    const int *shape,
    float scale,
    float mean,
    data_type_t idtype,
    data_type_t odtype);

void tpu_impl_ceiladd_global(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    const int *shape,
    float b_val,
    data_type_t dtype);

void tpu_impl_swapchannel_global(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    const int *shape,
    const int *order,
    data_type_t dtype);

void tpu_impl_crop_global(
    global_addr_t input_global_addr,
    global_addr_t output_global_addr,
    const int *shape,
    int hoffset,
    int woffset,
    int hnew,
    int wnew,
    data_type_t dtype);

int get_absadd_local_bfsz(
    const int *shape,
    data_type_t dtype);

int get_ceiladd_local_bfsz(
    const int *shape,
    data_type_t dtype);

void tpu_impl_absadd_local(
    local_addr_t input_addr,
    local_addr_t output_addr,
    local_addr_t buffer_addr,
    const int *shape,
    float b_val,
    data_type_t dtype);

void tpu_impl_ceiladd_local(
    local_addr_t input_addr,
    local_addr_t output_addr,
    local_addr_t buffer_addr,
    const int *shape,
    float b_val,
    data_type_t dtype);

void tpu_impl_crop_local(
    local_addr_t input_addr,
    local_addr_t output_addr,
    const int *shape,
    int hoffset,
    int woffset,
    int hnew,
    int wnew,
    data_type_t dtype);
