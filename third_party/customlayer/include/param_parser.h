#pragma once
#include "api_common.h"
#include "backend_custom_param.h"

static absadd_param_t absadd_parse_param(const void* param) {
    absadd_param_t absadd_param = {0};
    absadd_param.b_val = ((custom_param_t *)param)[0].float_t;
    return absadd_param;
}

static ceiladd_param_t ceiladd_parse_param(const void* param) {
    ceiladd_param_t ceiladd_param = {0};
    ceiladd_param.b_val = ((custom_param_t *)param)[0].float_t;
    return ceiladd_param;
}

static crop_param_t crop_parse_param(const void* param) {
    crop_param_t crop_param = {0};
    crop_param.hoffset = ((custom_param_t*)param)[0].int_t;
    crop_param.woffset = ((custom_param_t*)param)[1].int_t;
    crop_param.hnew = ((custom_param_t*)param)[2].int_t;
    crop_param.wnew = ((custom_param_t*)param)[3].int_t;
    return crop_param;
}

static swapchannel_param_t swapchannel_parse_param(const void* param) {
    swapchannel_param_t sc_param = {0};
    for (int i = 0; i < 3; i++) {
        sc_param.order[i] = ((custom_param_t *)param)[0].int_arr_t[i];
    }
    return sc_param;
}

static preprocess_param_t preprocess_parse_param(const void* param) {
    preprocess_param_t sc_param = {0};
    sc_param.scale = ((custom_param_t *)param)[0].float_t;
    sc_param.mean = ((custom_param_t *)param)[1].float_t;
    sc_param.type = ((custom_param_t *)param)[2].int_t;
    return sc_param;
}

