#ifndef BACKEND_CUSTOM_PARAM_H
#define BACKEND_CUSTOM_PARAM_H

typedef struct {
  int int_t;
  float float_t;
  // max size of int and float array is set as 16
  int int_arr_t[16];
  float float_arr_t[16];
} custom_param_t;

// start defining your custom op param from here
typedef struct swapchannel_param {
  int order[3];
} swapchannel_param_t;

typedef struct {
  float b_val;
} absadd_param_t;

typedef struct {
  float b_val;
} ceiladd_param_t;

#endif
