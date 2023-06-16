#ifndef BACKEND_API_PARAM_H
#define BACKEND_API_PARAM_H

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
