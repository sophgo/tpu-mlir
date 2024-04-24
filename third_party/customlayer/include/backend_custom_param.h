#ifndef BACKEND_CUSTOM_PARAM_H
#define BACKEND_CUSTOM_PARAM_H

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

typedef struct {
  int hoffset;
  int woffset;
  int hnew;
  int wnew;
} crop_param_t;

typedef struct {
  float scale;
  float mean;
  int type;
} preprocess_param_t;

#endif
