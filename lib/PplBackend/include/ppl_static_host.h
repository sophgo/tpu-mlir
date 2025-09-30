#ifndef PPL_STATIC_HOST_H
#define PPL_STATIC_HOST_H
#include <cstdint>
// #include "tpu_defs.h"
#include "tpu_mlir/Backend/BM168x/Param.h"

typedef tensor_spec_t local_tensor_spec_t;
typedef enum {
  /* 3D group if this group has CONV3D/DECONV3D/POOL3D
   * for 1684 float32, data in local memory storage as {d * n, c, h, w}
   * for 1684 int8, data in local memory storage as {n, d * c, h, w}
   * for 1684X, data in local memory storage as {d * n, c, h, w}
   * data in global memory always storage as {n, c, d, h, w}
   * group_type < 8, because 1684 dynamic compile reserved `3bit` for group_type
   */
  FW_GROUP_NORMAL = 0,
  GROUP_3D = 1,
  GROUP_MM_OPT3 = 5
} group_type_t;

typedef struct {
  int n, c, d, h, w;
} dim5;
typedef struct {
  int n, c, h, w;
} dim4;
typedef struct {
  int h, w;
} dim2;
typedef struct {
  int top, bottom, left, right;
} padding_t;
typedef struct {
  int start, end;
} range_t;

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

  int32_t hw_margins_opA;
  int32_t hw_margins_opB;
} local_sec_info_t;

static inline bool get_is_true3d(int group_type, int dims) {
  return group_type == GROUP_3D && dims > 4;
}

static inline void parse_NCDHW(int group_type, const int *shape, int dims,
                               dim5 *out_shape) {
  out_shape->n = dims > 0 ? shape[0] : 1;
  out_shape->c = dims > 1 ? shape[1] : 1;
  const int is_true3d = get_is_true3d(group_type, dims);
  out_shape->d = is_true3d ? shape[2] : 1;
  out_shape->h = is_true3d ? shape[3] : (dims > 2 ? shape[2] : 1);
  out_shape->w = 1;
  for (int i = is_true3d ? 4 : 3; i < dims; i++)
    out_shape->w *= shape[i];
}
// parse as NCDHW
static inline void
parse_input_slice_shape_with_d(const local_sec_info_t *sec_info,
                               const local_tensor_spec_t *spec,
                               int slice_shape[5]) {
  dim5 shape_5d;
  parse_NCDHW(sec_info->group_type, spec->shape, spec->dims, &shape_5d);
  slice_shape[0] = shape_5d.n != 1 ? sec_info->n_slice : 1;
  slice_shape[1] = shape_5d.c != 1 ? sec_info->c_slice : 1;
  slice_shape[2] = shape_5d.d != 1 ? sec_info->d_slice : 1;
  slice_shape[3] = shape_5d.h != 1 ? sec_info->h_slice : 1;
  slice_shape[4] = shape_5d.w != 1 ? sec_info->w_slice : 1;
}

static inline void
parse_output_slice_shape_with_d(const local_sec_info_t *sec_info,
                                const local_tensor_spec_t *spec,
                                int slice_shape[5]) {
  dim5 shape_5d;
  parse_NCDHW(sec_info->group_type, spec->shape, spec->dims, &shape_5d);
  slice_shape[0] = shape_5d.n != 1 ? sec_info->out_n_slice : 1;
  slice_shape[1] = shape_5d.c != 1 ? sec_info->c_slice : 1;
  slice_shape[2] = shape_5d.d != 1 ? sec_info->d_slice : 1;
  slice_shape[3] = shape_5d.h != 1 ? sec_info->out_h_slice : 1;
  slice_shape[4] = shape_5d.w != 1 ? sec_info->out_w_slice : 1;
}
// parse as NCHW
static inline void tpu_local_shape_5d_to_4d(const dim5 *shape_5d,
                                            dim4 *shape_4d) {
  shape_4d->n = shape_5d->n * shape_5d->d;
  shape_4d->c = shape_5d->c;
  shape_4d->h = shape_5d->h;
  shape_4d->w = shape_5d->w;
}
static inline void parse_input_slice_shape(const local_sec_info_t *sec_info,
                                           const local_tensor_spec_t *spec,
                                           int slice_shape[4]) {
  int slice_shape_5d[5];
  parse_input_slice_shape_with_d(sec_info, spec, slice_shape_5d);
  tpu_local_shape_5d_to_4d((dim5 *)slice_shape_5d, (dim4 *)slice_shape);
}

static inline void parse_output_slice_shape(const local_sec_info_t *sec_info,
                                            const local_tensor_spec_t *spec,
                                            int slice_shape[4]) {
  int slice_shape_5d[5];
  parse_output_slice_shape_with_d(sec_info, spec, slice_shape_5d);
  tpu_local_shape_5d_to_4d((dim5 *)slice_shape_5d, (dim4 *)slice_shape);
}
#endif