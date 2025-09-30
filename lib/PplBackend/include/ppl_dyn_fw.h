// DO NOT EDIT
#ifndef PPL_DYN_FW_H
#define PPL_DYN_FW_H
#include "common.h"
#include "tpu_kernel.h"
#include <stddef.h>
#include <stdint.h>

/*============================================================================*/
/*========================= edit PPL_FW_LAYER_TYPE here ======================*/
/*============================================================================*/
#define PPL_FW_LAYER_START 10000
typedef enum ppl_fw_layer_type {
  PPL_FW_ADD_CONST = PPL_FW_LAYER_START,
  PPL_FW_FLASH_ATTENTION = 10001,
  PPL_FW_FLASH_ATTENTION_HEIGH_PRECISION = 10002,
  PPL_FW_INTERP_LINEAR = 10003,
  PPL_FW_INTERP_NEAREST = 10004,
  PPL_FW_LAYER_TYPE_UNKNOWN,
} PPL_FW_LAYER_TYPE_T;

enum DynamicTensorType {
  DYNAMIC_NEURON = 0,
  DYNAMIC_COEFF = 1,
  DYNAMIC_SHAPE = 2
};

/*============================================================================*/
/*========================= code borrowed from TPU1686 =======================*/
/*============================================================================*/

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

// dynamic local ref relate
typedef enum fw_data_type {
  FW_DTYPE_FP32 = 0,
  FW_DTYPE_FP16 = 1,
  FW_DTYPE_INT8 = 2,
  FW_DTYPE_UINT8 = 3,
  FW_DTYPE_INT16 = 4,
  FW_DTYPE_UINT16 = 5,
  FW_DTYPE_INT32 = 6,
  FW_DTYPE_UINT32 = 7,
  FW_DTYPE_BFP16 = 8,
} FW_DATA_TYPE_T;

typedef struct local_tensor_ref {
  int type;
  int32_t id;
  int pad_h_top;
  int h_idx;
  int h_slice;
  int pad_w_left;
  int w_idx;
  int w_slice;
  int consume_num;
  int sign;
  local_tensor_spec_t *spec;
} local_tensor_ref_t;

typedef struct {
  int id;
  int pad_h_top;
  int h_idx;
  int h_slice;
  int pad_w_left;
  int w_idx;
  int w_slice;
  int dims;
  int shape[MAX_SHAPE_DIMS];
  u8 consume_num;
  u8 sign;
  int elem_num; // record the real element number. If 0, do not need to count
                // elem num
  FW_DATA_TYPE_T dtype;
} fw_local_tensor_info_t;

typedef struct {
  u32 reference_id;
  u32 local_offset;
  u32 real_h_slice;
  u32 real_c_slice;
  fw_local_tensor_info_t info;
} local_output_info_t;

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

typedef struct dynamic_local_tensor_info {
  int input_num;
  local_tensor_ref_t *in_refs;
  local_tensor_spec_t *in_specs;
  int output_num;
  local_tensor_ref_t *out_refs;
  local_tensor_spec_t *out_specs;
} dynamic_local_tensor_info_t;

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

u64 addr_relocate(void *ctx, u64 addr, int mode);
void set_out_info(local_output_info_t *top, local_tensor_ref_t *in_ref,
                  local_tensor_ref_t *out_ref, local_sec_info_t *sec_info);
/*============================================================================*/
/*============================== init func map ===============================*/
/*============================================================================*/
extern void *ppl_global_func_map[];
extern void *ppl_local_func_map[];

#define GLOBAL_FUNC_BIND(t, v)                                                 \
  do {                                                                         \
    const int index = (t - PPL_FW_LAYER_START);                                \
    if (index >= 0 &&                                                          \
        index < (PPL_FW_LAYER_TYPE_UNKNOWN - PPL_FW_LAYER_START)) {            \
      ppl_global_func_map[index] = (v);                                        \
    }                                                                          \
  } while (0)

#define LOCAL_FUNC_BIND(t, v)                                                  \
  do {                                                                         \
    const int index = (t - PPL_FW_LAYER_START);                                \
    if (index >= 0 &&                                                          \
        index < (PPL_FW_LAYER_TYPE_UNKNOWN - PPL_FW_LAYER_START)) {            \
      ppl_local_func_map[index] = (v);                                         \
    }                                                                          \
  } while (0)

#define REGISTER_PPL_DYN_OP(op_type, global_func, local_func)                  \
  __attribute__((constructor)) static void init_##op_type##_bindings(void) {   \
    GLOBAL_FUNC_BIND(op_type, global_func);                                    \
    LOCAL_FUNC_BIND(op_type, local_func);                                      \
  }
#endif
