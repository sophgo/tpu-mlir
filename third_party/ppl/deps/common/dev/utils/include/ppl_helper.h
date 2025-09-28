#ifndef __PPL_HELPER__
#define __PPL_HELPER__

#include "common.h"
#include "tpu_kernel.h"
#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#define ALIGN_UP(val, align) (val + align - 1) & ~(align - 1)

typedef struct {
  dim4 shape;
  int stride[4];
  global_addr_t addr;
  data_type_t dtype;
  int mode;
  int align_mode;
  long size;
  long offset;
  bool unsigned_flag;
  bool default_stride;
} ppl_tensor_t;

void set_id_node(CMD_ID_NODE *pid_node);

void get_id_node(CMD_ID_NODE *pid_node);

char *__ppl_to_string(const ppl_tensor_t *tensor);

void print_local_mem_data(local_addr_t local_offset, int start_idx,
                          const dim4 *shape, const dim4 *stride,
                          data_type_t dtype, int lane_num, bool is_rv);

void print_tensor_data(void *tensor, bool is_rv);

static void check_tensor_overflow(void *__parent, void *__sub_tensor,
                                  const char *var_name) {
  ppl_tensor_t *parent = (ppl_tensor_t *)__parent;
  ppl_tensor_t *sub_tensor = (ppl_tensor_t *)__sub_tensor;
  size_t parent_addr = parent->mode == 1
                           ? parent->addr - tpu_l2_sram_get_start_addr()
                           : parent->addr;
  size_t sub_tensor_addr = sub_tensor->mode == 1
                               ? sub_tensor->addr - tpu_l2_sram_get_start_addr()
                               : sub_tensor->addr % LOCAL_MEM_SIZE;

  if (sub_tensor_addr < parent_addr) {
    printf("[ERROR] tensor %s start addr:[%ld] is smallest than parent start "
           "addr:[%ld]\n",
           var_name, sub_tensor_addr, parent_addr);
    assert(0);
  }
  if (tpu_data_type_bits(sub_tensor->dtype) < 8) {
    return;
  }
  size_t sub_size = 0;
  if (sub_tensor->align_mode == 4) {
    size_t max_size = 1;
    max_size += (sub_tensor->shape.n - 1) * sub_tensor->stride[0];
    int c_shape = sub_tensor->shape.c;
    if (sub_tensor->mode == 0) {
      c_shape = (c_shape + NPU_NUM - 1) / NPU_NUM;
    }
    max_size += (c_shape - 1) * sub_tensor->stride[1];
    max_size += (sub_tensor->shape.h - 1) * sub_tensor->stride[2];
    max_size += (sub_tensor->shape.w - 1) * sub_tensor->stride[3];
    max_size *= tpu_data_type_size(sub_tensor->dtype);
    sub_size = max_size;
  } else {
    sub_size = sub_tensor->shape.n * sub_tensor->stride[0] *
               tpu_data_type_size(sub_tensor->dtype);
  }
  int parent_size = 0;
  if (parent->size == 0) {
    parent_size =
        parent->shape.n * parent->stride[0] * tpu_data_type_size(parent->dtype);
  } else {
    parent_size = parent->size;
  }
  if (sub_tensor_addr + sub_size > parent_addr + parent_size) {
    printf("[ERROR] tensor %s memory range:[%ld, %ld] is largest than parent "
           "memory "
           "range:[%ld, %ld]\n",
           var_name, sub_tensor_addr, sub_tensor_addr + sub_size, parent_addr,
           parent_addr + parent_size);
    printf("[ERROR] tensor %s shape:[%d, %d, %d, %d] parent shape:[%d, %d, %d, "
           "%d]\n",
           var_name, sub_tensor->shape.n, sub_tensor->shape.c,
           sub_tensor->shape.h, sub_tensor->shape.w, parent->shape.n,
           parent->shape.c, parent->shape.h, parent->shape.w);
    assert(0);
  }
}

void print_global_mem_data(global_addr_t addr, const dim4 *shape,
                           const dim4 *stride, data_type_t dtype,
                           bool is_global, bool is_rv);
void __attribute__((weak)) fw_log(char *fmt, ...) {}

static int compare_float(const void *a, const void *b, bool ascending) {
  float fa = *((float *)a);
  float fb = *((float *)b);

  if (ascending) {
    if (fabs(fa - fb) < 1e-5) {
      return 0;
    }
    return (fa > fb) ? 1 : -1;
  } else {
    if (fabs(fa - fb) < 1e-5) {
      return 0;
    }
    return (fa < fb) ? 1 : -1;
  }
}
static int compare_float_asc(const void *a, const void *b) {
  return compare_float(a, b, true);
}
static int compare_float_desc(const void *a, const void *b) {
  return compare_float(a, b, false);
}

#define COMPARE_FUNC_INT(type)                                                 \
  static int compare_##type(const void *a, const void *b, bool ascending) {    \
    type ia = *((type *)a);                                                    \
    type ib = *((type *)b);                                                    \
    if (ascending) {                                                           \
      return (ia > ib) - (ia < ib);                                            \
    } else {                                                                   \
      return (ib > ia) - (ib < ia);                                            \
    }                                                                          \
  }                                                                            \
  static int compare_##type##_desc(const void *a, const void *b) {             \
    return compare_##type(a, b, false);                                        \
  }                                                                            \
  static int compare_##type##_asc(const void *a, const void *b) {              \
    return compare_##type(a, b, true);                                         \
  }

COMPARE_FUNC_INT(int32_t)
COMPARE_FUNC_INT(uint32_t)
COMPARE_FUNC_INT(int16_t)
COMPARE_FUNC_INT(uint16_t)
COMPARE_FUNC_INT(int8_t)
COMPARE_FUNC_INT(uint8_t)

static inline bool is_l2sram_addr(u64 addr) {
  return ((addr >= tpu_l2_sram_get_start_addr()) &&
          (addr < tpu_l2_sram_get_start_addr() + L2_SRAM_SIZE));
}

static inline u64 get_global_ptr(u64 addr) {
  u64 ptr = 0;
  if (is_l2sram_addr(addr)) {
    ptr = (u64)(uintptr_t)tpu_l2_sram_addr(addr);
  } else {
    ptr = (u64)(uintptr_t)tpu_global_mem_addr(addr);
  }
  return ptr;
}

static data_type_t __ppl_get_dtype(int type) {
  data_type_t __dtype[] = {DT_FP32,
                           DT_FP32,
                           DT_FP16,
                           DT_BFP16,
                           DT_FP8E5M2,
                           DT_FP8E4M3,
                           DT_FP20,
                           DT_TF32,
                           DT_INT32,
                           DT_UINT32,
                           DT_INT16,
                           DT_UINT16,
                           DT_INT8,
                           DT_UINT8,
                           DT_INT4,
                           DT_UINT4
#if defined(__tpub_9_0__) || defined(__tpub_9_3__)
                           ,
                           DT_FP4};
#else
  };
#endif
  return __dtype[type];
}

#endif
