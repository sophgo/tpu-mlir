#include "ppl_helper.h"

void set_id_node(CMD_ID_NODE *pid_node) {
  if (pid_node == NULL) {
    return;
  }
  if (pid_node->in_parallel_state) {
    tpu_set_parallel_id_node(pid_node, NULL);
  } else {
    tpu_set_id_node(pid_node);
  }
  tpu_disable_check_id_node();
}

void get_id_node(CMD_ID_NODE *pid_node) {
  if (!pid_node)
    return;
  if (pid_node->in_parallel_state) {
    tpu_get_parallel_id_node(pid_node, NULL);
  } else {
    tpu_get_id_node(pid_node);
  }
  tpu_enable_check_id_node();
}

static const char *data_type_to_string(data_type_t dtype) {
  switch (dtype) {
  case DT_INT8:
    return "DT_INT8";
  case DT_UINT8:
    return "DT_UINT8";
  case DT_INT16:
    return "DT_INT16";
  case DT_UINT16:
    return "DT_UINT16";
  case DT_FP16:
    return "DT_FP16";
  case DT_BFP16:
    return "DT_BFP16";
  case DT_INT32:
    return "DT_INT32";
  case DT_UINT32:
    return "DT_UINT32";
  case DT_FP32:
    return "DT_FP32";
  case DT_INT4:
    return "DT_INT4";
  case DT_UINT4:
    return "DT_UINT4";
  case DT_FP8E5M2:
    return "DT_FP8E5M2";
  case DT_FP8E4M3:
    return "DT_FP8E4M3";
  case DT_FP20:
    return "DT_FP20";
  case DT_TF32:
    return "DT_TF32";
  default:
    return "UNKNOWN_DTYPE";
  }
}

static const char *align_mode_to_string(int align_mode) {
  switch (align_mode) {
  case 0:
    return "CONTINUOUS";
  case 1:
    return "TPU_ALIGN";
  case 2:
    return "TPU_COMPACT";
  case 3:
    return "TPU_ROW_ALIGN";
  case 4:
    return "NONE_ALIGN";
  default:
    return "UNKNOWN_ALIGN_MODE";
  }
}

static const char *tensor_mode_to_string(int mode) {
  switch (mode) {
  case 0:
    return "Local_Tensor";
  case 1:
    return "L2_Tensor";
  case 2:
    return "Global_Tensor";
  default:
    return "UNKNOWN_TENSOR_MODE";
  }
}

char *__ppl_to_string(const void *tensor) {
  const char *base = (const char *)tensor;
  // shape
  const dim4 *shape = (const dim4 *)(base + 0);
  // stride
  const dim4 *stride = (const dim4 *)(base + sizeof(dim4));
  // addr
  const global_addr_t *addr = (const global_addr_t *)(base + 2 * sizeof(dim4));
  // dtype
  const data_type_t *dtype =
      (const data_type_t *)(base + 2 * sizeof(dim4) + sizeof(global_addr_t));
  // mode
  const int *mode = (const int *)(base + 2 * sizeof(dim4) +
                                  sizeof(global_addr_t) + sizeof(data_type_t));
  // align_mode
  const int *align_mode =
      (const int *)(base + 2 * sizeof(dim4) + sizeof(global_addr_t) +
                    sizeof(data_type_t) + sizeof(int));
  // offset
  const int *offset =
      (const int *)(base + 2 * sizeof(dim4) + sizeof(global_addr_t) +
                    sizeof(data_type_t) + 3 * sizeof(int));
  // unsigned_flag
  const bool *unsigned_flag =
      (const bool *)(base + 2 * sizeof(dim4) + sizeof(global_addr_t) +
                     sizeof(data_type_t) + 4 * sizeof(int));
  // default_stride
  const bool *default_stride =
      (const bool *)(base + 2 * sizeof(dim4) + sizeof(global_addr_t) +
                     sizeof(data_type_t) + 4 * sizeof(int) + sizeof(bool));

  static char buffer[512];
  snprintf(buffer, sizeof(buffer),
           "\n\tshape: {%d, %d, %d, %d}\n"
           "\tstride: {%d, %d, %d, %d}\n"
           "\taddr: %llu\n"
           "\tdtype: %s\n"
           "\tmode: %s\n"
           "\talign_mode: %s\n"
           "\toffset: %d\n"
           "\tunsigned_flag: %s\n"
           "\tdefault_stride: %s\n",
           shape->n, shape->c, shape->h, shape->w, stride->n, stride->c,
           stride->h, stride->w, *addr, data_type_to_string(*dtype),
           tensor_mode_to_string(*mode), align_mode_to_string(*align_mode),
           *offset, *unsigned_flag ? "true" : "false",
           *default_stride ? "true" : "false");
  return buffer;
}

void print_value(const void *data, data_type_t dtype) {
  if (dtype == DT_INT32) {
    printf("%d  ", *(int32_t *)data);
  } else if (dtype == DT_UINT32) {
    printf("%u  ", *(uint32_t *)data);
  } else if (dtype == DT_FP32) {
    printf("%f  ", *(float *)data);
  } else if (dtype == DT_INT16) {
    printf("%hd  ", *(int16_t *)data);
  } else if (dtype == DT_UINT16) {
    printf("%hu  ", *(uint16_t *)data);
  } else if (dtype == DT_INT8) {
    printf("%d  ", *(int8_t *)data);
  } else if (dtype == DT_UINT8) {
    printf("%u  ", *(uint8_t *)data);
  } else if (dtype == DT_UINT4) {
    uint8_t val = *(uint8_t *)data & 0x0F;
    printf("%u  ", val);
  } else if (dtype == DT_INT4) {
    int8_t val = (*(int8_t *)data << 4) >> 4;
    printf("%d  ", val);
  } else if (dtype == DT_FP8E5M2) {
    scalar_t v;
    v.f8e5m2 = *(float8e5m2 *)data;
    v = tpu_cast(v, DT_FP32, dtype, RM_HALF_TO_EVEN);
    printf("%f  ", v.f32);
  } else if (dtype == DT_FP8E4M3) {
    scalar_t v;
    v.f8e3m4 = *(float8e4m3 *)data;
    v = tpu_cast(v, DT_FP32, dtype, RM_HALF_TO_EVEN);
    printf("%f  ", v.f32);
  } else if (dtype == DT_FP16) {
    scalar_t v;
    v.f16 = *(float16 *)data;
    v = tpu_cast(v, DT_FP32, dtype, RM_HALF_AWAY_FROM_ZERO);
    printf("%f  ", v.f32);
  } else if (dtype == DT_BFP16) {
    scalar_t v;
    v.bf16 = *(bfloat16 *)data;
    v = tpu_cast(v, DT_FP32, dtype, RM_HALF_AWAY_FROM_ZERO);
    printf("%f  ", v.f32);
  }
}

void print_local_mem_data(local_addr_t local_offset, int start_idx,
                          const dim4 *shape, const dim4 *stride,
                          data_type_t dtype, int lane_num) {
#if defined(__sg2260e__) || defined(__sg2262__)
  rvt_sync_i(0xdeadbeef, 0);
#else
  tpu_poll();
#endif
  if (!stride) {
    dim4 true_stride;
    tpu_aligned_stride(&true_stride, start_idx, shape, dtype);
    print_local_mem_data(local_offset, start_idx, shape, &true_stride, dtype,
                         lane_num);
    return;
  }
  unsigned char *ptrs[lane_num];
  for (int i = 0; i < lane_num; i++) {
    ptrs[i] = tpu_local_mem_addr(i, local_offset);
  }
  printf("\nLOCAL offset=0x%x, idx=%d, shape=(%d %d %d %d), "
         "stride=(%d,%d,%d,%d)\n",
         local_offset, start_idx, shape->n, shape->c, shape->h, shape->w,
         stride->n, stride->c, stride->h, stride->w);
  dim4 offset;
  int type_len = tpu_data_type_size(dtype);
  unsigned char *ptr = NULL;
  for (int n = 0; n < shape->n; n++) {
    offset.n = n * stride->n;
    for (int c = 0; c < shape->c; c++) {
      ptr = ptrs[(c + start_idx) % lane_num];
      offset.c = ((c + start_idx) / lane_num) * stride->c;
      for (int h = 0; h < shape->h; h++) {
        offset.h = h * stride->h;
        printf("(%d, %d, %d):\n", n, c, h);
        for (int w = 0; w < shape->w; w++) {
          offset.w = w * stride->w;
          print_value(ptr + (offset.n + offset.c + offset.h + offset.w) *
                                type_len,
                      dtype);
        }
        printf("\n");
      }
    }
  }
}

void print_global_mem_data(global_addr_t addr, const dim4 *shape,
                           const dim4 *stride, data_type_t dtype,
                           bool is_global) {
#if defined(__sg2260e__) || defined(__sg2262__)
  rvt_sync_i(0xdeadbeef, 0);
  // tpu_poll();
#else
  tpu_poll();
#endif
  if (!stride) {
    dim4 true_stride;
    tpu_continuous_stride(&true_stride, shape);
    print_global_mem_data(addr, shape, &true_stride, dtype, is_global);
    return;
  }
  printf("GLOBAL addr=0x%llx, shape=(%d %d %d %d), stride=(%d,%d,%d,%d)\n",
         addr, shape->n, shape->c, shape->h, shape->w, stride->n, stride->c,
         stride->h, stride->w);
  unsigned char *ptr = NULL;
  if (is_global) {
    ptr = tpu_global_mem_addr(addr);
  } else {
    ptr = tpu_l2_sram_addr(addr);
  }
  dim4 offset;
  int type_len = tpu_data_type_size(dtype);
  for (int n = 0; n < shape->n; n++) {
    offset.n = n * stride->n;
    for (int c = 0; c < shape->c; c++) {
      offset.c = c * stride->c;
      for (int h = 0; h < shape->h; h++) {
        offset.h = h * stride->h;
        printf("(%d,%d,%d): ", n, c, h);
        for (int w = 0; w < shape->w; w++) {
          offset.w = w * stride->w;
          print_value(ptr + (offset.n + offset.c + offset.h + offset.w) *
                                type_len,
                      dtype);
        }
        printf("\n");
      }
    }
  }
}
