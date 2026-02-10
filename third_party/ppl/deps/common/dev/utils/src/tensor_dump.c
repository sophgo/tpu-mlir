#include "tensor_dump.h"
#include <cast.h>
#include "ppl_helper.h"

#if defined(__tpub_7_1_e__) || defined(__tpub_9_0__) || defined(__tpub_9_3__)
#include "rvt_api.h"
#endif


static int g_reset_done = 0;
static char g_reset_path[1024] = {0};

void ppl_helper_reset_npz(const char* npz_path) {
  if (!npz_path || !npz_path[0]) return;
  size_t n = strlen(npz_path);
  if (n >= sizeof(g_reset_path)) return;
  memcpy(g_reset_path, npz_path, n + 1);
  FILE* f = fopen(g_reset_path, "rb");
  if (f) { fclose(f); (void)remove(g_reset_path); }
  g_reset_done = 1;
}

static inline void ensure_reset_npz_once(const char* file_path) {
  if (!file_path || !file_path[0]) return;
  if (g_reset_done == 0) {
    size_t n = strlen(file_path);
    if (n < sizeof(g_reset_path)) {
      memcpy(g_reset_path, file_path, n + 1);
      FILE* f = fopen(g_reset_path, "rb");
      if (f) { fclose(f); (void)remove(g_reset_path); }
    }
    g_reset_done = 1;
  } else {
    if (g_reset_path[0] != '\0' && strcmp(g_reset_path, file_path) != 0) {
      size_t n = strlen(file_path);
      if (n < sizeof(g_reset_path)) {
        memcpy(g_reset_path, file_path, n + 1);
        FILE* f = fopen(g_reset_path, "rb");
        if (f) { fclose(f); (void)remove(g_reset_path); }
      }
    }
  }
}


static void map_dtype_to_cnpy(data_type_t dtype,
                              cn_dtype_class_t *out_cls,
                              size_t *out_elem_size,
                              int *needs_cast_to_fp32) {
  *needs_cast_to_fp32 = 0;
  switch (dtype) {
    case DT_FP32:
      *out_cls = CN_DTYPE_FLOAT; *out_elem_size = 4; break;
    case DT_INT32:
      *out_cls = CN_DTYPE_INT;   *out_elem_size = 4; break;
    case DT_UINT32:
      *out_cls = CN_DTYPE_UINT;  *out_elem_size = 4; break;
    case DT_INT16:
      *out_cls = CN_DTYPE_INT;   *out_elem_size = 2; break;
    case DT_UINT16:
      *out_cls = CN_DTYPE_UINT;  *out_elem_size = 2; break;
    case DT_INT8:
      *out_cls = CN_DTYPE_INT;   *out_elem_size = 1; break;
    case DT_UINT8:
      *out_cls = CN_DTYPE_UINT;  *out_elem_size = 1; break;

    case DT_FP16:
    case DT_BFP16:
    case DT_FP8E5M2:
    case DT_FP8E4M3:
    case DT_INT4:
    case DT_UINT4:
    case DT_FP20:
    case DT_TF32:
      *out_cls = CN_DTYPE_FLOAT; *out_elem_size = 4; *needs_cast_to_fp32 = 1; break;

    default:
      *out_cls = CN_DTYPE_UINT; *out_elem_size = 1; break;
  }
}

static int build_shape_string_from_dim4(char *out, size_t out_sz, const dim4 *shape) {
  size_t dims[4] = { (size_t)shape->n, (size_t)shape->c, (size_t)shape->h, (size_t)shape->w };
  size_t ndims = 0;
  if (dims[0] > 0) ndims = 4;
  return cnpy_build_shape(out, out_sz, dims, ndims);
}

static int gather_local_to_buffer(unsigned char *dst,
                                  local_addr_t local_offset,
                                  int start_idx,
                                  const dim4 *shape,
                                  const dim4 *stride,
                                  data_type_t dtype,
                                  int lane_num) {
  unsigned char *ptrs[lane_num];
  for (int i = 0; i < lane_num; i++) {
    ptrs[i] = tpu_local_mem_addr(i, local_offset);
  }
  dim4 offset;
  size_t raw_size = tpu_data_type_size(dtype);
  size_t write_pos = 0;

  for (int n = 0; n < shape->n; n++) {
    offset.n = n * stride->n;
    for (int c = 0; c < shape->c; c++) {
      unsigned char *lane_ptr = ptrs[(c + start_idx) % lane_num];
      offset.c = ((c + start_idx) / lane_num) * stride->c;
      for (int h = 0; h < shape->h; h++) {
        offset.h = h * stride->h;
        for (int w = 0; w < shape->w; w++) {
          offset.w = w * stride->w;
          const unsigned char *src = lane_ptr + (offset.n + offset.c + offset.h + offset.w) * raw_size;
          memcpy(dst + write_pos, src, raw_size);
          write_pos += raw_size;
        }
      }
    }
  }
  return 0;
}

static int gather_global_to_buffer(unsigned char *dst,
                                   unsigned char *base_ptr,
                                   const dim4 *shape,
                                   const dim4 *stride,
                                   data_type_t dtype) {
  dim4 offset;
  size_t raw_size = tpu_data_type_size(dtype);
  size_t write_pos = 0;

  for (int n = 0; n < shape->n; n++) {
    offset.n = n * stride->n;
    for (int c = 0; c < shape->c; c++) {
      offset.c = c * stride->c;
      for (int h = 0; h < shape->h; h++) {
        offset.h = h * stride->h;
        for (int w = 0; w < shape->w; w++) {
          offset.w = w * stride->w;
          const unsigned char *src = base_ptr + (offset.n + offset.c + offset.h + offset.w) * raw_size;
          memcpy(dst + write_pos, src, raw_size);
          write_pos += raw_size;
        }
      }
    }
  }
  return 0;
}

static int cast_buffer_to_fp32(float *dst_fp32,
                               const unsigned char *src_raw,
                               size_t elem_cnt,
                               data_type_t dtype) {
  size_t pos = 0;
  for (size_t i = 0; i < elem_cnt; ++i) {
    float outv = 0.0f;
    switch (dtype) {
      case DT_FP32: {
        outv = *(const float *)(src_raw + pos);
        pos += 4;
        break;
      }
      case DT_INT32: {
        outv = (float)(*(const int32_t *)(src_raw + pos));
        pos += 4; break;
      }
      case DT_UINT32: {
        outv = (float)(*(const uint32_t *)(src_raw + pos));
        pos += 4; break;
      }
      case DT_INT16: {
        outv = (float)(*(const int16_t *)(src_raw + pos));
        pos += 2; break;
      }
      case DT_UINT16: {
        outv = (float)(*(const uint16_t *)(src_raw + pos));
        pos += 2; break;
      }
      case DT_INT8: {
        outv = (float)(*(const int8_t *)(src_raw + pos));
        pos += 1; break;
      }
      case DT_UINT8: {
        outv = (float)(*(const uint8_t *)(src_raw + pos));
        pos += 1; break;
      }
      case DT_FP16: {
        scalar_t v; v.f16 = *(const float16 *)(src_raw + pos);
        v = tpu_cast(v, DT_FP32, DT_FP16, RM_HALF_AWAY_FROM_ZERO);
        outv = v.f32; pos += sizeof(float16); break;
      }
      case DT_BFP16: {
        scalar_t v; v.bf16 = *(const bfloat16 *)(src_raw + pos);
        v = tpu_cast(v, DT_FP32, DT_BFP16, RM_HALF_AWAY_FROM_ZERO);
        outv = v.f32; pos += sizeof(bfloat16); break;
      }
      case DT_FP8E5M2: {
        uint8_t v = *(const uint8_t *)(src_raw + pos);
        outv = fp8_to_fp32(v, /*is_e5m2*/ true); pos += 1; break;
      }
      case DT_FP8E4M3: {
        uint8_t v = *(const uint8_t *)(src_raw + pos);
        outv = fp8_to_fp32(v, /*is_e5m2*/ false); pos += 1; break;
      }
      case DT_INT4: {
        int8_t v = (*(const int8_t *)(src_raw + pos) << 4) >> 4;
        outv = (float)v; pos += 1; break;
      }
      case DT_UINT4: {
        uint8_t v = *(const uint8_t *)(src_raw + pos) & 0x0F;
        outv = (float)v; pos += 1; break;
      }
      case DT_TF32: {
        outv = *(const float *)(src_raw + pos);
        pos += 4; break;
      }
      case DT_FP20: {
        outv = *(const float *)(src_raw + pos);
        pos += 4; break;
      }
      default: {
        outv = (float)(*(const uint8_t *)(src_raw + pos));
        pos += 1; break;
      }
    }
    dst_fp32[i] = outv;
  }
  return 0;
}

void dump_local_mem_data(local_addr_t local_offset, int start_idx,
                         const dim4 *shape, const dim4 *stride,
                         data_type_t dtype, const char *file_path,
                         const char *tensor_name, int lane_num, bool is_rv) {

  if (!shape || !file_path || !file_path[0] || !tensor_name) return;
  ensure_reset_npz_once(file_path);

#if defined(__tpub_7_1_e__) || defined(__tpub_9_3__) || defined(__tpub_9_0__)
  if (is_rv) {
#if defined(__tpub_7_1_e__)
    rvt_sync_i(0xdeadbeef, 0);
#else
    rvt_sync_i(__SR(0xdeadbeef), 0);
#endif
  } else {
    tpu_poll_with_check_parallel();
  }
#else
  tpu_poll_with_check_parallel();
#endif

  dim4 true_stride;
  if (!stride) {
    tpu_aligned_stride(&true_stride, start_idx, shape, dtype);
    stride = &true_stride;
  }

  size_t n = (size_t)shape->n, c = (size_t)shape->c, h = (size_t)shape->h, w = (size_t)shape->w;
  if (n == 0 || c == 0 || h == 0 || w == 0) return;
  size_t elem_cnt = n * c * h * w;

  size_t raw_size = tpu_data_type_size(dtype);
  unsigned char *raw_buf = (unsigned char *)malloc(elem_cnt * raw_size);
  if (!raw_buf) return;
  gather_local_to_buffer(raw_buf, local_offset, start_idx, shape, stride, dtype, lane_num);

  cn_dtype_class_t cls; size_t elem_size_out; int cast_to_fp32 = 0;
  map_dtype_to_cnpy(dtype, &cls, &elem_size_out, &cast_to_fp32);

  const void *data_ptr = NULL;
  size_t elem_size_to_write = elem_size_out;

  float *fp32_buf = NULL;
  if (cast_to_fp32) {
    fp32_buf = (float *)malloc(elem_cnt * sizeof(float));
    if (!fp32_buf) { free(raw_buf); return; }
    cast_buffer_to_fp32(fp32_buf, raw_buf, elem_cnt, dtype);
    data_ptr = fp32_buf;
    elem_size_to_write = sizeof(float);
    cls = CN_DTYPE_FLOAT;
  } else {
    data_ptr = raw_buf;
    elem_size_to_write = raw_size;
  }

  char shape_str[128];
  if (build_shape_string_from_dim4(shape_str, sizeof(shape_str), shape) != 0) {
    if (fp32_buf) free(fp32_buf);
    free(raw_buf);
    return;
  }

  cnpy_npz_add(file_path, tensor_name, data_ptr, elem_size_to_write, elem_cnt,
               shape_str, cls, "a");

  if (fp32_buf) free(fp32_buf);
  free(raw_buf);
}

void dump_global_mem_data(global_addr_t addr, const dim4 *shape,
                          const dim4 *stride, data_type_t dtype,
                          const char *file_path, const char *tensor_name,
                          int is_l2, bool is_rv) {
  if (!shape || !file_path || !file_path[0] || !tensor_name) return;
  ensure_reset_npz_once(file_path);

#if defined(__sg2260e__) || defined(__sg2262__)
  if (is_rv) {
#if defined(__sg2262__)
    rvt_sync_i(__SR(0xdeadbeef), 0);
#else
    rvt_sync_i(0xdeadbeef, 0);
#endif
  } else {
    tpu_poll_with_check_parallel();
  }
#else
  tpu_poll_with_check_parallel();
#endif

  dim4 true_stride;
  if (!stride) {
    tpu_continuous_stride(&true_stride, shape);
    stride = &true_stride;
  }

  unsigned char *base_ptr = NULL;
  if (is_l2)
    base_ptr = tpu_l2_sram_addr(addr);
  else
    base_ptr = tpu_global_mem_addr(addr);
  if (!base_ptr) return;

  size_t n = (size_t)shape->n, c = (size_t)shape->c, h = (size_t)shape->h, w = (size_t)shape->w;
  if (n == 0 || c == 0 || h == 0 || w == 0) return;
  size_t elem_cnt = n * c * h * w;

  size_t raw_size = tpu_data_type_size(dtype);
  unsigned char *raw_buf = (unsigned char *)malloc(elem_cnt * raw_size);
  if (!raw_buf) return;
  gather_global_to_buffer(raw_buf, base_ptr, shape, stride, dtype);

  cn_dtype_class_t cls; size_t elem_size_out; int cast_to_fp32 = 0;
  map_dtype_to_cnpy(dtype, &cls, &elem_size_out, &cast_to_fp32);

  const void *data_ptr = NULL;
  size_t elem_size_to_write = elem_size_out;

  float *fp32_buf = NULL;
  if (cast_to_fp32) {
    fp32_buf = (float *)malloc(elem_cnt * sizeof(float));
    if (!fp32_buf) { free(raw_buf); return; }
    cast_buffer_to_fp32(fp32_buf, raw_buf, elem_cnt, dtype);
    data_ptr = fp32_buf;
    elem_size_to_write = sizeof(float);
    cls = CN_DTYPE_FLOAT;
  } else {
    data_ptr = raw_buf;
    elem_size_to_write = raw_size;
  }

  char shape_str[128];
  if (build_shape_string_from_dim4(shape_str, sizeof(shape_str), shape) != 0) {
    if (fp32_buf) free(fp32_buf);
    free(raw_buf);
    return;
  }

  cnpy_npz_add(file_path, tensor_name, data_ptr, elem_size_to_write, elem_cnt,
               shape_str, cls, "a");

  if (fp32_buf) free(fp32_buf);
  free(raw_buf);
}
