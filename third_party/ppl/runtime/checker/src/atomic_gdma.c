#include "atomic_gdma.h"

#define ASSERT_TENSOR_NSIZE(n) \
    ASSERT_FS_INFO(n>0 && n<=GDMA_MAX_N, #n "=%d", n);

#define ASSERT_TENSOR_CSIZE(c) \
    ASSERT_FS_INFO(c>0 && c<=GDMA_MAX_C, #c "=%d", c);

#define ASSERT_TENSOR_HSIZE(h) \
    ASSERT_FS_INFO(h>0 && h<=GDMA_MAX_H, #h "=%d", h);

#define ASSERT_TENSOR_WSIZE(w) \
    ASSERT_FS_INFO(w>0 && w<=GDMA_MAX_W, #w "=%d", w);

#define ASSERT_TENSOR_SIZE(n,c,h,w) \
    ASSERT_TENSOR_NSIZE(n); \
    ASSERT_TENSOR_CSIZE(c); \
    ASSERT_TENSOR_HSIZE(h); \
    ASSERT_TENSOR_WSIZE(w)

#define ASSERT_WSTRIDE(wstr, byte_len) \
    ASSERT_FS_INFO((wstr * byte_len <= MAX_WSTRIDE_BYTE_LEN) && (wstr != 0), "W stride byte len = %d", wstr * byte_len);

#define ASSERT_WSTRIDE_FP20(wstr) \
    ASSERT_FS_INFO(wstr == 1, "When data type is fp20, W stride should be 1");

#define ASSERT_COMPACT_FP20(n,c,h,w,nstr,cstr,hstr,wstr) \
    ASSERT_WSTRIDE_FP20(wstr); \
    ASSERT_FS_INFO(hstr == (w), "When data type is fp20, 51 elements constitute fp20 block"); \
    ASSERT_FS_INFO(cstr == (h * hstr), "When data type is fp20, c stride should be compacted"); \
    ASSERT_FS_INFO(nstr == (c * cstr), "When data type is fp20, n stride should be compacted");

static inline int get_eu_num_from_type(int format) {
    switch (format)
    {
    case 2: //GDMA_FP32
    case 4: //GDMA_INT32
        return EU_NUM_32BIT;
    case 1: //GDMA_FP16
    case 3://GDMA_INT16
    case 5://GDMA_BF16
        return EU_NUM_16BIT;
    case 0:// GDMA_INT8:
    case 7://GDMA_FP8_E4M3
    case 8://GDMA_FP8_E5M2
        return EU_NUM_8BIT;
        return 0;
    default:
        ASSERT(0);
        break;
    }
    return -1;
}

void tensor_align_move_check(
  int local_mem_start_addr,
  int local_mem_idx,
  u64 sys_mem_start_addr,
  int src_N,
  int src_C,
  int src_H,
  int src_W,
  int src_format,
  int direction,
  int transpose // N/C transpose
) {
#ifdef USING_CMODEL
  ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
  ASSERT_FS_INFO(direction == GDMA_L2S ||
                     direction == GDMA_S2L,
                 "direction=%d", direction);
  ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                 "can't be static memory sys_addr:0x%llx",
                 sys_mem_start_addr);
#ifndef __sg2262__
  ASSERT(src_format != GDMA_FP20);
#endif
#endif
}

void tensor_stride_move_check(
    int local_mem_start_addr,
    int local_mem_idx,
    u64 sys_mem_start_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int src_format,
    int direction,
    int transpose // N/C transpose
) {
#ifdef __sg2262__
#ifdef USING_CMODEL
    ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    // int type_len = get_gdma_format_type_len(src_format);
    // ASSERT_WSTRIDE(src_W_stride, type_len);
    // ASSERT_WSTRIDE(dst_W_stride, type_len);
    ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                   "can't be static memory sys_addr:0x%llx",
                   sys_mem_start_addr);
    ASSERT_FS_INFO(direction == GDMA_L2S ||
                       direction == GDMA_S2L,
                   "directin=%d", direction);
    ASSERT(src_format < GDMA_FORMAT_NUM);
#endif
#else
  u64 dst_N = transpose ? src_C : src_N;
  u64 dst_C = transpose ? src_N : src_C;
  if (direction == GDMA_S2L) {
    tensor_general_move_check(sys_mem_start_addr, 0, src_N, src_C, src_H, src_W,
                              src_N_stride, src_C_stride, src_H_stride,
                              src_W_stride, src_format, local_mem_start_addr,
                              local_mem_idx, dst_N, dst_C, src_H, src_W,
                              dst_N_stride, dst_C_stride, dst_H_stride,
                              dst_W_stride, direction, transpose);
  } else if (direction == GDMA_L2S) {
    tensor_general_move_check(
        local_mem_start_addr, local_mem_idx, src_N, src_C, src_H, src_W,
        src_N_stride, src_C_stride, src_H_stride, src_W_stride, src_format,
        sys_mem_start_addr, 0, dst_N, dst_C, src_H, src_W, dst_N_stride,
        dst_C_stride, dst_H_stride, dst_W_stride, direction, transpose);
  }
#endif
}

void tensor_general_move_check(
    u64 src_addr,      // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    int src_format,
    u64 dst_addr,      // local_addr or global_addr
    int dst_local_idx, // use only to local_mem
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int direction,
    int transpose // N/C transpose
) {
#ifdef USING_CMODEL
  ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
  ASSERT_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
  ASSERT_FS_INFO(dst_N * dst_C * dst_H * dst_W ==
                       src_N * src_C * src_H * src_W,
                   "dst_count=%d, src_count=%d", dst_N * dst_C * dst_H * dst_W,
                   src_N * src_C * src_H * src_W);
#ifdef __sg2262__
  int type_len = get_gdma_format_type_len(src_format);
  ASSERT_WSTRIDE(src_W_stride, type_len);
  ASSERT_WSTRIDE(dst_W_stride, type_len);
#else
  if (src_format == GDMA_FP20) {
    ASSERT(direction == GDMA_S2S);
    ASSERT(transpose == 0);
    ASSERT(src_addr % 128 == 0);
    ASSERT(dst_addr % 128 == 0);
    ASSERT_WSTRIDE_FP20(src_W_stride);
    ASSERT_WSTRIDE_FP20(dst_W_stride);
  } else {
    // int type_len = get_gdma_format_type_len(src_format);
    // ASSERT_WSTRIDE(src_W_stride, type_len);
    // ASSERT_WSTRIDE(dst_W_stride, type_len);
  }
#endif
  ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                  "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                  src_addr, dst_addr);
#endif
}

void tensor_compact_move_check(
  int local_mem_start_addr,
  int local_mem_idx,
  u64 sys_mem_start_addr,
  int src_N, int src_C, int src_H, int src_W,
  int src_format,
  int direction,
  int transpose // N/C transpose
) {

#ifdef USING_CMODEL
  ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
  ASSERT(direction == GDMA_L2S || direction == GDMA_S2L);
  ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                  "can't be static memory sys_addr:0x%llx",
                  sys_mem_start_addr);
#ifndef __sg2262__
  ASSERT(src_format != GDMA_FP20);
#endif
#endif

  u64 sm_addr = sys_mem_start_addr;
  u64 lm_addr = CALC_LOCAL_ADDR(local_mem_idx, local_mem_start_addr);
  stride_type W_stride = 1;
  stride_type H_stride = src_W * W_stride;
  stride_type C_stride = src_H * H_stride;

  int is_local_to_sys = direction == GDMA_L2S;

  u64 src_addr = 0; // is_local_to_sys? local_addr : sys_addr;
  u64 dst_addr = 0; // is_local_to_sys ? sys_addr : local_addr;
  stride_type src_nstride = 0;
  stride_type dst_nstride = 0;
  int dst_C = transpose ? src_N : src_C;
  int dst_N = transpose ? src_C : src_N;
  if (is_local_to_sys) {
    src_addr = lm_addr;
    dst_addr = sm_addr;
    src_nstride = (src_C + local_mem_idx + NPU_NUM - 1) / NPU_NUM * C_stride;
    dst_nstride = dst_C * C_stride;
  } else {
    src_addr = sm_addr;
    dst_addr = lm_addr;
    src_nstride = src_C * C_stride;
    dst_nstride = (dst_C + local_mem_idx + NPU_NUM - 1) / NPU_NUM * C_stride;
  }

  int special_func = transpose ? GDMA_FUNC_TRANS: GDMA_FUNC_NONE;
  (void)special_func;
  (void)src_addr;
  (void)dst_addr;
  (void)dst_C;
  (void)dst_N;
  if (is_local_to_sys) {
    tensor_general_move_check(lm_addr, local_mem_idx, src_N, src_C, src_H,
                              src_W, src_nstride, C_stride, H_stride, W_stride,
                              src_format, sm_addr, 0, dst_N, dst_C, src_H,
                              src_W, dst_nstride, C_stride, H_stride, W_stride,
                              direction, transpose);
  } else {
    tensor_general_move_check(
        sm_addr, 0, src_N, src_C, src_H, src_W, src_nstride, C_stride, H_stride,
        W_stride, src_format, lm_addr, local_mem_idx, dst_N, dst_C, src_H,
        src_W, dst_nstride, C_stride, H_stride, W_stride, direction, transpose);
  }
}

void matrix_move_check(int local_mem_start_addr, int local_mem_idx,
                       u64 sys_mem_start_addr, int sec_size, int row_num,
                       int col_num, // means matrix in sys_mem  is row*col,
                       int src_format, int direction, int transpose) {
#ifdef USING_CMODEL
  ASSERT_FS_INFO(direction == GDMA_L2S ||
                      direction == GDMA_S2L,
                  "direction=%d", direction);
  ASSERT_FS_INFO(sec_size > 0 && row_num > 0 && col_num > 0,
                  "row=%d, col=%d, sec=%d", row_num, col_num, sec_size);
  ASSERT(row_num <= GDMA_MAX_H && col_num <= GDMA_MAX_W);
  ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                  "can't be static memory sys_addr:0x%llx",
                  sys_mem_start_addr);
#ifdef __sg2262__
  ASSERT(transpose == 0);
#endif
#endif
}

void matrix_stride_move_check(int local_mem_start_addr, int local_mem_idx,
                              u64 sys_mem_start_addr, int sec_size, int row_num,
                              int col_num, stride_type global_row_stride,
                              stride_type local_row_stride,
                              stride_type local_sec_stride, int src_format,
                              int direction, int transpose) {

#ifdef USING_CMODEL
  ASSERT(transpose == 0);
  ASSERT_FS_INFO(direction == GDMA_L2S ||
                      direction == GDMA_S2L,
                  "direction=%d", direction);
  ASSERT_FS_INFO(sec_size > 0 && row_num > 0 && col_num > 0,
                  "row=%d, col=%d, sec=%d", row_num, col_num, sec_size);
  ASSERT(row_num <= GDMA_MAX_H && col_num <= GDMA_MAX_W);
  ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                    "can't be static memory sys_addr:0x%llx",
                    sys_mem_start_addr);
#endif
}

void matrix_stride_move_txp_check(
    int local_mem_start_addr, int local_mem_idx, u64 sys_mem_start_addr,
    int sec_size, int row_num, int col_num, stride_type global_n_stride,
    stride_type global_c_stride, stride_type global_row_stride,
    stride_type global_w_stride, stride_type local_row_stride,
    stride_type local_sec_stride, stride_type local_h_stride, int src_format,
    int direction, int transpose) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT(transpose == 0);
  ASSERT(direction == GDMA_L2S || direction == GDMA_S2L);
  ASSERT(sec_size > 0 && row_num > 0 && col_num > 0);
  ASSERT(row_num <= GDMA_MAX_H && col_num <= GDMA_MAX_W);
  ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                    "can't be static memory sys_addr:0x%llx",
                    sys_mem_start_addr);
#endif
#endif
}

void general_matrix_move_check(int local_mem_start_addr, int local_mem_idx,
                               u64 sys_mem_start_addr, int sec_size,
                               int row_num, int col_num, stride_type row_stride,
                               int src_format, int direction, int transpose) {
  int lm_col = transpose ? row_num : col_num;
  int sec_num = (lm_col + sec_size - 1) / sec_size;
  int align_factor = 4 / get_gdma_format_type_len(src_format);
  stride_type lm_sec_stride = ALIGN(sec_size, EU_NUM * align_factor);
  stride_type lm_row_stride =
      (local_mem_idx + sec_num + NPU_NUM - 1) / NPU_NUM * lm_sec_stride;
  matrix_stride_move_check(local_mem_start_addr, local_mem_idx,
                           sys_mem_start_addr, sec_size, row_num, col_num,
                           row_stride, lm_row_stride, lm_sec_stride, src_format,
                           direction, transpose);
}

void fill_constant_gen_local_cmd_stride(
    int local_mem_start_addr, int local_mem_idx, const void *const_val,
    int data_format, int dst_N, int dst_C, int dst_H, int dst_W,
    stride_type dst_N_stride, stride_type dst_C_stride,
    stride_type dst_H_stride, stride_type dst_W_stride, int stride_enable,
    int use_broadcast) {

#ifdef USING_CMODEL
  ASSERT_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
  if (stride_enable) {
    // ASSERT_WSTRIDE(dst_W_stride, get_gdma_format_type_len(data_format));
  }
  if (use_broadcast) {
    if (stride_enable) {
      ASSERT_FS_INFO(dst_W_stride == 1,
                      "broadcast only support wstride == 1, stride:%d",
                      dst_W_stride);
    }
    ASSERT_FS_INFO(local_mem_idx + dst_C <= NPU_NUM,
                    "broadcast cannot overflow NPU_NUM");
  }
#endif
}

void fill_constant_check_global_stride(
    u64 sys_mem_start_addr, const void *const_val, int data_format, int dst_N,
    int dst_C, int dst_H, int dst_W, stride_type dst_N_stride,
    stride_type dst_C_stride, stride_type dst_H_stride,
    stride_type dst_W_stride, int stride_enable) {

#ifdef USING_CMODEL
  ASSERT_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
  if (data_format == GDMA_FP20) {
    ASSERT(sys_mem_start_addr % 128 == 0);
    ASSERT_WSTRIDE_FP20(dst_W_stride);
  } else {
    if (stride_enable) {
#ifdef __sg2262__
      ASSERT_WSTRIDE(dst_W_stride, get_gdma_format_type_len(data_format));
#endif
    }
  }
    ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                   "can't be static memory sys_addr:0x%llx",
                   sys_mem_start_addr);
#endif
}

void general_gdma_check_2262(
        u64 src_addr,
        u64 dst_addr,
        int src_format,
        u16 n,
        u16 c,
        u16 h,
        u32 w,
        int src_is_const
) {

#ifdef USING_CMODEL
    ASSERT_FS_INFO(h > 0, "src_count=%d", h);
    ASSERT(src_is_const == 0 || src_is_const == 1);
#endif
}

void general_gdma_check(u64 src_addr, u64 dst_addr, int src_format,
                        stride_type src_count, int src_is_const) {

#ifdef USING_CMODEL
  ASSERT(src_format != GDMA_FP20);
  ASSERT_FS_INFO(src_count > 0, "src_count=%d", src_count);
  ASSERT(src_is_const == 0 || src_is_const == 1);
#endif
}

void general_gdma_txp_check(u64 src_addr, u64 dst_addr, int src_format, u16 n,
                            u16 c, u16 h, u16 w, int src_is_const) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT(src_format != GDMA_FP20);
  ASSERT(src_is_const == 0 || src_is_const == 1);
#endif
#endif
}

void general_gdma_common_check(u64 src_addr, // Absolute addr
                               u64 dst_addr, // Absolute addr
                               int src_format,
                               stride_type src_count, // tv_gen: default=1
                               int src_is_const, int n, int c, int h, int w,
                               int mode_4d) {
#ifdef __sg2262__
  if (mode_4d) {
      general_gdma_check_2262(src_addr, dst_addr, src_format, n, c, h, w, src_is_const);
  } else {
      general_gdma_check_2262(src_addr, dst_addr, src_format, 1, 1, 1, src_count, src_is_const);
  }
#else
  if (mode_4d) {
    general_gdma_txp_check(src_addr, dst_addr, src_format, n, c, h, w,
                           src_is_const);
  } else {
    general_gdma_check(src_addr, dst_addr, src_format, src_count, src_is_const);
  }
#endif
}

void general_broadcast_check(
  u64 src_addr, // src_addr or constant
  int local_mem_start_addr,
  int local_mem_idx,
  int src_format,
  stride_type src_count,
  int dst_c, // Broadcast src_count data to dst_c lanes, local_idx + dst_c <= NPU_NUM
  int src_is_const // 0: not const, 1: is const
) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT(src_count > 0);
  ASSERT(local_mem_idx + dst_c <= NPU_NUM &&
         "broadcast cannot over NPU_NUM\n");
  ASSERT(local_mem_start_addr < LOCAL_MEM_SIZE &&
         "broadcast need offset for per npu\n");
#endif
#endif
}

void general_broadcast_txp_check(
    u64 src_addr, // src_addr or constant
    int local_mem_start_addr, int local_mem_idx, int src_format, u16 n, u16 c,
    u16 h, u16 w,
    int dst_c, // Broadcast src_count data to dst_c lanes, local_idx + dst_c <=
               // NPU_NUM
    int src_is_const // 0: not const, 1: is const
) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT(local_mem_idx + dst_c <= NPU_NUM &&
         "broadcast cannot over NPU_NUM");
  ASSERT(local_mem_start_addr < LOCAL_MEM_SIZE &&
         "broadcast need offset for per npu");
#endif
#endif
}

void general_broadcast_common_check(
    u64 src_addr, // src_addr(absolute addr) or constant
    int local_mem_start_addr,
    int local_mem_idx,
    int src_format,
    stride_type src_count,
    int dst_c, // Broadcast src_count data to dst_c lanes, local_idx + dst_c <= NPU_NUM
    int src_is_const, // 0: not const, 1: is const
    int n, int c, int h, int w,
    int mode_4d) {
#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  if (mode_4d) {
    general_broadcast_txp_check(src_addr, local_mem_start_addr, local_mem_idx,
                                src_format, n, c, h, w, dst_c, src_is_const);
  } else {
    general_broadcast_check(src_addr, local_mem_start_addr, local_mem_idx,
                            src_format, src_count, dst_c, src_is_const);
  }
#endif
#endif
}

void general_cwtrans_check(
    u64 src_addr,      // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    u64 dst_addr,      // local_addr or global_addr
    int dst_local_idx, // use only from local_mem
    int src_N, int src_C,
    int src_H, int src_W,
    int src_format,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int stride_enable,
    int direction // Support combination of Global、Local and L2 for src and dst
) {

  ASSERT_FS_INFO(0, "Not support");

  ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
  ASSERT(!is_smem(src_addr) && !is_smem(dst_addr) && "can't be static memory\n");
}

void tensor_general_move_with_mask_check(
    u64 src_addr,       // local_addr or global_addr
    int src_local_idx,  // use only from local_mem
    u64 mask_addr,      // local_addr or global_addr
    int mask_local_idx, // use only from local_mem
    int mask_in_lmem,   // 1: mask is in local mem, 0: mask is in global mem
    u64 dst_addr,       // global addr only
    int src_format,
    int mask_format,
    u32 N,
    u32 C,
    u32 H,
    u32 W,
    int direction // src to dst direction, support L2S, S2S, L22S, L2L2, S2L2,
                  // L22L2
) {
#ifdef USING_CMODEL
#ifdef __sg2262__
    ASSERT(H > 0 && H <= UINT32_MAX);
    ASSERT(W > 0 && W <= UINT32_MAX);
#else
  ASSERT(src_format != GDMA_FP20);
  ASSERT(mask_format != GDMA_FP20);
#endif
  ASSERT_TENSOR_NSIZE(N);
  ASSERT_TENSOR_CSIZE(C);

  ASSERT(direction == GDMA_L2S || direction == GDMA_S2S);
  int src_from_lmem = direction == GDMA_L2S;
  if (src_from_lmem) {
      src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
      ASSERT(src_addr % ALIGN_BYTES == 0);
  }
  if (mask_in_lmem) {
      mask_addr = CALC_LOCAL_ADDR(mask_local_idx, mask_addr);
      ASSERT(mask_addr % ALIGN_BYTES == 0);
  }
#endif
}

void tensor_move_nonzero_check(
    u64 src_addr,      // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    u64 dst_addr,      // global addr only
    int src_format,    // INT8/INT16/INT32/FP32/FP16/BF16
    int dst_format,    // INT8/INT16/INT32
    u32 N,
    u32 C,
    u32 H,
    u32 W,
    u32 base_idx,
    int direction // only support L2S, S2S
) {
#ifdef USING_CMODEL
  ASSERT_TENSOR_NSIZE(N);
  ASSERT_TENSOR_CSIZE(C);
#ifdef __sg2262__
  ASSERT(H > 0 && H <= UINT32_MAX);
  ASSERT(W > 0 && W <= UINT32_MAX);
#endif
  ASSERT(direction == GDMA_L2S || direction == GDMA_S2S);

  int src_from_lmem = direction == GDMA_L2S;
  if (src_from_lmem) {
      ASSERT(src_addr < LOCAL_MEM_SIZE);
      src_addr = CALC_LOCAL_ADDR(src_local_idx, src_addr);
      ASSERT(src_addr % ALIGN_BYTES == 0);
  }
#ifndef __sg2262__
  ASSERT(src_format != GDMA_FP20);
#endif
  ASSERT(dst_format == GDMA_INT8 || dst_format == GDMA_INT16 || dst_format == GDMA_INT32);

#endif
}

void tensor_broadcast_move_check(
    u64 src_addr,            // local_addr or global_addr
    int src_local_idx,       // use only from local_mem
    int dst_lmem_start_addr, // local_addr
    int dst_local_idx, int src_N, int src_H, int src_W,
    int dst_C, // Restriction: dst_local_idx + dst_C <= NPU_NUM
    stride_type src_N_stride, stride_type src_H_stride,
    stride_type dst_N_stride, stride_type dst_H_stride, int data_format,
    int stride_enable,
    int direction // Only support, S2L, L2L, L22L
) {
#ifdef USING_CMODEL
  ASSERT_TENSOR_SIZE(src_N, dst_C, src_H, src_W);
  ASSERT_FS_INFO(direction == GDMA_S2L || direction == GDMA_L2L, "direction=%d", direction);
#ifndef __sg2262__
  ASSERT(data_format != GDMA_FP20);
#endif
  ASSERT_FS_INFO(dst_C + dst_local_idx <= NPU_NUM,
                  "tensor broadcast dst_c + npu_idx <= NPU_NUM, dst_C + npu_idx:%d",
                  dst_C + dst_local_idx);
#endif
}

// index addr aligned to 512byte can get better performance
// wsize is larger can get better performance
void tensor_gdma_gather_check(
    u64 src_addr,        // local_addr or global_addr
    int src_local_idx,   // use only from local_mem
    u64 index_addr,      // local_addr or global_addr
    int index_local_idx, // use only from local_mem
    int index_in_lmem,   // 1: index is in local mem, 0: index is in global mem
    u64 dst_addr,        // local_addr or global_addr
    int dst_local_idx,   // use only from local_mem
    u64 const_val,
    u32 C,
    u32 src_H,
    u32 src_W,
    u32 index_H,
    u32 start_pos,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type index_C_stride,
    stride_type index_H_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int src_format,
    int src_C_is1,
    int index_C_is1,
    int stride_enable,
    int direction // Support S2S, S2L, L2S, L2L
) {
#ifdef USING_CMODEL
  ASSERT_TENSOR_CSIZE(C);
  ASSERT_TENSOR_WSIZE(src_W);
#ifndef __sg2262__
  ASSERT(src_format != GDMA_FP20);
#endif
  if (stride_enable)
    ASSERT(index_H_stride == 1);
#endif
}

// index addr aligned to 512byte can get better performance
// wsize is larger can get better performance
void tensor_gdma_scatter_check(
    u64 src_addr,        // local_addr or global_addr
    int src_local_idx,   // use only from local_mem
    u64 index_addr,      // local_addr or global_addr
    int index_local_idx, // use only from local_mem
    int index_in_lmem,   // 1: index is in local mem, 0: index is in global mem
    u64 dst_addr,        // local_addr or global_addr
    int dst_local_idx,   // use only from local_mem
    u32 C,
    u32 src_H,
    u32 src_W,
    u32 dst_H,
    u32 start_pos,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type index_C_stride,
     stride_type index_H_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int src_format,
    int src_C_is1,
    int index_C_is1,
    int stride_enable,
    int direction, // Support S2S, S2L, L2S, L2L, S2L2, L22S, L2L2, L22L, L22L2
    int inplace_add) {
#ifdef USING_CMODEL
  ASSERT(C <= GDMA_MAX_C);
  ASSERT(src_W <= GDMA_MAX_W);
  ASSERT(src_format != GDMA_FP20);
  if (stride_enable)
    ASSERT(index_H_stride == 1);
#endif
}

void tensor_gdma_scatter_txp_check(
    u64 src_addr, // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    u64 index_addr, // local_addr or global_addr
    int index_local_idx, // use only from local_mem
    int index_in_lmem, // 1: index is in local mem, 0: index is in global mem
    u64 dst_addr, // local_addr or global_addr
    int dst_local_idx, // use only from local_mem
    u32 C,
    u32 src_H,
    u32 src_W,
    u32 dst_H0,
    u32 dst_H,
    u32 start_pos,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type index_C_stride,
    stride_type index_H_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int src_format,
    int src_C_is1,
    int index_C_is1,
    int stride_enable,
    int direction, // Support S2S, S2L, L2S, L2L, S2L2, L22S, L2L2, L22L, L22L2
    int inplace_add) {
#ifdef USING_CMODEL
#ifndef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT(C <= GDMA_MAX_C);
  ASSERT(src_W <= GDMA_MAX_W);
  ASSERT(inplace_add == 0);
  ASSERT(stride_enable == 1);
  if (stride_enable) ASSERT(index_H_stride == 1);
#endif
#endif
}

// only support l2s
void tensor_normal_compress_check(
  uint32_t local_mem_addr, u64 sys_mem_addr,
  int32_t N,
  int32_t C,
  int32_t H,
  int32_t W,
  uint32_t local_n_stride,
  uint32_t local_c_stride,
  uint32_t local_h_stride,
  uint8_t bias0,
  uint8_t bias1,
  int32_t is_signed,
  int32_t zero_guard, // only valid for fp16
  int32_t data_format) {
#ifdef USING_CMODEL
#ifndef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  u64 src_addr = CALC_LOCAL_ADDR(0, local_mem_addr);
  u64 dst_addr = sys_mem_addr;
  ASSERT_TENSOR_SIZE(N, C, H, W);
  ASSERT_FS_INFO(data_format == GDMA_INT8 ||
                  data_format == GDMA_INT16 ||
                  data_format == GDMA_FP16 ||
                  data_format == GDMA_BF16,
                  "format:%d", data_format);
  ASSERT(is_signed >= 0 && is_signed < 2 && zero_guard >= 0 &&
          zero_guard < 2);
  ASSERT(data_format == GDMA_FP16 ||
          data_format == GDMA_BF16 ||
          !is_signed || (bias0 < 128 && bias1 < 128));
  ASSERT(get_npu_index(local_mem_addr) < NPU_NUM);
#endif
#endif
}

// only support s2l
void tensor_normal_decompress_check(
  uint32_t local_mem_addr,
  u64 sys_mem_addr,
  int32_t N,
  int32_t C,
  int32_t H,
  int32_t W,
  uint32_t local_n_stride,
  uint32_t local_c_stride,
  uint32_t local_h_stride,
  uint8_t bias0,
  uint8_t bias1,
  int32_t is_signed,
  int32_t zero_guard,
  int32_t data_format) {
#ifdef USING_CMODEL
#ifndef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  u64 dst_addr = CALC_LOCAL_ADDR(0, local_mem_addr);
  u64 src_addr = sys_mem_addr;
  ASSERT_TENSOR_SIZE(N, C, H, W);
  ASSERT_FS_INFO(data_format == GDMA_INT8 ||
                  data_format == GDMA_INT16 ||
                  data_format == GDMA_FP16 ||
                  data_format == GDMA_BF16,
                  "format:%d", data_format);
  ASSERT(is_signed >= 0 && is_signed < 2 && zero_guard >= 0 && zero_guard < 2);
  ASSERT(get_npu_index(local_mem_addr) < NPU_NUM);
#endif
#endif
}

// only support l2s
void tensor_racu_compress_check(
  uint32_t local_mem_addr,
  u64 racu_sys_mem_addr,
  u64 meta_sys_mem_addr,
  int32_t N,
  int32_t C,
  int32_t H,
  int32_t W,
  uint32_t local_n_stride,
  uint32_t local_c_stride,
  uint32_t local_h_stride,
  uint32_t racu_n_stride,
  uint32_t racu_c_stride,
  uint32_t racu_h_stride,
  uint32_t meta_n_stride,
  uint32_t meta_c_stride,
  uint8_t bias0,
  uint8_t bias1,
  int32_t is_signed,
  int32_t zero_guard,
  int32_t data_format) {
#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else

  int type_len = get_gdma_format_type_len(data_format);
  ASSERT(get_npu_index(local_mem_addr) == 0);
  u64 src_laddr = CALC_LOCAL_ADDR(0, local_mem_addr);
  u64 racu_gaddr = racu_sys_mem_addr;
  u64 meta_gaddr = meta_sys_mem_addr;
  ASSERT_TENSOR_SIZE(N, C, H, W);
  // because max enc_size is 19bit
  ASSERT((W * sg_min(C, NPU_NUM) * type_len) <=
         (1 << (12 + NNVLC_ALIGN_SHIFT)));
  ASSERT(data_format == GDMA_INT8 || data_format == GDMA_INT16 ||
             data_format == GDMA_FP16 || data_format == GDMA_BF16);
  ASSERT(is_signed >= 0 && is_signed < 2 && zero_guard >= 0 && zero_guard < 2);
  ASSERT(data_format == GDMA_FP16 || data_format == GDMA_BF16 || !is_signed ||
         (bias0 < 128 && bias1 < 128));
  ASSERT((racu_h_stride & (NNVLC_ALIGN_BYTES - 1)) == 0 &&
         (racu_c_stride & (NNVLC_ALIGN_BYTES - 1)) == 0 &&
         (racu_n_stride & (NNVLC_ALIGN_BYTES - 1)) == 0);
#endif
#endif
}

// only support s2l
void tensor_racu_decompress_check(
    uint32_t local_mem_addr,
    u64 racu_sys_mem_addr,
    u64 meta_sys_mem_addr,
    int32_t N,
    int32_t C,
    int32_t H,
    int32_t W,
    uint32_t local_n_stride,
    uint32_t local_c_stride,
    uint32_t local_h_stride,
    uint32_t racu_n_stride,
    uint32_t racu_c_stride,
    uint32_t racu_h_stride,
    uint32_t meta_n_stride,
    uint32_t meta_c_stride,
    uint8_t bias0,
    uint8_t bias1,
    int32_t is_signed,
    int32_t zero_guard,
    int32_t data_format) {
#ifdef USING_CMODEL
#ifndef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  int type_len = get_gdma_format_type_len(data_format);
  ASSERT(get_npu_index(local_mem_addr) == 0);
  u64 dst_laddr = CALC_LOCAL_ADDR(0, local_mem_addr);
  u64 racu_gaddr = racu_sys_mem_addr;
  u64 meta_gaddr = meta_sys_mem_addr;
  ASSERT_TENSOR_SIZE(N, C, H, W);
  // because max enc_size is 19bit
  ASSERT((W * sg_min(C, NPU_NUM) * type_len) <=
         (1 << (12 + NNVLC_ALIGN_SHIFT)));
  ASSERT(data_format == GDMA_INT8 || data_format == GDMA_INT16 ||
             data_format == GDMA_FP16 || data_format == GDMA_BF16);
  ASSERT(is_signed >= 0 && is_signed < 2);
  ASSERT(data_format == GDMA_FP16 || data_format == GDMA_BF16 || !is_signed ||
         (bias0 < 128 && bias1 < 128));
  ASSERT((racu_h_stride & (NNVLC_ALIGN_BYTES - 1)) == 0 &&
         (racu_c_stride & (NNVLC_ALIGN_BYTES - 1)) == 0 &&
         (racu_n_stride & (NNVLC_ALIGN_BYTES - 1)) == 0);
#endif
#endif
}

void tensor_gdma_reverse_check(
  u64 src_addr,
  u64 dst_addr,
  int32_t N,
  int32_t C,
  int32_t H,
  int32_t W,
  uint32_t src_n_stride,
  uint32_t src_c_stride,
  uint32_t src_h_stride,
  uint32_t dst_n_stride,
  uint32_t dst_c_stride,
  uint32_t dst_h_stride,
  int32_t reverse_axis,
  int32_t data_format,
  int32_t direction) {

  ASSERT_TENSOR_SIZE(N, C, H, W);
  ASSERT(reverse_axis >= 0 && reverse_axis < 4);
#ifndef __sg2262__
  ASSERT(data_format != GDMA_FP20);
#endif
  uint32_t src_npu_idx = 0, dst_npu_idx = 0;
  if (SRC_IS_LOCAL(direction)) {
      src_npu_idx = src_addr / LOCAL_MEM_SIZE;
      src_addr = CALC_LOCAL_ADDR(0, src_addr);
      ASSERT(reverse_axis == 1);
  }
  if (DST_IS_LOCAL(direction)) {
      dst_npu_idx = dst_addr / LOCAL_MEM_SIZE;
      dst_addr = CALC_LOCAL_ADDR(0, dst_addr);
  }
#ifdef __sg2262__
  if (SRC_IS_LOCAL(direction) && DST_IS_LOCAL(direction)) {
      //local mem data should aligned
      ASSERT(src_h_stride == (uint32_t)W);
      ASSERT(src_c_stride == ALIGN((uint32_t)H * src_h_stride, get_eu_num_from_type(data_format)));
      ASSERT(src_n_stride == ALIGN((uint32_t)C + src_npu_idx, NPU_NUM) / NPU_NUM * src_c_stride);
      ASSERT(dst_h_stride == (uint32_t)W);
      ASSERT(dst_c_stride == ALIGN((uint32_t)H * dst_h_stride, get_eu_num_from_type(data_format)));
      ASSERT(dst_n_stride == ALIGN((uint32_t)C + dst_npu_idx, NPU_NUM) / NPU_NUM * dst_c_stride);
  } else if (SRC_IS_LOCAL(direction) && !DST_IS_LOCAL(direction)) {
      //local mem data should aligned
      ASSERT(src_h_stride == (uint32_t)W);
      ASSERT(src_c_stride == ALIGN((uint32_t)H * src_h_stride, get_eu_num_from_type(data_format)));
      ASSERT(src_n_stride == ALIGN((uint32_t)C + src_npu_idx, NPU_NUM) / NPU_NUM * src_c_stride);
      //global mem data should CONTINUOUS
      ASSERT(dst_h_stride == (uint32_t)W);
      ASSERT(dst_c_stride == (uint32_t)(H * W));
      ASSERT(dst_n_stride == (uint32_t)(C * H * W));
  } else if (!SRC_IS_LOCAL(direction) && DST_IS_LOCAL(direction)) {
      //local mem data should aligned
      ASSERT(dst_h_stride == (uint32_t)W);
      ASSERT(dst_c_stride == ALIGN((uint32_t)H * dst_h_stride, get_eu_num_from_type(data_format)));
      ASSERT(dst_n_stride == ALIGN((uint32_t)C + dst_npu_idx, NPU_NUM) / NPU_NUM * dst_c_stride);
      //global mem data should CONTINUOUS
      ASSERT(src_h_stride == (uint32_t)W);
      ASSERT(src_c_stride == (uint32_t)(H * W));
      ASSERT(src_n_stride == (uint32_t)(C * H * W));
  } else {
      //global mem data should CONTINUOUS
      ASSERT(src_h_stride == (uint32_t)W);
      ASSERT(src_c_stride == (uint32_t)(H * W));
      ASSERT(src_n_stride == (uint32_t)(C * H * W));
      ASSERT(dst_h_stride == (uint32_t)W);
      ASSERT(dst_c_stride == (uint32_t)(H * W));
      ASSERT(dst_n_stride == (uint32_t)(C * H * W));
  }
#endif
}

void gdma_lossy_compress_check(u64 src_addr,      // local_addr or sys
                               int src_local_idx, // use only from local_mem
                               int N, int C, int H, int W,
                               stride_type src_N_stride,
                               stride_type src_C_stride,
                               stride_type src_H_stride,
                               u64 dst_addr, // sys
                               stride_type dst_N_stride,
                               stride_type dst_C_stride,
                               stride_type dst_H_stride,
                               int direction // Support S2S, L2S
) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT_TENSOR_SIZE(N, C, H, W);
  ASSERT(src_addr % 4 == 0);
  ASSERT(dst_addr % 128 == 0);
  ASSERT(!is_smem(src_addr) && !is_smem(dst_addr) && "can't be static memory\n");
  ASSERT((is_gmem(dst_addr) || is_l2mem(dst_addr)) && "must be sys memory\n");
#endif
#endif
}

void gdma_lossy_decompress_check(u64 src_addr, // sys
                                 int N, int C, int H, int W,
                                 stride_type src_N_stride,
                                 stride_type src_C_stride,
                                 stride_type src_H_stride,
                                 u64 dst_addr,      // local_addr or sys
                                 int dst_local_idx, // use only from local_mem
                                 stride_type dst_N_stride,
                                 stride_type dst_C_stride,
                                 stride_type dst_H_stride,
                                 int direction // S2S or S2L
) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT_TENSOR_SIZE(N, C, H, W);
  ASSERT(dst_addr % 4 == 0);
  ASSERT(src_addr % 128 == 0);
  ASSERT((is_gmem(src_addr) || is_l2mem(src_addr)) && "must be sys memory\n");
  ASSERT(!is_smem(src_addr) && !is_smem(dst_addr) && "can't be static memory\n");
#endif
#endif
}

void gdma_lossy_compress_reduce_check(
    u64 src_addr,      // local_addr, global_addr or l2_addr
    int src_local_idx, // use only from local_mem
    int N, int C, int H, int W, stride_type src_N_stride,
    stride_type src_C_stride, stride_type src_H_stride,
    u64 dst_addr, // l2_addr
    stride_type dst_N_stride, stride_type dst_C_stride,
    stride_type dst_H_stride,
    int direction, // Support S2S, L2S
    int reduce_psum_op, int reduce_opcode) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT_TENSOR_SIZE(N, C, H, W);
  ASSERT(src_addr % 4 == 0);
  ASSERT(dst_addr % 128 == 0);
  ASSERT(!is_smem(src_addr) && "can't be static memory\n");
  // ASSERT(is_l2mem(dst_addr),
  //                "must be l2 memory, dst_ddr:0x%llx", dst_addr);
#endif
#endif
}

void gdma_lossy_decompress_reduce_check(
    u64 src_addr, // sys
    int N, int C, int H, int W, stride_type src_N_stride,
    stride_type src_C_stride, stride_type src_H_stride,
    u64 dst_addr, // local_addr or sys
    int dst_local_idx, stride_type dst_N_stride, stride_type dst_C_stride,
    stride_type dst_H_stride,
    int direction, // only S2S
    int reduce_psum_op, int reduce_opcode) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT_TENSOR_SIZE(N, C, H, W);
  ASSERT(dst_addr % 4 == 0);
  ASSERT(src_addr % 128 == 0);
  ASSERT((is_gmem(src_addr) || is_l2mem(src_addr)) && "must be sys memory");
  // ASSERT(is_l2mem(dst_addr),
  //                "must be l2 memory, dst_ddr:0x%llx", dst_addr);
#endif
#endif
}

// Only support GDMA write
// derived from tensor_stride_move_check
void tensor_stride_move_reduce_check(
    int local_mem_start_addr,
    int local_mem_idx,
    u64 sys_mem_start_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int src_format, int direction,
    int transpose, // N/C transpose
    int reduce_psum_op,
    int reduce_opcode) {

#ifdef USING_CMODEL
  ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                  "can't be static memory sys_addr:0x%llx",
                  sys_mem_start_addr);
  ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
  ASSERT_FS_INFO(direction == GDMA_L2S ||
                      direction == GDMA_S2L,
                  "directin=%d", direction);
  // int type_len = get_gdma_format_type_len(src_format);
  // ASSERT_WSTRIDE(src_W_stride, type_len);
  // ASSERT_WSTRIDE(dst_W_stride, type_len);
#ifdef __sg2262__
  ASSERT(transpose == 0);
#else
  ASSERT(src_format != GDMA_FP20);
#endif
  if (src_format == GDMA_INT32) {
    ASSERT(reduce_opcode != GDMA_ARE_MUL);
  }
#endif
}

void tensor_general_move_local_cross_core_check(
    u64 src_addr,      // local_addr or smem_addr
    int src_local_idx, // use only from local_mem
    int src_core_idx, int src_N, int src_C, int src_H, int src_W,
    stride_type src_N_stride, stride_type src_C_stride,
    stride_type src_H_stride, stride_type src_W_stride, int src_format,
    u64 dst_addr,      // local_addr or smem_addr
    int dst_local_idx, // use only to local_mem
    int dst_core_idx, int dst_N, int dst_C, int dst_H, int dst_W,
    stride_type dst_N_stride, stride_type dst_C_stride,
    stride_type dst_H_stride, stride_type dst_W_stride) {
  // sg2260e deleted this instr
  ASSERT(0);
#ifdef USING_CMODEL
  ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
  ASSERT_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
  ASSERT(dst_N * dst_C * dst_H * dst_W == src_N * src_C * src_H * src_W);
  ASSERT(src_format != GDMA_FP20);
  ASSERT(0 <= src_local_idx && src_local_idx < NPU_NUM);
  ASSERT(0 <= dst_local_idx && dst_local_idx < NPU_NUM);
  ASSERT(0 <= dst_core_idx && dst_core_idx < MAX_TPU_CORE_NUM);
  ASSERT(0 <= src_core_idx && src_core_idx < MAX_TPU_CORE_NUM);
  ASSERT(src_core_idx != dst_core_idx);
  ASSERT(src_W_stride == 1);
  ASSERT(dst_W_stride == 1);
  int type_len = get_gdma_format_type_len(src_format);
  ASSERT_WSTRIDE(src_W_stride, type_len);
  ASSERT_WSTRIDE(dst_W_stride, type_len);
#endif
}

void atomic_transfer_general_check(
    u64 src_addr, // local_addr or smem_addr
    int src_core_idx, int src_N, int src_C, int src_H, int src_W,
    stride_type src_N_stride, stride_type src_C_stride,
    stride_type src_H_stride, stride_type src_W_stride, int src_format,
    u64 dst_addr, // local_addr or smem_addr
    int dst_core_idx, int dst_N, int dst_C, int dst_H, int dst_W,
    stride_type dst_N_stride, stride_type dst_C_stride,
    stride_type dst_H_stride, stride_type dst_W_stride) {
  // sg2260e deleted this instr
  ASSERT(0);
#ifdef USING_CMODEL
  ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
  ASSERT_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
  ASSERT(dst_N * dst_C * dst_H * dst_W == src_N * src_C * src_H * src_W);
  ASSERT(src_format != GDMA_FP20);
  ASSERT(0 <= dst_core_idx && dst_core_idx < MAX_TPU_CORE_NUM);
  ASSERT(0 <= src_core_idx && src_core_idx < MAX_TPU_CORE_NUM);
  ASSERT(src_core_idx != dst_core_idx);
  ASSERT(src_W_stride == 1);
  ASSERT(dst_W_stride == 1);
  int type_len = get_gdma_format_type_len(src_format);
  ASSERT_WSTRIDE(src_W_stride, type_len);
  ASSERT_WSTRIDE(dst_W_stride, type_len);
  ASSERT(is_smem(src_addr) || is_lmem(src_addr));
  ASSERT(is_smem(dst_addr) || is_lmem(dst_addr));
#endif
}

// Only support GDMA write
// derived from tensor_general_move_check
void tensor_general_move_reduce_check(
    u64 src_addr,      // local_addr or global_addr
    int src_local_idx, // use only from local_mem
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    int src_format,
    u64 dst_addr,      // local_addr or global_addr
    int dst_local_idx, // use only to local_mem
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int direction,
    int transpose, // N/C transpose
    int reduce_psum_op,
    int reduce_opcode) {

#ifdef USING_CMODEL
  ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
  ASSERT_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
  ASSERT_FS_INFO(dst_N * dst_C * dst_H * dst_W ==
                      src_N * src_C * src_H * src_W,
                  "dst_count=%d, src_count=%d", dst_N * dst_C * dst_H * dst_W,
                  src_N * src_C * src_H * src_W);
#ifdef __sg2262__
  ASSERT(transpose == 0);
#else
  if (src_format == GDMA_FP20) {
    ASSERT(direction == GDMA_S2S);
    ASSERT(transpose == 0);
    ASSERT(src_addr % 128 == 0);
    ASSERT(dst_addr % 128 == 0);
    ASSERT_WSTRIDE_FP20(src_W_stride);
    ASSERT_WSTRIDE_FP20(dst_W_stride);
    ASSERT_COMPACT_FP20((u32)src_N, (u32)src_C, (u32)src_H, (u32)src_W,
                        src_N_stride, src_C_stride, src_H_stride, src_W_stride)
    ASSERT_COMPACT_FP20((u32)dst_N, (u32)dst_C, (u32)dst_H, (u32)dst_W,
                        dst_N_stride, dst_C_stride, dst_H_stride, dst_W_stride)
  } else {
    // int type_len = get_gdma_format_type_len(src_format);
    // ASSERT_WSTRIDE(src_W_stride, type_len);
    // ASSERT_WSTRIDE(dst_W_stride, type_len);
  }
#endif
  ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                  "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                  src_addr, dst_addr);
  if (src_format == GDMA_INT32) {
    ASSERT(reduce_opcode != GDMA_ARE_MUL);
  }
  ASSERT(!DST_IS_LOCAL(direction));
#endif
}

void random_mask_init_seed_check(u64 src_addr, u64 dst_addr, int dst_local_idx,
                                 int src_N, int src_C, int src_H, int src_W,
                                 uint64_t size, stride_type dst_N_stride,
                                 stride_type dst_C_stride,
                                 stride_type dst_H_stride,
                                 stride_type dst_W_stride, int src_format) {

  // sg2260e deleted this instr
  ASSERT(0);
#ifdef USING_CMODEL
  ASSERT(!is_smem(src_addr) && "can't be static memory\n");
  ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
  ASSERT(src_format != GDMA_FP20);
  ASSERT(src_N == 1);
  ASSERT(dst_W_stride == 1);
#endif
}

void random_mask_check(u64 src_addr, u64 dst_addr, int dst_local_idx, int src_N,
                       int src_C, int src_H, int src_W, uint64_t size,
                       stride_type dst_N_stride, stride_type dst_C_stride,
                       stride_type dst_H_stride, stride_type dst_W_stride,
                       int src_format, int inter_state) {

  // sg2260e deleted this instr
  ASSERT(0);
#ifdef USING_CMODEL
  ASSERT(!is_smem(src_addr) && "can't be static memory\n");
  ASSERT_TENSOR_SIZE(src_N, src_C, src_H, src_W);
  ASSERT(src_format != GDMA_FP20);
  ASSERT(src_N == 1);
  ASSERT(dst_W_stride == 1);
#endif
}
