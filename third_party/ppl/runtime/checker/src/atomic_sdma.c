#include "atomic_sdma.h"

#define ASSERT_SDMA_TENSOR_NSIZE(n) \
    ASSERT_FS_INFO(n>0 && n<=SDMA_MAX_N, #n "=%d", n)

#define ASSERT_SDMA_TENSOR_CSIZE(c) \
    ASSERT_FS_INFO(c>0 && c<=SDMA_MAX_C, #c "=%d", c)

#define ASSERT_SDMA_TENSOR_HSIZE(h) \
    ASSERT_FS_INFO(h>0 && h<=SDMA_MAX_H, #h "=%d", h)

#define ASSERT_SDMA_TENSOR_WSIZE(w) \
    ASSERT_FS_INFO(w>0 && w<=SDMA_MAX_W, #w "=%d", w)

#define ASSERT_SDMA_TENSOR_SIZE(n,c,h,w) \
    ASSERT_SDMA_TENSOR_NSIZE(n); \
    ASSERT_SDMA_TENSOR_CSIZE(c); \
    ASSERT_SDMA_TENSOR_HSIZE(h); \
    ASSERT_SDMA_TENSOR_WSIZE(w)

#define ASSERT_SDMA_WSTRIDE(wstr, byte_len) \
    ASSERT_FS_INFO((wstr * byte_len <= SDMA_MAX_WSTRIDE_BYTE_LEN) && (wstr != 0), "W stride byte len = %d", wstr * byte_len)

#define ASSERT_SDMA_WSTRIDE_FP20(wstr) \
    ASSERT_FS_INFO(wstr == 1, "When data type is fp20, W stride should be 1")

#define ASSERT_SDMA_COMPACT_FP20(n,c,h,w,nstr,cstr,hstr,wstr) \
    ASSERT_SDMA_WSTRIDE_FP20(wstr); \
    ASSERT_FS_INFO(hstr == (w), "When data type is fp20, 51 elements constitute fp20 block"); \
    ASSERT_FS_INFO(cstr == (h * hstr), "When data type is fp20, c stride should be compacted"); \
    ASSERT_FS_INFO(nstr == (c * cstr), "When data type is fp20, n stride should be compacted");

inline static u64 sdma_get_lane_mask() {
    return 0xffffffffffffffff;
}

typedef enum {
  SDMA_ARE_NOP = 0,
  SDMA_ARE_MUL = 1,
  SDMA_ARE_MAX = 2,
  SDMA_ARE_MIN = 3,
  SDMA_ARE_ADD = 4,
} SDMA_ARE_OPCODE_TYPE;

typedef enum {
  SDMA_INT8 = 0,
  SDMA_FP16 = 1,
  SDMA_FP32 = 2,
  SDMA_INT16 = 3,
  SDMA_INT32 = 4,
  SDMA_BF16 = 5,
  SDMA_FP20 = 6,
  SDMA_FP8_E4M3 = 7,
  SDMA_FP8_E5M2 = 8,
  SDMA_FORMAT_NUM,
} SDMA_FORMAT;

static inline int get_sdma_format_type_len(int t) {
  switch (t) {
    case SDMA_INT8:
    case SDMA_FP8_E4M3:
    case SDMA_FP8_E5M2:
      return 1;
    case SDMA_FP16:
    case SDMA_BF16:
    case SDMA_INT16:
      return 2;
    case SDMA_FP32:
    case SDMA_INT32:
      return 4;
  }
  return 0;
}

void sdma_tensor_general_move_check(
    u64 src_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    int src_format,
    u64 dst_addr,
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int transpose  // N/C transpose
) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
    ASSERT_SDMA_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT_SDMA_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
    ASSERT_FS_INFO(dst_N * dst_C * dst_H * dst_W ==
                       src_N * src_C * src_H * src_W,
                   "dst_count=%d, src_count=%d", dst_N * dst_C * dst_H * dst_W,
                   src_N * src_C * src_H * src_W);
    int type_len = get_sdma_format_type_len(src_format);
    ASSERT_SDMA_WSTRIDE(src_W_stride, type_len);
    ASSERT_SDMA_WSTRIDE(dst_W_stride, type_len);
    if (src_format == SDMA_FP20) {
        ASSERT(transpose == 0);
        ASSERT(src_addr % ALIGN_BYTES == 0);
        ASSERT(dst_addr % ALIGN_BYTES == 0);
        ASSERT_SDMA_WSTRIDE_FP20(src_W_stride);
        ASSERT_SDMA_WSTRIDE_FP20(dst_W_stride);
    } else {
        // int type_len = get_sdma_format_type_len(src_format);
        // ASSERT_SDMA_WSTRIDE(src_W_stride, type_len);
        // ASSERT_SDMA_WSTRIDE(dst_W_stride, type_len);
    }
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
    ASSERT_FS_INFO(!is_lmem(src_addr) && !is_lmem(dst_addr),
                   "can't be local memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
#endif
#endif
}

void sdma_fill_constant_check_global_stride(
    u64 sys_mem_start_addr,
    const void* const_val,
    int data_format,
    int dst_N, int dst_C, int dst_H, int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int stride_enable
) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
    ASSERT_SDMA_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
    if (data_format == SDMA_FP20) {
        ASSERT(sys_mem_start_addr % ALIGN_BYTES == 0);
        ASSERT_SDMA_WSTRIDE_FP20(dst_W_stride);
    } else {
        // if (stride_enable) {
        //     ASSERT_SDMA_WSTRIDE(dst_W_stride, get_sdma_format_type_len(data_format));
        // }
    }
    ASSERT_FS_INFO(!is_smem(sys_mem_start_addr),
                   "can't be static memory sys_addr:0x%llx",
                   sys_mem_start_addr);
    ASSERT_FS_INFO(!is_lmem(sys_mem_start_addr),
                   "can't be local memory sys_addr:0x%llx",
                   sys_mem_start_addr);
#endif
#endif
}

void sdma_general_check(
    u64 src_addr,
    u64 dst_addr,
    int src_format,
    stride_type src_count,
    int src_is_const
) {

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
    ASSERT(src_format != SDMA_FP20);
    ASSERT_FS_INFO(src_count > 0, "src_count=%d", src_count);
    ASSERT(src_is_const == 0 || src_is_const == 1);
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
    ASSERT_FS_INFO(!is_lmem(src_addr) && !is_lmem(dst_addr),
                   "can't be local memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
#endif
#endif
}

void sdma_general_cwtrans_check(
    u64 src_addr,
    u64 dst_addr,
    int src_N,  int src_C,
    int src_H,  int src_W,
    int src_format,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int stride_enable
) {
    ASSERT_FS_INFO(0, "Not support");
}

void sdma_tensor_general_move_with_mask_check(
    u64 src_addr, // global addr or local addr
    u64 mask_addr, // global addr or local addr
    u64 dst_addr, // global addr or local addr
    int src_format,
    int mask_format,
    int N,
    int C,
    int H,
    int W
) {
    ASSERT_FS_INFO(0, "Not support");
}

void sdma_tensor_move_nonzero_check(
    u64 src_addr,
    u64 dst_addr,
    int src_format, // Only INT8/INT16/INT32
    int dst_format, // Only INT8/INT16/INT32
    int N,
    int C,
    int H,
    int W,
    u32 base_idx
) {
    ASSERT_FS_INFO(0, "Not support");
}

// index addr aligned to 512byte can get better performance
// wsize is larger can get better performance
void sdma_tensor_gather_check(
    u64 src_addr,
    u64 index_addr,
    u64 dst_addr,
    u32 const_val,
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
    int stride_enable
) {
#ifdef USING_CMODEL
#ifdef __sg2262__
    ASSERT_FS_INFO(0, "Not support");
#else

    ASSERT_SDMA_TENSOR_CSIZE(C);
    ASSERT_SDMA_TENSOR_WSIZE(src_W);
    ASSERT(src_format != SDMA_FP20);

    if (stride_enable) ASSERT(index_H_stride == 1);
#endif
#endif
}

// index addr aligned to 512byte can get better performance
// wsize is larger can get better performance
void sdma_tensor_scatter_check(
    u64 src_addr,
    u64 index_addr,
    u64 dst_addr,
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
    int inplace_add) {

#ifdef USING_CMODEL
#ifdef __sg2262__
    ASSERT_FS_INFO(0, "Not support");
#else
    ASSERT(C <= SDMA_MAX_C);
    ASSERT(src_W <= SDMA_MAX_W);
    if (stride_enable) ASSERT(index_H_stride == 1);
    if(src_format == SDMA_FP8_E4M3 || src_format == SDMA_FP8_E5M2) {
        ASSERT(inplace_add = 0);
    }
#endif
#endif
}

void sdma_tensor_reverse_check(
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
    int32_t data_format)
{
  ASSERT_FS_INFO(0, "Not support");
}

void sdma_lossy_compress_check(
    u64 src_addr,
    int N,
    int C,
    int H,
    int W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr, // sys_addr
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride
) {

#ifdef USING_CMODEL
#ifdef __sg2262__
    ASSERT_FS_INFO(0, "Not support");
#else
    ASSERT_SDMA_TENSOR_SIZE(N, C, H, W);
    ASSERT(src_addr % 4 == 0);
    ASSERT(dst_addr % 128 == 0);
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
    ASSERT_FS_INFO(!is_lmem(src_addr) && !is_lmem(dst_addr),
                   "can't be local memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
#endif
#endif
}

void sdma_lossy_decompress_check(
    u64 src_addr,
    int N,
    int C,
    int H,
    int W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride
) {
#ifdef USING_CMODEL
#ifdef __sg2262__
    ASSERT_FS_INFO(0, "Not support");
#else
    ASSERT_SDMA_TENSOR_SIZE(N, C, H, W);
    ASSERT(dst_addr % 4 == 0);
    ASSERT(src_addr % 128 == 0);
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
    ASSERT_FS_INFO(!is_lmem(src_addr) && !is_lmem(dst_addr),
                   "can't be local memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
#endif
#endif
}

void sdma_lossy_compress_reduce_check(
    u64 src_addr,
    int N,
    int C,
    int H,
    int W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr, // sys_addr
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int reduce_psum_op,
    int reduce_opcode
) {
#ifdef USING_CMODEL
#ifdef __sg2262__
    ASSERT_FS_INFO(0, "Not support");
#else
    ASSERT_SDMA_TENSOR_SIZE(N, C, H, W);
    ASSERT(src_addr % 4 == 0);
    ASSERT(dst_addr % 128 == 0);
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_lmem(src_addr),
                   "can't be static or local memory, src_addr:0x%llx",
                   src_addr);
    ASSERT_FS_INFO(is_l2mem(dst_addr),
                   "must be l2 memory, dst_ddr:0x%llx", dst_addr);
#endif
#endif
}

void sdma_lossy_decompress_reduce_check(
    u64 src_addr,
    int N,
    int C,
    int H,
    int W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    u64 dst_addr,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    int reduce_psum_op,
    int reduce_opcode
) {

#ifdef USING_CMODEL
#ifdef __sg2262__
    ASSERT_FS_INFO(0, "Not support");
#else
    ASSERT_SDMA_TENSOR_SIZE(N, C, H, W);
    ASSERT(dst_addr % 4 == 0);
    ASSERT(src_addr % 128 == 0);
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_lmem(src_addr),
                   "can't be static or local memory, src_addr:0x%llx",
                   src_addr);
    ASSERT_FS_INFO(is_l2mem(dst_addr),
                   "must be l2 memory, dst_ddr:0x%llx", dst_addr);
#endif
#endif
}

// Only support GDMA write
void sdma_tensor_reduce_check(
    u64 src_addr,
    int src_N,
    int src_C,
    int src_H,
    int src_W,
    stride_type src_N_stride,
    stride_type src_C_stride,
    stride_type src_H_stride,
    stride_type src_W_stride,
    int src_format,
    u64 dst_addr,
    int dst_N,
    int dst_C,
    int dst_H,
    int dst_W,
    stride_type dst_N_stride,
    stride_type dst_C_stride,
    stride_type dst_H_stride,
    stride_type dst_W_stride,
    int transpose,  // N/C transpose, fp20 not support
    int reduce_psum_op,
    int reduce_opcode
) {

#ifdef USING_CMODEL
#ifdef __sg2262__
    ASSERT_FS_INFO(0, "Not support");
#else
    ASSERT_SDMA_TENSOR_SIZE(src_N, src_C, src_H, src_W);
    ASSERT_SDMA_TENSOR_SIZE(dst_N, dst_C, dst_H, dst_W);
    ASSERT_FS_INFO(dst_N * dst_C * dst_H * dst_W ==
                       src_N * src_C * src_H * src_W,
                   "dst_count=%d, src_count=%d", dst_N * dst_C * dst_H * dst_W,
                   src_N * src_C * src_H * src_W);
    if (src_format == SDMA_FP20) {
        // ASSERT(direction == SDMA_VALUE_DIR_S2S);
        ASSERT(transpose == 0);
        ASSERT(src_addr % 128 == 0);
        ASSERT(dst_addr % 128 == 0);
        ASSERT_SDMA_COMPACT_FP20((u32)src_N, (u32)src_C, (u32)src_H, (u32)src_W, src_N_stride, src_C_stride, src_H_stride, src_W_stride)
        ASSERT_SDMA_COMPACT_FP20((u32)dst_N, (u32)dst_C, (u32)dst_H, (u32)dst_W, dst_N_stride, dst_C_stride, dst_H_stride, dst_W_stride)
    } else {
        // int type_len = get_sdma_format_type_len(src_format);
        // ASSERT_SDMA_WSTRIDE(src_W_stride, type_len);
        // ASSERT_SDMA_WSTRIDE(dst_W_stride, type_len);
    }
    ASSERT_FS_INFO(!is_smem(src_addr) && !is_smem(dst_addr),
                   "can't be static memory, src_addr:0x%llx, dst_addr:0x%llx",
                   src_addr, dst_addr);
    if(src_format == SDMA_INT32) {
        ASSERT(reduce_opcode != SDMA_ARE_MUL);
    }
#endif
#endif
}
