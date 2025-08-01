#include "atomic_tiu.h"

// src_shape=[N,1,H,W], dst_shape=[N,C,H,W]
// storage aligned in local memory
void atomic_lane_broad_check(
    u32 src_addr, // in local memory
    u32 dst_addr, // in local memory
    int N,
    int H,
    int W,
    int dst_C,
    u64 lane_mask,
    PREC prec)
{

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT(prec != FP4 && prec != INT4);
#else
  ASSERT(prec < (1 << 3) && prec >= 0 && prec != INT4);
#endif
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(dst_C < (1 << 16) && dst_C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(src_addr % ALIGN_BYTES == 0);
  ASSERT(dst_addr % ALIGN_BYTES == 0);
  int start_npu = get_npu_index(dst_addr);
  ASSERT(start_npu + dst_C <= NPU_NUM);
#endif
}

void atomic_lane_broad_txp_check(
    u32 src_addr, // in local memory
    u32 dst_addr, // in local memory
    int N,
    int H,
    int W,
    int dst_C,
    PREC prec)
{

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  ASSERT(prec < (1 << 3) && prec >= 0 && prec != INT4);
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(dst_C < (1 << 16) && dst_C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(src_addr % ALIGN_BYTES == 0);
  ASSERT(dst_addr % ALIGN_BYTES == 0);
  int start_npu = get_npu_index(dst_addr);
  ASSERT(start_npu + dst_C <= NPU_NUM);
#endif
#endif
}
// src_shape=dst_shape=[N,C,H,W]
// storage aligned in local memory
void atomic_lane_copy_check(
    u32 src_addr, // in local memory
    u32 dst_addr, // in local memory
    int N,
    int C,
    int H,
    int W,
    PREC prec)
{

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT(prec != FP4 && prec != INT4);
#else
  ASSERT(prec < (1 << 3) && prec >= 0 && prec != INT4);
#endif
  ASSERT(N < (1 << 16) && N > 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(H < (1 << 16) && H > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(dst_addr != src_addr);
  ASSERT(src_addr % ALIGN_BYTES == 0);
  ASSERT(dst_addr % ALIGN_BYTES == 0);
#endif
}

// src_shape=[1,1,1,W], dst_shape=[1,C,1,W]
// storage aligned in local memory
void atomic_static_broad_check(
    u32 src_addr, // in static memory
    u32 dst_addr, // in local memory
    int C,
    int W,
    u64 lane_mask,
    PREC prec)
{

#ifdef USING_CMODEL
  int start_npu = get_npu_index(dst_addr);
  ASSERT(start_npu + C <= NPU_NUM);
#ifdef __sg2262__
  ASSERT(prec != FP4 && prec != INT4);
#else
  ASSERT(prec < (1 << 3) && prec >= 0 && prec != INT4);
#endif
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(src_addr % ALIGN_BYTES == 0);
  ASSERT(dst_addr % ALIGN_BYTES == 0);
#endif
}

void atomic_static_broad_txp_check(
    u32 src_addr, // in static memory
    u32 dst_addr, // in local memory
    int C,
    int W,
    PREC prec)
{

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  int start_npu = get_npu_index(dst_addr);
  ASSERT(start_npu + C <= NPU_NUM);
  ASSERT(prec < (1 << 3) && prec >= 0 && prec != INT4);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(W < (1 << 16) && W > 0);
  ASSERT(src_addr % ALIGN_BYTES == 0);
  ASSERT(dst_addr % ALIGN_BYTES == 0);
#endif
#endif
}

// src_shape=dst_shape=[1,C,1,1]
// storage compacted in local memory
void atomic_static_distribute_check(
    u32 src_addr, // in static memory
    u32 dst_addr, // in local memory
    int C,
    u64 lane_mask,
    PREC prec)
{

#ifdef USING_CMODEL
  int start_npu = get_npu_index(dst_addr);
  ASSERT(start_npu == 0);
  ASSERT(C < (1 << 16) && C > 0);
#ifdef __sg2262__
  ASSERT(prec != FP4 && prec != INT4);
#else
  ASSERT(prec < (1 << 3) && prec >= 0 && prec != INT4);
#endif
  ASSERT(src_addr % ALIGN_BYTES == 0);
  ASSERT(dst_addr % get_bytesize(prec) == 0);
#endif
}


void atomic_static_distribute_txp_check(
    u32 src_addr, // in static memory
    u32 dst_addr, // in local memory
    int C,
    PREC prec)
{

#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT_FS_INFO(0, "Not support");
#else
  int start_npu = get_npu_index(dst_addr);
  ASSERT(start_npu == 0);
  ASSERT(C < (1 << 16) && C > 0);
  ASSERT(prec < (1 << 3) && prec >= 0 && prec != INT4);
  ASSERT(src_addr % ALIGN_BYTES == 0);
  ASSERT(dst_addr % get_bytesize(prec) == 0);
#endif
#endif
}

