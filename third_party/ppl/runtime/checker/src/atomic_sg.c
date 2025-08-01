#include "atomic_tiu.h"

#ifdef USING_CMODEL
typedef struct {
  int n;
  int c;
  int h;
  int w;
} dim4_t;

#define SWAP(a, b, type)  \
  do {                    \
    type tmp = a;         \
    a = b;                \
    b = tmp;              \
  } while (0)

static inline int ceiling_func(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

static inline int get_local_cstride(int h, int w, bool align, PREC precision) {
  return align ? ALIGN(h * w, get_eu_num(precision)) : (h * w);
}

static inline int get_local_nstride(int c_stride, int c, u32 local_addr) {
  int npu_idx = (local_addr & (NPU_NUM * LOCAL_MEM_SIZE - 1)) / LOCAL_MEM_SIZE;
  return (ceiling_func(c + npu_idx, NPU_NUM) * c_stride);
}

void assert_diff_bank(u32* addr, dim4_t* shape, int* align, PREC* prec, int num) {
  u32* start_addr = (u32*)malloc(num * sizeof(int));
  int* index = (int*)malloc(num * sizeof(int));
  for (int i = 0; i < num; i++) {
    start_addr[i] = addr[i] - get_npu_index(addr[i]) * LOCAL_MEM_SIZE;
    index[i] = i;
  }
  for (int i = 0; i < num; i++) {
    for (int j = i + 1; j < num; j++) {
      if (start_addr[j] < start_addr[i]) {
        SWAP(start_addr[i], start_addr[j], u32);
        SWAP(index[i], index[j], int);
      }
    }
  }

  int* tensor_size = (int*)malloc(num * sizeof(int));
  for (int i = 0; i < num; i++) {
    int k = index[i];
    int cstride = get_local_cstride(shape[k].h, shape[k].w, align[k], prec[k]);
    int nstride = get_local_nstride(cstride, shape[k].c, addr[k]);
    tensor_size[i] = shape[k].n * nstride * get_bytesize(prec[k]);
  }

  for (int i = 0; i < num; i++) {
    int begin = start_addr[i] / LOCAL_BANK_SIZE;
    int end = (start_addr[i] + tensor_size[i] - get_bytesize(prec[index[i]]))
              / LOCAL_BANK_SIZE;
    if (i+1 < num) {
      int next = start_addr[i+1] / LOCAL_BANK_SIZE;
      ASSERT_FS_INFO(begin != next && end != next,
                     "bank confilict: addr: %u vs. %u, "
                     "%d vs. %d, %d vs. %d\n",
                     start_addr[i], start_addr[i+1],
                     begin, next, end, next);
    }
  }
  free(start_addr);
  free(index);
  free(tensor_size);
}
#endif

/* PL_gather_d1coor :A=[N,C,1,Wa], B=[1,Wr,1,1], R=[N,C,1,Wr]
 * PL_scatter_d1coor:A=[N,C,1,Wa], B=[1,Wa,1,1], R=[N,C,1,Wr]
 * A and R is aligned in local memory, A_start_npu == R_start_npu
 * B is compacted in local mem, support U8 and U16,
 * but aligned to 16 byte, B_start_npu = 0
 */
void atomic_pl_sgd1_check(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  int tensorA_w,
  int tensorR_n,
  int tensorR_c,
  int tensorR_w,
  int if_fill_const,
  u32 fill_const_val,
  int limit_enable,
  PREC B_prec,
  PREC R_prec,
  SG_OP op)
{

#ifdef USING_CMODEL
  int tensorB_c = op == PL_gather_d1coor ? tensorR_w : tensorA_w;
  ASSERT(op == PL_gather_d1coor || op == PL_scatter_d1coor);
  ASSERT(tensorA_addr % ALIGN_BYTES == 0);
  ASSERT(tensorB_addr % ALIGN_BYTES == 0);
  ASSERT(tensorR_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(tensorB_addr) == 0);
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorR_addr));
#ifdef __sg2262__
  ASSERT(R_prec != FP4 && R_prec != INT4); //only 4bit not support
#else
  ASSERT(R_prec != INT4); //only 4bit not support
#endif
  ASSERT(B_prec == INT8 || B_prec == INT16);
  ASSERT(if_fill_const == 0 || if_fill_const == 1);
  ASSERT(tensorR_n < (1 << 16) && tensorR_n > 0);
  ASSERT(tensorR_c < (1 << 16) && tensorR_c > 0);
  ASSERT(tensorR_w < (1 << 16) && tensorR_w > 0);
  ASSERT(tensorA_w < (1 << 16) && tensorA_w > 0);
  ASSERT(tensorB_c < (1 << 16) && tensorB_c > 0);
  ASSERT(limit_enable == 0 || limit_enable == 1);
  if (op == PL_scatter_d1coor) {
    ASSERT(if_fill_const == 0);
  }
#endif
}

/* PL_gather_d2coor :A=[N,C,Ha,Wa], B=[1,Wr,1,1], R=[N,C,1,Wr]
 * PL_scatter_d2coor:A=[N,C,1,Wa], B=[1,Wa,1,1], R=[N,C,Hr,Wr]
 * A and R is aligned in local memory, A_start_npu == R_start_npu
 * B is compacted in local mem, but aligned to 16 byte
 * opd1 is uint16, but storaged as INT32 with [h, w], B_start_npu = 0
 */
void atomic_pl_sgd2_check(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  int tensorA_h,
  int tensorA_w,
  int tensorR_n,
  int tensorR_c,
  int tensorR_h,
  int tensorR_w,
  int if_fill_const,
  u32 fill_const_val,
  int limit_enable,
  PREC R_prec,
  SG_OP op)
{
#ifdef USING_CMODEL
  int tensorB_c = op == PL_gather_d2coor ? tensorR_w : tensorA_w;
  ASSERT(op == PL_gather_d2coor || op == PL_scatter_d2coor);
  if(op == PL_gather_d2coor) {
    ASSERT(tensorR_h == 1);
  } else {
    ASSERT(tensorA_h == 1);
  }
  ASSERT(tensorA_addr % ALIGN_BYTES == 0);
  ASSERT(tensorB_addr % ALIGN_BYTES == 0);
  ASSERT(tensorR_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(tensorB_addr) == 0);
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorR_addr));
#ifdef __sg2262__
  ASSERT(R_prec != FP4 && R_prec != INT4); //only 4bit not support
#else
  ASSERT(R_prec != INT4); //only 4bit not support
#endif
  ASSERT(if_fill_const == 0 || if_fill_const == 1);
  ASSERT(tensorR_n < (1 << 16) && tensorR_n > 0);
  ASSERT(tensorR_c < (1 << 16) && tensorR_c > 0);
  ASSERT(tensorR_h < (1 << 16) && tensorR_h > 0);
  ASSERT(tensorR_w < (1 << 16) && tensorR_w > 0);
  ASSERT(tensorA_h < (1 << 16) && tensorA_w > 0);
  ASSERT(tensorA_w < (1 << 16) && tensorA_w > 0);
  ASSERT(tensorB_c < (1 << 16) && tensorB_c > 0);
  ASSERT(limit_enable == 0 || limit_enable == 1);
  if (op == PL_scatter_d2coor) {
    ASSERT(if_fill_const == 0);
  }
#endif
}

/* PE_S_gather_d1coor: do not support bank confilict,
 * PE_S_gather_hzd: support bank confilict,
 * A=[1,C,1,A_w], B=[N,C,1,R_w],R=[N,C,1,R_w]
 * PE_S_scatter_d1coor: do not support bank confilict
 * PE_S_scatter_hzd: support bank confilict
 * A=[1,C,1,A_w], B=[N,C,1,A_w],R=[N,C,1,R_w]
 */
/* A is aligned in local memory, if A_cstride is 0
 * A_wstride=1;A_hstride=ceil(w,EU_NUM),cstride=0;nstride=0
 * B is aligned in local memory, support U8 and U16,
 * R is aligned in local memory
 * all start_npu is same
 */
void atomic_pes_sg_d1hzd_check(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  int tensorA_w,
  int tensorR_n,
  int tensorR_c,
  int tensorR_w,
  int A_cstride_is0,
  int if_fill_const,
  u32 fill_const_val,
  int limit_enable,
  PREC B_prec,
  PREC R_prec,
  SG_OP op)
{

#ifdef USING_CMODEL
  ASSERT(B_prec == INT8 || B_prec == INT16);
#ifdef __sg2262__
  ASSERT(R_prec != FP4 && R_prec != INT4); //only 4bit not support
#else
  ASSERT(R_prec != INT4); //only 4bit not support
#endif
  ASSERT(A_cstride_is0 == 0 || A_cstride_is0 == 1);
  ASSERT(if_fill_const < (1 << 1) && if_fill_const >= 0);
  ASSERT(limit_enable == 0 || limit_enable == 1);
  ASSERT(tensorR_n < (1 << 16) && tensorR_n > 0);
  ASSERT(tensorR_c < (1 << 16) && tensorR_c > 0);
  ASSERT(tensorR_w < (1 << 16) && tensorR_w > 0);
  ASSERT(tensorA_w < (1 << 16) && tensorA_w > 0);
  ASSERT(op == PE_S_gather_d1coor || op == PE_S_gather_hzd ||
         op == PE_S_scatter_d1coor || op == PE_S_scatter_hzd);
  ASSERT(tensorA_addr % ALIGN_BYTES == 0);
  ASSERT(tensorB_addr % ALIGN_BYTES == 0);
  ASSERT(tensorR_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorB_addr));
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorR_addr));
  if (op == PE_S_scatter_d1coor || op == PE_S_scatter_hzd) {
    ASSERT(if_fill_const == 0);
  }
  if (op == PE_S_gather_d1coor || op == PE_S_scatter_d1coor) {
    int B_w = op == PE_S_gather_d1coor ? tensorR_w : tensorA_w;
    int A_npu = get_npu_index(tensorA_addr);
    u32 addr[3] = {tensorA_addr, tensorB_addr, tensorR_addr};
    dim4_t shape[3] = {{1, A_cstride_is0 ? (NPU_NUM - A_npu) : tensorR_c, 1, tensorA_w},
                       {tensorR_n, tensorR_c, 1, B_w},
                       {tensorR_n, tensorR_c, 1, tensorR_w}};
    PREC prec_list[3] = {R_prec, B_prec, R_prec};
    int align[3] = {1, 1, 1};
    assert_diff_bank(addr, shape, align, prec_list, 3);
  }
#endif
}

/* PE_S_mask_select: do not support bank confilict
 * PE_S_mask_selhzd: support bank confilict
 * A, B, R are aligned in local memory,
 * if A_cstride is 0, A_wstride=1;A_hstride=ceil(w,EU_NUM),cstride=0;nstride=0
 * B support uint8/uint16/uint32
 * and mask_num is compacted in local mem which only support uint16
 * A=[1,C,1,A_w], B=[N, C, 1, A_w], R=[N,C,1,R_w], mask_num=[N,C,1,1]
 */
void atomic_pes_mask_sel_check(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  u32 mask_num_addr,
  int tensorA_w,
  int tensorB_n,
  int tensorB_c,
  int A_cstride_is0,
  PREC B_prec,
  PREC R_prec,
  SG_OP op)
{
#ifdef USING_CMODEL
#ifdef __sg2262__
  ASSERT(R_prec != FP4 && R_prec != INT4);  //only 4bit not support
#else
  ASSERT(R_prec != INT4);  //only 4bit not support
#endif
  ASSERT(B_prec == INT8 || B_prec == INT16 || B_prec == INT32);
  ASSERT(op == PE_S_mask_select || op == PE_S_mask_selhzd);
  ASSERT(A_cstride_is0 == 0 || A_cstride_is0 == 1);
  ASSERT(tensorB_n < (1 << 16) && tensorB_n > 0);
  ASSERT(tensorB_c < (1 << 16) && tensorB_c > 0);
  ASSERT(tensorA_w < (1 << 16) && tensorA_w > 0);
  ASSERT(tensorA_addr % ALIGN_BYTES == 0);
  ASSERT(tensorB_addr % ALIGN_BYTES == 0);
  ASSERT(tensorR_addr % ALIGN_BYTES == 0);
  ASSERT(mask_num_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorB_addr));
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorR_addr));
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(mask_num_addr));
  if (op == PE_S_mask_select) {
    u32 addr[4] = {tensorA_addr, tensorB_addr, tensorR_addr, mask_num_addr};
    int A_npu = get_npu_index(tensorA_addr);
    dim4_t shape[4] = {{1, A_cstride_is0 ? (NPU_NUM - A_npu) : tensorB_c, 1, tensorA_w},
                       {tensorB_n, tensorB_c, 1, tensorA_w},
                       {tensorB_n, tensorB_c, 1, tensorA_w},
                       {tensorB_n, tensorB_c, 1, 1}};
    PREC prec_list[4] = {R_prec, B_prec, R_prec, INT16};
    int align[4] = {1, 1, 1, 0};
    assert_diff_bank(addr, shape, align, prec_list, 4);
  }
#endif
}

/* PE_S_nonsero: do not support bank confilict
 * PE_S_nonzero_hzd: support bank confilict
 * A, R are aligned in local memory,
 * A support INT8/INT16/INT32, R support INT16/INT32
 * and mask_num is compacted in local mem which only support uint16
 * A=[N,C,1,W], R=[N,C,1,W],mask_num=[N,C,1,1]
 * all start_npu is same
 */
void atomic_pes_nonzero_check(
  u32 tensorA_addr,
  u32 tensorR_addr,
  u32 mask_num_addr,
  int tensorA_n,
  int tensorA_c,
  int tensorA_w,
  PREC A_prec,
  PREC R_prec,
  SG_OP op)
{
#ifdef USING_CMODEL
  ASSERT(A_prec == INT8 || A_prec == INT16 || A_prec == INT32);
  ASSERT(R_prec == INT16 || R_prec == INT32);
  ASSERT(op == PE_S_nonzero_hzd || op == PE_S_nonzero);
  ASSERT(tensorA_n < (1 << 16) && tensorA_n > 0);
  ASSERT(tensorA_c < (1 << 16) && tensorA_c > 0);
  ASSERT(tensorA_w < (1 << 16) && tensorA_w > 0);
  ASSERT(tensorA_addr % ALIGN_BYTES == 0);
  ASSERT(tensorR_addr % ALIGN_BYTES == 0);
  ASSERT(mask_num_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorR_addr));
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(mask_num_addr));
  if (op == PE_S_nonzero) {
    u32 addr[3] = {tensorA_addr, tensorR_addr, mask_num_addr};
    dim4_t shape[3] = {{tensorA_n, tensorA_c, 1, tensorA_w},
                       {tensorA_n, tensorA_c, 1, tensorA_w},
                       {tensorA_n, tensorA_c, 1, 1}};
    PREC prec_list[3] = {A_prec, R_prec, INT16};
    int align[3] = {1, 1, 0};
    assert_diff_bank(addr, shape, align, prec_list, 3);
  }
#endif
}
