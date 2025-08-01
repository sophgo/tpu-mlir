#include "atomic_tiu.h"

/* ha = tensorB(n, c, h, 0), and before call this func,
 * do if (tensorB(n, c, h, 0) >= tensorA_h
 *      tensorB(n,c,h,0) = tensorA_h - 1
 * if (ha != 0xff/0xffff || !limit_enable)
 *   tensorR(n,c,h,w) = tensorA(0,c,ha,w)
 * else if (if_fill_const)
 *   tensorR(n,c,h,w) = fill_const_val
 * else
 *   tensorR(n,c,h,w) = tensorR(n,c,h,w)
 */
/* Note:tensorA is aligned in local memory, and its stride is
 * wstride=1,hstride=ceil(w,EU_NUM) * EU_NUM,
 * cstride = A_cstride_is0 ? 0 : h*hstride,
 * nstride = c_per_npu * cstride.
 * And tensorR stride is wstride=1, hstride=ceil(w,EU_NUM) * EU_NUM
 * cstride=h*hstride, nstride=c_per_npu * cstride.
 * And tensorB is aligned in local memory with normal storage.
 * if (PE_S_gather_line) A=[1, C, A_h, A_w], B=[N, C, R_h, 1], R=[N, C, R_h, A_w]
 * else A=[1, C, A_h, A_w], B=[N, C, A_h, 1], R=[N, C, R_h, A_w]
 */
void atomic_sgl_check(
  u32 tensorA_addr,
  u32 tensorB_addr,
  u32 tensorR_addr,
  int tensorA_h,
  int tensorR_n,
  int tensorR_c,
  int tensorR_h,
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
  ASSERT(tensorA_addr % ALIGN_BYTES == 0);
  ASSERT(tensorB_addr % ALIGN_BYTES == 0);
  ASSERT(tensorR_addr % ALIGN_BYTES == 0);
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorB_addr));
  ASSERT(get_npu_index(tensorA_addr) == get_npu_index(tensorR_addr));
  ASSERT(op == PE_S_scatter_line || op == PE_S_gather_line);
#ifdef __sg2262__
  ASSERT(R_prec != FP4 && R_prec != INT4); //only 4bit not support
  ASSERT(A_cstride_is0 == 0);
#else
  ASSERT(R_prec != INT4); //only 4bit not support
  ASSERT(A_cstride_is0 == 0 || A_cstride_is0 == 1);
#endif
  ASSERT(B_prec == INT8 || B_prec == INT16);
  ASSERT(if_fill_const < (1 << 1) && if_fill_const >= 0);
  ASSERT(tensorR_n < (1 << 16) && tensorR_n > 0);
  ASSERT(tensorR_c < (1 << 16) && tensorR_c > 0);
  ASSERT(tensorR_h < (1 << 16) && tensorR_h > 0);
  ASSERT(tensorR_w < (1 << 16) && tensorR_w > 0);
  ASSERT(tensorA_h < (1 << 16) && tensorA_h > 0);
  ASSERT(limit_enable == 0 || limit_enable == 1);
  if (op == PE_S_scatter_line) {
    ASSERT(if_fill_const == 0);
  }
#endif
}
