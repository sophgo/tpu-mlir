#include "atomic_check.h"

static inline ppl_tensor_t *get_tensor(ppl_variable_t *var) {
  return (var && var->type == 0) ? (ppl_tensor_t *)(var->context) : NULL;
}
static inline scalar_t *get_scalar(ppl_variable_t *var) {
  return (var && var->type == 1) ? (scalar_t *)(var->context) : NULL;
}

static inline uint32_t get_addr_or_scalar(ppl_variable_t *var) {
  if (!var)
    return 0;
  if (var->type == 0) {
    if (var->context == NULL) {
      return 0;
    }
    return ((ppl_tensor_t *)(var->context))->addr;
  } else {
    if (var->context == NULL) {
      return 0;
    }
    return *((uint32_t *)(var->context));
  }
}

static inline int aligned_or_user(ppl_tensor_t *t) {
  bool aligned = t ? t->default_stride : true;
  return aligned ? 0 : 3;
}

static inline bool shape_equal(dim4 a, dim4 b) {
  return a.n == b.n && a.c == b.c && a.h == b.h && a.w == b.w;
}

static inline int dim4_get(dim4 d, int idx) {
  switch (idx) {
    case 0: return d.n;
    case 1: return d.c;
    case 2: return d.h;
    case 3: return d.w;
    default: return 0;
  }
}

static inline void print_dim4(const char* tag, dim4 d) {
  fprintf(stderr, "%s(N=%d C=%d H=%d W=%d)\n",
  tag,
  dim4_get(d, 0),
  dim4_get(d, 1),
  dim4_get(d, 2),
  dim4_get(d, 3));
  fflush(stderr);
}

static inline int match_with_broadcast(dim4 in_perm, dim4 out, dim4 bmask) {
  struct { int in; int out; int b; } dims[4] = {
    { in_perm.n, out.n, bmask.n },
    { in_perm.c, out.c, bmask.c },
    { in_perm.h, out.h, bmask.h },
    { in_perm.w, out.w, bmask.w },
  };
  for (int i = 0; i < 4; ++i) {
    if (dims[i].b) {
      if (!(dims[i].in == 1 || dims[i].in == dims[i].out)) {
        return i;
      }
    } else {
      if (dims[i].in != dims[i].out) {
        return i;
      }
    }
  }
  return -1;
}

static inline dim4 permute_dim4(dim4 src, dim4 perm) {
  dim4 dst;
  dst.n = dim4_get(src, perm.n);
  dst.c = dim4_get(src, perm.c);
  dst.h = dim4_get(src, perm.h);
  dst.w = dim4_get(src, perm.w);
  return dst;
}

static inline bool is_valid_perm(dim4 perm) {
  int used[4] = {0,0,0,0};
  int p[4] = {perm.n, perm.c, perm.h, perm.w};
  for (int i = 0; i < 4; ++i) {
    if (p[i] < 0 || p[i] > 3) return false;
    if (used[p[i]]) return false;
    used[p[i]] = 1;
  }
  return true;
}

static inline void check_shapes_all_equal(ppl_tensor_t** tensors, int count, const char* func_tag) {
  assert(count >= 2 && "[check_shapes_all_equal] need at least two tensors to compare");
  for (int i = 0; i < count; ++i) {
    assert(tensors[i] != NULL && "[check_shapes_all_equal] tensor pointer is NULL");
  }

  const char* tag = func_tag ? func_tag : "[check_shapes_all_equal] check_all_equal";

  dim4 base = tensors[0]->shape;
  for (int i = 1; i < count; ++i) {
    if (!shape_equal(base, tensors[i]->shape)) {
      assert(0 && "[check_shapes_all_equal] Shape mismatch between tensors");
    }
  }
}

static inline void get_dims_ab(dim4 d, int a, int b, int* row, int* col) {
  assert(row && col);
  assert(a >= 0 && a < 4 && b >= 0 && b < 4 && a != b);
  *row = dim4_get(d, a);
  *col = dim4_get(d, b);
}

static inline void check_other_dims_equal(dim4 lhs, dim4 rhs, dim4 out, int a, int b) {
  int used[4] = {0,0,0,0};
  used[a] = 1; used[b] = 1;
  for (int i = 0; i < 4; ++i) {
    if (used[i]) continue;
    int lv = dim4_get(lhs, i);
    int rv = dim4_get(rhs, i);
    int ov = dim4_get(out, i);
    if (!(lv == rv && rv == ov)) {
      const char* names[] = {"N","C","H","W"};
      assert(0 && "Non-matmul axes must be equal across lhs/rhs/output");
      return;
    }
  }
}

//------------------------------------------------------------------

// tiu
void arithmetic_check(ppl_tensor_t *rst, ppl_variable_t *lhs,
                      ppl_variable_t *rhs, int satu, int arith_mode) {
  bool l_scalar = lhs->type == 1;
  bool r_scalar = rhs->type == 1;

  ppl_tensor_t *lhs_tensor = get_tensor(lhs);
  ppl_tensor_t *rhs_tensor = get_tensor(rhs);

  int *lhs_stride = l_scalar ? NULL : (int *)(&lhs_tensor->stride);
  int *rhs_stride = r_scalar ? NULL : (int *)(&rhs_tensor->stride);

  data_type_t src_dtype = l_scalar ? rhs_tensor->dtype : lhs_tensor->dtype;

  uint32_t lhs_addr = get_addr_or_scalar(lhs);
  uint32_t rhs_addr = get_addr_or_scalar(rhs);

  int short_str[3] = {aligned_or_user(lhs_tensor), aligned_or_user(rhs_tensor),
                      aligned_or_user(rst)};

  int sign[3];
  int tempValue1, tempValue2;
  if (IS_FLOAT(src_dtype)) {
    tempValue1 = FP8TYPE(src_dtype);
    tempValue2 = FP8TYPE(rst->dtype);
  } else {
    tempValue1 = SIGN(src_dtype);
    tempValue2 = SIGN(rst->dtype);
  }
  sign[0] = tempValue1;
  sign[1] = tempValue1;
  sign[2] = tempValue2;

  int prec[3] = {
      l_scalar ? PRECISION(rhs_tensor->dtype) : PRECISION(lhs_tensor->dtype),
      r_scalar ? PRECISION(lhs_tensor->dtype) : PRECISION(rhs_tensor->dtype),
      PRECISION(rst->dtype)};

  atomic_tensor_arithmetic_check(
      lhs_addr, rhs_addr, rst->addr, rst->shape.n, rst->shape.c, rst->shape.h,
      rst->shape.w, lhs_stride, rhs_stride, (int *)&rst->stride, l_scalar,
      r_scalar, short_str, sign, satu, (PREC *)prec, arith_mode);
}

void cmp_check(ppl_tensor_t *dst, ppl_variable_t *src0, ppl_variable_t *src1,
               uint32_t true_val, AR_OP op) {
  bool l_scalar = src0->type == 1;
  bool r_scalar = src1->type == 1;

  ppl_tensor_t *src0_tensor = get_tensor(src0);
  ppl_tensor_t *src1_tensor = get_tensor(src1);

  uint32_t src0_addr = get_addr_or_scalar(src0);
  uint32_t src1_addr = get_addr_or_scalar(src1);

  int *src0_stride = l_scalar ? NULL : (int *)&src0_tensor->stride;
  int *src1_stride = r_scalar ? NULL : (int *)&src1_tensor->stride;
  data_type_t src_dtype = l_scalar ? src1_tensor->dtype : src0_tensor->dtype;

  int short_str[3] = {aligned_or_user(src0_tensor),
                      aligned_or_user(src1_tensor), aligned_or_user(dst)};
  int sign[3];
  int tempValue1, tempValue2;

  if (IS_FLOAT(src_dtype)) {
    tempValue1 = FP8TYPE(src_dtype);
    tempValue2 = FP8TYPE(dst->dtype);
  } else {
    tempValue1 = SIGN(src_dtype);
    tempValue2 = SIGN(dst->dtype);
  }
  sign[0] = tempValue1;
  sign[1] = tempValue1;
  sign[2] = tempValue2;
  int prec[2] = {PRECISION(src_dtype), PRECISION(dst->dtype)};
  atomic_tensor_arithmetic_select_check(
      src0_addr, src1_addr, true_val, dst->addr, dst->shape.n, dst->shape.c,
      dst->shape.h, dst->shape.w, src0_stride, src1_stride, (int *)&dst->stride,
      l_scalar, r_scalar, short_str, sign, (PREC *)prec, op);
}

void arithmetic_shift_check(ppl_tensor_t *rst, ppl_variable_t *lhs,
                            ppl_variable_t *rhs, unsigned int shift, int satu,
                            int arith_mode, int round) {
  bool l_tensor = (lhs && lhs->type == 0);
  bool r_tensor = (rhs && rhs->type == 0);

  ppl_tensor_t *lhs_tensor = get_tensor(lhs);
  ppl_tensor_t *rhs_tensor = get_tensor(rhs);

  uint32_t lhs_addr = get_addr_or_scalar(lhs);
  uint32_t rhs_addr = get_addr_or_scalar(rhs);

  int *lhs_stride = l_tensor ? (int *)&lhs_tensor->stride : NULL;
  int *rhs_stride = r_tensor ? (int *)&rhs_tensor->stride : NULL;
  data_type_t lhs_dtype = l_tensor ? lhs_tensor->dtype : 0;
  data_type_t rhs_dtype = r_tensor ? rhs_tensor->dtype : 0;

  data_type_t C_dtype = DT_INT8;
  int prec[4] = {PRECISION(lhs_dtype), PRECISION(rhs_dtype), PRECISION(C_dtype),
                 PRECISION(rst->dtype)};

  scalar_t scalar = {.s8 = shift};

  int short_str[3] = {aligned_or_user(lhs_tensor), aligned_or_user(rhs_tensor),
                      aligned_or_user(rst)};
  int sign[3] = {GETSIGN(lhs_dtype), GETSIGN(rhs_dtype), GETSIGN(rst->dtype)};

  atomic_tensor_arithmetic_ternary_check(
      lhs_addr, rhs_addr, scalar.u32, rst->addr, rst->shape.n, rst->shape.c,
      rst->shape.h, rst->shape.w, lhs_stride, rhs_stride, (int *)&rst->stride,
      l_tensor ? 0 : 1, // l_is_const
      r_tensor ? 0 : 1, // r_is_const
      1, short_str, sign, satu, (PREC *)prec, arith_mode, (ROUND_MODE)round);
}

void int_mac_check(ppl_tensor_t *rst, ppl_variable_t *src0,
                   ppl_variable_t *src1, unsigned char lshift,
                   unsigned char rshift, int round) {
  bool src0_const = (src0 && src0->type == 1);
  bool src1_const = (src1 && src1->type == 1);

  ppl_tensor_t *src0_tensor = get_tensor(src0);
  ppl_tensor_t *src1_tensor = get_tensor(src1);

  uint32_t src0_addr = get_addr_or_scalar(src0);
  uint32_t src1_addr = get_addr_or_scalar(src1);

  int *src0_stride = src0_const ? NULL : (int *)&src0_tensor->stride;
  int *src1_stride = src1_const ? NULL : (int *)&src1_tensor->stride;

  data_type_t src0_dtype = src0_const ? 0 : src0_tensor->dtype;
  data_type_t src1_dtype = src1_const ? 0 : src1_tensor->dtype;

  int prec[4] = {PRECISION(src0_dtype), PRECISION(src1_dtype),
                 PRECISION(DT_INT16), PRECISION(rst->dtype)};
  int short_str[3] = {aligned_or_user(src0_tensor),
                      aligned_or_user(src1_tensor), aligned_or_user(rst)};
  int sign[3] = {GETSIGN(src0_dtype), GETSIGN(src1_dtype), GETSIGN(rst->dtype)};

  unsigned int shift = lshift | (rshift << 8);
  scalar_t scalar = {.s8 = shift};
  atomic_tensor_arithmetic_ternary_check(
      src0_addr, src1_addr, scalar.u32, rst->addr, rst->shape.n, rst->shape.c,
      rst->shape.h, rst->shape.w, src0_stride, src1_stride, (int *)&rst->stride,
      src0_const, // src0_is_const
      src1_const, // src1_is_const
      true,       // shift_is_const
      short_str, sign, 0, (PREC *)prec, AR_MAC, (ROUND_MODE)round);
}

void arithmetic_single_opd_check(ppl_tensor_t *dst, ppl_tensor_t *src,
                                 AR_OP op) {
  int short_str[2] = {aligned_or_user(src), aligned_or_user(dst)};
  atomic_tensor_arithmetic_single_opd_check(
      src->addr, dst->addr, dst->shape.n, dst->shape.c, dst->shape.h,
      dst->shape.w, (int *)&src->stride, (int *)&dst->stride, false,
      short_str,        // num = 2, opd0, res0
      SIGN(src->dtype), // for abs and FP8
      PRECISION(src->dtype), op);
}

// dma
void dma_stride_move_check(ppl_tensor_t *dst, ppl_tensor_t *src, int direction,
                           int trans) {
  u32 local_index = 0;
  u32 local_addr = 0;
  u64 sys_addr = 0;
  if (direction == GDMA_S2L) {
    local_index = tpu_npu_index(dst->addr);
    local_addr = tpu_npu_addr(dst->addr);
    sys_addr = src->addr;
  } else if (direction == GDMA_L2S) {
    local_index = tpu_npu_index(src->addr);
    local_addr = tpu_npu_addr(src->addr);
    sys_addr = dst->addr;
  } else {
    assert(0 && "unsupported direction");
  }

  tensor_stride_move_check(
      local_addr, local_index, sys_addr, src->shape.n, src->shape.c,
      src->shape.h, src->shape.w, src->stride.n, src->stride.c, src->stride.h,
      src->stride.w, dst->stride.n, dst->stride.c, dst->stride.h, dst->stride.w,
      tpu_get_dma_dtype(src->dtype), direction, trans);
}

void dma_compact_move_check(ppl_tensor_t *dst, ppl_tensor_t *src, int direction,
                            int trans) {
  tensor_compact_move_check(tpu_npu_addr(dst->addr), tpu_npu_index(dst->addr),
                            src->addr, src->shape.n, src->shape.c, src->shape.h,
                            src->shape.w, tpu_get_dma_dtype(src->dtype),
                            direction, trans);
}

void broadcast_move_check(ppl_tensor_t *dst, ppl_tensor_t *src, int direction) {
  u64 src_addr = 0;
  int src_local_idx = 0;
  int dst_lmem_start_addr = 0;
  int dst_local_idx = 0;

  int src_N = 0, src_H = 0, src_W = 0, dst_C = 0;
  int src_N_stride = NO_USE, src_H_stride = NO_USE;
  int dst_N_stride = NO_USE, dst_H_stride = NO_USE;

  if (direction == GDMA_S2L) {
    // src: global, dst: local
    src_addr = src->addr;
    src_local_idx = NO_USE; // global
    dst_lmem_start_addr = tpu_npu_addr(dst->addr);
    dst_local_idx = tpu_npu_index(dst->addr);
  } else if (direction == GDMA_L2L) {
    // src: local, dst: local
    src_addr = tpu_npu_addr(src->addr);
    src_local_idx = tpu_npu_index(src->addr);
    dst_lmem_start_addr = tpu_npu_addr(dst->addr);
    dst_local_idx = tpu_npu_index(dst->addr);
  } else {
    assert(0 && "unsupported direction");
  }

  src_N = src->shape.n;
  src_H = src->shape.h;
  src_W = src->shape.w;
  dst_C = dst->shape.c;

  bool stride_enable = false;
  stride_enable = true;
  src_N_stride = src->stride.n;
  src_H_stride = src->stride.h;
  dst_N_stride = dst->stride.n;
  dst_H_stride = dst->stride.h;

  tensor_broadcast_move_check(
      src_addr, src_local_idx, dst_lmem_start_addr, dst_local_idx, src_N, src_H,
      src_W, dst_C, src_N_stride, src_H_stride, dst_N_stride, dst_H_stride,
      tpu_get_dma_dtype(direction != GDMA_S2L ? src->dtype : dst->dtype),
      stride_enable, direction);
}

void dma_general_move_check(ppl_tensor_t *dst, ppl_tensor_t *src, int direction,
                            int trans) {
  u64 src_addr = 0;
  int src_local_idx = 0;
  int src_N = 0, src_C = 0, src_H = 0, src_W = 0;
  int src_N_stride = 0, src_C_stride = 0, src_H_stride = 0, src_W_stride = 0;
  int src_format = 0;

  u64 dst_addr = 0;
  int dst_local_idx = 0;
  int dst_N = 0, dst_C = 0, dst_H = 0, dst_W = 0;
  int dst_N_stride = 0, dst_C_stride = 0, dst_H_stride = 0, dst_W_stride = 0;

  // src info
  src_N = src->shape.n;
  src_C = src->shape.c;
  src_H = src->shape.h;
  src_W = src->shape.w;
  src_N_stride = src->stride.n;
  src_C_stride = src->stride.c;
  src_H_stride = src->stride.h;
  src_W_stride = src->stride.w;
  src_format = tpu_get_dma_dtype(src->dtype);

  // dst info
  dst_N = dst->shape.n;
  dst_C = dst->shape.c;
  dst_H = dst->shape.h;
  dst_W = dst->shape.w;
  dst_N_stride = dst->stride.n;
  dst_C_stride = dst->stride.c;
  dst_H_stride = dst->stride.h;
  dst_W_stride = dst->stride.w;

  if (direction == GDMA_S2L) {
    // src: global, dst: local
    src_addr = src->addr;
    src_local_idx = 0;
    dst_addr = tpu_npu_addr(dst->addr);
    dst_local_idx = tpu_npu_index(dst->addr);
  } else if (direction == GDMA_L2S) {
    // src: local, dst: global
    src_addr = tpu_npu_addr(src->addr);
    src_local_idx = tpu_npu_index(src->addr);
    dst_addr = dst->addr;
    dst_local_idx = 0;
  } else if (direction == GDMA_S2S) {
    // src: global, dst: global
    src_addr = src->addr;
    src_local_idx = 0;
    dst_addr = dst->addr;
    dst_local_idx = 0;
  } else {
    // src: local, dst: local
    src_addr = tpu_npu_addr(src->addr);
    src_local_idx = tpu_npu_index(src->addr);
    dst_addr = tpu_npu_addr(dst->addr);
    dst_local_idx = tpu_npu_index(dst->addr);
  }

  tensor_general_move_check(
      src_addr, src_local_idx, src_N, src_C, src_H, src_W, src_N_stride,
      src_C_stride, src_H_stride, src_W_stride, src_format, dst_addr,
      dst_local_idx, dst_N, dst_C, dst_H, dst_W, dst_N_stride, dst_C_stride,
      dst_H_stride, dst_W_stride, direction, trans);
}

// direction: GDMA_L2S, GDMA_S2S
void dma_nonzero_move_check(ppl_tensor_t *dst, ppl_tensor_t *src, u32 base_idx,
                            int direction) {
  u64 src_addr = 0;
  int src_local_idx = 0;
  u64 dst_addr = 0;
  int src_format = 0;
  int dst_format = 0;
  u32 N, C, H, W;

  // shape
  N = src->shape.n;
  C = src->shape.c;
  H = src->shape.h;
  W = src->shape.w;

  src_format = tpu_get_dma_dtype(src->dtype);
  dst_format = tpu_get_dma_dtype(dst->dtype);

  if (direction == GDMA_L2S) {
    src_addr = tpu_npu_addr(src->addr);
    src_local_idx = tpu_npu_index(src->addr);
    dst_addr = dst->addr; // global
  } else if (direction == GDMA_S2S) {
    src_addr = src->addr; // global
    src_local_idx = 0;
    dst_addr = dst->addr; // global
  } else {
    assert(0 && "unsupported direction for nonzero move");
  }

  tensor_move_nonzero_check(src_addr, src_local_idx, dst_addr, src_format,
                            dst_format, N, C, H, W, base_idx, direction);
}

void dma_reverse_gen_cmd_check(ppl_tensor_t *dst, ppl_tensor_t *src,
                               int32_t reverse_axis, int32_t direction) {
  u64 src_addr = 0;
  u64 dst_addr = 0;
  int32_t N = src->shape.n;
  int32_t C = src->shape.c;
  int32_t H = src->shape.h;
  int32_t W = src->shape.w;
  uint32_t src_n_stride = src->stride.n;
  uint32_t src_c_stride = src->stride.c;
  uint32_t src_h_stride = src->stride.h;
  uint32_t dst_n_stride = dst->stride.n;
  uint32_t dst_c_stride = dst->stride.c;
  uint32_t dst_h_stride = dst->stride.h;

  if (direction == GDMA_L2L) {
    src_addr = tpu_npu_addr(src->addr);
    dst_addr = tpu_npu_addr(dst->addr);
  } else if (direction == GDMA_L2S) {
    src_addr = tpu_npu_addr(src->addr); // local
    dst_addr = dst->addr;               // global
  } else if (direction == GDMA_S2L) {
    src_addr = src->addr;               // global
    dst_addr = tpu_npu_addr(dst->addr); // local
  } else {
    src_addr = src->addr; // global
    dst_addr = dst->addr; // global
  }

  tensor_gdma_reverse_check(src_addr, dst_addr, N, C, H, W, src_n_stride,
                            src_c_stride, src_h_stride, dst_n_stride,
                            dst_c_stride, dst_h_stride, reverse_axis,
                            tpu_get_dma_dtype(src->dtype), direction);
}

void fill_constant_check_tensor(const ppl_tensor_t *dst,
                                const void *const_val) {
  fill_constant_check_global_stride(dst->addr, const_val, dst->dtype,
                                    dst->shape.n, dst->shape.c, dst->shape.h,
                                    dst->shape.w, dst->stride.n, dst->stride.c,
                                    dst->stride.h, dst->stride.w, 1);
}

// dtype convert checker
void cvt_check(ppl_tensor_t *dst, ppl_tensor_t *src,
               rounding_mode_t round_mode) {
  data_type_t src_dtype = src->dtype;
  data_type_t dst_dtype = dst->dtype;
  int short_str[2] = {aligned_or_user(src), aligned_or_user(dst)};
  dim4 shape = dst->shape;
  if (dst_dtype != src_dtype) {
    int sign[2];
    sign[0] =
        tpu_is_data_type_fp8(src_dtype) ? FP8TYPE(src_dtype) : SIGN(src_dtype);
    sign[1] =
        tpu_is_data_type_fp8(dst_dtype) ? FP8TYPE(dst_dtype) : SIGN(dst_dtype);
    int prec[2] = {PRECISION(src_dtype), PRECISION(dst_dtype)};
    // fp convert will do saturation
    int satu_mode =
        tpu_is_data_type_fp(src_dtype) && tpu_is_data_type_fp(dst_dtype) &&
        (tpu_data_type_size(src_dtype) > tpu_data_type_size(dst_dtype));
    if (!satu_mode && src_dtype == DT_FP8E5M2) {
      satu_mode = (dst_dtype == DT_FP8E4M3);
    }
    atomic_tensor_arithmetic_dtype_convert_check(
        src->addr, dst->addr, shape.n, shape.c, shape.h, shape.w,
        (int *)&src->stride, (int *)&dst->stride, false,
        short_str, // num = 2, opd0, res0
        sign,      // num = 2, opd0, res0
        satu_mode,
        (PREC *)prec, // num = 2, opd0, res0
        round_mode);
  } else {
    atomic_tensor_arithmetic_single_opd_check(
        src->addr, dst->addr, shape.n, shape.c, shape.h, shape.w,
        (int *)&src->stride, (int *)&dst->stride, false, short_str,
        SIGN(dst_dtype), PRECISION(dst_dtype), AR_COPY);
  }
}

// fuse linear checker
void fused_linear_check(ppl_tensor_t *dst, ppl_tensor_t *A, ppl_variable_t *B,
                        ppl_variable_t *C, int B_is_const, int C_is_const,
                        LIN_OP op_lin, int satu) {
  ppl_tensor_t *B_tensor = get_tensor(B);
  ppl_tensor_t *C_tensor = get_tensor(C);
  uint32_t B_addr = get_addr_or_scalar(B);
  uint32_t C_addr = get_addr_or_scalar(C);

  u32 A_addr = A->addr;
  u32 dst_addr = dst->addr;
  dim4 shape = dst->shape;
  data_type_t dtype = dst->dtype;
  atomic_fused_linear_check(A_addr, B_addr, C_addr, dst_addr, shape.n, shape.c,
                            shape.h, shape.w, B_is_const, C_is_const,
                            PRECISION(dtype), PRECISION(dtype), FP8TYPE(dtype),
                            op_lin, satu);
}

void arithmetic_div_check(ppl_tensor_t *rst, ppl_variable_t *lhs,
                          ppl_variable_t *rhs, int satu, int num_iter) {

  ppl_tensor_t *lhs_tensor = get_tensor(lhs);
  ppl_tensor_t *rhs_tensor = get_tensor(rhs);
  uint32_t lhs_addr = get_addr_or_scalar(lhs);
  uint32_t rhs_addr = get_addr_or_scalar(rhs);
  bool l_scalar = lhs->type == 1;
  bool r_scalar = rhs->type == 1;
  int *lhs_stride = l_scalar ? NULL : (int *)&lhs_tensor->stride;
  int *rhs_stride = r_scalar ? NULL : (int *)&rhs_tensor->stride;

  int short_str[3] = {aligned_or_user(lhs_tensor), aligned_or_user(rhs_tensor),
                      aligned_or_user(rst)};

  atomic_tensor_arithmetic_div_check(
      lhs_addr, rhs_addr, rst->addr, rst->shape.n, rst->shape.c, rst->shape.h,
      rst->shape.w, lhs_stride, rhs_stride, (int *)&rst->stride, l_scalar,
      r_scalar, short_str, PRECISION(rst->dtype), num_iter, satu);
}

void sfu_check(ppl_tensor_t *dst, ppl_tensor_t *src, ppl_tensor_t *coeff,
               int num, SFU_OP sfu_op) {
  dim4 shape = dst->shape;
  u32 Y1_addr = NO_USE;
  atomic_sfu_check(src->addr, dst->addr,
                   Y1_addr, // for frexp mantissa
                   shape.n, shape.c, shape.h, shape.w, num, sfu_op,
                   coeff ? coeff->addr : NO_USE, PRECISION(dst->dtype),
                   PRECISION(src->dtype));
}

void fused_cmp_check(ppl_tensor_t *R0, ppl_tensor_t *R1, ppl_variable_t *A,
                     ppl_variable_t *B, ppl_variable_t *C, ppl_variable_t *D,
                     int side, int bin_w, CMP_OP op) {
  uint32_t A_addr = get_addr_or_scalar(A);
  uint32_t B_addr = get_addr_or_scalar(B);
  uint32_t C_addr = get_addr_or_scalar(C);
  uint32_t D_addr = get_addr_or_scalar(D);

  bool A_is_const = (A && A->type == 1);
  bool B_is_const = (B && B->type == 1);
  bool C_is_const = (C && C->type == 1);
  bool D_is_const = (D && D->type == 1);

  ppl_tensor_t *A_tensor = (A && A->type == 0) ? get_tensor(A) : NULL;
  ppl_tensor_t *B_tensor = (B && B->type == 0) ? get_tensor(B) : NULL;
  ppl_tensor_t *C_tensor = (C && C->type == 0) ? get_tensor(C) : NULL;
  ppl_tensor_t *D_tensor = (D && D->type == 0) ? get_tensor(D) : NULL;

  int *A_stride = A_tensor ? (int *)&A_tensor->stride : NULL;
  int *B_stride = B_tensor ? (int *)&B_tensor->stride : NULL;
  int A_short_str = aligned_or_user(A_tensor);
  int B_short_str = aligned_or_user(B_tensor);

  dim4 shape = R0->shape;

  data_type_t A_dtype = A_tensor ? A_tensor->dtype : 0;
  data_type_t B_dtype = B_tensor ? B_tensor->dtype : 0;
  data_type_t C_dtype = C_tensor ? C_tensor->dtype : 0;
  data_type_t D_dtype = D_tensor ? D_tensor->dtype : 0;
  data_type_t AB_dtype = A_tensor ? A_dtype : B_dtype;
  data_type_t CD_dtype = C_tensor ? C_dtype : D_dtype;

  PREC AB_prec = PRECISION(AB_dtype);
  PREC CD_prec = PRECISION(CD_dtype);
  PREC RES0_dtype = PRECISION(R0->dtype);

  int sign = tpu_is_data_type_fp(AB_dtype) ? FP8TYPE(AB_dtype) : SIGN(AB_dtype);
  atomic_fused_cmp_check(A_addr, B_addr, C_addr, D_addr, R0->addr,
                         R1 ? R1->addr : 0, shape.n, shape.c, shape.h, shape.w,
                         A_is_const, B_is_const, C_is_const, D_is_const, sign,
                         side, bin_w, A_short_str, B_short_str, AB_prec,
                         CD_prec, RES0_dtype, op);
}

void conv_quant_check(ppl_tensor_t *input, ppl_tensor_t *output,
                      ppl_variable_t *weight, ppl_variable_t *bias,
                      ppl_variable_t *pad_ins, ppl_variable_t *kzp,
                      ppl_variable_t *requant, int kh, int kw, int stride_h,
                      int stride_w, int ins_h, int ins_w, int dilation_h,
                      int dilation_w, int pad_h_t, int pad_h_b, int pad_w_l,
                      int pad_w_r, int kernel_rotate, int result_add,
                      u32 ins_const_val, int do_relu, int sym_saturate,
                      int do_requant, int shift_num, int ozp,
                      ROUND_MODE rm_mode, PAD_MODE pad_mode) {

  ppl_tensor_t *weight_tensor =
      (weight && weight->type == 0) ? get_tensor(weight) : NULL;
  ppl_tensor_t *bias_tensor =
      (bias && bias->type == 0) ? get_tensor(bias) : NULL;

  dim4 input_shape = input->shape;
  int *input_stride = (int *)&input->stride;
  PREC input_prec = PRECISION(input->dtype);

  PREC output_prec = PRECISION(output->dtype);
  int output_c = output->shape.c;

  int kernel_is_const = (weight && weight->type == 1) ? 1 : 0;
  uint32_t weight_addr = get_addr_or_scalar(weight);
  PREC weight_prec = kernel_is_const ? INT8 : PRECISION(weight_tensor->dtype);
  int weight_sign = kernel_is_const ? 0 : SIGN(weight_tensor->dtype);

  uint32_t bias_addr = get_addr_or_scalar(bias);
  int bias_sign = bias ? SIGN(bias_tensor->dtype) : 0;
  int bias_is_const = bias ? bias->type == 1 : 1;

  uint32_t pad_ins_addr = pad_ins ? get_addr_or_scalar(pad_ins) : 0;
  uint32_t kzp_addr = kzp ? get_addr_or_scalar(kzp) : 0;
  uint32_t requant_addr = requant ? get_addr_or_scalar(requant) : 0;

  int pad_ins_is_const = (pad_ins && pad_ins->type == 1) ? 1 : 0;
  int kzp_is_const = (kzp && kzp->type == 1) ? 1 : 0;
  int requant_is_const = (requant && requant->type == 1) ? 1 : 0;

  int input_sign = SIGN(input->dtype);
  int res_sign = SIGN(output->dtype);

  atomic_conv_quant_check(input->addr,            // input_addr
                          weight_addr,            // weight_addr
                          bias_addr,              // bias_addr
                          pad_ins_addr,           // pad_ins_addr
                          kzp_addr,               // kzp_addr
                          requant_addr,           // requant_addr
                          output->addr,           // output_addr
                          input_shape.n,          // input_n
                          input_shape.c,          // input_c
                          input_shape.h,          // input_h
                          input_shape.w,          // input_w
                          output_c,               // output_c
                          kh, kw,                 // kh, kw
                          stride_h, stride_w,     // stride_h, stride_w
                          ins_h, ins_w,           // ins_h, ins_w
                          dilation_h, dilation_w, // dilation_h, dilation_w
                          pad_h_t, pad_h_b,       // pad_h_t, pad_h_b
                          pad_w_l, pad_w_r,       // pad_w_l, pad_w_r
                          kernel_is_const,        // kernel_is_const
                          bias_is_const,          // bias_is_const
                          pad_ins_is_const,       // pad_ins_is_const
                          kzp_is_const,           // kzp_is_const
                          kernel_rotate,          // kernel_rotate
                          result_add,             // result_add
                          ins_const_val,          // ins_const_val
                          input_sign,             // input_sign
                          weight_sign,            // weight_sign
                          bias_sign,              // bias_sign
                          res_sign,               // res_sign
                          input_stride,           // input_stride
                          do_relu,                // do_relu
                          sym_saturate,           // sym_saturate
                          do_requant,             // do_requant
                          requant_is_const,       // requant_is_const
                          shift_num,              // shift_num
                          ozp,                    // ozp
                          rm_mode,                // rm_mode
                          input_prec,             // input_prec
                          weight_prec,            // weight_prec
                          output_prec,            // output_prec
                          pad_mode                // pad_mode
  );
}

void conv_fp_check(ppl_tensor_t *input, ppl_tensor_t *output,
                   ppl_variable_t *weight, ppl_variable_t *bias,
                   ppl_variable_t *pad_ins, ppl_variable_t *rescale, int kh,
                   int kw, int stride_h, int stride_w, int ins_h, int ins_w,
                   int dilation_h, int dilation_w, int pad_h_t, int pad_h_b,
                   int pad_w_l, int pad_w_r, int kernel_rotate, int result_add,
                   u32 ins_const_val, int do_relu, int saturate,
                   PAD_MODE pad_mode) {
  // weight
  int kernel_is_const = (weight && weight->type == 1) ? 1 : 0;
  u32 weight_addr =
      weight ? (weight->type == 0 ? ((ppl_tensor_t *)weight->context)->addr
                                  : *((u32 *)weight->context))
             : 0;
  int weight_sign =
      kernel_is_const ? 0 : SIGN(((ppl_tensor_t *)weight->context)->dtype);

  // bias
  int bias_is_const = (bias && bias->type == 1) ? 1 : 0;
  u32 bias_addr = bias
                      ? (bias->type == 0 ? ((ppl_tensor_t *)bias->context)->addr
                                         : *((u32 *)bias->context))
                      : 0;
  int bias_sign =
      bias
          ? (bias->type == 0 ? SIGN(((ppl_tensor_t *)bias->context)->dtype) : 0)
          : 0;

  // pad_ins
  int pad_ins_is_const = (pad_ins && pad_ins->type == 1) ? 1 : 0;
  u32 pad_ins_addr =
      pad_ins ? (pad_ins->type == 0 ? ((ppl_tensor_t *)pad_ins->context)->addr
                                    : *((u32 *)pad_ins->context))
              : 0;

  // rescale
  int do_rescale = (rescale != NULL);
  int rescale_is_const = (rescale && rescale->type == 1) ? 1 : 0;
  u32 rescale_addr =
      rescale ? (rescale->type == 0 ? ((ppl_tensor_t *)rescale->context)->addr
                                    : *((u32 *)rescale->context))
              : 0;

  int input_sign = SIGN(input->dtype);
  int res_sign = SIGN(output->dtype);
  int input_prec = PRECISION(input->dtype);
  int output_prec = PRECISION(output->dtype);
  int bias_prec =
      bias
          ? (bias->type == 0 ? PRECISION(((ppl_tensor_t *)bias->context)->dtype)
                             : PRECISION(DT_FP32))
          : PRECISION(DT_FP32);

  atomic_conv_check(
      input->addr, weight_addr, bias_addr, pad_ins_addr, rescale_addr,
      output->addr, input->shape.n, input->shape.c, input->shape.h,
      input->shape.w, output->shape.c, kh, kw, stride_h, stride_w, ins_h, ins_w,
      dilation_h, dilation_w, pad_h_t, pad_h_b, pad_w_l, pad_w_r,
      kernel_is_const, bias_is_const, pad_ins_is_const, kernel_rotate,
      result_add, ins_const_val, (int *)&input->stride, do_relu, saturate,
      input_prec, output_prec, bias_prec, input_sign, weight_sign, res_sign,
      bias_sign, do_rescale, rescale_is_const, pad_mode);
}

void cpy_cross_npu_check(ppl_tensor_t *src, ppl_tensor_t *dst) {
  u32 src_addr = src->addr;
  u32 dst_addr = dst->addr;
  int N = dst->shape.n;
  int C = dst->shape.c;
  int H = dst->shape.h;
  int W = dst->shape.w;
  PREC prec = PRECISION(dst->dtype);
  atomic_lane_copy_check(src_addr, dst_addr, N, C, H, W, prec);
}

void dq0_check(ppl_tensor_t *input, ppl_variable_t *B_tensor,
               ppl_tensor_t *output, int offset, float scale, int round_mode) {
  int A_sign = SIGN(input->dtype);
  int R_sign = SIGN(output->dtype);
  PREC A_prec = PRECISION(input->dtype);
  PREC R_prec = PRECISION(output->dtype);

  uint32_t B_addr = get_addr_or_scalar(B_tensor);
  int B_is_const = (B_tensor == NULL || B_tensor->type != 0) ? 1 : 0;

  atomic_dq_f32mode_check(input->addr, B_addr, output->addr, output->shape.n,
                          output->shape.c, output->shape.h, output->shape.w,
                          B_is_const, scale, (short)offset, A_sign, R_sign,
                          A_prec, R_prec, round_mode);
}

void dq1_check(ppl_tensor_t *input, ppl_variable_t *B_tensor,
               ppl_tensor_t *output, int zp_value, int scale_factor,
               int shift_num, int round_mode) {
  int N = output->shape.n;
  int C = output->shape.c;
  int H = output->shape.h;
  int W = output->shape.w;

  int A_sign = SIGN(input->dtype);
  int R_sign = SIGN(output->dtype);
  PREC A_prec = PRECISION(input->dtype);
  PREC R_prec = PRECISION(output->dtype);

  uint32_t B_addr = get_addr_or_scalar(B_tensor);
  int B_is_const = (B_tensor == NULL || B_tensor->type != 0) ? 1 : 0;

  atomic_dq_i32mode_check(input->addr,  // A_addr
                          B_addr,       // B_addr
                          output->addr, // R_addr
                          N, C, H, W,
                          B_is_const,            // B_is_const
                          (short)zp_value,       // zp_value
                          scale_factor,          // scale_factor
                          (char)shift_num,       // shift_num
                          A_sign,                // A_sign
                          R_sign,                // R_sign
                          0,                     // sym_range
                          A_prec,                // A_prec
                          R_prec,                // R_prec
                          (ROUND_MODE)round_mode // shift_rd
  );
}

void rq0_check(ppl_tensor_t *input, ppl_variable_t *scale, ppl_tensor_t *output,
               float scale_value, float offset, int output_round_mode,
               int input_round_mode) {
  int N = output->shape.n;
  int C = output->shape.c;
  int H = output->shape.h;
  int W = output->shape.w;
  int A_sign = SIGN(input->dtype);
  int R_sign = SIGN(output->dtype);
  PREC A_prec = PRECISION(input->dtype);
  PREC R_prec = PRECISION(output->dtype);

  uint32_t B_addr = get_addr_or_scalar(scale);
  int B_is_const = (scale == NULL || scale->type != 0) ? 1 : 0;

  float scale_val = (B_is_const ? scale_value : 1.0f);
  float zp_val = (B_is_const ? offset : 0.0f);

  int sym_range = 0;

  atomic_rq_f32mode_check(
      input->addr, B_addr, output->addr, N, C, H, W, B_is_const, scale_val,
      zp_val, A_sign, R_sign, sym_range, A_prec, R_prec,
      (ROUND_MODE)input_round_mode, (ROUND_MODE)output_round_mode);
}

void rq1_check(ppl_tensor_t *input, ppl_variable_t *quant, ppl_tensor_t *output,
               int scale_val, char shift_val, short zp_val, int round_mode) {
  ppl_tensor_t *quant_tensor = get_tensor(quant);
  u32 A_addr = input->addr;
  u32 R_addr = output->addr;
  u32 B_addr = 0;
  int B_is_const = (quant == NULL || quant->type == 1) ? 1 : 0;

  B_addr = get_addr_or_scalar(quant);

  int N = output->shape.n;
  int C = output->shape.c;
  int H = output->shape.h;
  int W = output->shape.w;

  int A_sign = SIGN(input->dtype);
  int R_sign = SIGN(output->dtype);
  PREC A_prec = PRECISION(input->dtype);
  PREC R_prec = PRECISION(output->dtype);

  atomic_rq_i32mode_check(
      A_addr, B_addr, R_addr, N, C, H, W, B_is_const,
      scale_val, // scale_value
      shift_val, // shift_value (negative: right shift, positive: left shift)
      zp_val,    // zero point value
      A_sign, R_sign,
      0, // sym_range (default 0)
      A_prec, R_prec, (ROUND_MODE)round_mode);
}

void depthwise_check(ppl_tensor_t *input, ppl_tensor_t *output,
                     ppl_variable_t *weight, ppl_variable_t *bias,
                     ppl_variable_t *pad_ins, ppl_variable_t *rq, int kh,
                     int kw, int stride_h, int stride_w, int ins_h, int ins_w,
                     int dh, int dw, int pad_h_t, int pad_h_b, int pad_w_l,
                     int pad_w_r, int kernel_rotate, int do_relu, int saturate,
                     PAD_MODE pad_mode) {
  u32 input_addr = input->addr;
  int input_n = input->shape.n;
  int input_c = input->shape.c;
  int input_h = input->shape.h;
  int input_w = input->shape.w;

  u32 weight_addr = 0;
  int kernel_is_const = 0;
  if (weight) {
    kernel_is_const = (weight->type == 1);
    weight_addr = get_addr_or_scalar(weight);
  }

  u32 bias_addr = 0;
  int bias_is_const = 0;
  if (bias) {
    bias_is_const = (bias->type == 1);
    bias_addr = get_addr_or_scalar(bias);
  }

  u32 pad_ins_addr = 0;
  int pad_ins_is_const = 0;
  int ins_const_val = 0;
  if (pad_ins) {
    pad_ins_is_const = (pad_ins->type == 1);
    pad_ins_addr = get_addr_or_scalar(pad_ins);
  }

  int do_rq = false;
  int rq_is_const = (rq == NULL || rq->type == 1);
  u32 rq_addr = get_addr_or_scalar(rq);

  PREC in_prec = PRECISION(input->dtype);
  PREC out_prec = PRECISION(output->dtype);

  ppl_tensor_t *weight_tensor = get_tensor(weight);
  FP8_TYPE input_type = FP8TYPE(input->dtype);
  FP8_TYPE kernel_type =
      FP8TYPE(kernel_is_const ? input->dtype : weight_tensor->dtype);
  FP8_TYPE res_type = FP8TYPE(output->dtype);

  atomic_depthwise_check(input_addr,       // input_addr,
                         weight_addr,      // weight_addr,
                         bias_addr,        // bias_addr,
                         pad_ins_addr,     // pad_ins_addr,
                         rq_addr,          // rq_addr,
                         output->addr,     // output_addr,
                         input_n,          // input_n,
                         input_c,          // input_c,
                         input_h,          // input_h,
                         input_w,          // input_w,
                         kh,               // kh,
                         kw,               // kw,
                         stride_h,         // stride_h,
                         stride_w,         // stride_w,
                         ins_h,            // ins_h,
                         ins_w,            // ins_w,
                         dh,               // dh,
                         dw,               // dw,
                         pad_h_t,          // pad_h_t,
                         pad_h_b,          // pad_h_b,
                         pad_w_l,          // pad_w_l,
                         pad_w_r,          // pad_w_r,
                         kernel_is_const,  // kernel_is_const,
                         bias_is_const,    // bias_is_const,
                         pad_ins_is_const, // pad_ins_is_const,
                         ins_const_val,    // ins_const_val,
                         kernel_rotate,    // kernel_rotate,
                         do_relu,          // do_relu,
                         saturate,         // saturate,
                         do_rq,            // do_rq,
                         rq_is_const,      // rq_is_const,
                         in_prec,          // in_prec,
                         out_prec,         // out_prec,
                         input_type,       // input_type,
                         kernel_type,      // kernel_type,
                         res_type,         // res_type,
                         pad_mode          // pad_mode);
  );
}

void depthwise_quant_check(ppl_tensor_t *input, ppl_tensor_t *output,
                           ppl_variable_t *weight, ppl_variable_t *bias,
                           ppl_variable_t *pad_ins, ppl_variable_t *requant,
                           int kh, int kw, int stride_h, int stride_w,
                           int ins_h, int ins_w, int dh, int dw, int pad_h_t,
                           int pad_h_b, int pad_w_l, int pad_w_r,
                           int kernel_rotate, int do_relu, int sym_saturate,
                           int do_requant, int shift_num, int ozp,
                           ROUND_MODE rm_mode, PAD_MODE pad_mode) {
  u32 input_addr = input->addr;
  int input_n = input->shape.n;
  int input_c = input->shape.c;
  int input_h = input->shape.h;
  int input_w = input->shape.w;

  ppl_tensor_t *weight_tensor = get_tensor(weight);
  ppl_tensor_t *bias_tensor = get_tensor(bias);
  ppl_tensor_t *pad_ins_tensor = get_tensor(pad_ins);
  ppl_tensor_t *requant_tensor = get_tensor(requant);

  u32 weight_addr = 0;
  int kernel_is_const = 0;
  if (weight) {
    kernel_is_const = (weight->type == 1);
    weight_addr = get_addr_or_scalar(weight);
  }

  u32 bias_addr = 0;
  int bias_is_const = 0;
  if (bias) {
    bias_is_const = (bias->type == 1);
    bias_addr = get_addr_or_scalar(bias);
  }

  u32 pad_ins_addr = 0;
  int pad_ins_is_const = 0;
  int ins_const_val = 0;
  if (pad_ins) {
    pad_ins_is_const = (pad_ins->type == 1);
    pad_ins_addr = get_addr_or_scalar(pad_ins);
  }

  u32 requant_addr = 0;
  int requant_is_const = 0;
  if (requant) {
    requant_is_const = (requant->type == 1);
    requant_addr = get_addr_or_scalar(requant);
  }

  PREC input_prec = PRECISION(input->dtype);
  PREC output_prec = PRECISION(output->dtype);

  int input_sign = SIGN(input->dtype);
  int weight_sign = weight ? SIGN(weight_tensor->dtype) : 0;
  int bias_sign = bias ? SIGN(bias_tensor->dtype) : 0;
  int output_sign = SIGN(output->dtype);

  atomic_depthwise_quant_check(
      input_addr, weight_addr, bias_addr, pad_ins_addr, requant_addr,
      output->addr, input_n, input_c, input_h, input_w, kh, kw, stride_h,
      stride_w, ins_h, ins_w, dh, dw, pad_h_t, pad_h_b, pad_w_l, pad_w_r,
      kernel_is_const, bias_is_const, pad_ins_is_const, ins_const_val,
      kernel_rotate, input_sign, weight_sign, bias_sign, output_sign, do_relu,
      sym_saturate, do_requant, requant_is_const, shift_num, ozp, rm_mode,
      input_prec, output_prec, pad_mode);
}

void fp_exponent_part_check(ppl_tensor_t *dst, ppl_tensor_t *src) {
  // dst: exponent part（int16/int32），src: fp16/bfp16/fp32
  dim4 shape = dst->shape;
  atomic_sfu_check(src->addr, // A_addr
                   dst->addr, // Y_addr
                   NO_USE,    // Y1_addr (frexp only)
                   shape.n, shape.c, shape.h, shape.w,
                   NO_USE,   // n
                   SFU_NORM, // sfu_op
                   NO_USE,   // table_start_addr
                   PRECISION(dst->dtype), PRECISION(src->dtype));
}

void sfu_taylor_check(ppl_tensor_t *dst, ppl_tensor_t *src, ppl_tensor_t *table,
                      int num) {
  dim4 shape = dst->shape;
  atomic_sfu_check(src->addr, // A_addr
                   dst->addr, // Y_addr
                   NO_USE,    // Y1_addr, not used for taylor
                   shape.n, shape.c, shape.h, shape.w,
                   num,           // n, not used
                   SFU_TAYLOR_4X, // sfu_op,
                   table->addr,   // table_start_addr
                   PRECISION(dst->dtype), PRECISION(src->dtype));
}

void sgl_hgather_check(ppl_tensor_t *tensorR, ppl_tensor_t *tensorA,
                       ppl_tensor_t *tensorB, int A_cstride_is0,
                       int if_fill_const, u32 fill_const_val, int limit_enable,
                       SG_OP op) {

  dim4 shapeA = tensorA->shape;
  dim4 shapeR = tensorR->shape;
  atomic_sgl_check(tensorA->addr,             // tensorA_addr
                   tensorB->addr,             // tensorB_addr
                   tensorR->addr,             // tensorR_addr
                   shapeA.h,                  // tensorA_h
                   shapeR.n,                  // tensorR_n
                   shapeR.c,                  // tensorR_c
                   shapeR.h,                  // tensorR_h
                   shapeR.w,                  // tensorR_w
                   A_cstride_is0,             // A_cstride_is0
                   if_fill_const,             // if_fill_const
                   fill_const_val,            // fill_const_val
                   limit_enable,              // limit_enable
                   PRECISION(tensorB->dtype), // B_prec (index tensor)
                   PRECISION(tensorR->dtype), // R_prec (output tensor)
                   op);
}

void pes_sg_d1hzd_check(ppl_tensor_t *tensorR, ppl_tensor_t *tensorA,
                        ppl_tensor_t *tensorB, int A_cstride_is0,
                        int if_fill_const, u32 fill_const_val,
                        int limit_enable) {

  range_t ranges[3];
  data_type_t dtype = tensorR->dtype;
  dim4 stride;
  dim4 *shape = &(tensorR->shape);
  int start_idx = tpu_npu_index(tensorR->addr);
  tpu_aligned_stride(&stride, start_idx, shape, dtype);
  ranges[0] = tpu_bank_range(tensorR->addr, shape->n * stride.n * DSIZE(dtype));
  dim4 index_shape = {
      .n = shape->n, .c = shape->c, .h = 1, .w = tensorB->shape.w};
  data_type_t index_dtype = tensorB->dtype;
  tpu_aligned_stride(&stride, start_idx, &index_shape, index_dtype);
  ranges[1] = tpu_bank_range(tensorB->addr,
                             index_shape.n * stride.n * DSIZE(index_dtype));
  dim4 param_shape = {
      .n = 1, .c = A_cstride_is0 ? 1 : shape->c, .h = 1, .w = tensorA->shape.w};
  tpu_aligned_stride(&stride, start_idx, &param_shape, dtype);
  ranges[2] =
      tpu_bank_range(tensorA->addr, param_shape.n * stride.n * DSIZE(dtype));
  bool conflicting = tpu_any_range_overlapped(ranges, 3);
  dim4 shapeA = tensorA->shape;
  dim4 shapeR = tensorR->shape;
  atomic_pes_sg_d1hzd_check(
      tensorA->addr, tensorB->addr, tensorR->addr, shapeA.w, shapeR.n, shapeR.c,
      shapeR.w, A_cstride_is0, if_fill_const, fill_const_val, limit_enable,
      PRECISION(tensorB->dtype), PRECISION(tensorR->dtype),
      conflicting ? PE_S_gather_hzd : PE_S_gather_d1coor);
}

void pl_sgd2_check(ppl_tensor_t *tensorR, ppl_tensor_t *tensorA,
                   ppl_tensor_t *tensorB, int if_fill_const, u32 fill_const_val,
                   int limit_enable, SG_OP op) {
  dim4 shapeA = tensorA->shape;
  dim4 shapeR = tensorR->shape;
  atomic_pl_sgd2_check(tensorA->addr,             // tensorA_addr
                       tensorB->addr,             // tensorB_addr
                       tensorR->addr,             // tensorR_addr
                       shapeA.h,                  // tensorA_h
                       shapeA.w,                  // tensorA_w
                       shapeR.n,                  // tensorR_n
                       shapeR.c,                  // tensorR_c
                       shapeR.h,                  // tensorR_h
                       shapeR.w,                  // tensorR_w
                       if_fill_const,             // if_fill_const
                       fill_const_val,            // fill_const_val
                       limit_enable,              // limit_enable
                       PRECISION(tensorR->dtype), // R_prec
                       op                         // op
  );
}

// direction: GDMA_S2L, GDMA_L2S, GDMA_L2L, GDMA_S2S
void hgather_gdma_check(ppl_tensor_t *dst, ppl_tensor_t *src,
                        ppl_tensor_t *index, u64 const_val,
                        int direction, // S2S, S2L, L2S, L2L
                        int index_in_lmem, int start_pos) {
  u64 src_addr = 0, index_addr = 0, dst_addr = 0;
  int src_local_idx = 0, index_local_idx = 0, dst_local_idx = 0;

  if (direction == GDMA_S2L || direction == GDMA_S2S) {
    src_addr = src->addr;
    src_local_idx = 0;
  } else { // L2S, L2L
    src_addr = tpu_npu_addr(src->addr);
    src_local_idx = tpu_npu_index(src->addr);
  }

  // index
  if (index_in_lmem) {
    index_addr = tpu_npu_addr(index->addr);
    index_local_idx = tpu_npu_index(index->addr);
  } else {
    index_addr = index->addr;
    index_local_idx = 0;
  }

  // dst
  if (direction == GDMA_L2S || direction == GDMA_S2S) {
    dst_addr = dst->addr;
    dst_local_idx = 0;
  } else { // S2L, L2L
    dst_addr = tpu_npu_addr(dst->addr);
    dst_local_idx = tpu_npu_index(dst->addr);
  }

  u32 C = src->shape.c;
  u32 src_H = src->shape.h;
  u32 src_W = src->shape.w;
  u32 index_H = index->shape.h;

  stride_type src_C_stride = src->stride.c;
  stride_type src_H_stride = src->stride.h;
  stride_type index_C_stride = index->stride.c;
  stride_type index_H_stride = index->stride.h;
  stride_type dst_C_stride = dst->stride.c;
  stride_type dst_H_stride = dst->stride.h;

  int src_format = tpu_get_dma_dtype(src->dtype);
  int src_C_is1 = (src->shape.c == 1);
  int index_C_is1 = (index->shape.c == 1);
  int stride_enable = !src->default_stride && !dst->default_stride;

  tensor_gdma_gather_check(src_addr, src_local_idx, index_addr, index_local_idx,
                           index_in_lmem, dst_addr, dst_local_idx, const_val, C,
                           src_H, src_W, index_H, start_pos, src_C_stride,
                           src_H_stride, index_C_stride, index_H_stride,
                           dst_C_stride, dst_H_stride, src_format, src_C_is1,
                           index_C_is1, stride_enable, direction);
}

void pl_sgd1_check(ppl_tensor_t *tensorR, // output, [N,C,1,Wr]
                   ppl_tensor_t *tensorA, // param,  [N,C,1,Wa]
                   ppl_tensor_t *tensorB, // index,  [1,Wr,1,1]
                   int if_fill_const, u32 fill_const_val, int limit_enable,
                   SG_OP op) {

  dim4 shapeA = tensorA->shape;
  dim4 shapeR = tensorR->shape;

  atomic_pl_sgd1_check(tensorA->addr,             // tensorA_addr
                       tensorB->addr,             // tensorB_addr
                       tensorR->addr,             // tensorR_addr
                       shapeA.w,                  // tensorA_w
                       shapeR.n,                  // tensorR_n
                       shapeR.c,                  // tensorR_c
                       shapeR.w,                  // tensorR_w
                       if_fill_const,             // if_fill_const
                       fill_const_val,            // fill_const_val
                       limit_enable,              // limit_enable
                       PRECISION(tensorB->dtype), // B_prec (index tensor)
                       PRECISION(tensorR->dtype), // R_prec (output tensor)
                       op                         // op
  );
}

void static_broad_txp_check(ppl_tensor_t *dst, u32 src_addr) {
  int C = dst->shape.c;
  int W = dst->shape.w;
  PREC prec = PRECISION(dst->dtype);
  atomic_static_broad_txp_check(src_addr, dst->addr, C, W, prec);
}

void static_distribute_txp_check(ppl_tensor_t *dst, u32 src_addr) {
  int C = dst->shape.c;
  PREC prec = PRECISION(dst->dtype);
  atomic_static_distribute_txp_check(src_addr, dst->addr, 0, prec);
}

void static_broad_check(ppl_tensor_t *dst, u32 src_addr) {
  int C = dst->shape.c;
  int W = dst->shape.w;
  PREC prec = PRECISION(dst->dtype);
  atomic_static_broad_check(src_addr, dst->addr, C, W, 0, prec);
}

// mask select
// MaskSelect
void mask_select_check(ppl_tensor_t *dst, ppl_tensor_t *src, ppl_tensor_t *mask,
                       int direction) {

  u64 src_addr = 0;
  u64 mask_addr = 0;
  int src_local_idx = 0;
  int mask_local_idx = 0;
  int mask_in_lmem = 0;

  if (direction == GDMA_S2S) {
    src_addr = src->addr;
    mask_addr = mask->addr;
    src_local_idx = 0;
    mask_local_idx = 0;
    mask_in_lmem = 0;
  } else { // L2S
    src_addr = tpu_npu_addr(src->addr);
    mask_addr = tpu_npu_addr(mask->addr);
    src_local_idx = tpu_npu_index(src_addr);
    mask_in_lmem = 1;
    mask_local_idx = tpu_npu_index(mask_addr);
  }

  u64 dst_addr = dst->addr;

  int src_format = tpu_get_dma_dtype(src->dtype);
  int mask_format = tpu_get_dma_dtype(mask->dtype);

  dim4 shape = src->shape;
  u32 N = shape.n;
  u32 C = shape.c;
  u32 H = shape.h;
  u32 W = shape.w;

  tensor_general_move_with_mask_check(
      src_addr, src_local_idx, mask_addr, mask_local_idx, mask_in_lmem,
      dst_addr, src_format, mask_format, N, C, H, W, direction);
}
// MaskSelectBatchBcast
void mask_select_batch_bcast_check(ppl_tensor_t *dst,    /* tout          */
                                   ppl_tensor_t *count,  /* tout_cnt      */
                                   ppl_tensor_t *src,    /* ta            */
                                   ppl_tensor_t *mask) { /* tmask         */

  u32 tensorA_addr = src->addr;
  u32 tensorB_addr = mask->addr;
  u32 tensorR_addr = dst->addr;
  u32 mask_num_addr = count->addr;

  int tensorA_w = src->shape.w;
  int tensorB_n = mask->shape.n;
  int tensorB_c = mask->shape.c;

  int A_cstride_is0 = (src->default_stride || src->stride.c == 0) ? 1 : 0;

  PREC B_prec = PRECISION(mask->dtype);
  PREC R_prec = PRECISION(dst->dtype);

  SG_OP op = PE_S_mask_select;

  atomic_pes_mask_sel_check(tensorA_addr, tensorB_addr, tensorR_addr,
                            mask_num_addr, tensorA_w, tensorB_n, tensorB_c,
                            A_cstride_is0, B_prec, R_prec, op);
}

// nonzero
void nonzero_check(ppl_tensor_t *src, ppl_tensor_t *dst_idx,
                   ppl_tensor_t *dst_cnt, int support_hzd) {
  SG_OP op = support_hzd ? PE_S_nonzero_hzd : PE_S_nonzero;
  atomic_pes_nonzero_check(
      src->addr, dst_idx->addr, dst_cnt->addr, src->shape.n, src->shape.c,
      src->shape.w, PRECISION(src->dtype), PRECISION(dst_idx->dtype), op);
}

// normal
void normal_check(ppl_tensor_t *dst, ppl_tensor_t *src) {
  atomic_sfu_check(src->addr, dst->addr, NO_USE, dst->shape.n, dst->shape.c,
                   dst->shape.h, dst->shape.w, NO_USE, SFU_NORM, NO_USE,
                   PRECISION(dst->dtype), PRECISION(src->dtype));
}

// NPU bcast
void npu_bcast_check(ppl_tensor_t *dst, ppl_tensor_t *src) {
  atomic_lane_broad_check(src->addr, dst->addr, dst->shape.n, dst->shape.h,
                          dst->shape.w, dst->shape.c, NO_USE,
                          PRECISION(dst->dtype));
}

// scatter
//  1 BatchBcastHScatterOp
void batch_bcast_hscatter_check(ppl_tensor_t *output, ppl_tensor_t *param,
                                ppl_tensor_t *index) {
  // param = A, index = B, output = R
  atomic_sgl_check(param->addr, index->addr, output->addr,
                   param->shape.h,           // tensorA_h
                   output->shape.n,          // tensorR_n
                   output->shape.c,          // tensorR_c
                   output->shape.h,          // tensorR_h
                   output->shape.w,          // tensorR_w
                   0,                        // A_cstride_is0
                   0,                        // if_fill_const
                   0,                        // fill_const_val
                   0,                        // limit_enable
                   PRECISION(index->dtype),  // B_prec
                   PRECISION(output->dtype), // R_prec
                   PE_S_scatter_line);
}

// 2 BatchBcastWScatterOp
void batch_bcast_wscatter_check(ppl_tensor_t *output, ppl_tensor_t *param,
                                ppl_tensor_t *index) {
  // param = A, index = B, output = R
  atomic_pes_sg_d1hzd_check(param->addr, index->addr, output->addr,
                            param->shape.w,  // tensorA_w
                            output->shape.n, // tensorR_n
                            output->shape.c, // tensorR_c
                            output->shape.w, // tensorR_w
                            0,               // A_cstride_is0
                            0,               // if_fill_const
                            0,               // fill_const_val
                            0,               // limit_enable
                            PRECISION(index->dtype), PRECISION(output->dtype),
                            PE_S_scatter_hzd);
}

// 3 HScatterOp
void hscatter_check(ppl_tensor_t *output, ppl_tensor_t *param,
                    ppl_tensor_t *index) {
  // output = dst, param = src, index = index
  tensor_gdma_scatter_check(
      param->addr,                                       // src_addr
      0,                                                 // src_local_idx
      index->addr,                                       // index_addr
      0,                                                 // index_local_idx
      0,                                                 // index_in_lmem
      output->addr,                                      // dst_addr
      0,                                                 // dst_local_idx
      output->shape.c,                                   // C
      param->shape.h,                                    // src_H
      param->shape.w,                                    // src_W
      output->shape.h,                                   // dst_H
      0,                                                 // start_pos
      param->stride.c,                                   // src_C_stride
      param->stride.h,                                   // src_H_stride
      index->stride.c,                                   // index_C_stride
      index->stride.h,                                   // index_H_stride
      output->stride.c,                                  // dst_C_stride
      output->stride.h,                                  // dst_H_stride
      tpu_get_dma_dtype(param->dtype),                   // src_format
      (param->shape.c == 1),                             // src_C_is1
      (index->shape.c == 1),                             // index_C_is1
      (!param->default_stride && !output->default_stride), // stride_enable
      0,                                                 // direction (GDMA_S2S)
      0                                                  // inplace_add
  );
}

// 4 HWScatterOp
void hwscatter_check(ppl_tensor_t *output, ppl_tensor_t *param,
                     ppl_tensor_t *index) {
  // output = R, param = A, index = B
  atomic_pl_sgd2_check(param->addr,              // tensorA_addr
                       index->addr,              // tensorB_addr
                       output->addr,             // tensorR_addr
                       param->shape.h,           // tensorA_h
                       param->shape.w,           // tensorA_w
                       output->shape.n,          // tensorR_n
                       output->shape.c,          // tensorR_c
                       output->shape.h,          // tensorR_h
                       output->shape.w,          // tensorR_w
                       0,                        // if_fill_const
                       0,                        // fill_const_val
                       0,                        // limit_enable
                       PRECISION(output->dtype), // R_prec
                       PL_scatter_d2coor);
}
// 5 WScatterOp
void wscatter_check(ppl_tensor_t *output, ppl_tensor_t *param,
                    ppl_tensor_t *index) {
  // output = R, param = A, index = B
  atomic_pl_sgd1_check(param->addr,              // tensorA_addr
                       index->addr,              // tensorB_addr
                       output->addr,             // tensorR_addr
                       param->shape.w,           // tensorA_w
                       output->shape.n,          // tensorR_n
                       output->shape.c,          // tensorR_c
                       output->shape.w,          // tensorR_w
                       0,                        // if_fill_const
                       0,                        // fill_const_val
                       0,                        // limit_enable
                       PRECISION(index->dtype),  // B_prec
                       PRECISION(output->dtype), // R_prec
                       PL_scatter_d1coor);
}

// sdma
void sdma_general_move_check(ppl_tensor_t *dst, ppl_tensor_t *src,
                             int transpose) {
  int fmt = tpu_get_dma_dtype(src->dtype);
  sdma_tensor_general_move_check(
      src->addr, src->shape.n, src->shape.c, src->shape.h, src->shape.w,
      src->stride.n, src->stride.c, src->stride.h, src->stride.w, fmt,
      dst->addr, dst->shape.n, dst->shape.c, dst->shape.h, dst->shape.w,
      dst->stride.n, dst->stride.c, dst->stride.h, dst->stride.w, transpose);
}

void sdma_setc_check(ppl_tensor_t *dst, void *scalar) {
  int fmt = tpu_get_dma_dtype(dst->dtype);
  sdma_fill_constant_check_global_stride(
      dst->addr, scalar, fmt, dst->shape.n, dst->shape.c, dst->shape.h,
      dst->shape.w, dst->stride.n, dst->stride.c, dst->stride.h, dst->stride.w,
      1);
}

void sdma_hgather_check(ppl_tensor_t *output, ppl_tensor_t *param,
                        ppl_tensor_t *index, void *c) {
  int fmt = tpu_get_dma_dtype(param->dtype);
  sdma_tensor_gather_check(
      param->addr, index->addr, output->addr, *((u32 *)c), output->shape.c,
      param->shape.h, param->shape.w, index->shape.h, 0, param->stride.c,
      param->stride.h, index->stride.c, index->stride.h, output->stride.c,
      output->stride.h, fmt, param->shape.c == 1, index->shape.c == 1, 1);
}

void sdma_hscatter_check(ppl_tensor_t *output, ppl_tensor_t *param,
                         ppl_tensor_t *index) {
  int fmt = tpu_get_dma_dtype(param->dtype);
  sdma_tensor_scatter_check(
      param->addr, index->addr, output->addr, output->shape.c, param->shape.h,
      param->shape.w, output->shape.h, 0, param->stride.c, param->stride.h,
      index->stride.c, index->stride.h, output->stride.c, output->stride.h, fmt,
      param->shape.c == 1, index->shape.c == 1, 1, 0);
}

// shift
void shift_check(ppl_tensor_t *dst, ppl_tensor_t *src, ppl_variable_t *sft,
                 int op, int round_mode) {
  bool sft_scalar = sft->type == 1;
  ppl_tensor_t *sft_tensor = get_tensor(sft);
  int *sft_stride = sft_scalar ? NULL : (int *)&sft_tensor->stride;
  uint32_t sft_addr = get_addr_or_scalar(sft);

  int short_str[3] = {aligned_or_user(src), aligned_or_user(sft_tensor),
                      aligned_or_user(dst)};
  int sign[2] = {GETSIGN(src->dtype), GETSIGN(dst->dtype)};
  PREC prec[3] = {PRECISION(src->dtype),
                  PRECISION(sft_scalar ? DT_INT8 : sft->type),
                  PRECISION(dst->dtype)};
  if (op == AR_ARITH_SHIFT) {
    sign[0] = GETSIGN(src->dtype);
    sign[1] = 1;
  } else if (op == AR_LOGIC_SHIFT) {
    sign[0] = 0;
    sign[1] = 1;
  }
  atomic_tensor_arithmetic_with_round_check(
      src->addr, sft_addr, dst->addr, dst->shape.n, dst->shape.c, dst->shape.h,
      dst->shape.w, (int *)&src->stride, sft_stride, (int *)&dst->stride, false,
      sft_scalar, short_str, sign, prec, op, (ROUND_MODE)round_mode);
}

// tiu l2l
void l2l_copy_check(ppl_tensor_t *dst,   /* R_addr */
                    ppl_tensor_t *src) { /* A_addr */

  dim4 shape = dst->shape;
  int short_str[2] = {aligned_or_user(src), aligned_or_user(dst)};

  atomic_tensor_arithmetic_single_opd_check(
      src->addr, dst->addr, shape.n, shape.c, shape.h, shape.w,
      (int *)&src->stride, (int *)&dst->stride, false, short_str,
      SIGN(src->dtype), PRECISION(src->dtype), AR_COPY);
}

// tiu mask select
void tiu_mask_select_check(ppl_tensor_t *src, ppl_tensor_t *dst,
                           ppl_tensor_t *dst_cnt, ppl_tensor_t *mask_num,
                           int A_cstride_is0) {
  atomic_pes_mask_sel_check(src->addr, dst->addr, dst_cnt->addr, mask_num->addr,
                            src->shape.w, dst->shape.n, dst->shape.c,
                            A_cstride_is0, PRECISION(dst->dtype),
                            PRECISION(dst_cnt->dtype), PE_S_mask_select);
}

// tiu setC
void setc_check(ppl_tensor_t *dst, void *scalar, int scalar_dtype) {
  int n = dst->shape.n, c = dst->shape.c, h = dst->shape.h, w = dst->shape.w;
  int *dst_stride = (int *)&dst->stride;
  int short_str[2] = {0, aligned_or_user(dst)};
  int sign = SIGN(dst->dtype);
  int prec = PRECISION(dst->dtype);

  atomic_tensor_arithmetic_single_opd_check((u64)scalar, dst->addr, n, c, h, w,
                                            NULL, dst_stride, true, short_str,
                                            sign, prec, AR_COPY);
}

// tiu transpose
void cw_transpose_check(ppl_tensor_t *dst, ppl_tensor_t *src) {
  atomic_cw_transpose_check(src->addr, dst->addr, src->shape.n, src->shape.c,
                            src->shape.h, src->shape.w, PRECISION(src->dtype),
                            TRAN_C_W_TRANSPOSE);
}

void wc_transpose_check(ppl_tensor_t *dst, ppl_tensor_t *src) {
  atomic_cw_transpose_check(src->addr, dst->addr, src->shape.n, src->shape.c,
                            src->shape.h, src->shape.w, PRECISION(src->dtype),
                            TRAN_W_C_TRANSPOSE);
}

// vc arith
void vc_arith_check(ppl_tensor_t *rst, ppl_tensor_t *lhs, ppl_tensor_t *rhs,
                    AR_OP op, int round_mode) {
  int A_w = lhs->shape.w;
  int B_w = rhs->shape.w;
  int A_len = lhs->shape.h * A_w;
  int B_len = rhs->shape.h * B_w;

  atomic_vector_correlation_check(lhs->addr, rhs->addr, rst->addr, A_len, B_len,
                                  A_w, B_w, op, PRECISION(lhs->dtype),
                                  PRECISION(rhs->dtype), PRECISION(rst->dtype),
                                  (ROUND_MODE)round_mode, 0, SIGN(lhs->dtype),
                                  SIGN(rhs->dtype), SIGN(rst->dtype));
}

void fmm2_check(ppl_tensor_t *rst, ppl_variable_t *lhs, ppl_tensor_t *rhs,
                ppl_variable_t *bias, ppl_variable_t *requant, int add_result,
                int do_relu, int satu, bool is_L_trans, bool is_R_trans) {
  bool l_is_tensor = (lhs && lhs->type == 0);
  bool b_is_tensor = bias && bias->type == 0;
  bool requant_tensor = requant && requant->type == 0;

  uint32_t lhs_addr = get_addr_or_scalar(lhs);
  uint32_t rhs_addr = rhs->addr;
  uint32_t bias_addr = get_addr_or_scalar(bias);
  uint32_t requant_addr = get_addr_or_scalar(requant);
  uint32_t rst_addr = rst->addr;

  ppl_tensor_t *lhs_tensor = get_tensor(lhs);
  ppl_tensor_t *bias_tensor = get_tensor(bias);
  scalar_t *requant_scalar = get_scalar(requant);

  data_type_t rhs_dtype = rhs->dtype;
  bool tf32_mode = 0;
  if (rhs_dtype == DT_TF32) {
    tf32_mode = 1;
    rhs_dtype = DT_FP32;
  }
  data_type_t lhs_dtype = l_is_tensor ? lhs_tensor->dtype : rhs->dtype;
  data_type_t rst_dtype = rst->dtype;
  data_type_t bias_dtype = b_is_tensor ? bias_tensor->dtype : DT_FP32;

  int L_row_num = is_L_trans && is_R_trans ? rst->shape.w : rst->shape.c;
  int R_col_num = is_L_trans && is_R_trans ? rst->shape.c : rst->shape.w;
  int L_col_num = is_R_trans ? rhs->shape.w : rhs->shape.c;
  int do_rq = !(!requant_tensor && requant_scalar->u32 == 0);
  PREC LR_prec = PRECISION(rhs_dtype);
  PREC B_prec = PRECISION(bias_dtype);
  PREC Y_prec = PRECISION(rst_dtype);
  FP8_TYPE L_fp8type = l_is_tensor ? FP8TYPE(lhs_dtype) : FP8TYPE(rhs_dtype);
  FP8_TYPE R_fp8type = FP8TYPE(rhs_dtype);
  FP8_TYPE Y_fp8type = FP8TYPE(rst_dtype);

  atomic_mm2_check(lhs_addr, rhs_addr, rst_addr, bias_addr, requant_addr,
                   L_row_num, L_col_num, R_col_num, is_L_trans, is_R_trans,
                   !l_is_tensor, false, !b_is_tensor, add_result, do_relu,
                   do_rq, !requant_tensor, satu, LR_prec, B_prec, Y_prec,
                   L_fp8type, R_fp8type, Y_fp8type, tf32_mode);
}

void mm2_check(ppl_tensor_t *rst, ppl_variable_t *lhs, ppl_tensor_t *rhs,
               ppl_variable_t *bias, ppl_variable_t *requant,
               ppl_variable_t *rzp, int rshift_rd, int add_result, int do_relu,
               int satu, bool is_L_trans, bool is_R_trans) {
  bool l_is_tensor = (lhs && lhs->type == 0);
  bool b_is_tensor = bias && bias->type == 0;
  bool r_is_tensor = requant && requant->type == 0;
  bool rzp_tensor = rzp && rzp->type == 0;

  uint32_t lhs_addr = get_addr_or_scalar(lhs);
  uint32_t rhs_addr = rhs->addr;
  uint32_t bias_addr = get_addr_or_scalar(bias);
  uint32_t requant_addr = get_addr_or_scalar(requant);
  uint32_t rst_addr = rst->addr;
  uint32_t rzp_addr = get_addr_or_scalar(rzp);

  ppl_tensor_t *lhs_tensor = get_tensor(lhs);
  ppl_tensor_t *bias_tensor = get_tensor(bias);
  scalar_t *requant_scalar = get_scalar(requant);

  data_type_t rhs_dtype = rhs->dtype;
  data_type_t lhs_dtype = l_is_tensor ? lhs_tensor->dtype : rhs->dtype;
  data_type_t rst_dtype = rst->dtype;
  data_type_t bias_dtype = b_is_tensor ? bias_tensor->dtype : DT_INT32;

  int L_row_num = is_L_trans && is_R_trans ? rst->shape.w : rst->shape.c;
  int R_col_num = is_L_trans && is_R_trans ? rst->shape.c : rst->shape.w;
  int L_col_num = is_R_trans ? rhs->shape.w : rhs->shape.c;
  int do_rq = !(!r_is_tensor && requant_scalar->u32 == 0);
  PREC L_prec = PRECISION(lhs_dtype);
  PREC R_prec = PRECISION(rhs_dtype);
  PREC Y_prec = PRECISION(rst_dtype);
  int L_sign = SIGN(lhs_dtype);
  int R_sign = SIGN(rhs_dtype);
  int Y_sign = SIGN(rst_dtype);
  int B_sign = SIGN(bias_dtype);

  atomic_mm2_fixed_check(lhs_addr, rhs_addr, rst_addr, rzp_addr, bias_addr,
                         requant_addr, 0, 0, L_row_num, L_col_num, R_col_num,
                         is_L_trans, is_R_trans, !l_is_tensor, false,
                         !rzp_tensor, L_sign, R_sign, add_result, Y_sign,
                         B_sign, !b_is_tensor, !r_is_tensor, do_relu, satu,
                         do_rq, rshift_rd, L_prec, R_prec, Y_prec);
}

void fmm_check(ppl_tensor_t *rst, ppl_variable_t *lhs, ppl_tensor_t *rhs,
               ppl_variable_t *bias, int add_result, int do_relu,
               bool is_L_trans) {
  bool l_tensor = (lhs && lhs->type == 0);
  bool bias_tensor = bias && bias->type == 0;

  uint32_t lhs_addr = get_addr_or_scalar(lhs);
  uint32_t rhs_addr = rhs->addr;
  uint32_t bias_addr = get_addr_or_scalar(bias);
  uint32_t rst_addr = rst->addr;

  ppl_tensor_t *lhs_tensor = get_tensor(lhs);

  int L_tensor_W = l_tensor ? lhs_tensor->shape.w : 16;
  int L_tensor_C = l_tensor ? lhs_tensor->shape.c : DIV_UP(rst->shape.n, 16);
  int R_tensor_W = rhs->shape.w;
  int R_tensor_C = rhs->shape.c;
  int L_row_num = rst->shape.n;
  int L_col_num = rhs->shape.n;

  atomic_mm_check(lhs_addr, rhs_addr, rst_addr, bias_addr, L_tensor_W,
                  L_tensor_C, R_tensor_W, R_tensor_C, L_row_num, L_col_num,
                  is_L_trans, !l_tensor, !bias_tensor, add_result, do_relu);
}

void avg_pool_check(ppl_tensor_t *input, ppl_tensor_t *output, dim2 kernel,
                    dim2 stride, dim2 ins, dim2 dilation, int pt, int pb,
                    int pl, int pr, u32 scale) {
  uint32_t input_addr = input->addr;
  uint32_t output_addr = output->addr;

  PREC input_prec = PRECISION(input->dtype);
  PREC output_prec = PRECISION(output->dtype);

  FP8_TYPE input_fp8type = FP8TYPE(input->dtype);
  FP8_TYPE output_fp8type = FP8TYPE(output->dtype);

  atomic_avg_pooling_check(
      input_addr, 0, 0, output_addr, input->shape.n, input->shape.c,
      input->shape.h, input->shape.w, kernel.h, kernel.w, stride.h, stride.w,
      ins.h, ins.w, dilation.h, dilation.w, pt, pb, pl, pr, true, scale, 0, 0,
      0, 0, 0, 0, input_prec, output_prec, input_fp8type, 0, 0, PAD_CONSTANT,
      ROUND_HALF_TO_EVEN);
}

void avg_pool_fixed_check(ppl_tensor_t *input, ppl_tensor_t *output,
                          dim2 kernel, dim2 stride, dim2 ins, dim2 dilation,
                          int pt, int pb, int pl, int pr, u32 scale) {
  uint32_t input_addr = input->addr;
  uint32_t output_addr = output->addr;

  PREC input_prec = PRECISION(input->dtype);
  PREC output_prec = PRECISION(output->dtype);

  int input_sign = SIGN(input->dtype);
  int output_sign = SIGN(output->dtype);

  atomic_avg_pooling_fixed_check(
      input_addr, 0, output_addr, 0, input->shape.n, input->shape.c,
      input->shape.h, input->shape.w, kernel.h, kernel.w, stride.h, stride.w,
      ins.h, ins.w, dilation.h, dilation.w, pt, pb, pl, pr, true, scale, 0,
      input_sign, output_sign, 0 /*kernel sign*/, 0 /*do_relu*/, 0 /*do_rq*/,
      1 /*rq_is_const*/, 0 /*sym_range*/, 1 /*mul*/, 0 /*rshift*/, 0 /*yzp*/,
      ROUND_HALF_UP, input_prec, output_prec, PAD_CONSTANT);
}

void max_min_pool_check(ppl_tensor_t *input, ppl_tensor_t *output, dim2 kernel,
                        dim2 stride, dim2 ins, dim2 dilation, int pt, int pb,
                        int pl, int pr, int pool_op) {
  uint32_t input_addr = input->addr;
  uint32_t output_addr = output->addr;

  PREC input_prec = PRECISION(input->dtype);

  FP8_TYPE input_fp8type = FP8TYPE(input->dtype);

  atomic_max_min_pooling_check(
      input_addr, 0, output_addr, 0xFFFFFFFF, input->shape.n, input->shape.c,
      input->shape.h, input->shape.w, kernel.h, kernel.w, stride.h, stride.w,
      ins.h, ins.w, dilation.h, dilation.w, pt, pb, pl, pr, true, 0,
      input_fp8type, input_prec, 0 /*out_index_prec*/, PAD_CONSTANT,
      0 /*do_relu*/, 0 /*saturate*/, pool_op);
}

void roi_avg_pool_check(ppl_tensor_t *input, ppl_tensor_t *roi,
                        ppl_tensor_t *output, dim2 kernel, u32 scale,
                        int do_relu, int round_mode, int sym_range) {
  uint32_t input_addr = input->addr;
  uint32_t roi_addr = roi->addr;
  uint32_t output_addr = output->addr;

  PREC input_prec = PRECISION(input->dtype);
  PREC output_prec = PRECISION(output->dtype);

  FP8_TYPE input_fp8type = FP8TYPE(input->dtype);

  atomic_roi_avg_pooling_check(
      input_addr, roi_addr, output_addr, 0, input->shape.n, input->shape.c,
      input->shape.h, input->shape.w, output->shape.w, kernel.h, kernel.w,
      scale, 0 /*except_val*/, 0 /*do_rq*/, 0 /*rq_is_const*/, 0 /*rq scale*/,
      sym_range, do_relu, input_fp8type, 0 /*kernel_fp8*/, 0 /*res fp8*/,
      input_prec, output_prec, round_mode);
}

void roi_max_min_pool_check(ppl_tensor_t *input, ppl_tensor_t *roi,
                            ppl_tensor_t *output, dim2 kernel, int do_relu,
                            int pool_op) {
  uint32_t input_addr = input->addr;
  uint32_t roi_addr = roi->addr;
  uint32_t output_addr = output->addr;

  PREC input_prec = PRECISION(input->dtype);
  PREC output_prec = PRECISION(output->dtype);

  int input_sign = SIGN(input->dtype);

  atomic_roi_max_min_pooling_check(
      input_addr, roi_addr, output_addr, input->shape.n, input->shape.c,
      input->shape.h, input->shape.w, output->shape.w, kernel.h, kernel.w, 0,
      input_sign, input_prec, do_relu, pool_op);
}

//---------------------------------------------------------------------------------
static const dim4 kIdentityPerm = {0, 1, 2, 3};
static const dim4 kNoBroadcast = {0, 0, 0, 0};

void check_match_2(ppl_tensor_t *output,
                   ppl_tensor_t *input,
                   const dim4* perm_opt,
                   const dim4* bmask_opt) {
  assert(output != NULL && "[check_match_2] output tensor is NULL");
  assert(input != NULL && "[check_match_2] input tensor is NULL");

  dim4 perm  = perm_opt  ? *perm_opt  : kIdentityPerm;
  dim4 bmask = bmask_opt ? *bmask_opt : kNoBroadcast;

  assert(is_valid_perm(perm) && "[check_match_2] perm must be a permutation of {0,1,2,3}");

  dim4 input_perm_shape = permute_dim4(input->shape, perm);

  int err = match_with_broadcast(input_perm_shape, output->shape, bmask);
  if (err >= 0) {
    const char* names[] = {"N","C","H","W"};
    fprintf(stderr, "[check 2 tensor match] FAIL: dim=%s\n", names[err]);
    assert(!"[check_match_2] Shape mismatch between input and output");
    return;
  }
}

void check_match_3(ppl_tensor_t* a, ppl_tensor_t* b, ppl_tensor_t* c) {
  ppl_tensor_t* arr[3] = {a, b, c};
  check_shapes_all_equal(arr, 3, "check_match_3");
}

void check_match_4(ppl_tensor_t* a, ppl_tensor_t* b, ppl_tensor_t* c, ppl_tensor_t* d) {
  ppl_tensor_t* arr[4] = {a, b, c, d};
  check_shapes_all_equal(arr, 4, "check_match_4");
}

void check_match_5(ppl_tensor_t* a, ppl_tensor_t* b, ppl_tensor_t* c, ppl_tensor_t* d, ppl_tensor_t* e) {
  ppl_tensor_t* arr[5] = {a, b, c, d, e};
  check_shapes_all_equal(arr, 5, "check_match_5");
}

// t: 0=NN, 1=NT, 2=TT
//   NN: MK * KN = MN
//   NT: MK * NK = MN
//   TT: KM * NK = NM
void check_matmul(ppl_tensor_t* lhs,
                  ppl_tensor_t* rhs,
                  ppl_tensor_t* output,
                  int a, int b,
                  int t /* 0:NN, 1:NT, 2:TT */) {
  assert(lhs && "[check_matmul] lhs tensor is NULL");
  assert(rhs && "[check_matmul] rhs tensor is NULL");
  assert(output && "[check_matmul] output tensor is NULL");
  assert(a >= 0 && a < 4 && b >= 0 && b < 4 && a != b && "[check_matmul] invalid matmul axes (a,b)");

  dim4 lShape = lhs->shape;
  dim4 kShape = rhs->shape;
  dim4 rShape = output->shape;

  fprintf(stderr, "[check_matmul] shapes (t=%d, a=%d, b=%d):\n", t, a, b);
  fprintf(stderr, "[check_matmul] lhs shape = (%d, %d, %d, %d)\n",
  dim4_get(lShape, 0), dim4_get(lShape, 1), dim4_get(lShape, 2), dim4_get(lShape, 3));
  fprintf(stderr, " rhs shape = (%d, %d, %d, %d)\n",
  dim4_get(kShape, 0), dim4_get(kShape, 1), dim4_get(kShape, 2), dim4_get(kShape, 3));
  fprintf(stderr, " out shape = (%d, %d, %d, %d)\n",
  dim4_get(rShape, 0), dim4_get(rShape, 1), dim4_get(rShape, 2), dim4_get(rShape, 3));
  fflush(stderr);

  check_other_dims_equal(lShape, kShape, rShape, a, b);

  int L_row=0, L_col=0;
  int R_row=0, R_col=0;
  int O_row=0, O_col=0;

  get_dims_ab(lShape, a, b, &L_row, &L_col); // lhs: (a->row, b->col)
  get_dims_ab(kShape, a, b, &R_row, &R_col); // rhs
  get_dims_ab(rShape, a, b, &O_row, &O_col); // output

  int M_rst=0, N_rst=0;
  int M_lhs=0, K_lhs=0;
  int K_rhs=0, N_rhs=0;

  switch (t) {
    case 0: // NN: MK * KN = MN
      M_rst = dim4_get(rShape, a);
      N_rst = dim4_get(rShape, b);
      M_lhs = dim4_get(lShape, a);
      K_lhs = dim4_get(lShape, b);
      K_rhs = dim4_get(kShape, a);
      N_rhs = dim4_get(kShape, b);
      break;
    case 1: // NT: MK * NK = MN
      M_rst = dim4_get(rShape, a);
      N_rst = dim4_get(rShape, b);
      M_lhs = dim4_get(lShape, a);
      K_lhs = dim4_get(lShape, b);
      K_rhs = dim4_get(kShape, b);
      N_rhs = dim4_get(kShape, a);
      break;
    case 2: // TT: KM * NK = NM
      M_rst = dim4_get(rShape, b);
      N_rst = dim4_get(rShape, a);
      M_lhs = dim4_get(lShape, b);
      K_lhs = dim4_get(lShape, a);
      K_rhs = dim4_get(kShape, b);
      N_rhs = dim4_get(kShape, a);
      break;
    default:
      fprintf(stderr, "[check_matmul] invalid type t=%d (expected 0:NN,1:NT,2:TT)\n", t);
      assert(0 && "[check_matmul] invalid matmul type");
      return;
  }

  if (!(M_lhs > 0 && K_lhs > 0 && K_rhs > 0 && N_rhs > 0 && M_rst > 0 && N_rst > 0)) {
    fprintf(stderr, "[check_matmul] non-positive dims: M_lhs=%d K_lhs=%d K_rhs=%d N_rhs=%d M_rst=%d N_rst=%d\n",
            M_lhs, K_lhs, K_rhs, N_rhs, M_rst, N_rst);
    print_dim4("[check_matmul] lhs=", lShape); fprintf(stderr, "\n");
    print_dim4("[check_matmul] rhs=", kShape); fprintf(stderr, "\n");
    print_dim4("[check_matmul] out=", rShape); fprintf(stderr, "\n");
    fflush(stderr);
    assert(0 && "[check_matmul] MatMul dims must be positive");
    return;
  }

  if (K_lhs != K_rhs) {
    fprintf(stderr, "[check_matmul] FAIL: K mismatch (lhs.K=%d vs rhs.K=%d) [t=%d]\n", K_lhs, K_rhs, t);
    print_dim4("[check_matmul] lhs=", lShape); fprintf(stderr, "\n");
    print_dim4("[check_matmul] rhs=", kShape); fprintf(stderr, "\n");
    fflush(stderr);
    assert(0 && "[check_matmul] MatMul K dimension mismatch between lhs and rhs");
    return;
  }
  if (M_lhs != M_rst) {
    fprintf(stderr, "[check_matmul] FAIL: M mismatch (lhs.M=%d vs out.M=%d) [t=%d]\n", M_lhs, M_rst, t);
    print_dim4("[check_matmul] lhs=", lShape); fprintf(stderr, "\n");
    print_dim4("[check_matmul] out=", rShape); fprintf(stderr, "\n");
    fflush(stderr);
    assert(0 && "[check_matmul] MatMul M dimension mismatch between lhs and output");
    return;
  }
  if (N_rhs != N_rst) {
    fprintf(stderr, "[check_matmul] FAIL: N mismatch (rhs.N=%d vs out.N=%d) [t=%d]\n", N_rhs, N_rst, t);
    print_dim4("[check_matmul] rhs=", kShape); fprintf(stderr, "\n");
    print_dim4("[check_matmul] out=", rShape); fprintf(stderr, "\n");
    fflush(stderr);
    assert(0 && "[check_matmul] MatMul N dimension mismatch between rhs and output");
    return;
  }

}

static inline int d4(dim4 d, int i) { return dim4_get(d, i); }

void check_mm(ppl_tensor_t* left, ppl_tensor_t* right, ppl_tensor_t* dst) {
  assert(left && "[check_mm] left tensor is NULL");
  assert(right && "[check_mm] right tensor is NULL");
  assert(dst && "[check_mm] dst tensor is NULL");

  dim4 L = left->shape;
  dim4 R = right->shape;
  dim4 O = dst->shape;

  fprintf(stderr, "[check_mm] NN matmul with folded channels\n");
  fprintf(stderr, "[check_mm] left = (%d, %d, %d, %d)\n", d4(L,0), d4(L,1), d4(L,2), d4(L,3));
  fprintf(stderr, "[check_mm] right= (%d, %d, %d, %d)\n", d4(R,0), d4(R,1), d4(R,2), d4(R,3));
  fprintf(stderr, "[check_mm] dst  = (%d, %d, %d, %d)\n", d4(O,0), d4(O,1), d4(O,2), d4(O,3));
  fflush(stderr);

  int M_l = d4(L, 0);
  int K_l = d4(L, 1) * d4(L, 3);

  int K_r = d4(R, 0);
  int N_r = d4(R, 1) * d4(R, 3);

  int M_o = d4(O, 0);
  int N_o = d4(O, 1) * d4(O, 3);

  if (!(M_l > 0 && K_l > 0 && K_r > 0 && N_r > 0 && M_o > 0 && N_o > 0)) {
    fprintf(stderr, "[check_mm] non-positive dims: "
                    "[check_mm] M_l=%d K_l=%d K_r=%d N_r=%d M_o=%d N_o=%d\n",
            M_l, K_l, K_r, N_r, M_o, N_o);
    fflush(stderr);
    assert(0 && "[check_mm] MatMul dims must be positive");
    return;
  }

  if (K_l != K_r) {
    fprintf(stderr, "[check_mm] FAIL: K mismatch (left.K=%d vs right.K=%d)\n", K_l, K_r);
    fflush(stderr);
    assert(0 && "[check_mm] MatMul K dimension mismatch between left and right");
    return;
  }

  if (M_l != M_o) {
    fprintf(stderr, "[check_mm] FAIL: M mismatch (left.M=%d vs dst.M=%d)\n", M_l, M_o);
    fflush(stderr);
    assert(0 && "[check_mm] MatMul M dimension mismatch between left and dst");
    return;
  }

  if (N_r != N_o) {
    fprintf(stderr, "[check_mm] FAIL: N mismatch (right.N=%d vs dst.N=%d)\n", N_r, N_o);
    fflush(stderr);
    assert(0 && "[check_mm] MatMul N dimension mismatch between right and dst");
    return;
  }

}

void check_size(ppl_tensor_t* a, ppl_tensor_t* b) {
  assert(a && "tensor a is NULL");
  assert(b && "tensor b is NULL");

  dim4 A = a->shape;
  dim4 B = b->shape;
  fflush(stderr);

  long long size_a = (long long)d4(A,0) * d4(A,1) * d4(A,2) * d4(A,3);
  long long size_b = (long long)d4(B,0) * d4(B,1) * d4(B,2) * d4(B,3);

  if (!(d4(A,0) > 0 && d4(A,1) > 0 && d4(A,2) > 0 && d4(A,3) > 0 &&
        d4(B,0) > 0 && d4(B,1) > 0 && d4(B,2) > 0 && d4(B,3) > 0)) {
    fflush(stderr);
    assert(0 && "[check_size] Tensor dims must be positive");
    return;
  }

  if (size_a != size_b) {
    fprintf(stderr, "[check_size] FAIL: size mismatch (a=%lld vs b=%lld)\n", size_a, size_b);
    fflush(stderr);
    assert(0 && "[check_size] Total element count mismatch between tensors");
    return;
  }

  fflush(stderr);
}
