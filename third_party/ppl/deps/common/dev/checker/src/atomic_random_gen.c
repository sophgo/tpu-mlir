#include "atomic_random_gen.h"

#define CHECK_RG_STRIDE(p_stride) \
      ASSERT((p_stride[0] < (((int)1) << 18)) && (p_stride[0] >= 0)); \
      ASSERT((p_stride[1] < (((int)1) << 18)) && (p_stride[1] >= 0)); \
      ASSERT((p_stride[2] < (((int)1) << 18)) && (p_stride[2] >= 0)); \
      ASSERT((p_stride[3] < (((int)1) << 18)) && (p_stride[3] >= 0)); \

#define UNUSED(x) (void)(x)

void atomic_random_gen_check(
    unsigned int addr,
    int n,
    int c,
    int h,
    int w,
    int * stride,
    int short_str,
    PREC prec,
    unsigned int load_state_addr,
    unsigned int store_state_addr,
    int need_store,
    RAND_OP op_type
) {
#ifdef USING_CMODEL
    ASSERT(short_str == 0);
    ASSERT(addr % ALIGN_BYTES == 0);
    ASSERT(prec != INT4 && prec != FP32 && prec != FP16 && prec != BFP16 && prec != FP8);
    ASSERT(n == 1);
    ASSERT(c < (((int)1) << 16) && (c > 0));
    ASSERT(h == 1);
    ASSERT(w < (((int)1) << 16) && (w > 0));
    ASSERT(w % get_eu_num(prec) == 0);
    if(need_store) {
        ASSERT(addr / LOCAL_MEM_SIZE == store_state_addr / LOCAL_MEM_SIZE);
    }
    if (op_type == PRNG_WITH_LOADED_STATES) {
        ASSERT(addr / LOCAL_MEM_SIZE == load_state_addr / LOCAL_MEM_SIZE);
    }
    ASSERT(op_type == PRNG || op_type == PRNG_WITH_LOADED_STATES);
    if(op_type == PRNG) {
        ASSERT(c <= NPU_NUM && c > 0);
    }
#endif
}

void atomic_random_gen_init_seed_check(
    unsigned int addr,
    int n,
    int c,
    int h,
    int w,
    int * stride,
    int short_str,
    PREC prec,
    int jump_cnt,
    int c_offset,
    unsigned int store_state_addr,
    int need_store
) {
#ifdef USING_CMODEL
    ASSERT((short_str == 0));
    ASSERT(addr % ALIGN_BYTES == 0);
    ASSERT(prec != INT4 && prec != FP32 && prec != FP16 && prec != BFP16 && prec != FP8);
    ASSERT(n == 1);
    ASSERT(c < (((int)1) << 16) && (c > 0));
    ASSERT(h == 1);
    ASSERT(w < (((int)1) << 16) && (w > 0));
    ASSERT(w % get_eu_num(prec) == 0);
    ASSERT(c_offset < (((int)1) << 16) && (c_offset >= 0));
    ASSERT((short_str == 0));
    if(need_store) {
        ASSERT(addr / LOCAL_MEM_SIZE == store_state_addr / LOCAL_MEM_SIZE);
    }
#endif
}
