#ifndef ATOMIC_RANDOM_GEN_H
#define ATOMIC_RANDOM_GEN_H
#include "checker_internel.h"
#ifdef __cplusplus
extern "C" {
#endif

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
    int need_store);

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
    RAND_OP op_type);

#ifdef __cplusplus
}
#endif

#endif  /* ATOMIC_RANDOM_GEN_H */
