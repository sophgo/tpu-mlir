#ifndef COMMON_H_
#define COMMON_H_

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "fp16.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef signed char s8;
typedef signed short s16;
typedef signed int s32;
typedef signed long long s64;

typedef signed char  s8;
typedef signed short s16;
typedef signed int   s32;
typedef signed long long int s64;

typedef u32 stride_type;
typedef u32 size_type;

typedef void *P_COMMAND;

typedef union {
  uint16_t bits;
  struct {
    uint16_t frac : 10; // mantissa
    uint16_t exp  : 5;  // exponent
    uint16_t sign : 1;  // sign
  } format;
} fp16;

typedef union {
  uint16_t bits;
  struct {
    uint16_t frac : 7; // mantissa
    uint16_t exp  : 8; // exponent
    uint16_t sign : 1; // sign
  } format;
} bf16;

typedef union {
  float    fval;
  uint32_t bits;
  struct {
    uint32_t frac : 23; // mantissa
    uint32_t exp  : 8;  // exponent
    uint32_t sign : 1;  // sign
  } format;
} fp32;

typedef union {
  double double_val;
  uint64_t bits;
} Double;

typedef union {
  float f32val;
  signed int i32val;
  unsigned int u32val;
  signed short i16val;
  unsigned short u16val;
  signed char i8val;
  unsigned char u8val;
  fp16 f16val;
  bf16 bf16val;
} DataUnion;


typedef enum {
    ROUND_HALF_TO_EVEN = 0, // -1.5 -> -2, -2.5 -> -2, 3.5 -> 4,
    ROUND_HALF_AWAY_FROM_ZERO = 1, // 1.5 -> 2, 1.9 -> 2, -1.5 -> -2, -1.9 -> -2
    ROUND_TOWARDS_ZERO = 2, // 1.5 -> 1, 1.9 -> 1, -1.5 -> -1, -1.9 -> -1
    ROUND_DOWN = 3, // floor 1.9 -> 1, -1.9 -> -2
    ROUND_UP = 4, // ceil 1.1 -> 2, -1.1 -> -1
    ROUND_HALF_UP = 5, // 1.5 -> 2, -1.5 -> -1
    ROUND_HALF_DOWN = 6, // 1.5 -> 1, -1.5 -> -2
} ROUND_MODE;

//#define INT8_SIZE 1
#define FLOAT_SIZE 4
//#define FLOAT_BITWIDTH 32
//#define GET_U64(U32_H, U32_L) (((u64)(U32_H) << 32) | (u64)(U32_L))

#define __ALIGN_MASK(x, mask) (((x) + (mask)) & ~(mask))
#ifndef ALIGN
#define ALIGN(x, a) __ALIGN_MASK(x, (__typeof__(x))(a)-1)
#endif

#define sg_min(x, y) (((x)) < ((y)) ? (x) : (y))
#define sg_max(x, y) (((x)) > ((y)) ? (x) : (y))
#define SWAP_VAL(a, b) \
  a ^= b;              \
  b ^= a;              \
  a ^= b

#define SG_IS_NAN(x) ((((x >> 23) & 0xff) == 0xff) && ((x & 0x7fffff) != 0))

#ifdef USING_CMODEL
  u8 *get_global_memaddr(int);
  u8 *get_local_memaddr_by_node(int, int);
  u8 *get_static_memaddr_by_node(int);
  u8 *get_l2_memaddr(int);
  #define GET_GLOBAL_ADDR(ADDR) \
    ((u8 *)get_global_memaddr(0) + (ADDR) - GLOBAL_MEM_START_ADDR)
  #define GET_LOCAL_ADDR(LOCALMEM_IDX, LOCALMEM_OFFSET) \
    ((u8 *)get_local_memaddr_by_node(0, LOCALMEM_IDX) + (LOCALMEM_OFFSET))
  #define GET_SMEM_ADDR(ADDR) \
    ((u8 *)get_static_memaddr_by_node(0) + (ADDR) - STATIC_MEM_START_ADDR)
  #define GET_L2_SRAM_ADDR(ADDR) \
    ((u8 *)get_l2_sram(0) + (ADDR) - L2_SRAM_START_ADDR)
#else
#if (defined(USING_PLD_TEST) || defined(USING_BAREMETAL_BUILD) || defined(USING_EDA))
  #define GET_GLOBAL_ADDR(ADDR) \
    (GLOBAL_MEM_START_ADDR_ARM + (ADDR) - GLOBAL_MEM_START_ADDR)
  #define GET_LOCAL_ADDR(LOCALMEM_IDX, LOCALMEM_OFFSET) \
    (LOCAL_MEM_START_ADDR + LOCALMEM_IDX * LOCAL_MEM_SIZE + (LOCALMEM_OFFSET))
  #define GET_SMEM_ADDR(ADDR) \
    (u64)ADDR
  #define GET_L2_SRAM_ADDR(ADDR) \
    (u64)ADDR
#else
  #define GET_GLOBAL_ADDR(ADDR) \
    (map_to_kaddr(GLOBAL_MEM_START_ADDR_ARM + (ADDR) - GLOBAL_MEM_START_ADDR))
  #define GET_LOCAL_ADDR(LOCALMEM_IDX, LOCALMEM_OFFSET) \
    (map_to_kaddr(LOCAL_MEM_START_ADDR + LOCALMEM_IDX * LOCAL_MEM_SIZE + (LOCALMEM_OFFSET)))
  #define GET_SMEM_ADDR(ADDR) \
    map_to_kaddr((u64)ADDR)
  #define GET_L2_SRAM_ADDR(ADDR) \
    map_to_kaddr((u64)ADDR)
#endif
#endif

#ifdef USING_CMODEL
#define GET_SHARE_MEM_ADDR(offset) cmodel_get_share_memory_addr(offset, get_cur_nodechip_idx())
#define GLOBAL_MEM_SIZE(node_idx) (cmodel_get_global_mem_size(node_idx))
#else
#define GET_SHARE_MEM_ADDR(offset) (u32 *)(SHARE_MEM_START_ADDR + (offset)*4)
#define GLOBAL_MEM_SIZE(node_idx) (CONFIG_GLOBAL_MEM_SIZE)
#endif

#define IN_L2_SRAM(addr) (((addr) >= L2_SRAM_START_ADDR) && ((addr) < L2_SRAM_START_ADDR + L2_SRAM_SIZE))
#define IN_GLOBAL_MEM(addr) ((addr) >= GLOBAL_MEM_START_ADDR)

/* info about cmd_node */
typedef struct gdma_cmd_node_info_s {
  int n;
  int c;
  int h;
  int w;
  int direction;
  int src_format;
  int dest_format;
  bool setted;
} gdma_cmd_node_info_t;

typedef struct inst_profile {
  unsigned long long cycle;
  unsigned long long gdma_size;
  int gdma_direction;
  int src_format;
  int dst_format;
  double power;
  double compute_ability;
  bool b_gdma_use_l2;
} INST_PROFILE;

typedef struct cmd_id_node {
  unsigned int bd_cmd_id;
  unsigned int gdma_cmd_id;
  bool in_parallel_state;
#if defined(SG_STAS_GEN) || defined(SG_TV_GEN)
  unsigned int cycle_count;
  unsigned int cur_op_cycle;
#endif
#ifdef SG_STAS_GEN
  char cmd_name[16];
  char name_prefix[64];
  gdma_cmd_node_info_t gdma_cmd_info;
  INST_PROFILE inst_profile;
#endif
} CMD_ID_NODE;

#ifdef SG_STAS_GEN
static inline void set_gdma_cmd_info(CMD_ID_NODE *pid_node, int n, int c, int h,
                                     int w, int direction, int src_format,
                                     int dest_format) {
  gdma_cmd_node_info_t *the_info = &pid_node->gdma_cmd_info;
  the_info->n = n;
  the_info->c = c;
  the_info->h = h;
  the_info->w = w;
  the_info->direction = direction;
  the_info->src_format = src_format;
  the_info->dest_format = dest_format;
  the_info->setted = true;
}
#else
  #define set_gdma_cmd_info(...) {}
#endif

typedef enum {
  ENGINE_BD   = 0,
  ENGINE_GDMA = 1,
  ENGINE_GDE  = 2,
  ENGINE_SORT = 3,
  ENGINE_NMS  = 4,
  ENGINE_CDMA = 5,
  ENGINE_END
} ENGINE_ID;

typedef enum {
  STORAGE_MODE_1N_FP32    = 0,
  STORAGE_MODE_1N_INT8    = 1,
  STORAGE_MODE_1N_INT16   = 2,
  STORAGE_MODE_2N_INT16   = 3,
  STORAGE_MODE_4N_INT8    = 4,
  STORAGE_MODE_2IC_FP32   = 5,  // special for 2IC weight
  STORAGE_MODE_4N_4IC_4OC = 6,
  STORAGE_MODE_4N_INT16   = 7,
  STORAGE_MODE_UNINITILIZED,
  STORAGE_MODE_END
} TENSOR_STORAGE_MODE;

#ifdef __cplusplus
}
#endif

#endif

