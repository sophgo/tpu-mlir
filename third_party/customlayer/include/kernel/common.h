#ifndef COMMON_H_
#define COMMON_H_

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "op_code.h"
#include "common_def.h"
#include "cv_def.h"
#include "macros.h"

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

typedef struct {
  int8_t val : 4;
} int4_t;

typedef struct {
  uint8_t val : 4;
} uint4_t;

typedef int4_t  s4;
typedef uint4_t  u4;

typedef union {
  unsigned long long u64val;
  long long i64val;
  float f32val;
  signed int i32val;
  unsigned int u32val;
  signed short i16val;
  unsigned short u16val;
  signed char i8val;
  unsigned char u8val;
  int4_t  i4val;
  uint4_t u4val;
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
  double op_dyn_energy; //nJ
  double sram_rw_energy; // nJ
  double compute_ability;
  bool b_gdma_use_l2;
} INST_PROFILE;

//bm1686 also use it
#define CONFIG_MAX_CDMA_NUM 10
typedef struct cmd_id_node {
  unsigned int bd_cmd_id;
  unsigned int gdma_cmd_id;
  unsigned int hau_cmd_id;
  bool in_parallel_state;
#if defined(SG_STAS_GEN) || defined(SG_TV_GEN)
  long long cycle_count;
  long long cur_op_cycle;
#endif
#ifdef SG_STAS_GEN
  char cmd_name[16];
  char name_prefix[64];
  gdma_cmd_node_info_t gdma_cmd_info;
  INST_PROFILE inst_profile;
#endif
  unsigned int sdma_cmd_id;
  unsigned int cdma_cmd_id[CONFIG_MAX_CDMA_NUM];
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
    INT8 = 0,
    FP16 = 1,
    FP32 = 2,
    INT16 = 3,
    INT32 = 4,
    BFP16 = 5,
    INT4 = 6,
} PREC;

typedef enum host_cdma_dir { HOST2CHIP, CHIP2HOST, CHIP2CHIP } HOST_CDMA_DIR;

INLINE static int ceiling_func(int numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

INLINE static int ceiling_func_shift(int numerator, int shift) {
  return (numerator + (1 << shift) - 1) >> shift;
}

INLINE static int get_bytesize(PREC precison) {
  int bytesize = 4;
  if (precison == INT8 || precison == INT4) {
    bytesize = 1;
  } else if (precison == INT16 || precison == FP16 || precison == BFP16) {
    bytesize = 2;
  }
  return bytesize;
}

inline static int get_bit_width(PREC precision) {
  int bit_width = 8;
  if (precision == INT4) {
    bit_width = 4;
  } else if (precision == INT8) {
    bit_width = 8;
  } else if (precision == INT16 || precision == FP16 || precision == BFP16) {
    bit_width = 16;
  } else if (precision == INT32 || precision == FP32) {
    bit_width = 32;
  } else {
    ASSERT(0 && "invalid precision");
  }
  return bit_width;
}


#define pipeline_move(array, num) do { \
  for (int i = (int)num - 1; i > 0; i--) { \
    array[i] = array[i - 1];\
  }\
} while(0)

inline static bool is_float_prec(PREC precision) {
  return (precision == FP32 || precision == FP16 || precision == BFP16);
}

inline static bool is_fixed_prec(PREC precision) {
  return (precision == INT4 || precision == INT8 ||
          precision == INT16 || precision == INT32);
}


INLINE static int pointer_wrap_around(u32 cur_pointer, int step, int len_bit_width) {
  u32 max_len     = (1 << len_bit_width);
  u32 new_pointer = 0;

  new_pointer = cur_pointer + step;
  if (new_pointer >= max_len) new_pointer -= max_len;

  return (int)new_pointer;
}

#ifdef __cplusplus
}
#endif

#endif
