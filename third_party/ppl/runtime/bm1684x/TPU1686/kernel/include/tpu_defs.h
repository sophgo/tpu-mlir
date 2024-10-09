#ifndef _TPU_DEFS_
#define _TPU_DEFS_
typedef unsigned int local_addr_t;
typedef unsigned int static_addr_t;
typedef unsigned long long l2_sram_addr_t;
typedef unsigned long long system_addr_t;
typedef unsigned long long global_addr_t;
typedef unsigned long long addr_t;
typedef unsigned long long u64;
typedef long long i64;
typedef unsigned int u32;
#define NO_USE 0

#define TENSOR_N_DIM 0
#define TENSOR_C_DIM 1
#define TENSOR_H_DIM 2
#define TENSOR_W_DIM 3

typedef enum {
    DT_INT8   = (0 << 1) | 1,
    DT_UINT8  = (0 << 1) | 0,
    DT_INT16  = (3 << 1) | 1,
    DT_UINT16 = (3 << 1) | 0,
    DT_FP16   = (1 << 1) | 1,
    DT_BFP16  = (5 << 1) | 1,
    DT_INT32  = (4 << 1) | 1,
    DT_UINT32 = (4 << 1) | 0,
    DT_FP32   = (2 << 1) | 1,
    DT_INT4   = (6 << 1) | 1,
    DT_UINT4  = (6 << 1) | 0,
    DT_FP8E5M2 = (0 << 5) | (7 << 1) | 1,
    DT_FP8E4M3 = (1 << 5) | (7 << 1) | 1,
    DT_FP20   = (8 << 1) | 1,
    DT_TF32   = (9 << 1) | 1,
} data_type_t;
typedef enum {
    RM_HALF_TO_EVEN        = 0,
    RM_HALF_AWAY_FROM_ZERO = 1,
    RM_TOWARDS_ZERO        = 2,
    RM_DOWN                = 3,   /* FLOOR */
    RM_UP                  = 4,   /* CEIL */
    RM_HALF_UP             = 5,
    RM_HALF_DOWN           = 6
} rounding_mode_t;

typedef enum {
  REDUCE_MEAN = 0,
  REDUCE_SUM  = 1,
  REDUCE_MAX  = 2,
  REDUCE_MIN  = 3,
  REDUCE_PROD = 4,
  REDUCE_L2   = 5,
  REDUCE_L1   = 6,
} reduce_method_t;

typedef struct {
    int n, c, d, h, w;
} dim5;
typedef struct {
    int n, c, h, w;
} dim4;
typedef struct {
    int h, w;
} dim2;
typedef struct {
    int top, bottom, left, right;
} padding_t;
typedef struct {
    int start, end;
} range_t;

typedef union {
    unsigned short bits;
    struct {
        unsigned short frac : 2; // mantissa
        unsigned short exp  : 5;  // exponent
        unsigned short sign : 1;  // sign
    };
} float8e5m2;

typedef union {
    unsigned short bits;
    struct {
        unsigned short frac : 3; // mantissa
        unsigned short exp  : 4;  // exponent
        unsigned short sign : 1;  // sign
    } format;
} float8e4m3;

typedef union {
    unsigned short bits;
    struct {
        unsigned short frac : 10; // mantissa
        unsigned short exp  : 5;  // exponent
        unsigned short sign : 1;  // sign
    } format;
} float16;
typedef union {
    unsigned short bits;
    struct {
        unsigned short frac : 7;  // mantissa
        unsigned short exp  : 8;  // exponent
        unsigned short sign : 1;  // sign
    } format;
} bfloat16;
typedef union {
    unsigned short bits;
    struct {
        unsigned short frac : 11; // mantissa
        unsigned short exp  : 8;  // exponent
        unsigned short sign : 1;  // sign
    } format;
} float20;
typedef union {
    char           s4;
    unsigned char  u4;
    char           s8;
    unsigned char  u8;
    float8e5m2     f8e5m2;
    float8e4m3     f8e3m4;
    short          s16;
    unsigned short u16;
    float16        f16;
    bfloat16       bf16;
    int            s32;
    unsigned int   u32;
    float          f32;
    float20        f20;
} scalar_t;

typedef struct
{
    int addr_num;
    int base_idx[30];
    u64 base_addr[30];
    u64 bdc_cmd_offset;
    u64 gdma_cmd_offset;
    u64 hau_cmd_addr;
    u64 sdma_cmd_addr;
    int bdc_cmd_num[8];
    int gdma_cmd_num[8];
    int hau_cmd_len;
    int sdma_cmd_len;
    u64 tiu_len[8];
    u64 gdma_len[8];
} launch_cache_multicore_t;

#endif /* _OK_DEFS_ */