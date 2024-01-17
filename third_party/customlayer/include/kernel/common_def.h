#ifndef _COMMON_DEF_H_
#define _COMMON_DEF_H_

typedef enum {
  SG_ROUND_INF = 0,     // 1.5 -> 2   -1.5 -> -2
  SG_ROUND_UP = 1,      // 1.5 -> 2   -1.5 -> -1
  SG_ROUND_DOWN = 2,    // 1.5 -> 1   -1.5 -> -2
  SG_ROUND_EVEN = 3,    // 1.5 -> 2    2.5 -> 2
  SG_ROUND_ODD = 4,     // 1.5 -> 1    0.5 -> 1
  SG_ROUND_ZERO = 5,    // 1.5 -> 1   -1.5 -> -1
  SG_TRIM_ZERO = 6,     // 1.6 -> 1   -1.6 -> -1
  SG_TRIM_INF = 7,      // 1.4 -> 2   -1.4 -> -2
  SG_TRIM_UP = 8,       // 1.4 -> 2   -1.6 -> -1
  SG_TRIM_DOWN = 9,     // 1.6 -> 1   -1.4 -> -2
} sg_round_mode_t;

typedef enum {
  SG_DTYPE_FP32 = 0,
  SG_DTYPE_FP16 = 1,
  SG_DTYPE_INT8 = 2,
  SG_DTYPE_UINT8 = 3,
  SG_DTYPE_INT16 = 4,
  SG_DTYPE_UINT16 = 5,
  SG_DTYPE_INT32 = 6,
  SG_DTYPE_UINT32 = 7,
  SG_DTYPE_BFP16 = 8,
  SG_DTYPE_INT4 = 9,
  SG_DTYPE_UINT4 = 10,
  SG_DTYPE_UNKNOWN = -1,
} sg_data_type_t;

#endif
