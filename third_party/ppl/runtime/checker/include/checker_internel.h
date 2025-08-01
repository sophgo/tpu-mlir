#pragma once
#include "ppl_helper.h"
#include "assert.h"
#ifdef __sg2262__
#include "rvt_api.h"
#else
typedef enum {
  INT8 = 0,
  FP16 = 1,
  FP32 = 2,
  INT16 = 3,
  INT32 = 4,
  BFP16 = 5,
  INT4 = 6,
  FP8 = 7,
  FP20 = 8,
  TF32 = 9,
} PREC;
#endif

typedef enum {
  FP8E5M2 = 0,
  FP8E4M3 = 1,
} FP8_TYPE;

// dma maroc
#define GDMA_VALUE_DIR_S2L  0
#define GDMA_VALUE_DIR_L2S  1
#define GDMA_VALUE_DIR_S2S  2
#define GDMA_VALUE_DIR_L2L  3
#define GDMA_VAULE_DIR_NUM  4

#define GDMA_VALUE_FORMAT_FLOAT32  0
#define GDMA_VALUE_FORMAT_INT16    1
#define GDMA_VALUE_FORMAT_UINT8    2
#define GDMA_VALUE_FORMAT_INT8     3
#define GDMA_VALUE_FORMAT_FLOAT16  4
#define GDMA_VALUE_FORMAT_NUM 5

#define LMEM_TAG                (0x1ful)

#if defined(__bm1686__)
  #define MAX_TPU_CORE_NUM 2
# elif defined(__mars3__)
  #define MAX_TPU_CORE_NUM 1
# elif defined(__sg2260__)
  #define MAX_TPU_CORE_NUM 8
# elif defined(__sg2260e__) || defined(__sg2380__)
  #define MAX_TPU_CORE_NUM 4
# elif defined(__sg2262__)
  #define MAX_TPU_CORE_NUM 64
#endif

#ifdef __mars3__
  #define STATIC_MEM_SIZE 0x800
#else
  #define STATIC_MEM_SIZE 0x10000
#endif
#if defined(__sg2260__) || defined(__sg2260e__)
  #define GLOBAL_MEM_START_ADDR 0x0UL
# elif defined(__mars3__) || defined(__sg2262__) || defined(__sg2380__)
  #define GLOBAL_MEM_START_ADDR 0x80000000UL
#else
  #define GLOBAL_MEM_START_ADDR 0x100000000UL
#endif

#ifdef __bm1684xe_
  #define MAX_GMEM_BIT            (35)
# elif defined(__bm1686__)
  #define MAX_GMEM_BIT            (36)
#else
  #define MAX_GMEM_BIT            (40)
#endif
#define MAX_GMEM_SIZE           (1ull << MAX_GMEM_BIT)
#define CONFIG_GLOBAL_MEM_SIZE 0x10000

#ifdef __bm1686_
 #define NNVLC_ALIGN_SHIFT       4
#else
 #define NNVLC_ALIGN_SHIFT       7
#endif
#define NNVLC_ALIGN_BYTES       (1 << NNVLC_ALIGN_SHIFT)

#define SRC_IS_LOCAL(direction) \
    ((direction) == GDMA_VALUE_DIR_L2L || (direction) == GDMA_VALUE_DIR_L2S)
#define DST_IS_LOCAL(direction) \
    ((direction) == GDMA_VALUE_DIR_S2L || (direction) == GDMA_VALUE_DIR_L2L)

#define FORMAT_IS_FLOAT(format) \
    ((format) == GDMA_VALUE_FORMAT_FLOAT32 || (format) == GDMA_VALUE_FORMAT_FLOAT16)

#define  ADD_TAG(addr, mask, tag) \
  (((addr) & (mask)) | (((u64)tag) << MAX_GMEM_BIT))
#define  CALC_STATIC_ADDR(addr) \
  (ADD_TAG(addr, STATIC_MEM_SIZE - 1, SMEM_TAG) | (1 << 26))

#define  CALC_LOCAL_ADDR(mem_idx, mem_offset) \
  ADD_TAG((mem_idx * LOCAL_MEM_SIZE + mem_offset), LOCAL_MEM_SIZE * NPU_NUM - 1, LMEM_TAG)

// tiu maroc
#define SIGN(dtype) ((dtype) & 0x1)
#define FP8TYPE(dtype) ((dtype) >> 5)
#define PRECISION(dtype) (((dtype) >> 1) & 0xf)
#define IS_FLOAT(dtype) (dtype==DT_FP32 || dtype==DT_FP16 || dtype==DT_BFP16 || dtype==DT_FP8E5M2 || dtype==DT_FP8E4M3 || dtype==DT_FP20)
#define GETSIGN(dtype) (IS_FLOAT(dtype) ? FP8TYPE(dtype) : SIGN(dtype))

#define WIDTH(dtype) tpu_data_type_bits(dtype)
#define DSIZE(dtype) tpu_data_type_size(dtype)
#define ALIGNED_OR_USER(stride) ((stride) == NULL ? 0 : 3)
#define IS_FLOAT(dtype) (dtype==DT_FP32 || dtype==DT_FP16 || dtype==DT_BFP16 || dtype==DT_FP8E5M2 || dtype==DT_FP8E4M3 || dtype==DT_FP20)
#define ASSERT(cond) assert(cond)
#define ASSERT_FS_INFO(_cond, fmt, ...)                  \
  do {                                                   \
    if (!(_cond)) {                                      \
    printf("ASSERT_FS %s: %s: %d: %s\n",                 \
          __FILE__, __func__, __LINE__, #_cond);         \
        printf("ASSERT info: " fmt "\n", ##__VA_ARGS__); \
        assert(0);                                       \
    }                                                    \
  } while(0)

typedef enum {
  GDMA_FUNC_NONE       = 0,
  GDMA_FUNC_TRANS      = 1, // NC Transpose or Matrix Transpose
  GDMA_FUNC_BROADCAST  = 3,
} GDMA_FUNC_TYPE;

typedef enum {
  GDMA_ARE_NOP = 0,
  GDMA_ARE_MUL = 1,
  GDMA_ARE_MAX = 2,
  GDMA_ARE_MIN = 3,
  GDMA_ARE_ADD = 4,
} GDMA_ARE_OPCODE_TYPE;

typedef enum {
  GDMA_INT8 = 0,
  GDMA_FP16 = 1,
  GDMA_FP32 = 2,
  GDMA_INT16 = 3,
  GDMA_INT32 = 4,
  GDMA_BF16 = 5,
  GDMA_FP20 = 6,
  GDMA_FP8_E4M3 = 7,
  GDMA_FP8_E5M2 = 8,
  GDMA_FORMAT_NUM,
} GDMA_FORMAT;

typedef enum {
  // S: systerm memory: dram and l2sram
  GDMA_S2L = 0,
  GDMA_L2S = 1,
  GDMA_S2S = 2,
  GDMA_L2L = 3,
  GDMA_DIR_NUM,
} GDMA_DIRECTION;

typedef enum {
    PAD_CONSTANT    = 0,
    PAD_REFLECTION  = 1,
    PAD_REPLICATION = 2,
    PAD_CIRCULAR    = 3,
    PAD_MODE_NUM    = 4
} PAD_MODE;

typedef enum {
    CONV_NORMAL = 0,
    CONV_BW = 1,
    CONV_TF32 = 2,
    CONV_DW_TF32 = 4,
    CONV_OP_NUM = 5
} CONV_OP;

typedef enum {
    LIN_MAC = 1,
    LIN_ADD_SQR = 20,
    LIN_SUB_SQR = 21,
    FUSE_MADD = 22
} LIN_OP;

typedef enum {
    SFU_TAYLOR_4X = 12,
    SFU_TAYLOR    = 13,
    SFU_NORM      = 15,
    SFU_RSQ       = 17,
    SFU_FREXP     = 18,
    SFU_RECIPROCAL = 19,
    SFU_FUSE_TAYLOR_MUL = 21,
    SFU_FUSE_EXP = 22
} SFU_OP;

typedef enum {
    CMP_GT_AND_SG = 22,
    CMP_SG = 23,
    CMP_SE = 24,
    CMP_LT_AND_SL = 25,
    CMP_SL = 26,
    CMP_SRCH_BIN = 27,
    FUSE_SUB_CLAMP = 28,
} CMP_OP;

typedef enum {
    MM_NORMAL = 1,
    MM_WRQ = 2,
    MM_WRQ_RELU = 3,
    MM_NN = 4,
    MM_NT = 5,
    MM_TT = 6,
    MM_NN_TF32 = 7,
    MM_NT_TF32 = 8,
    MM_TT_TF32 = 9,
    DQ2_MM2_NT = 11,
    DQ2_MM2_TT = 12,
    BQ_MM2_NT = 13,
} MM_OP;

typedef enum {
  AR_MUL = 0,
  AR_NOT = 1,
  AR_ADD = 2,
  AR_SUB = 3,
  AR_MAX = 4,
  AR_MIN = 5,
  AR_LOGIC_SHIFT = 6,
  AR_AND = 7,
  AR_OR = 8,
  AR_XOR = 9,
  AR_SG = 10,
  AR_SE = 11,
  AR_DIV = 12,
  AR_SL = 13,
  AR_DATA_CONVERT = 14,
  AR_ADD_SATU = 15,
  AR_SUB_SATU = 16,
  AR_CLAMP = 17,
  AR_MAC = 18,
  AR_COPY = 19,
  AR_MUL_SATU = 20,
  AR_ARITH_SHIFT = 21,
  AR_ROTATE_SHIFT = 22,
  // AR_MULDHR = 23, // not support
  // AR_EU_IDX_GEN = 24,
  AR_DIV2 = 25,
  AR_ABS = 26,
  AR_FSUBABS = 27,
  // AR_COPY_MB = 28, // not support
  AR_GET_FIRST_ONE = 29,
  AR_GET_FIRST_ZERO = 30,
  AR_FUSE_MUL_CAST = 31,
  AR_DIFF_ABS = 32
} AR_OP;

typedef enum {
    PD_DEPTHWISE = 0,
    PD_AVG_POOLING = 1,
    PD_MIN_POOLING = 3,
    PD_MAX_POOLING = 4,
    PD_ROI_DEPTHWISE = 5,
    PD_ROI_AVG_POOLING = 6,
    PD_ROI_MAX_POOLING = 7,
    PD_ROI_MIN_POOLING = 8
} PD_OP;

typedef enum {
    LANE_COPY = 2,
    LANE_BROAD = 3,
    STATIC_BROAD = 4,
    STATIC_DISTRIBUTE = 5,
} BC_OP;

typedef enum {
    TRAN_C_W_TRANSPOSE = 0,
    TRAN_W_C_TRANSPOSE = 1,
} TRAN_OP;

typedef enum {
    PL_gather_d1coor = 0,
    PL_gather_d2coor = 1,
    // PL_gather_rec = 2,
    PL_scatter_d1coor = 3,
    PL_scatter_d2coor = 4,
    PE_S_gather_d1coor = 5,
    PE_S_scatter_d1coor = 6,
    PE_M_gather_d1coor = 7,
    PE_S_mask_select = 8,
    PE_S_nonzero = 9,
    // PE_S_scatter_pp_d1coor = 10,
    PE_S_gather_hzd = 13,
    PE_S_scatter_hzd = 14,
    PE_S_mask_selhzd = 15,
    PE_S_nonzero_hzd = 16,
    PE_S_gather_line = 17,
    PE_S_scatter_line = 18,
    // PE_S_mask_seline = 19,
} SG_OP;

typedef enum {
    RQ_0 = 0,
    RQ_1 = 1,
    DQ_0 = 3,
    DQ_1 = 4,
    DQ_2 = 5,
    QT_0 = 6,
} RQDQ_OP;

typedef enum {
    PRNG = 0, // use global state to generate random number
    PRNG_WITH_INTIAL_SEED = 1, // set seed
    PRNG_WITH_LOADED_STATES = 2 // load state from lmem
} RAND_OP;

static inline int tpu_get_dma_dtype(data_type_t dtype) {
    switch (dtype)
    {
    case DT_INT8:
    case DT_UINT8:
        return GDMA_INT8;
    case DT_INT4:
    case DT_UINT4:
        return GDMA_INT8;
    case DT_INT16:
    case DT_UINT16:
        return GDMA_INT16;
    case DT_FP16:
        return GDMA_FP16;
    case DT_BFP16:
        return GDMA_BF16;
    case DT_INT32:
    case DT_UINT32:
        return GDMA_INT32;
    case DT_FP8E4M3:
        return GDMA_FP8_E4M3;
    case DT_FP8E5M2:
        return GDMA_FP8_E5M2;
    case DT_FP32:
    case DT_TF32:
        return GDMA_FP32;
    case DT_FP20:
        return GDMA_FP20;
    default:
        ASSERT(0);
        return -1;
    }
}

static inline int get_eu_num(PREC precision) {
    switch(precision) {
        case INT4: return EU_NUM_8BIT << 1;
        case FP8:
        case INT8: return EU_NUM_8BIT;
        case INT16:
        case FP16:
        case BFP16: return EU_NUM_16BIT;
        case INT32:
        case TF32:
        case FP32: return EU_NUM_32BIT;
        default: ASSERT_FS_INFO(0, "ERROR PREC!");
    }
    return 0;
}

static inline int get_gdma_format_type_len(int t) {
  switch (t) {
    case GDMA_INT8:
    case GDMA_FP8_E4M3:
    case GDMA_FP8_E5M2:
      return 1;
    case GDMA_FP16:
    case GDMA_BF16:
    case GDMA_INT16:
      return 2;
    case GDMA_FP32:
    case GDMA_INT32:
      return 4;
  }
  return 0;
}

typedef struct {
  dim4 shape;
  int stride[4];
  global_addr_t addr;
  data_type_t dtype;
  int mode;
  int align_mode;
  int size;
  int offset;
  bool unsigned_flag;
  bool default_stride;
} ppl_tensor_t;

typedef struct {
  int type; // 0:TENSOR, 1:SCALAR, 2:VECTOR
  void *context; // ppl_tensor_t*, scalar_t*
} ppl_variable_t;

inline static int get_npu_index(u32 local_addr) {
  return (local_addr / LOCAL_MEM_SIZE);
}

inline static int get_bytesize(PREC precision) {
  int bytesize = 4;
  if (precision == INT8 || precision == INT4 || precision == FP8) {
    bytesize = 1;
  } else if (precision == INT16 || precision == FP16 || precision == BFP16) {
    bytesize = 2;
  }
#ifdef __sg2262__
  else if (precision == FP4) {
    bytesize = 1;
  }
#endif
  return bytesize;
}

inline static bool is_float_prec(PREC precision) {
#ifdef __sg2262__
  return (precision == FP32 || precision == FP16 || precision == BFP16 || precision == FP4 ||
          precision == FP8 || precision == TF32);
#endif
  return (precision == FP32 || precision == FP16 || precision == BFP16 || precision == FP20 ||
          precision == FP8 || precision == TF32);
}

inline static bool is_fixed_prec(PREC precision) {
  return (precision == INT4 || precision == INT8 ||
          precision == INT16 || precision == INT32);
}

inline static bool is_half_fp_prec(PREC precision) {
  return (precision == FP16 || precision == BFP16);
}

static inline int is_lmem(u64 addr) {
  return (addr >= LOCAL_MEM_START_ADDR &&
          addr < (LOCAL_MEM_SIZE * NPU_NUM + LOCAL_MEM_START_ADDR));
}

static inline int is_gmem(u64 addr) {
  addr &= (MAX_GMEM_SIZE - 1);
  return addr >= 0 &&
         addr < (GLOBAL_MEM_START_ADDR + CONFIG_GLOBAL_MEM_SIZE);
}

static inline int is_l2mem(u64 addr) {
  addr &= (MAX_GMEM_SIZE - 1);
  return (addr >= L2_SRAM_START_ADDR &&
          addr < (L2_SRAM_START_ADDR  + L2_SRAM_SIZE));
}

static inline int is_smem(u64 addr) {
  #ifndef __sg2262__
  return addr >= STATIC_MEM_START_ADDR &&
         addr < (STATIC_MEM_START_ADDR + STATIC_MEM_SIZE);
  #else
  // sg2262 don't support smem
  return !is_lmem(addr) && !is_gmem(addr) && !is_l2mem(addr);
  #endif
}

static inline local_addr_t tpu_npu_addr(local_addr_t addr) {
    return addr & (LOCAL_MEM_SIZE - 1);
}



