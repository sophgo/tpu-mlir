#ifndef _COMMON_DEF_H_
#define _COMMON_DEF_H_

#define FW_MAX_SHAPE_DIMS 8
typedef enum {
    SG_FW_SUCCESS             = 0,
    SG_FW_ERR_AGAIN           = 1001,   /* Not ready yet */
    SG_FW_ERR_FAILURE         = 1002,   /* General failure */
    SG_FW_ERR_TIMEOUT         = 1003,   /* Timeout */
    SG_FW_ERR_PARAM           = 1004,   /* Parameters invalid */
    SG_FW_ERR_NOMEM           = 1005,   /* Not enough memory */
    SG_FW_ERR_DATA            = 1006,   /* Data error */
    SG_FW_ERR_BUSY            = 1007,   /* Busy */
    SG_FW_ERR_NOFEATURE       = 1008,    /* Not supported yet */
    SG_FW_ERR_ASSERT          = 1009    /* ASSERT */
} sg_fw_status_t;

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

static inline const char* sg_round_name(int t){
    #define CASE(t) case SG_##t: return #t;
    switch(t){
        CASE(ROUND_INF)
        CASE(ROUND_UP)
        CASE(ROUND_DOWN)
        CASE(ROUND_EVEN)
        CASE(ROUND_ODD)
        CASE(ROUND_ZERO)
        CASE(TRIM_ZERO)
        CASE(TRIM_INF)
        CASE(TRIM_UP)
        CASE(TRIM_DOWN)
    }
    #undef CASE
    return "UNKNOWN";
}

typedef enum {
  SG_REDUCE_MEAN = 0,
  SG_REDUCE_SUM  = 1,
  SG_REDUCE_MAX  = 2,
  SG_REDUCE_MIN  = 3,
  SG_REDUCE_PROD = 4,
  SG_REDUCE_L2   = 5,
  SG_REDUCE_L1   = 6,
} sg_reduce_method_t;

static inline const char* sg_reduce_name(int t){
    #define CASE(t) case SG_REDUCE_##t: return #t;
    switch(t){
        CASE(MEAN)
        CASE(SUM)
        CASE(MAX)
        CASE(MIN)
        CASE(PROD)
        CASE(L2)
        CASE(L1)
    }
    #undef CASE
    return "UNKNOWN";
}

typedef enum {
  BINARY_ADD          = 0,
  BINARY_SUB          = 1,
  BINARY_MUL          = 2,
  BINARY_DIV          = 3,
  BINARY_MAX          = 4,
  BINARY_MIN          = 10000,
  BINARY_GT           = 10001,
  BINARY_GE           = 10002,
  BINARY_LT           = 10003,
  BINARY_LE           = 10004,
  BINARY_EQ           = 10005,
  BINARY_NE           = 10006,
  BINARY_SQUARED_DIFF = 10007,
  BINARY_FLOOR_MOD    = 10008,
  BINARY_FLOOR_DIV    = 10009
} sg_binary_type_t;

static inline const char* sg_binary_name(int t){
    #define CASE(t) case BINARY_##t: return #t;
    switch(t){
        CASE(ADD)
        CASE(SUB)
        CASE(MUL)
        CASE(DIV)
        CASE(MAX)
        CASE(MIN)
        CASE(GT)
        CASE(GE)
        CASE(LT)
        CASE(LE)
        CASE(EQ)
        CASE(NE)
        CASE(SQUARED_DIFF)
        CASE(FLOOR_MOD)
        CASE(FLOOR_DIV)
    }
    #undef CASE
    return "UNKNOWN";
}

typedef enum {
  ACTIVE_TANH      = 0,
  ACTIVE_SIGMOID   = 1,
  ACTIVE_RELU      = 2,
  ACTIVE_EXP       = 3,
  ACTIVE_ELU       = 4,
  ACTIVE_SQRT      = 5,
  ACTIVE_SQUARE    = 6,
  ACTIVE_RSQRT     = 7,
  ACTIVE_ABSVAL    = 8,
  ACTIVE_LN        = 9,
  ACTIVE_ROUND     = 10,
  ACTIVE_CEIL      = 11,
  ACTIVE_FLOOR     = 12,
  ACTIVE_SIN       = 13,
  ACTIVE_COS       = 14,
  ACTIVE_IS_FINITE = 15,
  ACTIVE_MISH      = 16,
  ACTIVE_SWISH     = 17,
  ACTIVE_HSWISH    = 18,
  ACTIVE_SILU      = 19,
  ACTIVE_ARCSIN    = 20,
  ACTIVE_ARCCOS    = 21,
  ACTIVE_ARCSINH   = 22,
  ACTIVE_ARCCOSH   = 23,
  ACTIVE_ARCTANH   = 24,
  ACTIVE_SINH      = 25,
  ACTIVE_COSH      = 26,
  ACTIVE_TAN       = 27,
  ACTIVE_SIGN      = 28,
  ACTIVE_GELU      = 29,
  ACTIVE_ERF       = 30,
  ACTIVE_HSIGMOID  = 31,
  ACTIVE_LOG_SIGMOID = 32,
  ACTIVE_SOFT_PLUS = 33,
  ACTIVE_SOFT_SIGN = 34,
} sg_active_type_t;
static inline const char* sg_active_name(int t){
    #define CASE(t) case ACTIVE_##t: return #t;
    switch(t){
        CASE(TANH)
        CASE(SIGMOID)
        CASE(RELU)
        CASE(EXP)
        CASE(ELU)
        CASE(SQRT)
        CASE(SQUARE)
        CASE(RSQRT)
        CASE(ABSVAL)
        CASE(LN)
        CASE(ROUND)
        CASE(CEIL)
        CASE(FLOOR)
        CASE(SIN)
        CASE(COS)
        CASE(IS_FINITE)
        CASE(MISH)
        CASE(SWISH)
        CASE(HSWISH)
        CASE(SILU)
        CASE(ARCSIN)
        CASE(ARCCOS)
        CASE(ARCSINH)
        CASE(ARCCOSH)
        CASE(ARCTANH)
        CASE(SINH)
        CASE(COSH)
        CASE(TAN)
        CASE(SIGN)
        CASE(GELU)
        CASE(ERF)
        CASE(HSIGMOID)
        CASE(LOG_SIGMOID)
        CASE(SOFT_PLUS)
        CASE(SOFT_SIGN)
    }
    #undef CASE
    return "UNKNOWN";
}

// Channel shift macro(left,right,circle left,circle right)
typedef enum {
  SHIFT_L  = 0,
  SHIFT_R  = 1,
  SHIFT_CL = 2,
  SHIFT_CR = 3
} sg_shift_type_t;

// BATCH_FIRST: x = [batch, seq_len, input_size], y = [batch, seq_len, num_dir, hidden_size]
// BATCH_TORCH: x = [seq_len, batch, input_size], y = [seq_len, batch, num_dir, hidden_size]
// BATCH_ONNX:  x = [seq_len, batch, input_size], y = [seq_len, num_dir, batch, hidden_size]
typedef enum {
   BATCH_TORCH = 0,
   BATCH_FIRST = 1,
   BATCH_ONNX = 2
} sg_rnn_batch_t;

// The data type number is the same with bmcompiler
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


#define IS_INT4(t) (((t) == SG_DTYPE_UINT4) || ((t) == SG_DTYPE_INT4))
#define IS_INT8(t) (((t) == SG_DTYPE_UINT8) || ((t) == SG_DTYPE_INT8))
#define IS_INT16(t) (((t) == SG_DTYPE_UINT16) || ((t) == SG_DTYPE_INT16))
#define IS_INT32(t) (((t) == SG_DTYPE_UINT32) || ((t) == SG_DTYPE_INT32))
#define IS_FP16(t) ((t) == SG_DTYPE_FP16 || (t) == SG_DTYPE_BFP16)
#define IS_FP(t) (((t) == SG_DTYPE_FP32) || ((t) == SG_DTYPE_FP16) || ((t) == SG_DTYPE_BFP16))
#define IS_SIGN(t) (((t) == SG_DTYPE_INT4) || ((t) == SG_DTYPE_INT8) || ((t) == SG_DTYPE_INT16) || ((t) == SG_DTYPE_INT32))

static inline unsigned sg_dtype_len(sg_data_type_t t)
{
    switch (t)
    {
        case SG_DTYPE_FP32:
        case SG_DTYPE_UINT32:
        case SG_DTYPE_INT32:
            return 4;
        case SG_DTYPE_UINT8:
        case SG_DTYPE_INT8:
        case SG_DTYPE_UINT4:
        case SG_DTYPE_INT4:
            return 1;
        case SG_DTYPE_FP16:
        case SG_DTYPE_BFP16:
        case SG_DTYPE_UINT16:
        case SG_DTYPE_INT16:
            return 2;
        default:
            return 0;
    }
}
static inline const char* sg_dtype_name(int t){
    #define CASE(t) case SG_DTYPE_##t: return #t;
    switch(t){
        CASE(FP32)
        CASE(FP16)
        CASE(BFP16)
        CASE(INT32)
        CASE(UINT32)
        CASE(INT16)
        CASE(UINT16)
        CASE(INT8)
        CASE(UINT8)
        CASE(INT4)
        CASE(UINT4)
        CASE(UNKNOWN)
    }
    #undef CASE
    return "";
}

static inline long long sg_dtype_fixed_max(sg_data_type_t t)
{
    long long ret = 0;
    switch (t)
    {
        case SG_DTYPE_UINT32:
            ret = (1ll << 32) - 1;
            break;
        case SG_DTYPE_INT32:
            ret = (1ll << 31) - 1;
            break;
        case SG_DTYPE_UINT8:
            ret = 255;
            break;
        case SG_DTYPE_INT8:
            ret = 127;
            break;
        case SG_DTYPE_UINT16:
            ret = 65535;
            break;
        case SG_DTYPE_INT16:
            ret = 32767;
            break;
        case SG_DTYPE_INT4:
             ret = 7;
             break;
        case SG_DTYPE_UINT4:
             ret = 15;
             break;
        default:
            return 0;
    }
    return ret;
}

static inline long long sg_dtype_fixed_min(sg_data_type_t t)
{
    long long ret = 0;
    switch (t)
    {
        case SG_DTYPE_UINT4:
        case SG_DTYPE_UINT8:
        case SG_DTYPE_UINT16:
        case SG_DTYPE_UINT32:
            ret = 0;
            break;
        case SG_DTYPE_INT32:
            ret = -(1ll << 31);
            break;
        case SG_DTYPE_INT8:
            ret = -128;
            break;
        case SG_DTYPE_INT16:
            ret = -32768;
            break;
        case SG_DTYPE_INT4:
             ret = -8;
             break;
        default:
            return 0;
    }
    return ret;
}

// Keep in sync with bmcompiler_net_param.h
typedef enum {
  /* 3D group if this group has CONV3D/DECONV3D/POOL3D
   * for 1684 float32, data in local memory storage as {d * n, c, h, w}
   * for 1684 int8, data in local memory storage as {n, d * c, h, w}
   * for 1684X, data in local memory storage as {d * n, c, h, w}
   * data in global memory always storage as {n, c, d, h, w}
   * group_type < 8, because 1684 dynamic compile reserved `3bit` for group_type
   */
  FW_GROUP_NORMAL = 0,
  GROUP_3D = 1,
} group_type_t;

#define UNUSED(x) (void)(x)

typedef enum {
  CAFFE_SUPPORT = 0,
  TENSORFLOW_SUPPORT = 1,
  CAFFE_NEAREST = 2,
  TENSORFLOW_NEAREST = 3,
  PYTORCH_SUPPORT = 4,
  PYTORCH_NEAREST = 5,
  OPENCV_BILINEAR = 6,
  ONNX_NEAREST = 7,
} PLATFORM_SUPPORT;

typedef enum {
    GridSampleNearest = 0,
    GridSampleBilinear = 1,
} GridSampleInterpMode;

typedef enum {
    GridSampleZeros = 0,
    GridSampleBorder = 1,
    GridSampleReflection = 2,
} GridSamplePaddingMode;

#define MIN_PROPOSAL_NUM (1)
#define MAX_PROPOSAL_NUM (40000)//(65536)
#define MAX_SOFT_SUPPORT_PROPOSAL_NUM (22500)
#define ALL_MASK_IN_L2_MAX_SIZE (1400)
#define ALL_MASK_IN_L2_SOFT_NMS_MAX_SIZE (350)
typedef enum sgdnn_nms_alg_ {
    HARD_NMS = 0,
    SOFT_NMS,
    ADAPTIVE_NMS,
    SSD_NMS,
    MAX_NMS_TYPE
} sgdnn_nms_alg_e;
typedef enum {
    LINEAR_WEIGHTING = 0,
    GAUSSIAN_WEIGHTING,
    MAX_WEIGHTING_TYPE
} sgdnn_weighting_method_e;
typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
}__attribute__((packed)) face_rect_t;

typedef struct nms_proposal {
    int          size;
    face_rect_t  face_rect[MAX_PROPOSAL_NUM];
    int          capacity;
    face_rect_t *begin;
    face_rect_t *end;
} __attribute__((packed)) nms_proposal_t;
#endif // _COMMON_DEF_H_

#define SWAP_VAL(a, b) \
  a ^= b;              \
  b ^= a;              \
  a ^= b
