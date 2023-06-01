#ifndef MACROS_H__
#define MACROS_H__

#define __TRUE__     (1)
#define __FALSE__    (0)
#define WORD_SIZE    (32)
#define DWORD_SIZE   (64)
#define WORD_BITS    (5)
#define WORD_MASK    (0x1f)
#define LANE_SEC     ((NPU_NUM - 1) / WORD_SIZE + 1)

#define INT8_SIZE 1
#define INT16_SIZE 2
#define INT32_SIZE 4
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

#define NO_USE 0
#define UNUSED(x) (void)(x)
#define INLINE inline

#ifndef USING_CMODEL
    #ifdef __arm__
        extern jmp_buf error_stat;
        extern void fw_log(char *fmt, ...);
        #define hang(_ret) {              \
            fw_log("ASSERT %s: %s: %d: %s\n", __FILE__, __func__, __LINE__, #_cond); \
            longjmp(error_stat,1);   \
        }
    #else
        #define hang(_ret) while (1)
    #endif
#else
    #ifdef hang
    #undef hang
    #endif
    #define hang(_ret) _Exit(_ret)
#endif

#ifdef __cplusplus
extern "C" {
#endif
 int get_atomic_cmodel_assert_enable();
#ifdef __cplusplus
}
#endif

#include "print_trace.h"
#ifdef USING_BACKEND_API
#define ASSERT_INFO(_cond, fmt, ...)                                               \
  do {                                                                             \
    if (get_atomic_cmodel_assert_enable()) {                                       \
        if (!(_cond)) {                                                            \
          printf("ASSERT %s: %s: %d: %s\n", __FILE__, __func__, __LINE__, #_cond); \
          printf("ASSERT info: " fmt "\n", ##__VA_ARGS__);                         \
          _print_trace();                                                           \
          hang(-1);                                                                \
        }                                                                          \
      }                                                                            \
    } while (0)
#elif ! defined(ASSERT_INFO)
#define ASSERT_INFO(_cond, fmt, ...)                                             \
  do {                                                                           \
      if (!(_cond)) {                                                            \
        printf("ASSERT %s: %s: %d: %s\n", __FILE__, __func__, __LINE__, #_cond); \
        printf("ASSERT info: " fmt "\n", ##__VA_ARGS__);                         \
        _print_trace();                                                           \
        hang(-1);                                                                \
      }                                                                          \
    } while (0)
#endif
#define ASSERT(_cond) ASSERT_INFO(_cond, "none.")

#define ASSERT_OP(v0, v1, op) \
    ASSERT_INFO((v0) op (v1), #v0 "=%d vs " #v1 "=%d", (int)(v0), (int)(v1))

#define ASSERT_EQ(v0, v1) ASSERT_OP(v0, v1, ==)
#define ASSERT_NE(v0, v1) ASSERT_OP(v0, v1, !=)
#define ASSERT_LE(v0, v1) ASSERT_OP(v0, v1, <=)
#define ASSERT_LT(v0, v1) ASSERT_OP(v0, v1, <)
#define ASSERT_GE(v0, v1) ASSERT_OP(v0, v1, >=)
#define ASSERT_GT(v0, v1) ASSERT_OP(v0, v1, >)
#ifndef ASSERT_RANGE
#define ASSERT_RANGE(v, beg, end) \
  ASSERT_INFO((v)>=(beg) && (v)<(end), #v "=%d should in range [%d,%d)", (int)(v), (int)(beg), (int)(end))
#endif
#define ASSERT_SAME_SHAPE(shape0, dims0, shape1, dims1) \
    do {\
        ASSERT_INFO(dims0 == dims1, "dims0=%d, dims1=%d", dims0, dims1);\
        for(int i=0; i<(int)dims0; i++){\
            ASSERT_INFO(shape0[i] == shape1[i], "shape0[%d]=%d, shape1[%d]=%d", i, shape0[i], i, shape1[i]);\
        }\
    } while(0)

#endif
