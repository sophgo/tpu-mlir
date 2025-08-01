#ifndef __HOST_DEF_H__
#define __HOST_DEF_H__

#include <stdlib.h>

#ifndef gaddr_t
#define gaddr_t unsigned long long
#endif

inline const char* get_chip_str() {
  return getenv("CHIP");;
}

inline int get_core_num() {
  return atoi(getenv("CORE_NUM"));
}

enum PplChip {
  bm1684x = 0,
  bm1688 = 1,
  bm1690 = 2,
  sg2262 = 3,
  sg2380 = 4,
  mars3 = 5,
  bm1684xe = 6,
  sg2262rv = 7,
};

enum PplErrorCode_t {
  PplLocalAddrAssignErr = 0x11,
  FileErr = 0x12,
  LlvmFeErr = 0x13,
  PplFeErr = 0x14,
  PplOpt1Err = 0x15,
  PplOpt2Err = 0x16,
  PplFinalErr = 0x17,
  PplTransErr = 0x18,
  EnvErr = 0x19,
  PplL2AddrAssignErr = 0x1A,
  PplShapeInferErr = 0x1B,
  PplSetMemRefShapeErr = 0x1C,
  ToPplErr = 0x1D,
  PplTensorConvErr = 0x1E,
  PplDynBlockErr = 0x1F,
  CacheOpenKernelSoErr = 0x20,
  CacheGetKernelFunErr = 0x21,
  PplJitSetChipErr = 0x22,
  LocalMemSetErr = 0x23,
};
#endif
