#ifndef __HOST_DEF_H__
#define __HOST_DEF_H__

#include <stdlib.h>
#include <cassert>
#include <chip_map.h>

#ifndef gaddr_t
#define gaddr_t unsigned long long
#endif

inline const std::string get_chip_code() {
  auto chip_name = getenv("CHIP");
  if (chip_name == nullptr) {
    assert(0 && "CHIP is not set, please set CHIP env");
  }
  if (CHIP_MAP.count(chip_name)) {
    return CHIP_MAP[chip_name];
  }
  for (const auto& pair : CHIP_MAP) {
    if (chip_name == pair.second)
      return pair.second;
  }
  return "";
}

inline int get_core_num() {
  return atoi(getenv("CORE_NUM"));
}

enum PplChip {
  tpu_6_0 = 0,
  tpul_6_0 = 1,
  tpub_7_1 = 2,
  tpub_9_0 = 3,
  tpul_8_0 = 4,
  tpul_8_1 = 5,
  tpu_6_0_e = 6,
  tpub_9_0_rv = 7,
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
