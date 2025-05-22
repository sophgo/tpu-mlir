#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "ppl_jit.h"

#define CHECK_PPL_RET(ret)                                                     \
  do {                                                                         \
    switch (ret) {                                                             \
    case FileErr:                                                              \
      printf("PPL file load error, please check the path and whether the "     \
             "file exist!!!\n");                                               \
      exit(ret);                                                               \
    case LlvmFeErr:                                                            \
      printf("LLVM fe to PPL fe error, please check the pl code!!!\n");        \
      exit(ret);                                                               \
    case PplFeErr:                                                             \
      printf("PPL fe pass error, please check the error info!!!\n");           \
      exit(ret);                                                               \
    case PplOpt1Err:                                                           \
      printf("PPL opt1 pass error, please check the error info!!!\n");         \
      exit(ret);                                                               \
    case PplOpt2Err:                                                           \
      printf("PPL opt2 pass error, please check the error info!!!\n");         \
      exit(ret);                                                               \
    case PplFinalErr:                                                          \
      printf("PPL final pass error, please check the error info!!!\n");        \
      exit(ret);                                                               \
    case PplTransErr:                                                          \
      printf("PPL codegen error, please check the error info!!!\n");           \
      exit(ret);                                                               \
    case EnvErr:                                                               \
      printf("Env error, please check env path!!!\n");                         \
      exit(ret);                                                               \
    case PplShapeInferErr:                                                     \
      printf("Shape infer failed, please check tensor shape!!!\n");            \
      exit(ret);                                                               \
    case PplSetMemRefShapeErr:                                                 \
      printf("Set memRef shape failed, please check tesnor shape!!!\n");       \
      exit(ret);                                                               \
    case ToPplErr:                                                             \
      printf("LLVM fe to ppl fe error, please check the error info!!!\n");     \
      exit(ret);                                                               \
    case PplTensorConvErr:                                                     \
      printf("Tensor conversion error, please check the error info!!!\n");     \
      exit(ret);                                                               \
    case PplDynBlockErr:                                                       \
      printf("Dynamic block error, please check the error info!!!\n");         \
      exit(ret);                                                               \
    case CacheOpenKernelSoErr:                                                 \
      printf("Can not open the kernel so from cache, please check cache "      \
             "path!!!\n");                                                     \
      exit(ret);                                                               \
    case CacheGetKernelFunErr:                                                 \
      printf(                                                                  \
          "Can not get the kernel from cache, please check cache path!!!\n");  \
      exit(ret);                                                               \
    case PplJitSetChipErr:                                                     \
      printf("Check chip is set or is valild. 'echo ${CHIP}'!!!\n");           \
      exit(ret);                                                               \
    case LocalMemSetErr:                                                       \
      printf("Can not partially set local_mem, please check in pl file!!!\n"); \
      exit(ret);                                                               \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
  } while (0)
