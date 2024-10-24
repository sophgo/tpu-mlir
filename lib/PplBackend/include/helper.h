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
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
  } while (0)

template <typename T>
static inline T abs_ceiling_func(T numerator, T denominator) {
  return (std::abs(numerator + denominator) - 1) / std::abs(denominator);
}

template <typename U, typename V>
static inline auto
ceiling_func(U numerator, V denominator) -> decltype(numerator + denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename U, typename V>
static inline auto align_up(U x, V a) -> decltype(x + a) {
  return ceiling_func(x, a) * a;
}
