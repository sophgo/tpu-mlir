#ifndef BMCPU_MACRO_H_
#define BMCPU_MACRO_H_

#include <stdio.h>
#include <vector>
#include <map>
#include <string>
#include <set>
#include <list>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

using std::vector;
using std::map;
using std::pair;
using std::make_pair;
using std::string;
using std::cout;
using std::endl;
using std::set;
using std::list;

//namespace bmcpu {

#define CPU_ASSERT(_cond)                       \
  do {                                           \
    if (!(_cond)) {                              \
      printf("ASSERT %s: %s: %d: %s\n",          \
          __FILE__, __func__, __LINE__, #_cond); \
      exit(-1);                                  \
    }                                            \
  } while(0)

#define CPU_ASSERT_PAIR(var1, var2, op)\
    do {\
        if(!((var1) op (var2))){\
            std::cout<<#var1 #op #var2<<" should be true, but "\
                <<#var1<<"="<<(var1)<<" vs " <<#var2<<"="<<(var2)<<std::endl;\
            CPU_ASSERT(0);\
        }\
    } while(0)

#define CPU_ASSERT_EQ(var1, var2)\
    CPU_ASSERT_PAIR(var1, var2, ==)

#define CPU_ASSERT_NE(var1, var2)\
    CPU_ASSERT_PAIR(var1, var2, !=)

#define CPU_ASSERT_GT(var1, var2)\
    CPU_ASSERT_PAIR(var1, var2, >)

#define CPU_ASSERT_GE(var1, var2)\
    CPU_ASSERT_PAIR(var1, var2, >=)

#define CPU_ASSERT_LT(var1, var2)\
    CPU_ASSERT_PAIR(var1, var2, <)

#define CPU_ASSERT_LE(var1, var2)\
    CPU_ASSERT_PAIR(var1, var2, <=)

#define BMCPU_DECLARE_AND_UNPACK_PARAM(param_type, param_ptr, raw_param, size) \
    param_type *param_ptr;                                                     \
    param_type param_ptr##_inst;                                               \
    param_ptr = &(param_ptr##_inst);                                           \
    memset(reinterpret_cast<void *>(param_ptr), 0, sizeof(param_type));                                  \
    memcpy(reinterpret_cast<void *>(param_ptr), raw_param, size)

#endif
