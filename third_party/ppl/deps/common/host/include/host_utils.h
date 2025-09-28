#ifndef __HOST_UTILS_H__
#define __HOST_UTILS_H__
#include "host_def.h"
#include <algorithm>

template <typename T>
static inline T abs_ceiling_func(T numerator, T denominator) {
  return (std::abs(numerator + denominator) - 1) / std::abs(denominator);
}

template <typename U, typename V>
static inline auto ceiling_func(U numerator, V denominator)
    -> decltype(numerator + denominator) {
  return (numerator + denominator - 1) / denominator;
}

template <typename U, typename V>
static inline auto align_up(U x, V a) -> decltype(x + a) {
  return ceiling_func(x, a) * a;
}
int lane_num();
int eu_num();
int get_chip();

#define LANE_NUM lane_num()
#define EU_NUM eu_num()
#endif
