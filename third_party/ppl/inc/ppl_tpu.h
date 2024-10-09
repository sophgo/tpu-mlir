#pragma once

#define LANE_NUM ppl::lane_num()
#define NPU_NUM LANE_NUM
#define EU_BYTES ppl::eu_bytes()

namespace ppl {
int lane_num();
int eu_bytes();
template <typename DataType> int get_eu_num() {
  if constexpr (std::is_same_v<DataType, int4>) {
    return 2 * EU_BYTES;
  } else {
    return EU_BYTES / sizeof(DataType);
  }
}

template <typename DataType> int get_nic() {
  if constexpr (std::is_same_v<DataType, fp32>) {
    return 1;
  } else if constexpr (std::is_same_v<DataType, int4> ||
                       std::is_same_v<DataType, uint4>) {
    return LANE_NUM * 2;
  } else {
    return LANE_NUM / sizeof(DataType);
  }
}

} // namespace ppl
