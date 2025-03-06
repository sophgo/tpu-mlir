#include "host_utils.h"
#include <cassert>
#include <string>

int get_chip() {
  static int chip = -1;
  if (chip == -1) {
    const char *chip_name = get_chip_str();
    if (chip_name == nullptr) {
      assert(0 && "CHIP is not set, please set CHIP env");
    }
    std::string chip_str(chip_name);
    if (chip_str == "bm1684x") {
      chip = PplChip::bm1684x;
    } else if (chip_str == "bm1688") {
      chip = PplChip::bm1688;
    } else if (chip_str == "bm1690") {
      chip = PplChip::bm1690;
    } else if (chip_str == "sg2262") {
      chip = PplChip::sg2262;
    } else if (chip_str == "sg2380") {
      chip = PplChip::sg2380;
    } else if (chip_str == "mars3") {
      chip = PplChip::mars3;
    } else if (chip_str == "bm1684xe") {
      chip = PplChip::bm1684xe;
    } else {
      assert(0 && "CHIP is not supported");
    }
  }
  return chip;
}

int lane_num() {
  int lane_chip[] = {64, 32, 64, 32, 32, 8, 64};
  return lane_chip[get_chip()];
}

int eu_num() {
  int eu_chip[] = {64, 16, 64, 16, 16, 16, 64};
  return eu_chip[get_chip()];
}
