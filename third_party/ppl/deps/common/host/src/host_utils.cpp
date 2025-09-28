#include "host_utils.h"
#include <cassert>
#include <string>

int get_chip() {
  static int chip = -1;
  if (chip == -1) {
    std::string chip_code = get_chip_code();
    if (chip_code == "tpu_6_0") {
      chip = PplChip::tpu_6_0;
    } else if (chip_code == "tpul_6_0") {
      chip = PplChip::tpul_6_0;
    } else if (chip_code == "tpub_7_1") {
      chip = PplChip::tpub_7_1;
    } else if (chip_code == "tpub_9_0") {
      chip = PplChip::tpub_9_0;
    } else if (chip_code == "tpub_9_0_rv") {
      chip = PplChip::tpub_9_0_rv;
    } else if (chip_code == "tpul_8_0") {
      chip = PplChip::tpul_8_0;
    } else if (chip_code == "tpul_8_1") {
      chip = PplChip::tpul_8_1;
    } else if (chip_code == "tpu_6_0_e") {
      chip = PplChip::tpu_6_0_e;
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
