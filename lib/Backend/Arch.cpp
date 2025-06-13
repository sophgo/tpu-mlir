//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Backend/BM168x/BM1688.h"
#include "tpu_mlir/Backend/BM168x/BM1690.h"
#include "tpu_mlir/Backend/BM168x/MARS3.h"
#include "tpu_mlir/Backend/BM168x/SG2380.h"
#include "tpu_mlir/Backend/BM168x/SGTPUV8.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Support/MathUtils.h"
#include <filesystem>

namespace fs = std::filesystem;
using namespace tpu_mlir::backend;

int64_t Arch::NPU_NUM = 0;
int64_t Arch::EU_BYTES = 0;
int64_t Arch::LMEM_BYTES = 0;
int64_t Arch::LMEM_BANKS = 0;
int64_t Arch::LMEM_BANK_BYTES = 0;
bool Arch::ALIGN_4N = false;
llvm::StringRef Arch::LIB_BACKEND_NAME = "";
module::Chip Arch::chip;
uint64_t Arch::FREQ = 0;
Arch *Arch::inst = nullptr;

void Arch::init(uint64_t freq) {
  if (inst != nullptr) {
    return;
  }

  chip = module::getChip();
  if (chip == module::Chip::ALL) {
    // do nothing
    return;
  } else {
    Arch::FREQ = freq;
    if (chip == module::Chip::BM1684) {
      inst = &BM1684::instance();
    } else if (chip == module::Chip::BM1684X) {
      inst = &BM1684X::instance();
    } else if (chip == module::Chip::BM1688) {
      inst = &BM1688::instance(A2_1::value);
    } else if (chip == module::Chip::CV186X) {
      inst = &BM1688::instance(A2_2::value);
    } else if (module::isCV18xx()) {
      inst = &CV18xx::instance(chip);
    } else if (chip == module::Chip::BM1690) {
      inst = &BM1690::instance();
    } else if (chip == module::Chip::MARS3) {
      inst = &MARS3::instance(A2_1::value);
    } else if (chip == module::Chip::SGTPUV8) {
      inst = &SGTPUV8::instance(A2_1::value);
    } else if (chip == module::Chip::SG2380) {
      inst = &SG2380::instance();
    } else if (chip == module::Chip::SG2262) {
      inst = &BM1690::instance();
    } else {
      llvm_unreachable("unsupport chip\n");
    }
    // for ppl
    std::string chip_str;
    switch (chip) {
    case module::Chip::BM1684X:
      chip_str = PPL_BM1684X;
      break;
    case module::Chip::BM1688:
    case module::Chip::CV186X:
      chip_str = PPL_BM1688;
      break;
    case module::Chip::BM1690:
      chip_str = PPL_BM1690;
      break;
    case module::Chip::MARS3:
      chip_str = PPL_MARS3;
      break;
    default:
      // llvm::errs() << "ppl unsupport this chip\n";
      break;
    }
    setenv("CHIP", chip_str.c_str(), 1);
    setenv("CORE_NUM", std::to_string(module::getCoreNum()).c_str(), 1);
  }
}

int64_t Arch::eu_num(double dbytes) { return EU_BYTES / dbytes; }

size_t Arch::get_gmem_bytes(Value v) {
  if (!Arch::ALIGN_4N) {
    return module::getBytes(v);
  }
  if (v.getType().isa<NoneType>()) {
    return 0;
  }
  if (module::isWeight(v)) {
    return module::getBytes(v);
  }
  std::vector<int64_t> shape = module::getShape(v);
  auto stype = module::getStorageType(v);
  int elm_bytes = stype.getIntOrFloatBitWidth() / 8;
  assert(elm_bytes);
  shape[0] = ceiling_func(shape[0], (int64_t)4 / elm_bytes);
  return 4 * std::accumulate(shape.begin(), shape.end(), 1,
                             std::multiplies<int64_t>());
}

int64_t Arch::get_tensor_lmem_bytes(Value v, int64_t n, int64_t c, int64_t d,
                                    int64_t h, int64_t w, bool eu_align) {
  auto type = module::getStorageType(v);
  int type_bits = type.getIntOrFloatBitWidth();
  double dbytes = (double)type_bits / 8;
  if (ALIGN_4N) {
    int64_t eu_num = Arch::eu_num(4);
    int64_t c_per_npu = ceiling_func(c, Arch::NPU_NUM);
    int64_t n_aligned = ceiling_func(n, 4 / (int64_t)dbytes);
    int64_t eu_aligned = eu_align ? align_up(h * w, eu_num) : (h * w);
    return n_aligned * d * c_per_npu * eu_aligned * 4;
  } else {
    int64_t eu_num = Arch::eu_num(dbytes);
    int64_t c_per_npu = ceiling_func(c, Arch::NPU_NUM);
    int64_t eu_aligned = eu_align ? align_up(h * w, eu_num) : (h * w);
    if (type_bits == 4) {
      return align_up((int64_t)(n * d * c_per_npu * eu_aligned), (int64_t)2) *
             dbytes;
    }
    return (int64_t)n * d * c_per_npu * eu_aligned * dbytes;
  }
}

int64_t Arch::get_tensor_lmem_bytes(Value v, int64_t slice_n, int64_t slice_c,
                                    int64_t slice_h, int64_t slice_d,
                                    int64_t slice_w, group_type_t group_type,
                                    bool eu_align) {
  int64_t n, c, d, h, w;
  module::getNCDHW(v, n, c, d, h, w, group_type);
  if (slice_n > 0) {
    n = slice_n;
  }
  if (slice_c > 0) {
    c = slice_c;
  }
  if (slice_h > 0) {
    h = slice_h;
  }
  if (slice_d > 0) {
    d = slice_d;
  }
  if (slice_w > 0) {
    w = slice_w;
  }
  return get_tensor_lmem_bytes(v, n, c, d, h, w, eu_align);
}

int64_t Arch::get_weight_lmem_bytes(Value v, group_type_t group_type,
                                    bool eu_align) {
  int64_t n, c, d, h, w;
  module::getNCDHW(v, n, c, d, h, w, group_type);
  auto type = module::getStorageType(v);
  int type_bits = type.getIntOrFloatBitWidth();
  double dbytes = (double)type_bits / 8;
  int64_t eu_num = Arch::eu_num(dbytes);
  int64_t c_per_npu = ceiling_func(c, Arch::NPU_NUM);
  int64_t eu_aligned = eu_align ? align_up(h * w, eu_num) : (h * w);
  if (type_bits == 4)
    return align_up((int64_t)(n * d * c_per_npu * eu_aligned), (int64_t)2) *
           dbytes;
  return (int64_t)n * d * c_per_npu * eu_aligned * dbytes;
}

Arch::~Arch() {}

void Arch::load_library() {
  std::string Err;
  if (!DL.isValid()) {
    DL = llvm::sys::DynamicLibrary::getPermanentLibrary(LIB_BACKEND_NAME.data(),
                                                        &Err);
    if (DL.isValid() == false) {
      llvm_unreachable(Err.c_str());
    }
  }
}

void Arch::load_ppl() {
  if (!PPL_DL.isValid()) {
    std::string Err;
    std::string ppl_so_name = "libppl_host.so";
    PPL_DL = llvm::sys::DynamicLibrary::getPermanentLibrary(ppl_so_name.c_str(),
                                                            &Err);
    if (PPL_DL.isValid() == false) {
      llvm_unreachable(Err.c_str());
    }
  }
}
