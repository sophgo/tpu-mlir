//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/BM168x/Athena2.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Backend/BM168x/BM1684x.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/Arch.h"
#include "tpu_mlir/Interfaces/LocalGenInterface.h"
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

using namespace tpu_mlir;
using namespace tpu_mlir::backend;
using namespace tpu_mlir::helper;

int64_t Arch::NPU_NUM = 0;
int64_t Arch::EU_BYTES = 0;
int64_t Arch::LMEM_BYTES = 0;
int64_t Arch::LMEM_BANKS = 0;
int64_t Arch::LMEM_BANK_BYTES = 0;
bool Arch::ALIGN_4N = false;
llvm::StringRef Arch::LIB_NAME = "";

Arch *Arch::inst = nullptr;

void Arch::init(const llvm::StringRef chip) {
  if (inst != nullptr) {
    return;
  }
  if (chip == Module::Chip::BM1684) {
    inst = &BM1684::instance();
  } else if (chip == Module::Chip::BM1684x) {
    inst = &BM1684x::instance();
  } else if (chip == Module::Chip::ATHENA2) {
    inst = &Athena2::instance();
  } else if (Module::isCV18xx(chip)) {
    inst = &CV18xx::instance(chip);
  } else {
    llvm_unreachable("unsupport chip");
  }
  inst->chip = chip;
}

int64_t Arch::get_lmem_bytes(int64_t n, int64_t c, int64_t h, int64_t w,
                             mlir::Type type, bool eu_align) {
  int64_t npu_num = Arch::NPU_NUM;
  int64_t dbytes = type.getIntOrFloatBitWidth() / 8;
  int64_t eu_num = Arch::eu_num(dbytes);
  int64_t c_per_npu = ceiling_func(c, npu_num);
  int64_t n_align = get_n_align(dbytes);
  int64_t n_aligned = align_up(n, n_align);
  int64_t eu_aligned =
      eu_align ? align_up(h * w, eu_num) * dbytes : (h * w * dbytes);
  return n_aligned * c_per_npu * eu_aligned;
}

int64_t Arch::get_tensor_lmem_bytes(mlir::Value v, int64_t slice_n,
                                    int64_t slice_h, bool eu_align) {
  int64_t n, c, h, w;
  Module::getNCHW(v, n, c, h, w);
  auto type = Module::getStorageType(v);
  return get_lmem_bytes(slice_n, c, slice_h, w, type, eu_align);
}

int64_t Arch::get_weight_lmem_bytes(mlir::Value v, bool eu_align) {
  int64_t n, c, h, w;
  Module::getNCHW(v, n, c, h, w);
  auto type = Module::getStorageType(v);
  return get_lmem_bytes(n, c, h, w, type, eu_align);
}

Arch::~Arch() {}

void Arch::load_library() {
  if (!DL.isValid()) {
    std::string Err;
    DL = llvm::sys::DynamicLibrary::getPermanentLibrary(LIB_NAME.data(), &Err);
    if (DL.isValid() == false) {
      llvm_unreachable(Err.c_str());
    }
  }
}
