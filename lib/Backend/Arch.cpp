//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Backend/BM168x/BM1686.h"
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Backend/BM168x/BM1684.h"
#include "tpu_mlir/Backend/BM168x/BM1684X.h"
#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Backend/Arch.h"
#include "tpu_mlir/Interfaces/LocalGenInterface.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

using namespace tpu_mlir::backend;

int64_t Arch::NPU_NUM = 0;
int64_t Arch::EU_BYTES = 0;
int64_t Arch::LMEM_BYTES = 0;
int64_t Arch::LMEM_BANKS = 0;
int64_t Arch::LMEM_BANK_BYTES = 0;
bool Arch::ALIGN_4N = false;
llvm::StringRef Arch::LIB_NAME = "";
llvm::StringRef Arch::chip = "";

Arch *Arch::inst = nullptr;

void Arch::init() {
  if (inst != nullptr) {
    return;
  }
  chip = module::getChip();
  if (chip == module::Chip::BM1684) {
    inst = &BM1684::instance();
  } else if (chip == module::Chip::BM1684X) {
    inst = &BM1684X::instance();
  } else if (chip == module::Chip::BM1686) {
    inst = &BM1686::instance();
  } else if (module::isCV18xx()) {
    inst = &CV18xx::instance(chip);
  } else if (chip == module::Chip::ALL) {
    // do nothing
    return;
  } else {
    llvm_unreachable("unsupport chip");
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
  auto type = v.getType().cast<RankedTensorType>();
  std::vector<int64_t> shape = module::getShape(v);
  auto stype = module::getStorageType(v);
  int elm_bytes = stype.getIntOrFloatBitWidth() / 8;
  assert(elm_bytes);
  shape[0] = ceiling_func(shape[0], (int64_t)4 / elm_bytes);
  return 4 * std::accumulate(shape.begin(), shape.end(), 1,
                             std::multiplies<int64_t>());
}

int64_t Arch::get_tensor_lmem_bytes(Value v, int64_t slice_n, int64_t slice_h,
                                    bool eu_align) {
  int64_t n, c, h, w;
  module::getNCHW(v, n, c, h, w);
  if (slice_n > 0) {
    n = slice_n;
  }
  if (slice_h > 0) {
    h = slice_h;
  }
  auto type = module::getStorageType(v);
  int type_bits = type.getIntOrFloatBitWidth();
  double dbytes = (double)type_bits / 8;
  if (ALIGN_4N) {
    int64_t eu_num = Arch::eu_num(4);
    int64_t c_per_npu = ceiling_func(c, Arch::NPU_NUM);
    int64_t n_aligned = ceiling_func(n, 4 / (int64_t)dbytes);
    int64_t eu_aligned = eu_align ? align_up(h * w, eu_num) : (h * w);
    return n_aligned * c_per_npu * eu_aligned * 4;
  } else {
    int64_t eu_num = Arch::eu_num(dbytes);
    int64_t c_per_npu = ceiling_func(c, Arch::NPU_NUM);
    int64_t eu_aligned = eu_align ? align_up(h * w, eu_num) : (h * w);
    if(type_bits == 4) {
      return align_up((int64_t)(n * c_per_npu * eu_aligned) , (int64_t)2) * dbytes;
    }
    return (int64_t)n * c_per_npu * eu_aligned * dbytes;
  }
}

int64_t Arch::get_weight_lmem_bytes(Value v, bool eu_align) {
  int64_t n, c, h, w;
  module::getNCHW(v, n, c, h, w);
  auto type = module::getStorageType(v);
  int type_bits = type.getIntOrFloatBitWidth();
  double dbytes = (double)type_bits / 8;
  int64_t eu_num = Arch::eu_num(dbytes);
  int64_t c_per_npu = ceiling_func(c, Arch::NPU_NUM);
  int64_t eu_aligned = eu_align ? align_up(h * w, eu_num) : (h * w);
  if(type_bits == 4)
    return align_up((int64_t)(n * c_per_npu * eu_aligned) , (int64_t)2) * dbytes;
  return (int64_t)n * c_per_npu * eu_aligned * dbytes;
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
