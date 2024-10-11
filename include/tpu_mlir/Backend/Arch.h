//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Builders.h"
#include "tpu_mlir/Support/Module.h"
#include "llvm/Support/DynamicLibrary.h"
#include <assert.h>
#include <cmath>
#include <cstring>
#include <vector>

namespace tpu_mlir {
namespace backend {

using A2_1 = std::integral_constant<int, 900>;
using A2_2 = std::integral_constant<int, 900>;

class Arch {
public:
  static void init(uint64_t freq);
  static int64_t NPU_NUM;
  static int64_t EU_BYTES;
  static int64_t LMEM_BYTES;
  static int64_t LMEM_BANKS;
  static int64_t LMEM_BANK_BYTES;
  static llvm::StringRef LIB_BACKEND_NAME;
  static bool ALIGN_4N;
  static module::Chip chip;
  static uint64_t FREQ;
  static uint64_t get_frequance() {return Arch::FREQ;}
  // dbytes is 0.5 for INT4
  static int64_t eu_num(double dbytes);
  static int64_t get_n_align(int64_t dtype_bytes) {
    return ALIGN_4N ? (4 / dtype_bytes) : 1;
  }
  static size_t get_gmem_bytes(Value v);
  static int64_t get_tensor_lmem_bytes(Value v, int64_t n, int64_t c, int64_t d, int64_t h,
                                       int64_t w, bool eu_align = true);
  static int64_t get_tensor_lmem_bytes(Value v, int64_t slice_n, int64_t slice_c, int64_t slice_d, int64_t slice_h,
                                       int64_t slice_w, group_type_t group_type, bool eu_align = true);
  static int64_t get_weight_lmem_bytes(Value v, group_type_t group_type, bool eu_align = true);

  template <typename FPtrTy> FPtrTy CastToFPtr(const char *symbolName) {
    assert(DL.isValid());
    auto fPtr = DL.getAddressOfSymbol(symbolName);
    if (fPtr == nullptr) {
      llvm::errs() << "can't find symbol: " << symbolName << "\n";
      llvm_unreachable(symbolName);
    }
    return reinterpret_cast<FPtrTy>(fPtr);
  }

  template <typename FPtrTy> FPtrTy PplCastToFPtr(const char *symbolName) {
    if (!PPL_DL.isValid()) {
      load_ppl();
    }
    if(!PPL_DL.isValid()) {
      llvm::errs() << "ppl can't be loaded!!\n";
      llvm_unreachable(symbolName);
    }
    auto fPtr = PPL_DL.getAddressOfSymbol(symbolName);
    if (fPtr == nullptr) {
      llvm::errs() << "can't find symbol: " << symbolName << "\n";
      llvm_unreachable(symbolName);
    }
    return reinterpret_cast<FPtrTy>(fPtr);
  }

  // the cast function only for dq custom op
  template <typename FPtrTy> FPtrTy CastToDQFPtr(const char *libName, const char *symbolName) {
    std::string Err;
    auto custom_dl = llvm::sys::DynamicLibrary::getPermanentLibrary(libName, &Err);
    if (!custom_dl.isValid()) {
      llvm::errs() << "[FATAL] lib name: " << libName << " is not valid\n";
      llvm_unreachable(libName);
    }
    auto fPtr = custom_dl.getAddressOfSymbol(symbolName);
    if (fPtr == nullptr) {
      llvm::errs() << "[FATAL] can't find symbol: " << symbolName << "\n";
      llvm_unreachable(symbolName);
    }
    return reinterpret_cast<FPtrTy>(fPtr);
  }

  // the cast function only for custom op
  template <typename FPtrTy> FPtrTy CastToCustomFPtr(const char *symbolName, bool fatal=true) {
    llvm::StringRef custom_lib_name = "libbackend_custom.so";
    std::string Err;
    auto custom_dl = llvm::sys::DynamicLibrary::getPermanentLibrary(custom_lib_name.data(), &Err);
    if (!custom_dl.isValid()) {
      llvm::errs() << "[FATAL] lib name: " << custom_lib_name.str() << " is not valid\n";
      llvm_unreachable("");
    }
    auto fPtr = custom_dl.getAddressOfSymbol(symbolName);
    if (fPtr == nullptr && fatal) {
      llvm::errs() << "[FATAL] can't find symbol: " << symbolName << "\n";
      llvm_unreachable(symbolName);
    }
    return reinterpret_cast<FPtrTy>(fPtr);
  }

  // the cast function only for custom op
  template <typename FPtrTy> FPtrTy CastToCustomPluginPtr(const char *symbolName) {
    llvm::StringRef custom_lib_name = "libplugin_custom.so";
    std::string Err;
    auto custom_dl = llvm::sys::DynamicLibrary::getPermanentLibrary(custom_lib_name.data(), &Err);
    assert(custom_dl.isValid());
    auto fPtr = custom_dl.getAddressOfSymbol(symbolName);
    return reinterpret_cast<FPtrTy>(fPtr);
  }

protected:
  static Arch *inst;
  llvm::sys::DynamicLibrary DL, PPL_DL;
  Arch(){};
  virtual ~Arch() = 0;
  void load_library();
  void load_ppl();
};
} // namespace backend
} // namespace tpu_mlir
