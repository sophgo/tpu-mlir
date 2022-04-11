#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

using namespace sophgo::backend;
using namespace sophgo::helper;
using namespace mlir;

uint64_t BM1686::get_gmem_start() {
  return 0x100000000ull;
}

uint64_t BM1686::get_ctx_start_addr() {
  return get_gmem_start();
}

template <typename FPtrTy> FPtrTy BM1686::CastToFPtr(const char *symbolName) {
  assert(DL.isValid());
  auto fPtr = DL.getAddressOfSymbol(symbolName);
  if (fPtr == nullptr) {
    llvm::errs() << "can't find symbol: " << symbolName << "\n";
    llvm_unreachable(symbolName);
  }
  return reinterpret_cast<FPtrTy>(fPtr);
}

#define CAST_FUNCTION(name) dl_##name = CastToFPtr<name>(#name)


BM1686::BM1686() {
  std::string Err;
  DL = llvm::sys::DynamicLibrary::getPermanentLibrary("libbackend_1686.so",
                                                      &Err);
  if (DL.isValid() == false) {
    llvm_unreachable(Err.c_str());
  }
  CAST_FUNCTION(cmodel_init);
  CAST_FUNCTION(cmodel_deinit);
  CAST_FUNCTION(get_global_memaddr);
  CAST_FUNCTION(set_cmd_buffer_ptr);
}

BM1686::~BM1686() {}

int64_t BM1686::get_eu_num(int64_t dtype_bytes) {
  return 16 * 4 / dtype_bytes;
}
