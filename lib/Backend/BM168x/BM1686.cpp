#include "sophgo/Backend/BM168x/BM1686.h"
#include "sophgo/Support/Helper/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

using namespace sophgo::backend;
using namespace sophgo::helper;
using namespace mlir;

constexpr llvm::StringRef BM1686::LIB_NAME;

uint64_t BM1686::get_gmem_start() { return 0x100000000ull; }

uint64_t BM1686::get_ctx_start_addr() { return get_gmem_start(); }

uint32_t BM1686::get_bdc_len(int bdc_num, int group_id) {
  if (bdc_num == 0) {
    return 0;
  }
  assert(group_id < bdc_bytes.size());
  return bdc_bytes[group_id];
}

uint32_t BM1686::get_gdma_len(int gdma_num, int group_id) {
  if (gdma_num == 0) {
    return 0;
  }
  assert(group_id < gdma_bytes.size());
  return gdma_bytes[group_id];
}

BM1686::BM1686() { chip = Module::Chip::BM1686; }

template <typename FPtrTy> FPtrTy BM1686::CastToFPtr(const char *symbolName) {
  assert(DL.isValid());
  auto fPtr = DL.getAddressOfSymbol(symbolName);
  if (fPtr == nullptr) {
    llvm::errs() << "can't find symbol: " << symbolName << "\n";
    llvm_unreachable(symbolName);
  }
  return reinterpret_cast<FPtrTy>(fPtr);
}

typedef int (*backend_api_t)(void *params, int param_size, void *pid_node);
void BM1686::call_global_func(const char *symbolName, void *params,
                              int param_size) {
  auto func = CastToFPtr<backend_api_t>(symbolName);
  func(params, param_size, cmdid_node);
}

#define CAST_FUNCTION(name) dl_##name = CastToFPtr<name>(#name)

void BM1686::load_functions() {
  BM168x::load_functions();
  CAST_FUNCTION(load_lookup_tables);
  CAST_FUNCTION(store_cmd_end);
  CAST_FUNCTION(set_cmd_len_ptr);
}

void BM1686::init() {
  BM168x::init();
  dl_set_cmd_len_ptr((void *)&gdma_bytes, (void *)&bdc_bytes);
}

void BM1686::after_codegen() {
  BM168x::after_codegen();
  dl_store_cmd_end();
}
