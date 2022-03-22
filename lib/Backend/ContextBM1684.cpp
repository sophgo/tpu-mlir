#include "sophgo/Backend/ContextBM1684.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Format.h"

using namespace sophgo::backend;

BM1684Context::BM1684Context() : Context(BM1684) {
  std::string Err;
  DL = sys::DynamicLibrary::getPermanentLibrary("libbackend_1684.so", &Err);
  if (DL.isValid() == false) {
    llvm_unreachable(Err.c_str());
  }
  cmodel_init = CastToFPtr<CModelInitFn>("cmodel_init");
  cmodel_deinit = CastToFPtr<CModelDeinitFn>("cmodel_deinit");
}
