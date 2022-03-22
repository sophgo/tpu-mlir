#pragma once
#include "llvm/Support/Casting.h"
#include "llvm/Support/DynamicLibrary.h"

using namespace llvm;

namespace sophgo {
namespace backend {

class Context {
public:
  enum ContextKind {
    BM1684,
    BM1686,
  };
  ContextKind getKind() const { return Kind; }
  template <typename FPtrTy> FPtrTy CastToFPtr(const char *symbolName);
  static Context *instance() { return inst.get(); }

protected:
  Context(ContextKind K) : Kind(K) {}
  virtual ~Context() = 0;

protected:
  const ContextKind Kind;
  sys::DynamicLibrary DL;
  static std::shared_ptr<Context> inst;
};
} // namespace backend
} // namespace sophgo
