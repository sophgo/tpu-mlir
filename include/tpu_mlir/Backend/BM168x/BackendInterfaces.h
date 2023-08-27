//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Support/Module.h"

namespace tpu_mlir {
namespace backend {

template <typename ConcreteType, typename BaseT>
class BackendInterfaceBase : public BaseT {
public:
  using Base = BackendInterfaceBase<ConcreteType, BaseT>;

  /// Get a unique id for the derived interface type.
  static TypeID getInterfaceID() { return TypeID::get<ConcreteType>(); }

protected:
  BackendInterfaceBase()
      // cast the final type to this interface: to resolve virtual slice.
      : BaseT(getInterfaceID(), static_cast<BaseT *>(this)) {}
};

class MultiCoreInterface {
public:

  template <typename ConcreteType>
  using Base = BackendInterfaceBase<ConcreteType, MultiCoreInterface>;

  virtual ~MultiCoreInterface() = default;
  virtual void setCoreNum(int core = 1) = 0;
  virtual int getCoreNum() = 0;
  virtual int getCurrentCoreID() = 0;
  virtual void useCore(int coreID = 0) = 0;
  virtual void syncAll() = 0;
  virtual void setupMultiCoreContext(int, int, int) = 0;
  virtual std::vector<std::shared_ptr<BM168x::Code>> const &getCodebuffer() = 0;

  static bool classof(const BM168x *bm168x) {
    return getBackends().contains(bm168x->getTypeID());
  };
  // provide cast<X> and dyn_cast<X>
  static MultiCoreInterface *doCast(BM168x *bm168x) {
    return getBackends().at(bm168x->getTypeID());
  };

protected:
  MultiCoreInterface(mlir::TypeID id, MultiCoreInterface *ptr) {
    getBackends().insert({id, ptr});
  }

private:
  // Provide interface storage; It is only validate for singleton(each
  // backend[target] should have one instance only.)
  // If this is not hold, we need to consider concept based polymorphism as
  // MLIR.
  static mlir::DenseMap<mlir::TypeID, MultiCoreInterface *> &getBackends() {
    static mlir::DenseMap<mlir::TypeID, MultiCoreInterface *>
        Backends; // derived classes
    return Backends;
  }
};

} // namespace backend
} // namespace tpu_mlir

namespace llvm {

// for interface which is not a derived class of BM168x
template <typename To>
struct CastInfo<To, backend::BM168x *,
                std::enable_if_t<!std::is_base_of_v<backend::BM168x, To>>> {
  using CastReturnType = typename cast_retty<To, backend::BM168x *>::ret_type;
  static CastReturnType castFailed() { return nullptr; }
  static bool isPossible(backend::BM168x *op) { return To::classof(op); }
  static CastReturnType doCast(backend::BM168x *val) { return To::doCast(val); }
  static CastReturnType doCastIfPossible(backend::BM168x *val) {
    if (!isPossible(val))
      return castFailed();
    return doCast(val);
  }
};
template <typename T>
struct CastInfo<T, const backend::BM168x *,
                std::enable_if_t<!std::is_base_of_v<backend::BM168x, T>>>
    : public ConstStrippingForwardingCast<T, const backend::BM168x *,
                                          CastInfo<T, backend::BM168x *>> {};

} // namespace llvm
