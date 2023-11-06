//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/InterfaceSupport.h"
#include "gtest/gtest.h"

class Backend {
public:
  /// Return the derived instance.
  Backend *getInstance() { return this; }
  mlir::TypeID getTypeID() const { return typeID; }

protected:
  Backend(mlir::TypeID typeID) : typeID(typeID){};
  mlir::TypeID typeID;
};

struct ExampleOpInterfaceTraits {
  struct Concept {
    virtual ~Concept() = default;
    virtual unsigned exampleInterfaceHook(Backend *op) = 0;
  };

  template <typename ConcreteOp>
  struct Model : public Concept {
    /// Override the method to dispatch on the concrete operation.
    unsigned exampleInterfaceHook(Backend *op) final {
      return ((ConcreteOp *)op)->exampleInterfaceHook();
    }
  };

  template <typename ConcreteType>
  struct FallbackModel : public Concept {
    unsigned exampleInterfaceHook(Backend *type) override { return 0; }
  };
  template <typename ConcreteModel, typename ConcreteType>
  struct ExternalModel : public FallbackModel<ConcreteModel> {
    unsigned exampleInterfaceHook(Backend *type) override { return 1; }
  };
};

template <typename ConcreteType, template <typename> class TraitType>
class DummyTraitBase {};

class BackendStroage {
public:
  BackendStroage() {}
  BackendStroage(const Backend *backend) : backend(backend) {}
  Backend *getInstance() { return const_cast<Backend *>(backend); }
  explicit operator bool() { return getInstance() != nullptr; }

private:
  const Backend *backend = nullptr;
};

template <typename ConcreteType, typename Traits>
class BackendInterface
    : public mlir::detail::Interface<ConcreteType, const Backend *, Traits,
                                     BackendStroage, DummyTraitBase> {
public:
  using Base = BackendInterface<ConcreteType, Traits>;
  using InterfaceBase =
      mlir::detail::Interface<ConcreteType, const Backend *, Traits,
                              BackendStroage, DummyTraitBase>;

  /// Inherit the base class constructor.
  using InterfaceBase::InterfaceBase;
};

class InterfaceMap {
public:
  InterfaceMap() = default;
  InterfaceMap(InterfaceMap &&) = default;
  InterfaceMap &operator=(InterfaceMap &&rhs) {
    for (auto &it : interfaces)
      free(it.second);
    interfaces = std::move(rhs.interfaces);
    return *this;
  }
  ~InterfaceMap() {
    for (auto &it : interfaces)
      free(it.second);
  }

  /// Returns true if the interface map contains an interface for the given id.
  bool contains(mlir::TypeID interfaceID) const { return lookup(interfaceID); }

  /// Insert the given interface models.
  template <typename... IfaceModels>
  void insertModels() {
    (insertModel<IfaceModels>(), ...);
  }

  /// Returns an instance of the concept object for the given interface id if it
  /// was registered to this map, null otherwise.
  void *lookup(mlir::TypeID id) const {
    const auto *it =
        llvm::lower_bound(interfaces, id, [](const auto &it, mlir::TypeID id) {
          return compare(it.first, id);
        });
    return (it != interfaces.end() && it->first == id) ? it->second : nullptr;
  }

private:
  /// Insert the given interface model into the map.
  template <typename InterfaceModel>
  void insertModel() {
    using ModelT = typename InterfaceModel::ModelT;
    static_assert(std::is_trivially_destructible_v<InterfaceModel>,
                  "interface models must be trivially destructible");
    auto *model = new (malloc(sizeof(ModelT))) ModelT();
    insert(InterfaceModel::getInterfaceID(), model);
  }

  void insert(mlir::TypeID interfaceId, void *conceptImpl) {
    // Insert directly into the right position to keep the interfaces sorted.
    auto *it = llvm::lower_bound(
        interfaces, interfaceId,
        [](const auto &it, mlir::TypeID id) { return compare(it.first, id); });
    if (it != interfaces.end() && it->first == interfaceId) {
      free(conceptImpl);
      return;
    }
    interfaces.insert(it, {interfaceId, conceptImpl});
  }

  /// Compare two TypeID instances by comparing the underlying pointer.
  static bool compare(mlir::TypeID lhs, mlir::TypeID rhs) {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  }

  /// A list of interface instances, sorted by TypeID.
  mlir::SmallVector<std::pair<mlir::TypeID, void *>> interfaces;
};

class ExampleOpInterface
    : public BackendInterface<ExampleOpInterface, ExampleOpInterfaceTraits> {
public:
  /// Inherit the base class constructor to support LLVM-style casting.
  using BackendInterface =
      BackendInterface<ExampleOpInterface, ExampleOpInterfaceTraits>;

  using BackendInterface::BackendInterface;

  template <typename ConcreteT>
  struct Traits : public BackendInterface::Trait<ConcreteT> {
    using ModelT = typename BackendInterface::Trait<ConcreteT>::ModelT;
    Traits() {
      ExampleOpInterface::getInterfaceMap().insertModels<Traits<ConcreteT>>();
    };
    static mlir::TypeID getInterfaceID() {
      return mlir::TypeID::get<ConcreteT>();
    }
  };
  /// The interface dispatches to 'getImpl()', a method provided by the base
  /// `BackendInterface` class that returns an instance of the concept.
  unsigned exampleInterfaceHook() {
    return getImpl()->exampleInterfaceHook(getInstance());
  }

protected:
  /// Returns the impl interface instance for the given operation.
  static InterfaceBase::Concept *getInterfaceFor(const Backend *op) {
    return (InterfaceBase::Concept *)getInterfaceMap().lookup(op->getTypeID());
  }
  static InterfaceMap &getInterfaceMap() {
    static InterfaceMap map;
    return map;
  }
  /// Allow access to `getInterfaceFor`.
  friend InterfaceBase;
};

namespace llvm {

template <typename T>
struct CastInfo<T, Backend *>
    : public ValueFromPointerCast<T, Backend, CastInfo<T, Backend *>> {
  static bool isPossible(Backend *op) { return T::classof(op); }
};
template <typename T>
struct CastInfo<T, const Backend *>
    : public ConstStrippingForwardingCast<T, const Backend *,
                                          CastInfo<T, Backend *>> {};

template <typename T>
struct CastInfo<T, Backend>
    : public NullableValueCastFailed<T>,
      public DefaultDoCastIfPossible<T, Backend &, CastInfo<T, Backend>> {
  // Provide isPossible here because here we have the const-stripping from
  // ConstStrippingCast.
  static bool isPossible(Backend &val) { return T::classof(&val); }
  static T doCast(Backend &val) { return T(&val); }
};

template <typename T>
struct CastInfo<T, const Backend>
    : public ConstStrippingForwardingCast<T, Backend, CastInfo<T, Backend>> {};

} // namespace llvm

class BM1684 : public Backend, ExampleOpInterface::Traits<BM1684> {
public:
  BM1684() : Backend(mlir::TypeID::get<BM1684>()){};
  unsigned exampleInterfaceHook() { return 1; }
  static bool classof(const Backend *bm168x) {
    return bm168x->getTypeID() == mlir::TypeID::get<BM1684>();
  }
};

TEST(InterfaceTest, MLIRInterface) {
  BM1684 bm1684;
  Backend *backend = &bm1684;
  EXPECT_TRUE(llvm::isa<ExampleOpInterface>(backend));
  if (auto imp = llvm::dyn_cast<ExampleOpInterface>(backend))
    EXPECT_EQ(imp.exampleInterfaceHook(), 1);
}

class MultiCoreInterface {
public:
  static bool classof(const Backend *bm168x) {
    return getBackends().contains(bm168x->getTypeID());
  };

  explicit MultiCoreInterface(Backend *bm168x) {
    impl = getBackends()[bm168x->getTypeID()];
  };
  explicit operator bool() { return impl != nullptr; }

  bool setCoreNum() { return impl->setCoreNum(); }

  template <typename ConcreteType>
  explicit MultiCoreInterface(ConcreteType *bm168x) {
    getBackends().insert(
        {mlir::TypeID::get<ConcreteType>(), new Model(bm168x)});
  };

private:
  struct Concept {
    virtual ~Concept() = default;
    virtual bool setCoreNum() = 0;
  };

  template <typename ConcreteType>
  struct Model : public Concept {
    Model(ConcreteType *concreteType) : concreteType(concreteType){};
    /// Override the method to dispatch on the concrete operation.
    bool setCoreNum() final { return concreteType->setCoreNum(); }
    ConcreteType *concreteType;
  };
  static mlir::DenseMap<mlir::TypeID, Concept *> &getBackends() {
    static mlir::DenseMap<mlir::TypeID, Concept *> Backends; // derived classes
    return Backends;
  }
  Concept *impl = nullptr;
};

class BM1688 : public Backend, MultiCoreInterface {
public:
  BM1688() : Backend(mlir::TypeID::get<BM1688>()), MultiCoreInterface(this){};
  bool setCoreNum() { return true; };
  static bool classof(const Backend *bm168x) {
    return bm168x->getTypeID() == mlir::TypeID::get<BM1684>();
  }
};

TEST(InterfaceTest, ConceptInterface) {
  BM1688 bm1688;
  Backend *backend = &bm1688;
  EXPECT_TRUE(llvm::isa<MultiCoreInterface>(backend));
  if (auto multi = llvm::dyn_cast<MultiCoreInterface>(backend))
    EXPECT_TRUE(multi.setCoreNum());
}
