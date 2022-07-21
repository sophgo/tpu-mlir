/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class OpAsmOpInterface;
namespace detail {
struct OpAsmOpInterfaceInterfaceTraits {
  struct Concept {
    void (*getAsmResultNames)(const Concept *impl, ::mlir::Operation *, ::mlir::OpAsmSetValueNameFn);
    void (*getAsmBlockArgumentNames)(const Concept *impl, ::mlir::Operation *, ::mlir::Region&, ::mlir::OpAsmSetValueNameFn);
    void (*getAsmBlockNames)(const Concept *impl, ::mlir::Operation *, ::mlir::OpAsmSetBlockNameFn);
    ::llvm::StringRef (*getDefaultDialect)();
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::OpAsmOpInterface;
    Model() : Concept{getAsmResultNames, getAsmBlockArgumentNames, getAsmBlockNames, getDefaultDialect} {}

    static inline void getAsmResultNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn);
    static inline void getAsmBlockArgumentNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Region& region, ::mlir::OpAsmSetValueNameFn setNameFn);
    static inline void getAsmBlockNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetBlockNameFn setNameFn);
    static inline ::llvm::StringRef getDefaultDialect();
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::OpAsmOpInterface;
    FallbackModel() : Concept{getAsmResultNames, getAsmBlockArgumentNames, getAsmBlockNames, getDefaultDialect} {}

    static inline void getAsmResultNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn);
    static inline void getAsmBlockArgumentNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Region& region, ::mlir::OpAsmSetValueNameFn setNameFn);
    static inline void getAsmBlockNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetBlockNameFn setNameFn);
    static inline ::llvm::StringRef getDefaultDialect();
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    void getAsmResultNames(::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn) const;
    void getAsmBlockArgumentNames(::mlir::Operation *tablegen_opaque_val, ::mlir::Region&region, ::mlir::OpAsmSetValueNameFn setNameFn) const;
    void getAsmBlockNames(::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetBlockNameFn setNameFn) const;
    static ::llvm::StringRef getDefaultDialect();
  };
};template <typename ConcreteOp>
struct OpAsmOpInterfaceTrait;

} // namespace detail
class OpAsmOpInterface : public ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::OpAsmOpInterfaceTrait<ConcreteOp> {};
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  void getAsmBlockArgumentNames(::mlir::Region& region, ::mlir::OpAsmSetValueNameFn setNameFn);
  void getAsmBlockNames(::mlir::OpAsmSetBlockNameFn setNameFn);
  ::llvm::StringRef getDefaultDialect();
};
namespace detail {
  template <typename ConcreteOp>
  struct OpAsmOpInterfaceTrait : public ::mlir::OpInterface<OpAsmOpInterface, detail::OpAsmOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
      return;
    }
    void getAsmBlockArgumentNames(::mlir::Region& region, ::mlir::OpAsmSetValueNameFn setNameFn) {
      return;
    }
    void getAsmBlockNames(::mlir::OpAsmSetBlockNameFn setNameFn) {
      ;
    }
    static ::llvm::StringRef getDefaultDialect() {
      return "";
    }
  };
}// namespace detail
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsmResultNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsmResultNames(setNameFn);
}
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsmBlockArgumentNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Region& region, ::mlir::OpAsmSetValueNameFn setNameFn) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsmBlockArgumentNames(region, setNameFn);
}
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAsmBlockNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetBlockNameFn setNameFn) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAsmBlockNames(setNameFn);
}
template<typename ConcreteOp>
::llvm::StringRef detail::OpAsmOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDefaultDialect() {
  return ConcreteOp::getDefaultDialect();
}
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAsmResultNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn) {
  return static_cast<const ConcreteOp *>(impl)->getAsmResultNames(tablegen_opaque_val, setNameFn);
}
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAsmBlockArgumentNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::Region& region, ::mlir::OpAsmSetValueNameFn setNameFn) {
  return static_cast<const ConcreteOp *>(impl)->getAsmBlockArgumentNames(tablegen_opaque_val, region, setNameFn);
}
template<typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAsmBlockNames(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetBlockNameFn setNameFn) {
  return static_cast<const ConcreteOp *>(impl)->getAsmBlockNames(tablegen_opaque_val, setNameFn);
}
template<typename ConcreteOp>
::llvm::StringRef detail::OpAsmOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDefaultDialect() {
  return ConcreteOp::getDefaultDialect();
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAsmResultNames(::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetValueNameFn setNameFn) const {
return;
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAsmBlockArgumentNames(::mlir::Operation *tablegen_opaque_val, ::mlir::Region&region, ::mlir::OpAsmSetValueNameFn setNameFn) const {
return;
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::OpAsmOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAsmBlockNames(::mlir::Operation *tablegen_opaque_val, ::mlir::OpAsmSetBlockNameFn setNameFn) const {
;
}
template<typename ConcreteModel, typename ConcreteOp>
::llvm::StringRef detail::OpAsmOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDefaultDialect() {
return "";
}
} // namespace mlir
