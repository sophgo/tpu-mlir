/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace arith {
class ArithFastMathInterface;
namespace detail {
struct ArithFastMathInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    FastMathFlagsAttr (*getFastMathFlagsAttr)(const Concept *impl, ::mlir::Operation *);
    StringRef (*getFastMathAttrName)();
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::arith::ArithFastMathInterface;
    Model() : Concept{getFastMathFlagsAttr, getFastMathAttrName} {}

    static inline FastMathFlagsAttr getFastMathFlagsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline StringRef getFastMathAttrName();
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::arith::ArithFastMathInterface;
    FallbackModel() : Concept{getFastMathFlagsAttr, getFastMathAttrName} {}

    static inline FastMathFlagsAttr getFastMathFlagsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline StringRef getFastMathAttrName();
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    FastMathFlagsAttr getFastMathFlagsAttr(::mlir::Operation *tablegen_opaque_val) const;
    static StringRef getFastMathAttrName();
  };
};template <typename ConcreteOp>
struct ArithFastMathInterfaceTrait;

} // namespace detail
class ArithFastMathInterface : public ::mlir::OpInterface<ArithFastMathInterface, detail::ArithFastMathInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ArithFastMathInterface, detail::ArithFastMathInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ArithFastMathInterfaceTrait<ConcreteOp> {};
  /// Returns a FastMathFlagsAttr attribute for the operation
  FastMathFlagsAttr getFastMathFlagsAttr();
  /// Returns the name of the FastMathFlagsAttr attribute
  ///                          for the operation
  StringRef getFastMathAttrName();
};
namespace detail {
  template <typename ConcreteOp>
  struct ArithFastMathInterfaceTrait : public ::mlir::OpInterface<ArithFastMathInterface, detail::ArithFastMathInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Returns a FastMathFlagsAttr attribute for the operation
    FastMathFlagsAttr getFastMathFlagsAttr() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getFastmathAttr();
    }
    /// Returns the name of the FastMathFlagsAttr attribute
    ///                          for the operation
    static StringRef getFastMathAttrName() {
      return "fastmath";
    }
  };
}// namespace detail
} // namespace arith
} // namespace mlir
namespace mlir {
namespace arith {
template<typename ConcreteOp>
FastMathFlagsAttr detail::ArithFastMathInterfaceInterfaceTraits::Model<ConcreteOp>::getFastMathFlagsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getFastMathFlagsAttr();
}
template<typename ConcreteOp>
StringRef detail::ArithFastMathInterfaceInterfaceTraits::Model<ConcreteOp>::getFastMathAttrName() {
  return ConcreteOp::getFastMathAttrName();
}
template<typename ConcreteOp>
FastMathFlagsAttr detail::ArithFastMathInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getFastMathFlagsAttr(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getFastMathFlagsAttr(tablegen_opaque_val);
}
template<typename ConcreteOp>
StringRef detail::ArithFastMathInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getFastMathAttrName() {
  return ConcreteOp::getFastMathAttrName();
}
template<typename ConcreteModel, typename ConcreteOp>
FastMathFlagsAttr detail::ArithFastMathInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getFastMathFlagsAttr(::mlir::Operation *tablegen_opaque_val) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getFastmathAttr();
}
template<typename ConcreteModel, typename ConcreteOp>
StringRef detail::ArithFastMathInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getFastMathAttrName() {
return "fastmath";
}
} // namespace arith
} // namespace mlir
