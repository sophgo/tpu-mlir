/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class TilingInterface;
namespace detail {
struct TilingInterfaceInterfaceTraits {
  struct Concept {
    SmallVector<Value> (*getDestinationOperands)(const Concept *impl, ::mlir::Operation *, OpBuilder &);
    SmallVector<StringRef> (*getLoopIteratorTypes)(const Concept *impl, ::mlir::Operation *);
    SmallVector<Range> (*getIterationDomain)(const Concept *impl, ::mlir::Operation *, OpBuilder &);
    SmallVector<Operation *> (*getTiledImplementation)(const Concept *impl, ::mlir::Operation *, OpBuilder &, ValueRange , ArrayRef<OpFoldResult> , ArrayRef<OpFoldResult> , bool );
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::TilingInterface;
    Model() : Concept{getDestinationOperands, getLoopIteratorTypes, getIterationDomain, getTiledImplementation} {}

    static inline SmallVector<Value> getDestinationOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b);
    static inline SmallVector<StringRef> getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<Range> getIterationDomain(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b);
    static inline SmallVector<Operation *> getTiledImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, ValueRange  dest, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, bool  tileDestOperands);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::TilingInterface;
    FallbackModel() : Concept{getDestinationOperands, getLoopIteratorTypes, getIterationDomain, getTiledImplementation} {}

    static inline SmallVector<Value> getDestinationOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b);
    static inline SmallVector<StringRef> getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<Range> getIterationDomain(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b);
    static inline SmallVector<Operation *> getTiledImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, ValueRange  dest, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, bool  tileDestOperands);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    SmallVector<Value> getDestinationOperands(::mlir::Operation *tablegen_opaque_val, OpBuilder &b) const;
    SmallVector<StringRef> getLoopIteratorTypes(::mlir::Operation *tablegen_opaque_val) const;
    SmallVector<Range> getIterationDomain(::mlir::Operation *tablegen_opaque_val, OpBuilder &b) const;
    SmallVector<Operation *> getTiledImplementation(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, ValueRange dest, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, bool tileDestOperands) const;
  };
};template <typename ConcreteOp>
struct TilingInterfaceTrait;

} // namespace detail
class TilingInterface : public ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::TilingInterfaceTrait<ConcreteOp> {};
  SmallVector<Value> getDestinationOperands(OpBuilder & b);
  SmallVector<StringRef> getLoopIteratorTypes();
  SmallVector<Range> getIterationDomain(OpBuilder & b);
  SmallVector<Operation *> getTiledImplementation(OpBuilder & b, ValueRange  dest, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, bool  tileDestOperands);
};
namespace detail {
  template <typename ConcreteOp>
  struct TilingInterfaceTrait : public ::mlir::OpInterface<TilingInterface, detail::TilingInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    SmallVector<Value> getDestinationOperands(OpBuilder & b) {
      return ValueRange{};
    }
    SmallVector<StringRef> getLoopIteratorTypes() {
      return {};
    }
    SmallVector<Range> getIterationDomain(OpBuilder & b) {
      return {};
    }
    SmallVector<Operation *> getTiledImplementation(OpBuilder & b, ValueRange  dest, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, bool  tileDestOperands) {
      return {};
    }
  };
}// namespace detail
template<typename ConcreteOp>
SmallVector<Value> detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::getDestinationOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDestinationOperands(b);
}
template<typename ConcreteOp>
SmallVector<StringRef> detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopIteratorTypes();
}
template<typename ConcreteOp>
SmallVector<Range> detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::getIterationDomain(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIterationDomain(b);
}
template<typename ConcreteOp>
SmallVector<Operation *> detail::TilingInterfaceInterfaceTraits::Model<ConcreteOp>::getTiledImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, ValueRange  dest, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, bool  tileDestOperands) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTiledImplementation(b, dest, offsets, sizes, tileDestOperands);
}
template<typename ConcreteOp>
SmallVector<Value> detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDestinationOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b) {
  return static_cast<const ConcreteOp *>(impl)->getDestinationOperands(tablegen_opaque_val, b);
}
template<typename ConcreteOp>
SmallVector<StringRef> detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getLoopIteratorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLoopIteratorTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<Range> detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getIterationDomain(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b) {
  return static_cast<const ConcreteOp *>(impl)->getIterationDomain(tablegen_opaque_val, b);
}
template<typename ConcreteOp>
SmallVector<Operation *> detail::TilingInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTiledImplementation(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, ValueRange  dest, ArrayRef<OpFoldResult>  offsets, ArrayRef<OpFoldResult>  sizes, bool  tileDestOperands) {
  return static_cast<const ConcreteOp *>(impl)->getTiledImplementation(tablegen_opaque_val, b, dest, offsets, sizes, tileDestOperands);
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<Value> detail::TilingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDestinationOperands(::mlir::Operation *tablegen_opaque_val, OpBuilder &b) const {
return ValueRange{};
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<StringRef> detail::TilingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLoopIteratorTypes(::mlir::Operation *tablegen_opaque_val) const {
return {};
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<Range> detail::TilingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getIterationDomain(::mlir::Operation *tablegen_opaque_val, OpBuilder &b) const {
return {};
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<Operation *> detail::TilingInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiledImplementation(::mlir::Operation *tablegen_opaque_val, OpBuilder &b, ValueRange dest, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, bool tileDestOperands) const {
return {};
}
} // namespace mlir
