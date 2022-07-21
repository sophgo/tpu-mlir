/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

class AffineMapAccessInterface;
namespace detail {
struct AffineMapAccessInterfaceInterfaceTraits {
  struct Concept {
    NamedAttribute (*getAffineMapAttrForMemRef)(const Concept *impl, ::mlir::Operation *, Value);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = AffineMapAccessInterface;
    Model() : Concept{getAffineMapAttrForMemRef} {}

    static inline NamedAttribute getAffineMapAttrForMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value memref);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = AffineMapAccessInterface;
    FallbackModel() : Concept{getAffineMapAttrForMemRef} {}

    static inline NamedAttribute getAffineMapAttrForMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value memref);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    NamedAttribute getAffineMapAttrForMemRef(::mlir::Operation *tablegen_opaque_val, Value memref) const;
  };
};template <typename ConcreteOp>
struct AffineMapAccessInterfaceTrait;

} // namespace detail
class AffineMapAccessInterface : public ::mlir::OpInterface<AffineMapAccessInterface, detail::AffineMapAccessInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AffineMapAccessInterface, detail::AffineMapAccessInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AffineMapAccessInterfaceTrait<ConcreteOp> {};
  NamedAttribute getAffineMapAttrForMemRef(Value memref);
};
namespace detail {
  template <typename ConcreteOp>
  struct AffineMapAccessInterfaceTrait : public ::mlir::OpInterface<AffineMapAccessInterface, detail::AffineMapAccessInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    NamedAttribute getAffineMapAttrForMemRef(Value memref) {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        assert(memref == op.getMemRef() &&
               "Expected memref argument to match memref operand");
        return {StringAttr::get(op.getContext(), op.getMapAttrName()),
                op.getAffineMapAttr()};
    }
  };
}// namespace detail
template<typename ConcreteOp>
NamedAttribute detail::AffineMapAccessInterfaceInterfaceTraits::Model<ConcreteOp>::getAffineMapAttrForMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value memref) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMapAttrForMemRef(memref);
}
template<typename ConcreteOp>
NamedAttribute detail::AffineMapAccessInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAffineMapAttrForMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, Value memref) {
  return static_cast<const ConcreteOp *>(impl)->getAffineMapAttrForMemRef(tablegen_opaque_val, memref);
}
template<typename ConcreteModel, typename ConcreteOp>
NamedAttribute detail::AffineMapAccessInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAffineMapAttrForMemRef(::mlir::Operation *tablegen_opaque_val, Value memref) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        assert(memref == op.getMemRef() &&
               "Expected memref argument to match memref operand");
        return {StringAttr::get(op.getContext(), op.getMapAttrName()),
                op.getAffineMapAttr()};
}
class AffineReadOpInterface;
namespace detail {
struct AffineReadOpInterfaceInterfaceTraits {
  struct Concept {
    Value (*getMemRef)(const Concept *impl, ::mlir::Operation *);
    MemRefType (*getMemRefType)(const Concept *impl, ::mlir::Operation *);
    Operation::operand_range (*getMapOperands)(const Concept *impl, ::mlir::Operation *);
    AffineMap (*getAffineMap)(const Concept *impl, ::mlir::Operation *);
    Value (*getValue)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = AffineReadOpInterface;
    Model() : Concept{getMemRef, getMemRefType, getMapOperands, getAffineMap, getValue} {}

    static inline Value getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline MemRefType getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Operation::operand_range getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value getValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = AffineReadOpInterface;
    FallbackModel() : Concept{getMemRef, getMemRefType, getMapOperands, getAffineMap, getValue} {}

    static inline Value getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline MemRefType getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Operation::operand_range getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value getValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    Value getMemRef(::mlir::Operation *tablegen_opaque_val) const;
    MemRefType getMemRefType(::mlir::Operation *tablegen_opaque_val) const;
    Operation::operand_range getMapOperands(::mlir::Operation *tablegen_opaque_val) const;
    AffineMap getAffineMap(::mlir::Operation *tablegen_opaque_val) const;
    Value getValue(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct AffineReadOpInterfaceTrait;

} // namespace detail
class AffineReadOpInterface : public ::mlir::OpInterface<AffineReadOpInterface, detail::AffineReadOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AffineReadOpInterface, detail::AffineReadOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AffineReadOpInterfaceTrait<ConcreteOp> {};
  Value getMemRef();
  MemRefType getMemRefType();
  Operation::operand_range getMapOperands();
  AffineMap getAffineMap();
  Value getValue();
};
namespace detail {
  template <typename ConcreteOp>
  struct AffineReadOpInterfaceTrait : public ::mlir::OpInterface<AffineReadOpInterface, detail::AffineReadOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    Value getMemRef() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getMemRefOperandIndex());
    }
    MemRefType getMemRefType() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getMemRef().getType().template cast<MemRefType>();
    }
    Operation::operand_range getMapOperands() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return llvm::drop_begin(op.getOperands(), 1);
    }
    AffineMap getAffineMap() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAffineMapAttr().getValue();
    }
    Value getValue() {
      return cast<ConcreteOp>(this->getOperation());
    }
  };
}// namespace detail
template<typename ConcreteOp>
Value detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRef();
}
template<typename ConcreteOp>
MemRefType detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRefType();
}
template<typename ConcreteOp>
Operation::operand_range detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMapOperands();
}
template<typename ConcreteOp>
AffineMap detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMap();
}
template<typename ConcreteOp>
Value detail::AffineReadOpInterfaceInterfaceTraits::Model<ConcreteOp>::getValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getValue();
}
template<typename ConcreteOp>
Value detail::AffineReadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMemRef(tablegen_opaque_val);
}
template<typename ConcreteOp>
MemRefType detail::AffineReadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMemRefType(tablegen_opaque_val);
}
template<typename ConcreteOp>
Operation::operand_range detail::AffineReadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMapOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
AffineMap detail::AffineReadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAffineMap(tablegen_opaque_val);
}
template<typename ConcreteOp>
Value detail::AffineReadOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getValue(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::AffineReadOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMemRef(::mlir::Operation *tablegen_opaque_val) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getMemRefOperandIndex());
}
template<typename ConcreteModel, typename ConcreteOp>
MemRefType detail::AffineReadOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMemRefType(::mlir::Operation *tablegen_opaque_val) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getMemRef().getType().template cast<MemRefType>();
}
template<typename ConcreteModel, typename ConcreteOp>
Operation::operand_range detail::AffineReadOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMapOperands(::mlir::Operation *tablegen_opaque_val) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return llvm::drop_begin(op.getOperands(), 1);
}
template<typename ConcreteModel, typename ConcreteOp>
AffineMap detail::AffineReadOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAffineMap(::mlir::Operation *tablegen_opaque_val) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAffineMapAttr().getValue();
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::AffineReadOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getValue(::mlir::Operation *tablegen_opaque_val) const {
return cast<ConcreteOp>(this->getOperation());
}
class AffineWriteOpInterface;
namespace detail {
struct AffineWriteOpInterfaceInterfaceTraits {
  struct Concept {
    Value (*getMemRef)(const Concept *impl, ::mlir::Operation *);
    MemRefType (*getMemRefType)(const Concept *impl, ::mlir::Operation *);
    Operation::operand_range (*getMapOperands)(const Concept *impl, ::mlir::Operation *);
    AffineMap (*getAffineMap)(const Concept *impl, ::mlir::Operation *);
    Value (*getValueToStore)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = AffineWriteOpInterface;
    Model() : Concept{getMemRef, getMemRefType, getMapOperands, getAffineMap, getValueToStore} {}

    static inline Value getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline MemRefType getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Operation::operand_range getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value getValueToStore(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = AffineWriteOpInterface;
    FallbackModel() : Concept{getMemRef, getMemRefType, getMapOperands, getAffineMap, getValueToStore} {}

    static inline Value getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline MemRefType getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Operation::operand_range getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value getValueToStore(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    Value getMemRef(::mlir::Operation *tablegen_opaque_val) const;
    MemRefType getMemRefType(::mlir::Operation *tablegen_opaque_val) const;
    Operation::operand_range getMapOperands(::mlir::Operation *tablegen_opaque_val) const;
    AffineMap getAffineMap(::mlir::Operation *tablegen_opaque_val) const;
    Value getValueToStore(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct AffineWriteOpInterfaceTrait;

} // namespace detail
class AffineWriteOpInterface : public ::mlir::OpInterface<AffineWriteOpInterface, detail::AffineWriteOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<AffineWriteOpInterface, detail::AffineWriteOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::AffineWriteOpInterfaceTrait<ConcreteOp> {};
  Value getMemRef();
  MemRefType getMemRefType();
  Operation::operand_range getMapOperands();
  AffineMap getAffineMap();
  Value getValueToStore();
};
namespace detail {
  template <typename ConcreteOp>
  struct AffineWriteOpInterfaceTrait : public ::mlir::OpInterface<AffineWriteOpInterface, detail::AffineWriteOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    Value getMemRef() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getMemRefOperandIndex());
    }
    MemRefType getMemRefType() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getMemRef().getType().template cast<MemRefType>();
    }
    Operation::operand_range getMapOperands() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return llvm::drop_begin(op.getOperands(), 2);
    }
    AffineMap getAffineMap() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAffineMapAttr().getValue();
    }
    Value getValueToStore() {
      ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getStoredValOperandIndex());
    }
  };
}// namespace detail
template<typename ConcreteOp>
Value detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRef();
}
template<typename ConcreteOp>
MemRefType detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMemRefType();
}
template<typename ConcreteOp>
Operation::operand_range detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getMapOperands();
}
template<typename ConcreteOp>
AffineMap detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getAffineMap();
}
template<typename ConcreteOp>
Value detail::AffineWriteOpInterfaceInterfaceTraits::Model<ConcreteOp>::getValueToStore(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getValueToStore();
}
template<typename ConcreteOp>
Value detail::AffineWriteOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMemRef(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMemRef(tablegen_opaque_val);
}
template<typename ConcreteOp>
MemRefType detail::AffineWriteOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMemRefType(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMemRefType(tablegen_opaque_val);
}
template<typename ConcreteOp>
Operation::operand_range detail::AffineWriteOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getMapOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getMapOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
AffineMap detail::AffineWriteOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getAffineMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getAffineMap(tablegen_opaque_val);
}
template<typename ConcreteOp>
Value detail::AffineWriteOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getValueToStore(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getValueToStore(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::AffineWriteOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMemRef(::mlir::Operation *tablegen_opaque_val) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getMemRefOperandIndex());
}
template<typename ConcreteModel, typename ConcreteOp>
MemRefType detail::AffineWriteOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMemRefType(::mlir::Operation *tablegen_opaque_val) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getMemRef().getType().template cast<MemRefType>();
}
template<typename ConcreteModel, typename ConcreteOp>
Operation::operand_range detail::AffineWriteOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getMapOperands(::mlir::Operation *tablegen_opaque_val) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return llvm::drop_begin(op.getOperands(), 2);
}
template<typename ConcreteModel, typename ConcreteOp>
AffineMap detail::AffineWriteOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getAffineMap(::mlir::Operation *tablegen_opaque_val) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getAffineMapAttr().getValue();
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::AffineWriteOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getValueToStore(::mlir::Operation *tablegen_opaque_val) const {
ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getOperand(op.getStoredValOperandIndex());
}
