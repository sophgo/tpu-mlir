/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
class DestinationStyleOpInterface;
namespace detail {
struct DestinationStyleOpInterfaceInterfaceTraits {
  struct Concept {
    /// The methods defined by the interface.
    std::pair<int64_t, int64_t> (*getDpsInitsPositionRange)(const Concept *impl, ::mlir::Operation *);
    int64_t (*getNumDpsInits)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OpOperandVector (*getDpsInitOperands)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OpOperand *(*getDpsInitOperand)(const Concept *impl, ::mlir::Operation *, int64_t);
    void (*setDpsInitOperand)(const Concept *impl, ::mlir::Operation *, int64_t, ::mlir::Value);
    int64_t (*getNumDpsInputs)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OpOperandVector (*getDpsInputOperands)(const Concept *impl, ::mlir::Operation *);
    ::mlir::OpOperand *(*getDpsInputOperand)(const Concept *impl, ::mlir::Operation *, int64_t);
    bool (*isDpsInput)(const Concept *impl, ::mlir::Operation *, ::mlir::OpOperand *);
    bool (*isDpsInit)(const Concept *impl, ::mlir::Operation *, ::mlir::OpOperand *);
    bool (*isScalar)(const Concept *impl, ::mlir::Operation *, ::mlir::OpOperand *);
    ::mlir::OpResult (*getTiedOpResult)(const Concept *impl, ::mlir::Operation *, ::mlir::OpOperand *);
    ::mlir::OpOperand *(*getTiedOpOperand)(const Concept *impl, ::mlir::Operation *, ::mlir::OpResult);
    bool (*hasBufferSemantics)(const Concept *impl, ::mlir::Operation *);
    bool (*hasTensorSemantics)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::DestinationStyleOpInterface;
    Model() : Concept{getDpsInitsPositionRange, getNumDpsInits, getDpsInitOperands, getDpsInitOperand, setDpsInitOperand, getNumDpsInputs, getDpsInputOperands, getDpsInputOperand, isDpsInput, isDpsInit, isScalar, getTiedOpResult, getTiedOpOperand, hasBufferSemantics, hasTensorSemantics} {}

    static inline std::pair<int64_t, int64_t> getDpsInitsPositionRange(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline int64_t getNumDpsInits(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperandVector getDpsInitOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand *getDpsInitOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i);
    static inline void setDpsInitOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i, ::mlir::Value value);
    static inline int64_t getNumDpsInputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperandVector getDpsInputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand *getDpsInputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i);
    static inline bool isDpsInput(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand);
    static inline bool isDpsInit(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand);
    static inline bool isScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand);
    static inline ::mlir::OpResult getTiedOpResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand);
    static inline ::mlir::OpOperand *getTiedOpOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpResult opResult);
    static inline bool hasBufferSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasTensorSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::DestinationStyleOpInterface;
    FallbackModel() : Concept{getDpsInitsPositionRange, getNumDpsInits, getDpsInitOperands, getDpsInitOperand, setDpsInitOperand, getNumDpsInputs, getDpsInputOperands, getDpsInputOperand, isDpsInput, isDpsInit, isScalar, getTiedOpResult, getTiedOpOperand, hasBufferSemantics, hasTensorSemantics} {}

    static inline std::pair<int64_t, int64_t> getDpsInitsPositionRange(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline int64_t getNumDpsInits(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperandVector getDpsInitOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand *getDpsInitOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i);
    static inline void setDpsInitOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i, ::mlir::Value value);
    static inline int64_t getNumDpsInputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperandVector getDpsInputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ::mlir::OpOperand *getDpsInputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i);
    static inline bool isDpsInput(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand);
    static inline bool isDpsInit(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand);
    static inline bool isScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand);
    static inline ::mlir::OpResult getTiedOpResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand);
    static inline ::mlir::OpOperand *getTiedOpOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpResult opResult);
    static inline bool hasBufferSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasTensorSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    int64_t getNumDpsInits(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::OpOperandVector getDpsInitOperands(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::OpOperand *getDpsInitOperand(::mlir::Operation *tablegen_opaque_val, int64_t i) const;
    void setDpsInitOperand(::mlir::Operation *tablegen_opaque_val, int64_t i, ::mlir::Value value) const;
    int64_t getNumDpsInputs(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::OpOperandVector getDpsInputOperands(::mlir::Operation *tablegen_opaque_val) const;
    ::mlir::OpOperand *getDpsInputOperand(::mlir::Operation *tablegen_opaque_val, int64_t i) const;
    bool isDpsInput(::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand *opOperand) const;
    bool isDpsInit(::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand *opOperand) const;
    bool isScalar(::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand *opOperand) const;
    ::mlir::OpResult getTiedOpResult(::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand *opOperand) const;
    ::mlir::OpOperand *getTiedOpOperand(::mlir::Operation *tablegen_opaque_val, ::mlir::OpResult opResult) const;
    bool hasBufferSemantics(::mlir::Operation *tablegen_opaque_val) const;
    bool hasTensorSemantics(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct DestinationStyleOpInterfaceTrait;

} // namespace detail
class DestinationStyleOpInterface : public ::mlir::OpInterface<DestinationStyleOpInterface, detail::DestinationStyleOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<DestinationStyleOpInterface, detail::DestinationStyleOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::DestinationStyleOpInterfaceTrait<ConcreteOp> {};
  /// Return start and end indices of the init operands range.
  std::pair<int64_t, int64_t> getDpsInitsPositionRange();
  /// Return the number of inits.
  int64_t getNumDpsInits();
  /// Return the init operands.
  ::mlir::OpOperandVector getDpsInitOperands();
  /// Return the `i`-th init operand.
  ::mlir::OpOperand *getDpsInitOperand(int64_t i);
  /// Set the `i`-th init operand.
  void setDpsInitOperand(int64_t i, ::mlir::Value value);
  /// Return the number of inputs.
  int64_t getNumDpsInputs();
  /// Return the input operands.
  ::mlir::OpOperandVector getDpsInputOperands();
  /// Return the `i`-th input operand.
  ::mlir::OpOperand *getDpsInputOperand(int64_t i);
  /// Return true if `opOperand` is an input.
  bool isDpsInput(::mlir::OpOperand * opOperand);
  /// Return true if `opOperand` is an init.
  bool isDpsInit(::mlir::OpOperand * opOperand);
  /// Return true if the `opOperand` is a scalar value.
  bool isScalar(::mlir::OpOperand * opOperand);
  /// Return the OpResult that is tied to the given OpOperand.
  ::mlir::OpResult getTiedOpResult(::mlir::OpOperand * opOperand);
  /// Return the OpOperand that is tied to the given OpResult.
  ::mlir::OpOperand *getTiedOpOperand(::mlir::OpResult opResult);
  /// Return whether the op has only ranked MemRef input/inits.
  bool hasBufferSemantics();
  /// Return whether the op has only ranked tensor inputs/inits.
  bool hasTensorSemantics();
};
namespace detail {
  template <typename ConcreteOp>
  struct DestinationStyleOpInterfaceTrait : public ::mlir::OpInterface<DestinationStyleOpInterface, detail::DestinationStyleOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Return the number of inits.
    int64_t getNumDpsInits() {
      auto [start, end] = (*static_cast<ConcreteOp *>(this)).getDpsInitsPositionRange();
        return end - start;
    }
    /// Return the init operands.
    ::mlir::OpOperandVector getDpsInitOperands() {
      auto [start, end] = (*static_cast<ConcreteOp *>(this)).getDpsInitsPositionRange();

        ::mlir::OpOperandVector result;
        result.reserve(end - start);
        for (int i = start; i < end; ++i)
          result.push_back(&(*static_cast<ConcreteOp *>(this))->getOpOperand(i));
        return result;
    }
    /// Return the `i`-th init operand.
    ::mlir::OpOperand *getDpsInitOperand(int64_t i) {
      assert(i >= 0 && i < (*static_cast<ConcreteOp *>(this)).getNumDpsInits());
        auto [start, end] = (*static_cast<ConcreteOp *>(this)).getDpsInitsPositionRange();
        return &(*static_cast<ConcreteOp *>(this))->getOpOperand(start + i);
    }
    /// Set the `i`-th init operand.
    void setDpsInitOperand(int64_t i, ::mlir::Value value) {
      assert(i >= 0 && i < (*static_cast<ConcreteOp *>(this)).getNumDpsInits());
        auto [start, end] = (*static_cast<ConcreteOp *>(this)).getDpsInitsPositionRange();
        (*static_cast<ConcreteOp *>(this))->setOperand(start + i, value);
    }
    /// Return the number of inputs.
    int64_t getNumDpsInputs() {
      return (*static_cast<ConcreteOp *>(this)).getNumOperands() - (*static_cast<ConcreteOp *>(this)).getNumDpsInits();
    }
    /// Return the input operands.
    ::mlir::OpOperandVector getDpsInputOperands() {
      auto [start, end] = (*static_cast<ConcreteOp *>(this)).getDpsInitsPositionRange();
        int64_t numInits = end - start;
        int64_t numOperands = (*static_cast<ConcreteOp *>(this)).getNumOperands();

        ::mlir::OpOperandVector result;
        result.reserve(numOperands - numInits);
        for (int i = 0; i < start; ++i)
          result.push_back(&(*static_cast<ConcreteOp *>(this))->getOpOperand(i));
        for (int i = end; i < numOperands; ++i)
          result.push_back(&(*static_cast<ConcreteOp *>(this))->getOpOperand(end + i));

        return result;
    }
    /// Return the `i`-th input operand.
    ::mlir::OpOperand *getDpsInputOperand(int64_t i) {
      assert(i >= 0 && i < getNumDpsInputs());
        auto [start, end] = (*static_cast<ConcreteOp *>(this)).getDpsInitsPositionRange();
        return &(*static_cast<ConcreteOp *>(this))->getOpOperand(i < start ? i : i + end - start) ;
    }
    /// Return true if `opOperand` is an input.
    bool isDpsInput(::mlir::OpOperand * opOperand) {
      auto [start, end] = (*static_cast<ConcreteOp *>(this)).getDpsInitsPositionRange();
        auto operandNumber = opOperand->getOperandNumber();
        return operandNumber < start || operandNumber >= end;
    }
    /// Return true if `opOperand` is an init.
    bool isDpsInit(::mlir::OpOperand * opOperand) {
      auto [start, end] = (*static_cast<ConcreteOp *>(this)).getDpsInitsPositionRange();
        auto operandNumber = opOperand->getOperandNumber();
        return operandNumber >= start && operandNumber < end;
    }
    /// Return true if the `opOperand` is a scalar value.
    bool isScalar(::mlir::OpOperand * opOperand) {
      assert(opOperand->getOwner() == (*static_cast<ConcreteOp *>(this)).getOperation());
        return !opOperand->get().getType().template isa<ShapedType>();
    }
    /// Return the OpResult that is tied to the given OpOperand.
    ::mlir::OpResult getTiedOpResult(::mlir::OpOperand * opOperand) {
      assert(opOperand->getOwner() == (*static_cast<ConcreteOp *>(this)).getOperation());

        auto [start, end] = (*static_cast<ConcreteOp *>(this)).getDpsInitsPositionRange();
        int64_t resultIndex = opOperand->getOperandNumber() - start;
        assert(resultIndex >= 0 &&
               resultIndex < (*static_cast<ConcreteOp *>(this))->getNumResults() );
        return (*static_cast<ConcreteOp *>(this))->getResult(resultIndex);
    }
    /// Return the OpOperand that is tied to the given OpResult.
    ::mlir::OpOperand *getTiedOpOperand(::mlir::OpResult opResult) {
      assert(opResult.getDefiningOp() == (*static_cast<ConcreteOp *>(this)).getOperation());
        return (*static_cast<ConcreteOp *>(this)).getDpsInitOperand(opResult.getResultNumber());
    }
    /// Return whether the op has only ranked MemRef input/inits.
    bool hasBufferSemantics() {
      return (*static_cast<ConcreteOp *>(this))->getNumResults() == 0 &&
          ::llvm::all_of((*static_cast<ConcreteOp *>(this))->getOpOperands(),
            [&](::mlir::OpOperand &opOperand) {
              return isScalar(&opOperand) ||
                     opOperand.get().getType().template isa<::mlir::MemRefType>();
            });
    }
    /// Return whether the op has only ranked tensor inputs/inits.
    bool hasTensorSemantics() {
      return ::llvm::all_of((*static_cast<ConcreteOp *>(this))->getOpOperands(),
          [&](::mlir::OpOperand &opOperand) {
            return isScalar(&opOperand) ||
                   opOperand.get().getType().template isa<::mlir::RankedTensorType>();
          });
    }
    static ::mlir::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      return detail::verifyDestinationStyleOpInterface(op);
    }
  };
}// namespace detail
} // namespace mlir
namespace mlir {
template<typename ConcreteOp>
std::pair<int64_t, int64_t> detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDpsInitsPositionRange(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitsPositionRange();
}
template<typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getNumDpsInits(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumDpsInits();
}
template<typename ConcreteOp>
::mlir::OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDpsInitOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitOperands();
}
template<typename ConcreteOp>
::mlir::OpOperand *detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDpsInitOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitOperand(i);
}
template<typename ConcreteOp>
void detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::setDpsInitOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i, ::mlir::Value value) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setDpsInitOperand(i, value);
}
template<typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getNumDpsInputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumDpsInputs();
}
template<typename ConcreteOp>
::mlir::OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDpsInputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInputOperands();
}
template<typename ConcreteOp>
::mlir::OpOperand *detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getDpsInputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInputOperand(i);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDpsInput(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDpsInput(opOperand);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::isDpsInit(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isDpsInit(opOperand);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::isScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isScalar(opOperand);
}
template<typename ConcreteOp>
::mlir::OpResult detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTiedOpResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTiedOpResult(opOperand);
}
template<typename ConcreteOp>
::mlir::OpOperand *detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTiedOpOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpResult opResult) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTiedOpOperand(opResult);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::hasBufferSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasBufferSemantics();
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::hasTensorSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasTensorSemantics();
}
template<typename ConcreteOp>
std::pair<int64_t, int64_t> detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDpsInitsPositionRange(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDpsInitsPositionRange(tablegen_opaque_val);
}
template<typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getNumDpsInits(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumDpsInits(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDpsInitOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDpsInitOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OpOperand *detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDpsInitOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i) {
  return static_cast<const ConcreteOp *>(impl)->getDpsInitOperand(tablegen_opaque_val, i);
}
template<typename ConcreteOp>
void detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setDpsInitOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i, ::mlir::Value value) {
  return static_cast<const ConcreteOp *>(impl)->setDpsInitOperand(tablegen_opaque_val, i, value);
}
template<typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getNumDpsInputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumDpsInputs(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDpsInputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getDpsInputOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
::mlir::OpOperand *detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getDpsInputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i) {
  return static_cast<const ConcreteOp *>(impl)->getDpsInputOperand(tablegen_opaque_val, i);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isDpsInput(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->isDpsInput(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isDpsInit(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->isDpsInit(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->isScalar(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
::mlir::OpResult detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTiedOpResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getTiedOpResult(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
::mlir::OpOperand *detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTiedOpOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ::mlir::OpResult opResult) {
  return static_cast<const ConcreteOp *>(impl)->getTiedOpOperand(tablegen_opaque_val, opResult);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::hasBufferSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasBufferSemantics(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::hasTensorSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasTensorSemantics(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumDpsInits(::mlir::Operation *tablegen_opaque_val) const {
auto [start, end] = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitsPositionRange();
        return end - start;
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDpsInitOperands(::mlir::Operation *tablegen_opaque_val) const {
auto [start, end] = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitsPositionRange();

        ::mlir::OpOperandVector result;
        result.reserve(end - start);
        for (int i = start; i < end; ++i)
          result.push_back(&(llvm::cast<ConcreteOp>(tablegen_opaque_val))->getOpOperand(i));
        return result;
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OpOperand *detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDpsInitOperand(::mlir::Operation *tablegen_opaque_val, int64_t i) const {
assert(i >= 0 && i < (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumDpsInits());
        auto [start, end] = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitsPositionRange();
        return &(llvm::cast<ConcreteOp>(tablegen_opaque_val))->getOpOperand(start + i);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setDpsInitOperand(::mlir::Operation *tablegen_opaque_val, int64_t i, ::mlir::Value value) const {
assert(i >= 0 && i < (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumDpsInits());
        auto [start, end] = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitsPositionRange();
        (llvm::cast<ConcreteOp>(tablegen_opaque_val))->setOperand(start + i, value);
}
template<typename ConcreteModel, typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumDpsInputs(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumOperands() - (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumDpsInits();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDpsInputOperands(::mlir::Operation *tablegen_opaque_val) const {
auto [start, end] = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitsPositionRange();
        int64_t numInits = end - start;
        int64_t numOperands = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumOperands();

        ::mlir::OpOperandVector result;
        result.reserve(numOperands - numInits);
        for (int i = 0; i < start; ++i)
          result.push_back(&(llvm::cast<ConcreteOp>(tablegen_opaque_val))->getOpOperand(i));
        for (int i = end; i < numOperands; ++i)
          result.push_back(&(llvm::cast<ConcreteOp>(tablegen_opaque_val))->getOpOperand(end + i));

        return result;
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OpOperand *detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getDpsInputOperand(::mlir::Operation *tablegen_opaque_val, int64_t i) const {
assert(i >= 0 && i < getNumDpsInputs());
        auto [start, end] = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitsPositionRange();
        return &(llvm::cast<ConcreteOp>(tablegen_opaque_val))->getOpOperand(i < start ? i : i + end - start) ;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isDpsInput(::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand *opOperand) const {
auto [start, end] = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitsPositionRange();
        auto operandNumber = opOperand->getOperandNumber();
        return operandNumber < start || operandNumber >= end;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isDpsInit(::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand *opOperand) const {
auto [start, end] = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitsPositionRange();
        auto operandNumber = opOperand->getOperandNumber();
        return operandNumber >= start && operandNumber < end;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isScalar(::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand *opOperand) const {
assert(opOperand->getOwner() == (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation());
        return !opOperand->get().getType().template isa<ShapedType>();
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OpResult detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiedOpResult(::mlir::Operation *tablegen_opaque_val, ::mlir::OpOperand *opOperand) const {
assert(opOperand->getOwner() == (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation());

        auto [start, end] = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitsPositionRange();
        int64_t resultIndex = opOperand->getOperandNumber() - start;
        assert(resultIndex >= 0 &&
               resultIndex < (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getNumResults() );
        return (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getResult(resultIndex);
}
template<typename ConcreteModel, typename ConcreteOp>
::mlir::OpOperand *detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiedOpOperand(::mlir::Operation *tablegen_opaque_val, ::mlir::OpResult opResult) const {
assert(opResult.getDefiningOp() == (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation());
        return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getDpsInitOperand(opResult.getResultNumber());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasBufferSemantics(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getNumResults() == 0 &&
          ::llvm::all_of((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getOpOperands(),
            [&](::mlir::OpOperand &opOperand) {
              return isScalar(&opOperand) ||
                     opOperand.get().getType().template isa<::mlir::MemRefType>();
            });
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasTensorSemantics(::mlir::Operation *tablegen_opaque_val) const {
return ::llvm::all_of((llvm::cast<ConcreteOp>(tablegen_opaque_val))->getOpOperands(),
          [&](::mlir::OpOperand &opOperand) {
            return isScalar(&opOperand) ||
                   opOperand.get().getType().template isa<::mlir::RankedTensorType>();
          });
}
} // namespace mlir
