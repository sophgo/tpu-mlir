/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace linalg {
class DestinationStyleOpInterface;
namespace detail {
struct DestinationStyleOpInterfaceInterfaceTraits {
  struct Concept {
    ValueRange (*inputs)(const Concept *impl, ::mlir::Operation *);
    int64_t (*getNumInputs)(const Concept *impl, ::mlir::Operation *);
    ValueRange (*outputs)(const Concept *impl, ::mlir::Operation *);
    int64_t (*getNumOutputs)(const Concept *impl, ::mlir::Operation *);
    int64_t (*getNumInputsAndOutputs)(const Concept *impl, ::mlir::Operation *);
    OpOperandVector (*getInputOperands)(const Concept *impl, ::mlir::Operation *);
    OpOperand*(*getInputOperand)(const Concept *impl, ::mlir::Operation *, int64_t);
    OpOperandVector (*getInputBufferOperands)(const Concept *impl, ::mlir::Operation *);
    OpOperandVector (*getInputTensorOperands)(const Concept *impl, ::mlir::Operation *);
    OpOperandVector (*getOutputOperands)(const Concept *impl, ::mlir::Operation *);
    OpOperand*(*getOutputOperand)(const Concept *impl, ::mlir::Operation *, int64_t);
    void (*setOutputOperand)(const Concept *impl, ::mlir::Operation *, int64_t, Value);
    OpOperandVector (*getOutputBufferOperands)(const Concept *impl, ::mlir::Operation *);
    OpOperandVector (*getOutputTensorOperands)(const Concept *impl, ::mlir::Operation *);
    SmallVector<MemRefType> (*getOutputBufferTypes)(const Concept *impl, ::mlir::Operation *);
    SmallVector<RankedTensorType> (*getOutputTensorTypes)(const Concept *impl, ::mlir::Operation *);
    OpOperandVector (*getInputAndOutputOperands)(const Concept *impl, ::mlir::Operation *);
    bool (*isInputTensor)(const Concept *impl, ::mlir::Operation *, OpOperand *);
    bool (*isOutputTensor)(const Concept *impl, ::mlir::Operation *, OpOperand *);
    bool (*isScalar)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    OpResult (*getTiedOpResult)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    bool (*hasBufferSemantics)(const Concept *impl, ::mlir::Operation *);
    bool (*hasTensorSemantics)(const Concept *impl, ::mlir::Operation *);
    Operation *(*clone)(const Concept *impl, ::mlir::Operation *, OpBuilder &, Location, TypeRange, ValueRange);
    Operation *(*cloneWithoutRegions)(const Concept *impl, ::mlir::Operation *, OpBuilder &, Location, TypeRange, ValueRange);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::linalg::DestinationStyleOpInterface;
    Model() : Concept{inputs, getNumInputs, outputs, getNumOutputs, getNumInputsAndOutputs, getInputOperands, getInputOperand, getInputBufferOperands, getInputTensorOperands, getOutputOperands, getOutputOperand, setOutputOperand, getOutputBufferOperands, getOutputTensorOperands, getOutputBufferTypes, getOutputTensorTypes, getInputAndOutputOperands, isInputTensor, isOutputTensor, isScalar, getTiedOpResult, hasBufferSemantics, hasTensorSemantics, clone, cloneWithoutRegions} {}

    static inline ValueRange inputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline int64_t getNumInputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ValueRange outputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline int64_t getNumOutputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline int64_t getNumInputsAndOutputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperandVector getInputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperand*getInputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i);
    static inline OpOperandVector getInputBufferOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperandVector getInputTensorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperandVector getOutputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperand*getOutputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i);
    static inline void setOutputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i, Value value);
    static inline OpOperandVector getOutputBufferOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperandVector getOutputTensorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<MemRefType> getOutputBufferTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<RankedTensorType> getOutputTensorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperandVector getInputAndOutputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isInputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isOutputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline OpResult getTiedOpResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline bool hasBufferSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasTensorSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Operation *clone(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
    static inline Operation *cloneWithoutRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::linalg::DestinationStyleOpInterface;
    FallbackModel() : Concept{inputs, getNumInputs, outputs, getNumOutputs, getNumInputsAndOutputs, getInputOperands, getInputOperand, getInputBufferOperands, getInputTensorOperands, getOutputOperands, getOutputOperand, setOutputOperand, getOutputBufferOperands, getOutputTensorOperands, getOutputBufferTypes, getOutputTensorTypes, getInputAndOutputOperands, isInputTensor, isOutputTensor, isScalar, getTiedOpResult, hasBufferSemantics, hasTensorSemantics, clone, cloneWithoutRegions} {}

    static inline ValueRange inputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline int64_t getNumInputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ValueRange outputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline int64_t getNumOutputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline int64_t getNumInputsAndOutputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperandVector getInputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperand*getInputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i);
    static inline OpOperandVector getInputBufferOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperandVector getInputTensorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperandVector getOutputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperand*getOutputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i);
    static inline void setOutputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i, Value value);
    static inline OpOperandVector getOutputBufferOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperandVector getOutputTensorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<MemRefType> getOutputBufferTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<RankedTensorType> getOutputTensorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline OpOperandVector getInputAndOutputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isInputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isOutputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline OpResult getTiedOpResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline bool hasBufferSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasTensorSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Operation *clone(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
    static inline Operation *cloneWithoutRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    int64_t getNumInputs(::mlir::Operation *tablegen_opaque_val) const;
    int64_t getNumOutputs(::mlir::Operation *tablegen_opaque_val) const;
    int64_t getNumInputsAndOutputs(::mlir::Operation *tablegen_opaque_val) const;
    OpOperandVector getInputOperands(::mlir::Operation *tablegen_opaque_val) const;
    OpOperand*getInputOperand(::mlir::Operation *tablegen_opaque_val, int64_t i) const;
    OpOperandVector getInputBufferOperands(::mlir::Operation *tablegen_opaque_val) const;
    OpOperandVector getInputTensorOperands(::mlir::Operation *tablegen_opaque_val) const;
    OpOperandVector getOutputOperands(::mlir::Operation *tablegen_opaque_val) const;
    OpOperand*getOutputOperand(::mlir::Operation *tablegen_opaque_val, int64_t i) const;
    void setOutputOperand(::mlir::Operation *tablegen_opaque_val, int64_t i, Value value) const;
    OpOperandVector getOutputBufferOperands(::mlir::Operation *tablegen_opaque_val) const;
    OpOperandVector getOutputTensorOperands(::mlir::Operation *tablegen_opaque_val) const;
    SmallVector<MemRefType> getOutputBufferTypes(::mlir::Operation *tablegen_opaque_val) const;
    SmallVector<RankedTensorType> getOutputTensorTypes(::mlir::Operation *tablegen_opaque_val) const;
    OpOperandVector getInputAndOutputOperands(::mlir::Operation *tablegen_opaque_val) const;
    bool isInputTensor(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const;
    bool isOutputTensor(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const;
    bool isScalar(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    OpResult getTiedOpResult(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
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
  /// Return the input shape operands.
  ValueRange inputs();
  /// Return the number of inputs.
  int64_t getNumInputs();
  /// Return the output shape operands.
  ValueRange outputs();
  /// Return the number of outputs.
  int64_t getNumOutputs();
  /// Return the number of inputs and outputs.
  int64_t getNumInputsAndOutputs();
  /// Return the input operands.
  OpOperandVector getInputOperands();
  /// Return the `i`-th input operand.
  OpOperand*getInputOperand(int64_t i);
  /// Return the subset of input operands that are of buffer type.
  OpOperandVector getInputBufferOperands();
  /// Return the subset of input operands that are of tensor type.
  OpOperandVector getInputTensorOperands();
  /// Return the output operands.
  OpOperandVector getOutputOperands();
  /// Return the `i`-th output operand.
  OpOperand*getOutputOperand(int64_t i);
  /// Set the `i`-th output operand.
  void setOutputOperand(int64_t i, Value value);
  /// Return the subset of output operands that are of buffer type.
  OpOperandVector getOutputBufferOperands();
  /// Return the subset of output operands that are of tensor type.
  OpOperandVector getOutputTensorOperands();
  /// Return the types of the subset of output operands that are of buffer type.
  SmallVector<MemRefType> getOutputBufferTypes();
  /// Return the types of the subset of output operands that are of tensor type.
  SmallVector<RankedTensorType> getOutputTensorTypes();
  /// Return the range over input and output operands.
  OpOperandVector getInputAndOutputOperands();
  /// Return true if `opOperand` is an input tensor.
  bool isInputTensor(OpOperand * opOperand);
  /// Return true if `opOperand` is an output tensor.
  bool isOutputTensor(OpOperand * opOperand);
  /// Return true if the `opOperand` is a scalar value.
  bool isScalar(OpOperand* opOperand);
  /// Return the result tied to `opOperand`.
  OpResult getTiedOpResult(OpOperand* opOperand);
  /// Return whether the op has only MemRef input and outputs.
  bool hasBufferSemantics();
  /// Return whether the op has only RankedTensor input and outputs.
  bool hasTensorSemantics();
  /// Clone the current operation with the given location and operands. This
  /// is used to abstract away the optional underlying region creation. This
  /// does not change the balance between input, output_buffer and
  /// init_tensors operands.
  Operation *clone(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
  /// Clone the current operation with the given location, operands
  /// and BlockAndValueMapping but leave the regions empty. This is
  /// used to abstract away the optional underlying region creation.
  /// This does not change the balance between input, output_buffer
  /// and init_tensors operands.
  Operation *cloneWithoutRegions(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
};
namespace detail {
  template <typename ConcreteOp>
  struct DestinationStyleOpInterfaceTrait : public ::mlir::OpInterface<DestinationStyleOpInterface, detail::DestinationStyleOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Return the number of inputs.
    int64_t getNumInputs() {
      return (*static_cast<ConcreteOp *>(this)).getInputs().size();
    }
    /// Return the number of outputs.
    int64_t getNumOutputs() {
      return (*static_cast<ConcreteOp *>(this)).outputs().size();
    }
    /// Return the number of inputs and outputs.
    int64_t getNumInputsAndOutputs() {
      return this->getOperation()->getNumOperands();
    }
    /// Return the input operands.
    OpOperandVector getInputOperands() {
      int64_t numInputs = getNumInputs();
        OpOperandVector result;
        result.reserve(numInputs);
        llvm::transform(
          this->getOperation()->getOpOperands().take_front(numInputs),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
    }
    /// Return the `i`-th input operand.
    OpOperand*getInputOperand(int64_t i) {
      assert(i >= 0 && i < getNumInputs());
        return &this->getOperation()->getOpOperand(i);
    }
    /// Return the subset of input operands that are of buffer type.
    OpOperandVector getInputBufferOperands() {
      OpOperandVector result;
        result.reserve(getNumInputs());
        llvm::copy_if(getInputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<MemRefType>();
          });
        return result;
    }
    /// Return the subset of input operands that are of tensor type.
    OpOperandVector getInputTensorOperands() {
      OpOperandVector result;
        result.reserve(getNumInputs());
        llvm::copy_if(getInputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<RankedTensorType>();
          });
        return result;
    }
    /// Return the output operands.
    OpOperandVector getOutputOperands() {
      int64_t numOutputs = getNumOutputs();
        OpOperandVector result;
        result.reserve(numOutputs);
        llvm::transform(
          this->getOperation()->getOpOperands()
            .take_back(numOutputs),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
    }
    /// Return the `i`-th output operand.
    OpOperand*getOutputOperand(int64_t i) {
      assert(i >= 0 && i < getNumOutputs());
        return &this->getOperation()->getOpOperand(getNumInputs() + i);
    }
    /// Set the `i`-th output operand.
    void setOutputOperand(int64_t i, Value value) {
      assert(i >= 0 && i < getNumOutputs());
        this->getOperation()->setOperand(getNumInputs() + i, value);
    }
    /// Return the subset of output operands that are of buffer type.
    OpOperandVector getOutputBufferOperands() {
      OpOperandVector result;
        result.reserve(getNumOutputs());
        llvm::copy_if(getOutputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<MemRefType>();
          });
        return result;
    }
    /// Return the subset of output operands that are of tensor type.
    OpOperandVector getOutputTensorOperands() {
      OpOperandVector result;
        result.reserve(getNumOutputs());
        llvm::copy_if(getOutputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<RankedTensorType>();
          });
        return result;
    }
    /// Return the types of the subset of output operands that are of buffer type.
    SmallVector<MemRefType> getOutputBufferTypes() {
      SmallVector<MemRefType> result;
        result.reserve(getNumOutputs());
        llvm::transform(getOutputBufferOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperands) {
            return opOperands->get().getType().cast<MemRefType>();
          });
        return result;
    }
    /// Return the types of the subset of output operands that are of tensor type.
    SmallVector<RankedTensorType> getOutputTensorTypes() {
      SmallVector<RankedTensorType> result;
        result.reserve(getNumOutputs());
        llvm::transform(getOutputTensorOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperands) {
            return opOperands->get().getType().cast<RankedTensorType>();
          });
        return result;
    }
    /// Return the range over input and output operands.
    OpOperandVector getInputAndOutputOperands() {
      int64_t numInputsAndOutputs = getNumInputsAndOutputs();
        OpOperandVector result;
        result.reserve(numInputsAndOutputs);
        llvm::transform(
          this->getOperation()->getOpOperands(),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
    }
    /// Return true if `opOperand` is an input tensor.
    bool isInputTensor(OpOperand * opOperand) {
      if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() < (*static_cast<ConcreteOp *>(this)).getNumInputs())
          return true;
        return false;
    }
    /// Return true if `opOperand` is an output tensor.
    bool isOutputTensor(OpOperand * opOperand) {
      if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() >= (*static_cast<ConcreteOp *>(this)).getNumInputs())
          return true;
        return false;
    }
    /// Return true if the `opOperand` is a scalar value.
    bool isScalar(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        return !opOperand->get().getType().template isa<ShapedType>();
    }
    /// Return the result tied to `opOperand`.
    OpResult getTiedOpResult(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        int64_t resultIndex = opOperand->getOperandNumber() - getNumInputs();
        assert(resultIndex >= 0 &&
               resultIndex < this->getOperation()->getNumResults() );
        return this->getOperation()->getResult(resultIndex);
    }
    /// Return whether the op has only MemRef input and outputs.
    bool hasBufferSemantics() {
      return this->getOperation()->getNumResults() == 0 &&
          llvm::all_of(this->getOperation()->getOpOperands(),
            [&](OpOperand &opOperand) {
              return isScalar(&opOperand) ||
                     opOperand.get().getType().template isa<MemRefType>();
            });
    }
    /// Return whether the op has only RankedTensor input and outputs.
    bool hasTensorSemantics() {
      return llvm::all_of(this->getOperation()->getOpOperands(),
          [&](OpOperand &opOperand) {
            return isScalar(&opOperand) ||
                   opOperand.get().getType().template isa<RankedTensorType>();
          });
    }
    static ::mlir::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      return detail::verifyDestinationStyleOpInterface(op);
    }
  };
}// namespace detail
template<typename ConcreteOp>
ValueRange detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::inputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inputs();
}
template<typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getNumInputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInputs();
}
template<typename ConcreteOp>
ValueRange detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::outputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).outputs();
}
template<typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getNumOutputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumOutputs();
}
template<typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getNumInputsAndOutputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInputsAndOutputs();
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getInputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputOperands();
}
template<typename ConcreteOp>
OpOperand*detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getInputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputOperand(i);
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getInputBufferOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputBufferOperands();
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getInputTensorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputTensorOperands();
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getOutputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputOperands();
}
template<typename ConcreteOp>
OpOperand*detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getOutputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputOperand(i);
}
template<typename ConcreteOp>
void detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::setOutputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i, Value value) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setOutputOperand(i, value);
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getOutputBufferOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputBufferOperands();
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getOutputTensorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputTensorOperands();
}
template<typename ConcreteOp>
SmallVector<MemRefType> detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getOutputBufferTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputBufferTypes();
}
template<typename ConcreteOp>
SmallVector<RankedTensorType> detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getOutputTensorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputTensorTypes();
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getInputAndOutputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputAndOutputOperands();
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::isInputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isInputTensor(opOperand);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::isOutputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isOutputTensor(opOperand);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::isScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isScalar(opOperand);
}
template<typename ConcreteOp>
OpResult detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::getTiedOpResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTiedOpResult(opOperand);
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
Operation *detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::clone(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
  BlockAndValueMapping bvm;
        OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttrs());
        for (Region &r : (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getRegions())
          r.cloneInto(state.addRegion(), bvm);
        return b.create(state);
}
template<typename ConcreteOp>
Operation *detail::DestinationStyleOpInterfaceInterfaceTraits::Model<ConcreteOp>::cloneWithoutRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
  OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttrs());
        for (size_t cnt = 0, e = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getNumRegions(); cnt < e; ++cnt)
          state.addRegion();
        return b.create(state);
}
template<typename ConcreteOp>
ValueRange detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::inputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->inputs(tablegen_opaque_val);
}
template<typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getNumInputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumInputs(tablegen_opaque_val);
}
template<typename ConcreteOp>
ValueRange detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::outputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->outputs(tablegen_opaque_val);
}
template<typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getNumOutputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumOutputs(tablegen_opaque_val);
}
template<typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getNumInputsAndOutputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumInputsAndOutputs(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getInputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getInputOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperand*detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getInputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i) {
  return static_cast<const ConcreteOp *>(impl)->getInputOperand(tablegen_opaque_val, i);
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getInputBufferOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getInputBufferOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getInputTensorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getInputTensorOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getOutputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOutputOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperand*detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getOutputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i) {
  return static_cast<const ConcreteOp *>(impl)->getOutputOperand(tablegen_opaque_val, i);
}
template<typename ConcreteOp>
void detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::setOutputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i, Value value) {
  return static_cast<const ConcreteOp *>(impl)->setOutputOperand(tablegen_opaque_val, i, value);
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getOutputBufferOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOutputBufferOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getOutputTensorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOutputTensorOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<MemRefType> detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getOutputBufferTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOutputBufferTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<RankedTensorType> detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getOutputTensorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOutputTensorTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getInputAndOutputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getInputAndOutputOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isInputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->isInputTensor(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isOutputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->isOutputTensor(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return static_cast<const ConcreteOp *>(impl)->isScalar(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
OpResult detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::getTiedOpResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getTiedOpResult(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::hasBufferSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasBufferSemantics(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::hasTensorSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasTensorSemantics(tablegen_opaque_val);
}
template<typename ConcreteOp>
Operation *detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::clone(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
  return static_cast<const ConcreteOp *>(impl)->clone(tablegen_opaque_val, b, loc, resultTypes, operands);
}
template<typename ConcreteOp>
Operation *detail::DestinationStyleOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::cloneWithoutRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
  return static_cast<const ConcreteOp *>(impl)->cloneWithoutRegions(tablegen_opaque_val, b, loc, resultTypes, operands);
}
template<typename ConcreteModel, typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumInputs(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputs().size();
}
template<typename ConcreteModel, typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumOutputs(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).outputs().size();
}
template<typename ConcreteModel, typename ConcreteOp>
int64_t detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumInputsAndOutputs(::mlir::Operation *tablegen_opaque_val) const {
return this->getOperation()->getNumOperands();
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getInputOperands(::mlir::Operation *tablegen_opaque_val) const {
int64_t numInputs = getNumInputs();
        OpOperandVector result;
        result.reserve(numInputs);
        llvm::transform(
          this->getOperation()->getOpOperands().take_front(numInputs),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperand*detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getInputOperand(::mlir::Operation *tablegen_opaque_val, int64_t i) const {
assert(i >= 0 && i < getNumInputs());
        return &this->getOperation()->getOpOperand(i);
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getInputBufferOperands(::mlir::Operation *tablegen_opaque_val) const {
OpOperandVector result;
        result.reserve(getNumInputs());
        llvm::copy_if(getInputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<MemRefType>();
          });
        return result;
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getInputTensorOperands(::mlir::Operation *tablegen_opaque_val) const {
OpOperandVector result;
        result.reserve(getNumInputs());
        llvm::copy_if(getInputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<RankedTensorType>();
          });
        return result;
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOutputOperands(::mlir::Operation *tablegen_opaque_val) const {
int64_t numOutputs = getNumOutputs();
        OpOperandVector result;
        result.reserve(numOutputs);
        llvm::transform(
          this->getOperation()->getOpOperands()
            .take_back(numOutputs),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperand*detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOutputOperand(::mlir::Operation *tablegen_opaque_val, int64_t i) const {
assert(i >= 0 && i < getNumOutputs());
        return &this->getOperation()->getOpOperand(getNumInputs() + i);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setOutputOperand(::mlir::Operation *tablegen_opaque_val, int64_t i, Value value) const {
assert(i >= 0 && i < getNumOutputs());
        this->getOperation()->setOperand(getNumInputs() + i, value);
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOutputBufferOperands(::mlir::Operation *tablegen_opaque_val) const {
OpOperandVector result;
        result.reserve(getNumOutputs());
        llvm::copy_if(getOutputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<MemRefType>();
          });
        return result;
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOutputTensorOperands(::mlir::Operation *tablegen_opaque_val) const {
OpOperandVector result;
        result.reserve(getNumOutputs());
        llvm::copy_if(getOutputOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperand) {
            return opOperand->get().getType().template isa<RankedTensorType>();
          });
        return result;
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<MemRefType> detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOutputBufferTypes(::mlir::Operation *tablegen_opaque_val) const {
SmallVector<MemRefType> result;
        result.reserve(getNumOutputs());
        llvm::transform(getOutputBufferOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperands) {
            return opOperands->get().getType().cast<MemRefType>();
          });
        return result;
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<RankedTensorType> detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOutputTensorTypes(::mlir::Operation *tablegen_opaque_val) const {
SmallVector<RankedTensorType> result;
        result.reserve(getNumOutputs());
        llvm::transform(getOutputTensorOperands(),
          std::back_inserter(result),
          [](OpOperand *opOperands) {
            return opOperands->get().getType().cast<RankedTensorType>();
          });
        return result;
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperandVector detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getInputAndOutputOperands(::mlir::Operation *tablegen_opaque_val) const {
int64_t numInputsAndOutputs = getNumInputsAndOutputs();
        OpOperandVector result;
        result.reserve(numInputsAndOutputs);
        llvm::transform(
          this->getOperation()->getOpOperands(),
          std::back_inserter(result),
          [](OpOperand &opOperand) { return &opOperand; });
        return result;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isInputTensor(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const {
if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() < (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInputs())
          return true;
        return false;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isOutputTensor(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const {
if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() >= (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInputs())
          return true;
        return false;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isScalar(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        return !opOperand->get().getType().template isa<ShapedType>();
}
template<typename ConcreteModel, typename ConcreteOp>
OpResult detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiedOpResult(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        int64_t resultIndex = opOperand->getOperandNumber() - getNumInputs();
        assert(resultIndex >= 0 &&
               resultIndex < this->getOperation()->getNumResults() );
        return this->getOperation()->getResult(resultIndex);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasBufferSemantics(::mlir::Operation *tablegen_opaque_val) const {
return this->getOperation()->getNumResults() == 0 &&
          llvm::all_of(this->getOperation()->getOpOperands(),
            [&](OpOperand &opOperand) {
              return isScalar(&opOperand) ||
                     opOperand.get().getType().template isa<MemRefType>();
            });
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::DestinationStyleOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasTensorSemantics(::mlir::Operation *tablegen_opaque_val) const {
return llvm::all_of(this->getOperation()->getOpOperands(),
          [&](OpOperand &opOperand) {
            return isScalar(&opOperand) ||
                   opOperand.get().getType().template isa<RankedTensorType>();
          });
}
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ContractionOpInterface;
namespace detail {
struct ContractionOpInterfaceInterfaceTraits {
  struct Concept {
    Value (*lhs)(const Concept *impl, ::mlir::Operation *);
    Value (*rhs)(const Concept *impl, ::mlir::Operation *);
    bool (*isRowMajorMatmul)(const Concept *impl, ::mlir::Operation *);
    bool (*isColumnMajorMatmul)(const Concept *impl, ::mlir::Operation *);
    bool (*isRowMajorBatchMatmul)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::linalg::ContractionOpInterface;
    Model() : Concept{lhs, rhs, isRowMajorMatmul, isColumnMajorMatmul, isRowMajorBatchMatmul} {}

    static inline Value lhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value rhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isRowMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isColumnMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isRowMajorBatchMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::linalg::ContractionOpInterface;
    FallbackModel() : Concept{lhs, rhs, isRowMajorMatmul, isColumnMajorMatmul, isRowMajorBatchMatmul} {}

    static inline Value lhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value rhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isRowMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isColumnMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool isRowMajorBatchMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
  };
};template <typename ConcreteOp>
struct ContractionOpInterfaceTrait;

} // namespace detail
class ContractionOpInterface : public ::mlir::OpInterface<ContractionOpInterface, detail::ContractionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ContractionOpInterface, detail::ContractionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ContractionOpInterfaceTrait<ConcreteOp> {};
  /// Returns the left-hand side operand.
  Value lhs();
  /// Returns the right-hand side operand.
  Value rhs();
  /// Returns whether the given op has indexing maps that correspond to a
  /// row-major matmul operation.
  bool isRowMajorMatmul();
  /// Returns whether the given op has indexing maps that correspond to a
  /// column-major matmul operation.
  bool isColumnMajorMatmul();
  /// Returns whether the given op has indexing maps that correspond to a
  /// row-major batch matmul operation.
  bool isRowMajorBatchMatmul();
};
namespace detail {
  template <typename ConcreteOp>
  struct ContractionOpInterfaceTrait : public ::mlir::OpInterface<ContractionOpInterface, detail::ContractionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    static ::mlir::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      return detail::verifyContractionInterface(op);
    }
  };
}// namespace detail
template<typename ConcreteOp>
Value detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::lhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getOperand(0);
}
template<typename ConcreteOp>
Value detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::rhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getOperand(1);
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isRowMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return mlir::isRowMajorMatmul((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps());
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isColumnMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return mlir::isColumnMajorMatmul((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps());
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isRowMajorBatchMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return mlir::isRowMajorBatchMatmul((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps());
}
template<typename ConcreteOp>
Value detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::lhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->lhs(tablegen_opaque_val);
}
template<typename ConcreteOp>
Value detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::rhs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->rhs(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isRowMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isRowMajorMatmul(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isColumnMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isColumnMajorMatmul(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::isRowMajorBatchMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->isRowMajorBatchMatmul(tablegen_opaque_val);
}
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ConvolutionOpInterface;
namespace detail {
struct ConvolutionOpInterfaceInterfaceTraits {
  struct Concept {
    Value (*image)(const Concept *impl, ::mlir::Operation *);
    Value (*filter)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::linalg::ConvolutionOpInterface;
    Model() : Concept{image, filter} {}

    static inline Value image(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value filter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::linalg::ConvolutionOpInterface;
    FallbackModel() : Concept{image, filter} {}

    static inline Value image(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value filter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    Value image(::mlir::Operation *tablegen_opaque_val) const;
    Value filter(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct ConvolutionOpInterfaceTrait;

} // namespace detail
class ConvolutionOpInterface : public ::mlir::OpInterface<ConvolutionOpInterface, detail::ConvolutionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ConvolutionOpInterface, detail::ConvolutionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ConvolutionOpInterfaceTrait<ConcreteOp> {};
  /// Return the image operand.
  Value image();
  /// Return the filter operand.
  Value filter();
};
namespace detail {
  template <typename ConcreteOp>
  struct ConvolutionOpInterfaceTrait : public ::mlir::OpInterface<ConvolutionOpInterface, detail::ConvolutionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Return the image operand.
    Value image() {
      return (*static_cast<ConcreteOp *>(this)).getOperation()->getOperand(0);
    }
    /// Return the filter operand.
    Value filter() {
      return (*static_cast<ConcreteOp *>(this)).getOperation()->getOperand(1);
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return detail::verifyConvolutionInterface(op);
    }
  };
}// namespace detail
template<typename ConcreteOp>
Value detail::ConvolutionOpInterfaceInterfaceTraits::Model<ConcreteOp>::image(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).image();
}
template<typename ConcreteOp>
Value detail::ConvolutionOpInterfaceInterfaceTraits::Model<ConcreteOp>::filter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).filter();
}
template<typename ConcreteOp>
Value detail::ConvolutionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::image(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->image(tablegen_opaque_val);
}
template<typename ConcreteOp>
Value detail::ConvolutionOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::filter(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->filter(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::ConvolutionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::image(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getOperand(0);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::ConvolutionOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::filter(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getOperand(1);
}
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class FillOpInterface;
namespace detail {
struct FillOpInterfaceInterfaceTraits {
  struct Concept {
    Value (*value)(const Concept *impl, ::mlir::Operation *);
    Value (*output)(const Concept *impl, ::mlir::Operation *);
    Value (*result)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::linalg::FillOpInterface;
    Model() : Concept{value, output, result} {}

    static inline Value value(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value output(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value result(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::linalg::FillOpInterface;
    FallbackModel() : Concept{value, output, result} {}

    static inline Value value(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value output(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Value result(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    Value value(::mlir::Operation *tablegen_opaque_val) const;
    Value output(::mlir::Operation *tablegen_opaque_val) const;
    Value result(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct FillOpInterfaceTrait;

} // namespace detail
class FillOpInterface : public ::mlir::OpInterface<FillOpInterface, detail::FillOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<FillOpInterface, detail::FillOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::FillOpInterfaceTrait<ConcreteOp> {};
  /// Return the fill value.
  Value value();
  /// Return the output operand.
  Value output();
  /// Return the result.
  Value result();
};
namespace detail {
  template <typename ConcreteOp>
  struct FillOpInterfaceTrait : public ::mlir::OpInterface<FillOpInterface, detail::FillOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    /// Return the fill value.
    Value value() {
      return (*static_cast<ConcreteOp *>(this)).getOperation()->getOperand(0);
    }
    /// Return the output operand.
    Value output() {
      return (*static_cast<ConcreteOp *>(this)).getOperation()->getOperand(1);
    }
    /// Return the result.
    Value result() {
      if ((*static_cast<ConcreteOp *>(this)).getOperation()->getResults().empty())
          return nullptr;
        return (*static_cast<ConcreteOp *>(this)).getOperation()->getResults().front();
    }
    static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
      return detail::verifyFillInterface(op);
    }
  };
}// namespace detail
template<typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::Model<ConcreteOp>::value(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).value();
}
template<typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::Model<ConcreteOp>::output(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).output();
}
template<typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::Model<ConcreteOp>::result(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).result();
}
template<typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::value(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->value(tablegen_opaque_val);
}
template<typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::output(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->output(tablegen_opaque_val);
}
template<typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::FallbackModel<ConcreteOp>::result(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->result(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::value(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getOperand(0);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::output(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getOperand(1);
}
template<typename ConcreteModel, typename ConcreteOp>
Value detail::FillOpInterfaceInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::result(::mlir::Operation *tablegen_opaque_val) const {
if ((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getResults().empty())
          return nullptr;
        return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation()->getResults().front();
}
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class LinalgOp;
namespace detail {
struct LinalgOpInterfaceTraits {
  struct Concept {
    unsigned (*getNumParallelLoops)(const Concept *impl, ::mlir::Operation *);
    void (*getParallelDims)(const Concept *impl, ::mlir::Operation *, SmallVectorImpl<unsigned> &);
    unsigned (*getNumReductionLoops)(const Concept *impl, ::mlir::Operation *);
    void (*getReductionDims)(const Concept *impl, ::mlir::Operation *, SmallVectorImpl<unsigned> &);
    unsigned (*getNumWindowLoops)(const Concept *impl, ::mlir::Operation *);
    void (*getWindowDims)(const Concept *impl, ::mlir::Operation *, SmallVectorImpl<unsigned> &);
    unsigned (*getNumLoops)(const Concept *impl, ::mlir::Operation *);
    bool (*hasSingleReductionLoop)(const Concept *impl, ::mlir::Operation *);
    bool (*payloadUsesValueFromOperand)(const Concept *impl, ::mlir::Operation *, OpOperand *);
    bool (*isInitTensor)(const Concept *impl, ::mlir::Operation *, OpOperand *);
    int64_t (*getRank)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    Block::BlockArgListType (*getRegionOutputArgs)(const Concept *impl, ::mlir::Operation *);
    ArrayRef<int64_t> (*getShape)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    BlockArgument (*getTiedBlockArgument)(const Concept *impl, ::mlir::Operation *, OpOperand *);
    OpOperand *(*getTiedOpOperand)(const Concept *impl, ::mlir::Operation *, BlockArgument);
    AffineMap (*getTiedIndexingMap)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    AffineMap (*getTiedIndexingMapForResult)(const Concept *impl, ::mlir::Operation *, OpResult);
    OpOperand *(*getTiedYieldValue)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    Block*(*getBlock)(const Concept *impl, ::mlir::Operation *);
    ArrayAttr (*iterator_types)(const Concept *impl, ::mlir::Operation *);
    bool (*hasDynamicIndexingMaps)(const Concept *impl, ::mlir::Operation *);
    LogicalResult (*verifyIndexingMapRequiredAttributes)(const Concept *impl, ::mlir::Operation *);
    ArrayAttr (*getIndexingMaps)(const Concept *impl, ::mlir::Operation *);
    SmallVector<AffineMap> (*getIndexingMapsArray)(const Concept *impl, ::mlir::Operation *);
    bool (*hasDynamicShape)(const Concept *impl, ::mlir::Operation *);
    std::string (*getLibraryCallName)(const Concept *impl, ::mlir::Operation *);
    bool (*hasIndexSemantics)(const Concept *impl, ::mlir::Operation *);
    AffineMap (*getLoopsToShapesMap)(const Concept *impl, ::mlir::Operation *);
    AffineMap (*getShapesToLoopsMap)(const Concept *impl, ::mlir::Operation *);
    bool (*canOpOperandsBeDropped)(const Concept *impl, ::mlir::Operation *, ArrayRef<OpOperand *>);
    SmallVector<int64_t> (*getStaticShape)(const Concept *impl, ::mlir::Operation *);
    SmallVector<int64_t, 4> (*getStaticLoopRanges)(const Concept *impl, ::mlir::Operation *);
    std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> (*getRegionBuilder)();
    bool (*hasOnlyProjectedPermutations)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::linalg::LinalgOp;
    Model() : Concept{getNumParallelLoops, getParallelDims, getNumReductionLoops, getReductionDims, getNumWindowLoops, getWindowDims, getNumLoops, hasSingleReductionLoop, payloadUsesValueFromOperand, isInitTensor, getRank, getRegionOutputArgs, getShape, getTiedBlockArgument, getTiedOpOperand, getTiedIndexingMap, getTiedIndexingMapForResult, getTiedYieldValue, getBlock, iterator_types, hasDynamicIndexingMaps, verifyIndexingMapRequiredAttributes, getIndexingMaps, getIndexingMapsArray, hasDynamicShape, getLibraryCallName, hasIndexSemantics, getLoopsToShapesMap, getShapesToLoopsMap, canOpOperandsBeDropped, getStaticShape, getStaticLoopRanges, getRegionBuilder, hasOnlyProjectedPermutations} {}

    static inline unsigned getNumParallelLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getParallelDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumReductionLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getReductionDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumWindowLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getWindowDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasSingleReductionLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool payloadUsesValueFromOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isInitTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline int64_t getRank(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline Block::BlockArgListType getRegionOutputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline BlockArgument getTiedBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline OpOperand *getTiedOpOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, BlockArgument blockArgument);
    static inline AffineMap getTiedIndexingMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline AffineMap getTiedIndexingMapForResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpResult result);
    static inline OpOperand *getTiedYieldValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline Block*getBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayAttr iterator_types(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasDynamicIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult verifyIndexingMapRequiredAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayAttr getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<AffineMap> getIndexingMapsArray(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasDynamicShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline std::string getLibraryCallName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasIndexSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getLoopsToShapesMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getShapesToLoopsMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool canOpOperandsBeDropped(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands);
    static inline SmallVector<int64_t> getStaticShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<int64_t, 4> getStaticLoopRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> getRegionBuilder();
    static inline bool hasOnlyProjectedPermutations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::linalg::LinalgOp;
    FallbackModel() : Concept{getNumParallelLoops, getParallelDims, getNumReductionLoops, getReductionDims, getNumWindowLoops, getWindowDims, getNumLoops, hasSingleReductionLoop, payloadUsesValueFromOperand, isInitTensor, getRank, getRegionOutputArgs, getShape, getTiedBlockArgument, getTiedOpOperand, getTiedIndexingMap, getTiedIndexingMapForResult, getTiedYieldValue, getBlock, iterator_types, hasDynamicIndexingMaps, verifyIndexingMapRequiredAttributes, getIndexingMaps, getIndexingMapsArray, hasDynamicShape, getLibraryCallName, hasIndexSemantics, getLoopsToShapesMap, getShapesToLoopsMap, canOpOperandsBeDropped, getStaticShape, getStaticLoopRanges, getRegionBuilder, hasOnlyProjectedPermutations} {}

    static inline unsigned getNumParallelLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getParallelDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumReductionLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getReductionDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumWindowLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getWindowDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasSingleReductionLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool payloadUsesValueFromOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isInitTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline int64_t getRank(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline Block::BlockArgListType getRegionOutputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline BlockArgument getTiedBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline OpOperand *getTiedOpOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, BlockArgument blockArgument);
    static inline AffineMap getTiedIndexingMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline AffineMap getTiedIndexingMapForResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpResult result);
    static inline OpOperand *getTiedYieldValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline Block*getBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayAttr iterator_types(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasDynamicIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult verifyIndexingMapRequiredAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayAttr getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<AffineMap> getIndexingMapsArray(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasDynamicShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline std::string getLibraryCallName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasIndexSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getLoopsToShapesMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getShapesToLoopsMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool canOpOperandsBeDropped(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands);
    static inline SmallVector<int64_t> getStaticShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<int64_t, 4> getStaticLoopRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> getRegionBuilder();
    static inline bool hasOnlyProjectedPermutations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    using ConcreteEntity = ConcreteOp;
    unsigned getNumParallelLoops(::mlir::Operation *tablegen_opaque_val) const;
    void getParallelDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> &res) const;
    unsigned getNumReductionLoops(::mlir::Operation *tablegen_opaque_val) const;
    void getReductionDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> &res) const;
    unsigned getNumWindowLoops(::mlir::Operation *tablegen_opaque_val) const;
    void getWindowDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> &res) const;
    unsigned getNumLoops(::mlir::Operation *tablegen_opaque_val) const;
    bool hasSingleReductionLoop(::mlir::Operation *tablegen_opaque_val) const;
    bool payloadUsesValueFromOperand(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const;
    bool isInitTensor(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const;
    int64_t getRank(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    Block::BlockArgListType getRegionOutputArgs(::mlir::Operation *tablegen_opaque_val) const;
    ArrayRef<int64_t> getShape(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    BlockArgument getTiedBlockArgument(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const;
    OpOperand *getTiedOpOperand(::mlir::Operation *tablegen_opaque_val, BlockArgument blockArgument) const;
    AffineMap getTiedIndexingMap(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    AffineMap getTiedIndexingMapForResult(::mlir::Operation *tablegen_opaque_val, OpResult result) const;
    OpOperand *getTiedYieldValue(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    Block*getBlock(::mlir::Operation *tablegen_opaque_val) const;
    ArrayAttr iterator_types(::mlir::Operation *tablegen_opaque_val) const;
    bool hasDynamicIndexingMaps(::mlir::Operation *tablegen_opaque_val) const;
    LogicalResult verifyIndexingMapRequiredAttributes(::mlir::Operation *tablegen_opaque_val) const;
    SmallVector<AffineMap> getIndexingMapsArray(::mlir::Operation *tablegen_opaque_val) const;
    bool hasDynamicShape(::mlir::Operation *tablegen_opaque_val) const;
    std::string getLibraryCallName(::mlir::Operation *tablegen_opaque_val) const;
    AffineMap getLoopsToShapesMap(::mlir::Operation *tablegen_opaque_val) const;
    AffineMap getShapesToLoopsMap(::mlir::Operation *tablegen_opaque_val) const;
    bool canOpOperandsBeDropped(::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands) const;
    SmallVector<int64_t> getStaticShape(::mlir::Operation *tablegen_opaque_val) const;
    SmallVector<int64_t, 4> getStaticLoopRanges(::mlir::Operation *tablegen_opaque_val) const;
  };
};template <typename ConcreteOp>
struct LinalgOpTrait;

} // namespace detail
class LinalgOp : public ::mlir::OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits> {
public:
  using ::mlir::OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::LinalgOpTrait<ConcreteOp> {};
  /// Return the number of parallel loops.
  unsigned getNumParallelLoops();
  /// Return the dims that are parallel loops.
  void getParallelDims(SmallVectorImpl<unsigned> & res);
  /// Return the number of reduction loops.
  unsigned getNumReductionLoops();
  /// Return the dims that are reduction loops.
  void getReductionDims(SmallVectorImpl<unsigned> & res);
  /// Return the number of window loops.
  unsigned getNumWindowLoops();
  /// Return the dims that are window loops.
  void getWindowDims(SmallVectorImpl<unsigned> & res);
  /// Return the total number of loops within the current operation.
  unsigned getNumLoops();
  /// Returns true if the current operation has only one loop and it's a
  /// reduction loop.
  bool hasSingleReductionLoop();
  /// Return true if the payload uses the value loaded from `opOperand`. This
  /// is useful to avoid loading from "write-only" memory that may be
  /// uninitialized, as well as properly cloning "read-write" operands.
  bool payloadUsesValueFromOperand(OpOperand * opOperand);
  /// Return true if `opOperand` is an init tensor. This is true when it is
  /// an output tensor operand whose value is used in the payload region.
  bool isInitTensor(OpOperand * opOperand);
  /// Return the `opOperand` rank or zero for scalars.
  int64_t getRank(OpOperand* opOperand);
  /// Return the output block arguments of the region.
  Block::BlockArgListType getRegionOutputArgs();
  /// Return the `opOperand` shape or an empty vector for scalars.
  ArrayRef<int64_t> getShape(OpOperand* opOperand);
  /// Return the block argument for an `opOperand`.
  BlockArgument getTiedBlockArgument(OpOperand * opOperand);
  /// Return the operand for a `blockArgument`.
  OpOperand *getTiedOpOperand(BlockArgument blockArgument);
  /// Return the input or output indexing map for `opOperand`.
  AffineMap getTiedIndexingMap(OpOperand* opOperand);
  /// Return the indexing map for a `result`.
  AffineMap getTiedIndexingMapForResult(OpResult result);
  /// Return the value yielded by the region corresponding to an output
  /// `opOperand`.
  OpOperand *getTiedYieldValue(OpOperand* opOperand);
  /// Return the single block constituting the body of the operation by
  /// calling the getBody method on the concrete operation.
  Block*getBlock();
  /// Return the iterator types attribute within the current operation.
  ArrayAttr iterator_types();
  /// Return true if the indexing map is depending on the current op instance.
  /// This means that the indexing map is dynamically synthesized by using the
  /// op instance's concrete attributes, instead of being static for all
  /// instances of the same op kind.
  bool hasDynamicIndexingMaps();
  /// Verify all attributes used by indexing maps are valid.
  LogicalResult verifyIndexingMapRequiredAttributes();
  /// Return the indexing maps attribute within the current operation.
  ArrayAttr getIndexingMaps();
  /// Return the indexing maps within the current operation.
  SmallVector<AffineMap> getIndexingMapsArray();
  /// Return true if any of the operands has a dynamic shape.
  bool hasDynamicShape();
  /// Return the name registered for this op when lowering to an external
  /// library call.
  std::string getLibraryCallName();
  /// Return whether the op accesses the iteration indices.
  bool hasIndexSemantics();
  /// Hook to provide a custom AffineMap used to compute all the operand
  /// subshapes given loop bounds. This is used to answer the question: "given
  /// an iteration space over the codomain, what are the subshapes of the
  /// operands involved in the computation".
  /// The default behavior is to just concatenate all the indexing maps.
  /// A custom AffineMap allows providing a map that can be used to
  /// compute subshapes even in cases where the concatenation of indexing maps
  /// (i.e. the data traversal order) is not a simple permutation of the loop
  /// traversal order. It is then possible to define ops with skewed data
  /// traversal order for which we can still easily compute hyperrectangular
  /// loop bounds and subviews.
  AffineMap getLoopsToShapesMap();
  /// Hook to provide a custom AffineMap used to construct the
  /// hyperrectangular loop iteration space given all the operand subshapes.
  /// This is used to answer the question:
  /// "Given a list of operand ranges, what is the subportion of the iteration
  /// space involved in the computation".
  /// This is the inverse problem of `getLoopsToShapesMap`.
  /// Return the empty AffineMap when such an AffineMap cannot be constructed.
  /// The default behavior is based on a very simple inference procedure that
  /// only works with permutation affine maps.
  /// A more advanced Tensor-Comprehension like inference is possible but has
  /// proven to be ambiguous in unfavorable case.
  /// A safer and more robust alternative is to allow each op to define
  /// its own AffineMap.
  AffineMap getShapesToLoopsMap();
  /// Checks if the given operands can be dropped, and the remaining
  /// operands can still compute the bounds of the op.
  bool canOpOperandsBeDropped(ArrayRef<OpOperand *> droppedOperands);
  /// Like `getShape`, but only returns statically-known information, without
  /// generating any new IR. For each shape dimension, returns >=0 if that
  /// dimension is statically known, or ShapeType::kDynamicSize otherwise.
  SmallVector<int64_t> getStaticShape();
  /// Returns the statically-known loop ranges. Composes
  /// `getShapesToLoopsMap()` with the result of `getStaticShape`.
  /// Returns ShapeType::kDynamicSize for non-statically-known loop ranges.
  /// This is expected to be called by a valid Linalg op
  SmallVector<int64_t, 4> getStaticLoopRanges();
  /// Returns the region builder for constructing the body for linalg.generic.
  /// Returns a null function if this named op does not define a region
  /// builder.
  std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> getRegionBuilder();
  /// Return true if all the indexing maps are projected permutations.
  /// Otherwise return false.
  bool hasOnlyProjectedPermutations();

    /// Return the flat list of all operand dimension sizes in the order they
    /// appear in the operands.
    SmallVector<OpFoldResult> createFlatListOfOperandDims(OpBuilder &, Location);

    /// Return the flat list of all operands' static dimension sizes in the
    /// order they appear in the operands. All operand dimension sizes have to
    /// be statically known.
    SmallVector<int64_t, 4> createFlatListOfOperandStaticDims();

    /// Create the loop ranges to materialize the computation over the current
    /// operands. This is done by applying `getShapesToLoopsMap` to
    /// `createFlatListOfOperandDims`.
    SmallVector<Range, 4> createLoopRanges(OpBuilder &b, Location loc);

    /// Compute the static loop sizes necessary to vectorize the computation.
    /// This is done by applying `getShapesToLoopsMap` to
    /// `createFlatListOfOperandStaticDims`.
    SmallVector<int64_t, 4> computeStaticLoopSizes();

    /// Returns the value that expresses the shape of the output in terms of
    /// shape of the input operands where possible
    LogicalResult reifyResultShapes(OpBuilder &b,
        ReifiedRankedShapedTypeDims &reifiedReturnShapes);

    // TODO: Remove once prefixing is flipped.
    ArrayAttr getIteratorTypes() { return iterator_types(); }

    //========================================================================//
    // Forwarding functions to access interface methods from the
    // DestinationStyleOpInterface.
    // MLIR currently does not support dependent interfaces or interface
    // inheritance. By construction all ops with StructuredOpInterface must
    // implement DestinationStyleOpInterface.
    // TODO: reevalute the need for a cast when a better mechanism exists.
    //========================================================================//

    ValueRange inputs() {
      return cast<DestinationStyleOpInterface>(*this->getOperation()).inputs();
    }

    int64_t getNumInputs() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getNumInputs();
    }

    ValueRange outputs() {
      return cast<DestinationStyleOpInterface>(*this->getOperation()).outputs();
    }

    int64_t getNumOutputs() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getNumOutputs();
    }

    int64_t getNumInputsAndOutputs() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getNumInputsAndOutputs();
    }

    OpOperandVector getInputOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getInputOperands();
    }

    OpOperand *getInputOperand(int64_t i) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getInputOperand(i);
    }

    OpOperandVector getInputBufferOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getInputBufferOperands();
    }

    OpOperandVector getInputTensorOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getInputTensorOperands();
    }

    OpOperandVector getOutputOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getOutputOperands();
    }

    OpOperand *getOutputOperand(int64_t i) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getOutputOperand(i);
    }

    void setOutputOperand(int64_t i, Value value) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .setOutputOperand(i, value);
    }

    OpOperandVector getOutputBufferOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getOutputBufferOperands();
    }

    OpOperandVector getOutputTensorOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getOutputTensorOperands();
    }

    SmallVector<MemRefType> getOutputBufferTypes() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getOutputBufferTypes();
    }

    SmallVector<RankedTensorType> getOutputTensorTypes() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getOutputTensorTypes();
    }

    OpOperandVector getInputAndOutputOperands() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getInputAndOutputOperands();
    }

    bool isInputTensor(OpOperand *opOperand) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .isInputTensor(opOperand);
    }

    bool isOutputTensor(OpOperand *opOperand) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .isOutputTensor(opOperand);
    }

    bool isScalar(OpOperand *opOperand) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .isScalar(opOperand);
    }

    OpResult getTiedOpResult(OpOperand *opOperand) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .getTiedOpResult(opOperand);
    }

    bool hasBufferSemantics() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .hasBufferSemantics();
    }

    bool hasTensorSemantics() {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .hasTensorSemantics();
    }

    Operation *clone(OpBuilder & b, Location loc, TypeRange resultTypes,
                     ValueRange operands) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .clone(b, loc, resultTypes, operands);
    }

    Operation *cloneWithoutRegions(OpBuilder & b, Location loc,
                                   TypeRange resultTypes, ValueRange operands) {
      return cast<DestinationStyleOpInterface>(*this->getOperation())
          .cloneWithoutRegions(b, loc, resultTypes, operands);
    }

    //========================================================================//
    // Helper functions to mutate the `operand_segment_sizes` attribute.
    // These are useful when cloning and changing operand types.
    //========================================================================//
    void setNumInputs(unsigned num) { setOperandSegmentAt(0, num); }
    void setNumOutputBuffers(unsigned num) { setOperandSegmentAt(1, num); }

    private:
    void setOperandSegmentAt(unsigned idx, unsigned val) {
      auto attr = (*this)->getAttr("operand_segment_sizes")
        .cast<DenseIntElementsAttr>();
      unsigned i = 0;
      auto newAttr = attr.mapValues(IntegerType::get(getContext(), 32),
        [&](const APInt &v) { return (i++ == idx) ? APInt(32, val) : v; });
      getOperation()->setAttr("operand_segment_sizes", newAttr);
    }
  
};
namespace detail {
  template <typename ConcreteOp>
  struct LinalgOpTrait : public ::mlir::OpInterface<LinalgOp, detail::LinalgOpInterfaceTraits>::Trait<ConcreteOp> {
    /// Return the number of parallel loops.
    unsigned getNumParallelLoops() {
      return getNumIterators(getParallelIteratorTypeName(),
                               (*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    /// Return the dims that are parallel loops.
    void getParallelDims(SmallVectorImpl<unsigned> & res) {
      return getDimsOfType((*static_cast<ConcreteOp *>(this)), getParallelIteratorTypeName(), res);
    }
    /// Return the number of reduction loops.
    unsigned getNumReductionLoops() {
      return getNumIterators(getReductionIteratorTypeName(),
                               (*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    /// Return the dims that are reduction loops.
    void getReductionDims(SmallVectorImpl<unsigned> & res) {
      return getDimsOfType((*static_cast<ConcreteOp *>(this)), getReductionIteratorTypeName(), res);
    }
    /// Return the number of window loops.
    unsigned getNumWindowLoops() {
      return getNumIterators(getWindowIteratorTypeName(),
                               (*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    /// Return the dims that are window loops.
    void getWindowDims(SmallVectorImpl<unsigned> & res) {
      return getDimsOfType((*static_cast<ConcreteOp *>(this)).getOperation(), getWindowIteratorTypeName(), res);
    }
    /// Return the total number of loops within the current operation.
    unsigned getNumLoops() {
      return getNumIterators((*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    /// Returns true if the current operation has only one loop and it's a
    /// reduction loop.
    bool hasSingleReductionLoop() {
      auto iters = (*static_cast<ConcreteOp *>(this)).iterator_types();
        return iters.size() == 1 &&
               getNumIterators(getReductionIteratorTypeName(), iters) == 1;
    }
    /// Return true if the payload uses the value loaded from `opOperand`. This
    /// is useful to avoid loading from "write-only" memory that may be
    /// uninitialized, as well as properly cloning "read-write" operands.
    bool payloadUsesValueFromOperand(OpOperand * opOperand) {
      unsigned bbArgNumber = opOperand->getOperandNumber();
        // Init tensors have uses.
        return !getBlock()->getArgument(bbArgNumber).use_empty();
    }
    /// Return true if `opOperand` is an init tensor. This is true when it is
    /// an output tensor operand whose value is used in the payload region.
    bool isInitTensor(OpOperand * opOperand) {
      if (!(*static_cast<ConcreteOp *>(this)).isOutputTensor(opOperand))
          return false;
        return payloadUsesValueFromOperand(opOperand);
    }
    /// Return the `opOperand` rank or zero for scalars.
    int64_t getRank(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              opOperand->get().getType().template dyn_cast<ShapedType>())
          return shapedType.getRank();
        return 0;
    }
    /// Return the output block arguments of the region.
    Block::BlockArgListType getRegionOutputArgs() {
      // MLIR currently does not support dependent interfaces or interface
        // inheritance. By construction all ops with StructuredOpInterface must
        // implement DestinationStyleOpInterface.
        // TODO: reevalute the need for a cast when a better mechanism exists.
        return getBlock()->getArguments().take_back(
            cast<DestinationStyleOpInterface>(*this->getOperation())
                .getNumOutputs());
    }
    /// Return the `opOperand` shape or an empty vector for scalars.
    ArrayRef<int64_t> getShape(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              opOperand->get().getType().template dyn_cast<ShapedType>())
          return shapedType.getShape();
        return {};
    }
    /// Return the block argument for an `opOperand`.
    BlockArgument getTiedBlockArgument(OpOperand * opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        return getBlock()->getArgument(opOperand->getOperandNumber());
    }
    /// Return the operand for a `blockArgument`.
    OpOperand *getTiedOpOperand(BlockArgument blockArgument) {
      assert(blockArgument.getOwner() == getBlock());
        return &this->getOperation()->getOpOperand(
            blockArgument.getArgNumber());
    }
    /// Return the input or output indexing map for `opOperand`.
    AffineMap getTiedIndexingMap(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        auto indexingMaps =
          (*static_cast<ConcreteOp *>(this)).getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + opOperand->getOperandNumber());
    }
    /// Return the indexing map for a `result`.
    AffineMap getTiedIndexingMapForResult(OpResult result) {
      assert(result.getOwner() == this->getOperation());
        auto indexingMaps =
          (*static_cast<ConcreteOp *>(this)).getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        // MLIR currently does not support dependent interfaces or interface
        // inheritance. By construction all ops with StructuredOpInterface must
        // implement DestinationStyleOpInterface.
        // TODO: reevalute the need for a cast when a better mechanism exists.
        return *(indexingMaps.begin() +
                 cast<DestinationStyleOpInterface>(*this->getOperation())
                     .getNumInputs() +
                 result.getResultNumber());
    }
    /// Return the value yielded by the region corresponding to an output
    /// `opOperand`.
    OpOperand *getTiedYieldValue(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        // MLIR currently does not support dependent interfaces or interface
        // inheritance. By construction all ops with StructuredOpInterface must
        // implement DestinationStyleOpInterface.
        // TODO: reevalute the need for a cast when a better mechanism exists.
        int64_t resultIndex =
            opOperand->getOperandNumber() -
            cast<DestinationStyleOpInterface>(*this->getOperation())
                .getNumInputs();
        assert(resultIndex >= 0 &&
               resultIndex < this->getOperation()->getNumResults());
        Operation *yieldOp = getBlock()->getTerminator();
        return &yieldOp->getOpOperand(resultIndex);
    }
    /// Return the single block constituting the body of the operation by
    /// calling the getBody method on the concrete operation.
    Block*getBlock() {
      // Assume the concrete operation implements the
        // SingleBlockImplicitTerminator trait.
        return (*static_cast<ConcreteOp *>(this)).getBody();
    }
    /// Return the iterator types attribute within the current operation.
    ArrayAttr iterator_types() {
      return (*static_cast<ConcreteOp *>(this)).iterator_types();
    }
    /// Return true if the indexing map is depending on the current op instance.
    /// This means that the indexing map is dynamically synthesized by using the
    /// op instance's concrete attributes, instead of being static for all
    /// instances of the same op kind.
    bool hasDynamicIndexingMaps() {
      return false;
    }
    /// Verify all attributes used by indexing maps are valid.
    LogicalResult verifyIndexingMapRequiredAttributes() {
      return success();
    }
    /// Return the indexing maps within the current operation.
    SmallVector<AffineMap> getIndexingMapsArray() {
      auto range = (*static_cast<ConcreteOp *>(this)).getIndexingMaps()
          .template getAsValueRange<AffineMapAttr>();
        return {range.begin(), range.end()};
    }
    /// Return true if any of the operands has a dynamic shape.
    bool hasDynamicShape() {
      return llvm::any_of(getStaticShape(), ShapedType::isDynamic);
    }
    /// Return the name registered for this op when lowering to an external
    /// library call.
    std::string getLibraryCallName() {
      return (*static_cast<ConcreteOp *>(this)).getLibraryCallName();
    }
    /// Hook to provide a custom AffineMap used to compute all the operand
    /// subshapes given loop bounds. This is used to answer the question: "given
    /// an iteration space over the codomain, what are the subshapes of the
    /// operands involved in the computation".
    /// The default behavior is to just concatenate all the indexing maps.
    /// A custom AffineMap allows providing a map that can be used to
    /// compute subshapes even in cases where the concatenation of indexing maps
    /// (i.e. the data traversal order) is not a simple permutation of the loop
    /// traversal order. It is then possible to define ops with skewed data
    /// traversal order for which we can still easily compute hyperrectangular
    /// loop bounds and subviews.
    AffineMap getLoopsToShapesMap() {
      auto maps =  (*static_cast<ConcreteOp *>(this)).getIndexingMapsArray();
        return concatAffineMaps(maps);
    }
    /// Hook to provide a custom AffineMap used to construct the
    /// hyperrectangular loop iteration space given all the operand subshapes.
    /// This is used to answer the question:
    /// "Given a list of operand ranges, what is the subportion of the iteration
    /// space involved in the computation".
    /// This is the inverse problem of `getLoopsToShapesMap`.
    /// Return the empty AffineMap when such an AffineMap cannot be constructed.
    /// The default behavior is based on a very simple inference procedure that
    /// only works with permutation affine maps.
    /// A more advanced Tensor-Comprehension like inference is possible but has
    /// proven to be ambiguous in unfavorable case.
    /// A safer and more robust alternative is to allow each op to define
    /// its own AffineMap.
    AffineMap getShapesToLoopsMap() {
      return inversePermutation(getLoopsToShapesMap());
    }
    /// Checks if the given operands can be dropped, and the remaining
    /// operands can still compute the bounds of the op.
    bool canOpOperandsBeDropped(ArrayRef<OpOperand *> droppedOperands) {
      return detail::canOpOperandsBeDroppedImpl((*static_cast<ConcreteOp *>(this)), droppedOperands);
    }
    /// Like `getShape`, but only returns statically-known information, without
    /// generating any new IR. For each shape dimension, returns >=0 if that
    /// dimension is statically known, or ShapeType::kDynamicSize otherwise.
    SmallVector<int64_t> getStaticShape() {
      SmallVector<int64_t> res;
        // MLIR currently does not support dependent interfaces or interface
        // inheritance. By construction all ops with StructuredOpInterface must
        // implement DestinationStyleOpInterface.
        // TODO: reevalute the need for a cast when a better mechanism exists.
        auto iface = cast<DestinationStyleOpInterface>(*this->getOperation());
        for (OpOperand *opOperand : iface.getInputAndOutputOperands())
          llvm::append_range(res, getShape(opOperand));
        return res;
    }
    /// Returns the statically-known loop ranges. Composes
    /// `getShapesToLoopsMap()` with the result of `getStaticShape`.
    /// Returns ShapeType::kDynamicSize for non-statically-known loop ranges.
    /// This is expected to be called by a valid Linalg op
    SmallVector<int64_t, 4> getStaticLoopRanges() {
      SmallVector<int64_t> viewSizes = getStaticShape();
        AffineMap invertedMap = getShapesToLoopsMap();
        assert(invertedMap && "expected a valid Linalg op to call the method");
        return invertedMap.compose(viewSizes);
    }
    static ::mlir::LogicalResult verifyRegionTrait(::mlir::Operation *op) {
      return detail::verifyStructuredOpInterface(op);
    }
  };
}// namespace detail
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumParallelLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumParallelLoops();
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getParallelDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getParallelDims(res);
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumReductionLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumReductionLoops();
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getReductionDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getReductionDims(res);
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumWindowLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumWindowLoops();
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getWindowDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getWindowDims(res);
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumLoops();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasSingleReductionLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasSingleReductionLoop();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::payloadUsesValueFromOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).payloadUsesValueFromOperand(opOperand);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::isInitTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isInitTensor(opOperand);
}
template<typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getRank(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRank(opOperand);
}
template<typename ConcreteOp>
Block::BlockArgListType detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getRegionOutputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getRegionOutputArgs();
}
template<typename ConcreteOp>
ArrayRef<int64_t> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShape(opOperand);
}
template<typename ConcreteOp>
BlockArgument detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getTiedBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTiedBlockArgument(opOperand);
}
template<typename ConcreteOp>
OpOperand *detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getTiedOpOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, BlockArgument blockArgument) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTiedOpOperand(blockArgument);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getTiedIndexingMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTiedIndexingMap(opOperand);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getTiedIndexingMapForResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpResult result) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTiedIndexingMapForResult(result);
}
template<typename ConcreteOp>
OpOperand *detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getTiedYieldValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTiedYieldValue(opOperand);
}
template<typename ConcreteOp>
Block*detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getBlock();
}
template<typename ConcreteOp>
ArrayAttr detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::iterator_types(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).iterator_types();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasDynamicIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasDynamicIndexingMaps();
}
template<typename ConcreteOp>
LogicalResult detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::verifyIndexingMapRequiredAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).verifyIndexingMapRequiredAttributes();
}
template<typename ConcreteOp>
ArrayAttr detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps();
}
template<typename ConcreteOp>
SmallVector<AffineMap> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getIndexingMapsArray(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMapsArray();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasDynamicShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasDynamicShape();
}
template<typename ConcreteOp>
std::string detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getLibraryCallName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLibraryCallName();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasIndexSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasIndexSemantics();
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getLoopsToShapesMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLoopsToShapesMap();
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getShapesToLoopsMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getShapesToLoopsMap();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::canOpOperandsBeDropped(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).canOpOperandsBeDropped(droppedOperands);
}
template<typename ConcreteOp>
SmallVector<int64_t> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getStaticShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticShape();
}
template<typename ConcreteOp>
SmallVector<int64_t, 4> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getStaticLoopRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getStaticLoopRanges();
}
template<typename ConcreteOp>
std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getRegionBuilder() {
  return ConcreteOp::getRegionBuilder();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasOnlyProjectedPermutations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return llvm::all_of((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMapsArray(),
                            [](AffineMap map) { return map.isProjectedPermutation(); });
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getNumParallelLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumParallelLoops(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getParallelDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res) {
  return static_cast<const ConcreteOp *>(impl)->getParallelDims(tablegen_opaque_val, res);
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getNumReductionLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumReductionLoops(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getReductionDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res) {
  return static_cast<const ConcreteOp *>(impl)->getReductionDims(tablegen_opaque_val, res);
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getNumWindowLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumWindowLoops(tablegen_opaque_val);
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getWindowDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res) {
  return static_cast<const ConcreteOp *>(impl)->getWindowDims(tablegen_opaque_val, res);
}
template<typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getNumLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumLoops(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::hasSingleReductionLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasSingleReductionLoop(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::payloadUsesValueFromOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->payloadUsesValueFromOperand(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::isInitTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->isInitTensor(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getRank(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getRank(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
Block::BlockArgListType detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getRegionOutputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getRegionOutputArgs(tablegen_opaque_val);
}
template<typename ConcreteOp>
ArrayRef<int64_t> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getShape(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
BlockArgument detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getTiedBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getTiedBlockArgument(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
OpOperand *detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getTiedOpOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, BlockArgument blockArgument) {
  return static_cast<const ConcreteOp *>(impl)->getTiedOpOperand(tablegen_opaque_val, blockArgument);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getTiedIndexingMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getTiedIndexingMap(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getTiedIndexingMapForResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpResult result) {
  return static_cast<const ConcreteOp *>(impl)->getTiedIndexingMapForResult(tablegen_opaque_val, result);
}
template<typename ConcreteOp>
OpOperand *detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getTiedYieldValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getTiedYieldValue(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
Block*detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getBlock(tablegen_opaque_val);
}
template<typename ConcreteOp>
ArrayAttr detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::iterator_types(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->iterator_types(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::hasDynamicIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasDynamicIndexingMaps(tablegen_opaque_val);
}
template<typename ConcreteOp>
LogicalResult detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::verifyIndexingMapRequiredAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->verifyIndexingMapRequiredAttributes(tablegen_opaque_val);
}
template<typename ConcreteOp>
ArrayAttr detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getIndexingMaps(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<AffineMap> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getIndexingMapsArray(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getIndexingMapsArray(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::hasDynamicShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasDynamicShape(tablegen_opaque_val);
}
template<typename ConcreteOp>
std::string detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getLibraryCallName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLibraryCallName(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::hasIndexSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasIndexSemantics(tablegen_opaque_val);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getLoopsToShapesMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getLoopsToShapesMap(tablegen_opaque_val);
}
template<typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getShapesToLoopsMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getShapesToLoopsMap(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::canOpOperandsBeDropped(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands) {
  return static_cast<const ConcreteOp *>(impl)->canOpOperandsBeDropped(tablegen_opaque_val, droppedOperands);
}
template<typename ConcreteOp>
SmallVector<int64_t> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getStaticShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getStaticShape(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<int64_t, 4> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getStaticLoopRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getStaticLoopRanges(tablegen_opaque_val);
}
template<typename ConcreteOp>
std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getRegionBuilder() {
  return ConcreteOp::getRegionBuilder();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::hasOnlyProjectedPermutations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasOnlyProjectedPermutations(tablegen_opaque_val);
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumParallelLoops(::mlir::Operation *tablegen_opaque_val) const {
return getNumIterators(getParallelIteratorTypeName(),
                               (llvm::cast<ConcreteOp>(tablegen_opaque_val)).iterator_types());
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getParallelDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> &res) const {
return getDimsOfType((llvm::cast<ConcreteOp>(tablegen_opaque_val)), getParallelIteratorTypeName(), res);
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumReductionLoops(::mlir::Operation *tablegen_opaque_val) const {
return getNumIterators(getReductionIteratorTypeName(),
                               (llvm::cast<ConcreteOp>(tablegen_opaque_val)).iterator_types());
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getReductionDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> &res) const {
return getDimsOfType((llvm::cast<ConcreteOp>(tablegen_opaque_val)), getReductionIteratorTypeName(), res);
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumWindowLoops(::mlir::Operation *tablegen_opaque_val) const {
return getNumIterators(getWindowIteratorTypeName(),
                               (llvm::cast<ConcreteOp>(tablegen_opaque_val)).iterator_types());
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getWindowDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> &res) const {
return getDimsOfType((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOperation(), getWindowIteratorTypeName(), res);
}
template<typename ConcreteModel, typename ConcreteOp>
unsigned detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumLoops(::mlir::Operation *tablegen_opaque_val) const {
return getNumIterators((llvm::cast<ConcreteOp>(tablegen_opaque_val)).iterator_types());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasSingleReductionLoop(::mlir::Operation *tablegen_opaque_val) const {
auto iters = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).iterator_types();
        return iters.size() == 1 &&
               getNumIterators(getReductionIteratorTypeName(), iters) == 1;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::payloadUsesValueFromOperand(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const {
unsigned bbArgNumber = opOperand->getOperandNumber();
        // Init tensors have uses.
        return !getBlock()->getArgument(bbArgNumber).use_empty();
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isInitTensor(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const {
if (!(llvm::cast<ConcreteOp>(tablegen_opaque_val)).isOutputTensor(opOperand))
          return false;
        return payloadUsesValueFromOperand(opOperand);
}
template<typename ConcreteModel, typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getRank(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              opOperand->get().getType().template dyn_cast<ShapedType>())
          return shapedType.getRank();
        return 0;
}
template<typename ConcreteModel, typename ConcreteOp>
Block::BlockArgListType detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getRegionOutputArgs(::mlir::Operation *tablegen_opaque_val) const {
// MLIR currently does not support dependent interfaces or interface
        // inheritance. By construction all ops with StructuredOpInterface must
        // implement DestinationStyleOpInterface.
        // TODO: reevalute the need for a cast when a better mechanism exists.
        return getBlock()->getArguments().take_back(
            cast<DestinationStyleOpInterface>(*this->getOperation())
                .getNumOutputs());
}
template<typename ConcreteModel, typename ConcreteOp>
ArrayRef<int64_t> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getShape(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              opOperand->get().getType().template dyn_cast<ShapedType>())
          return shapedType.getShape();
        return {};
}
template<typename ConcreteModel, typename ConcreteOp>
BlockArgument detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiedBlockArgument(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        return getBlock()->getArgument(opOperand->getOperandNumber());
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperand *detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiedOpOperand(::mlir::Operation *tablegen_opaque_val, BlockArgument blockArgument) const {
assert(blockArgument.getOwner() == getBlock());
        return &this->getOperation()->getOpOperand(
            blockArgument.getArgNumber());
}
template<typename ConcreteModel, typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiedIndexingMap(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        auto indexingMaps =
          (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + opOperand->getOperandNumber());
}
template<typename ConcreteModel, typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiedIndexingMapForResult(::mlir::Operation *tablegen_opaque_val, OpResult result) const {
assert(result.getOwner() == this->getOperation());
        auto indexingMaps =
          (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps().template getAsValueRange<AffineMapAttr>();
        // MLIR currently does not support dependent interfaces or interface
        // inheritance. By construction all ops with StructuredOpInterface must
        // implement DestinationStyleOpInterface.
        // TODO: reevalute the need for a cast when a better mechanism exists.
        return *(indexingMaps.begin() +
                 cast<DestinationStyleOpInterface>(*this->getOperation())
                     .getNumInputs() +
                 result.getResultNumber());
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperand *detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiedYieldValue(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        // MLIR currently does not support dependent interfaces or interface
        // inheritance. By construction all ops with StructuredOpInterface must
        // implement DestinationStyleOpInterface.
        // TODO: reevalute the need for a cast when a better mechanism exists.
        int64_t resultIndex =
            opOperand->getOperandNumber() -
            cast<DestinationStyleOpInterface>(*this->getOperation())
                .getNumInputs();
        assert(resultIndex >= 0 &&
               resultIndex < this->getOperation()->getNumResults());
        Operation *yieldOp = getBlock()->getTerminator();
        return &yieldOp->getOpOperand(resultIndex);
}
template<typename ConcreteModel, typename ConcreteOp>
Block*detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getBlock(::mlir::Operation *tablegen_opaque_val) const {
// Assume the concrete operation implements the
        // SingleBlockImplicitTerminator trait.
        return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getBody();
}
template<typename ConcreteModel, typename ConcreteOp>
ArrayAttr detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::iterator_types(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).iterator_types();
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasDynamicIndexingMaps(::mlir::Operation *tablegen_opaque_val) const {
return false;
}
template<typename ConcreteModel, typename ConcreteOp>
LogicalResult detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::verifyIndexingMapRequiredAttributes(::mlir::Operation *tablegen_opaque_val) const {
return success();
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<AffineMap> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getIndexingMapsArray(::mlir::Operation *tablegen_opaque_val) const {
auto range = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps()
          .template getAsValueRange<AffineMapAttr>();
        return {range.begin(), range.end()};
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasDynamicShape(::mlir::Operation *tablegen_opaque_val) const {
return llvm::any_of(getStaticShape(), ShapedType::isDynamic);
}
template<typename ConcreteModel, typename ConcreteOp>
std::string detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLibraryCallName(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLibraryCallName();
}
template<typename ConcreteModel, typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLoopsToShapesMap(::mlir::Operation *tablegen_opaque_val) const {
auto maps =  (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMapsArray();
        return concatAffineMaps(maps);
}
template<typename ConcreteModel, typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getShapesToLoopsMap(::mlir::Operation *tablegen_opaque_val) const {
return inversePermutation(getLoopsToShapesMap());
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::canOpOperandsBeDropped(::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands) const {
return detail::canOpOperandsBeDroppedImpl((llvm::cast<ConcreteOp>(tablegen_opaque_val)), droppedOperands);
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<int64_t> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getStaticShape(::mlir::Operation *tablegen_opaque_val) const {
SmallVector<int64_t> res;
        // MLIR currently does not support dependent interfaces or interface
        // inheritance. By construction all ops with StructuredOpInterface must
        // implement DestinationStyleOpInterface.
        // TODO: reevalute the need for a cast when a better mechanism exists.
        auto iface = cast<DestinationStyleOpInterface>(*this->getOperation());
        for (OpOperand *opOperand : iface.getInputAndOutputOperands())
          llvm::append_range(res, getShape(opOperand));
        return res;
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<int64_t, 4> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getStaticLoopRanges(::mlir::Operation *tablegen_opaque_val) const {
SmallVector<int64_t> viewSizes = getStaticShape();
        AffineMap invertedMap = getShapesToLoopsMap();
        assert(invertedMap && "expected a valid Linalg op to call the method");
        return invertedMap.compose(viewSizes);
}
} // namespace linalg
} // namespace mlir
