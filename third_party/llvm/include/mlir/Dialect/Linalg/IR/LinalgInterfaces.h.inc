/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Interface Declarations                                                     *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

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
  };
};template <typename ConcreteOp>
struct ContractionOpInterfaceTrait;

} // namespace detail
class ContractionOpInterface : public ::mlir::OpInterface<ContractionOpInterface, detail::ContractionOpInterfaceInterfaceTraits> {
public:
  using ::mlir::OpInterface<ContractionOpInterface, detail::ContractionOpInterfaceInterfaceTraits>::OpInterface;
  template <typename ConcreteOp>
  struct Trait : public detail::ContractionOpInterfaceTrait<ConcreteOp> {};
  Value lhs();
  Value rhs();
  bool isRowMajorMatmul();
  bool isColumnMajorMatmul();
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
  return mlir::isRowMajorMatmul((llvm::cast<ConcreteOp>(tablegen_opaque_val)).indexing_maps());
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isColumnMajorMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return mlir::isColumnMajorMatmul((llvm::cast<ConcreteOp>(tablegen_opaque_val)).indexing_maps());
}
template<typename ConcreteOp>
bool detail::ContractionOpInterfaceInterfaceTraits::Model<ConcreteOp>::isRowMajorBatchMatmul(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return mlir::isRowMajorBatchMatmul((llvm::cast<ConcreteOp>(tablegen_opaque_val)).indexing_maps());
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
  Value image();
  Value filter();
};
namespace detail {
  template <typename ConcreteOp>
  struct ConvolutionOpInterfaceTrait : public ::mlir::OpInterface<ConvolutionOpInterface, detail::ConvolutionOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    Value image() {
      return (*static_cast<ConcreteOp *>(this)).getOperation()->getOperand(0);
    }
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
  Value value();
  Value output();
  Value result();
};
namespace detail {
  template <typename ConcreteOp>
  struct FillOpInterfaceTrait : public ::mlir::OpInterface<FillOpInterface, detail::FillOpInterfaceInterfaceTraits>::Trait<ConcreteOp> {
    Value value() {
      return (*static_cast<ConcreteOp *>(this)).getOperation()->getOperand(0);
    }
    Value output() {
      return (*static_cast<ConcreteOp *>(this)).getOperation()->getOperand(1);
    }
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
    bool (*payloadUsesValueFromOperand)(const Concept *impl, ::mlir::Operation *, OpOperand *);
    bool (*isInputTensor)(const Concept *impl, ::mlir::Operation *, OpOperand *);
    bool (*isOutputTensor)(const Concept *impl, ::mlir::Operation *, OpOperand *);
    bool (*isInitTensor)(const Concept *impl, ::mlir::Operation *, OpOperand *);
    int64_t (*getRank)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    Block::BlockArgListType (*getRegionOutputArgs)(const Concept *impl, ::mlir::Operation *);
    ArrayRef<int64_t> (*getShape)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    bool (*isScalar)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    BlockArgument (*getTiedBlockArgument)(const Concept *impl, ::mlir::Operation *, OpOperand *);
    AffineMap (*getTiedIndexingMap)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    AffineMap (*getTiedIndexingMapForResult)(const Concept *impl, ::mlir::Operation *, OpResult);
    OpResult (*getTiedOpResult)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    OpOperand *(*getTiedYieldValue)(const Concept *impl, ::mlir::Operation *, OpOperand*);
    Block*(*getBlock)(const Concept *impl, ::mlir::Operation *);
    ArrayAttr (*iterator_types)(const Concept *impl, ::mlir::Operation *);
    bool (*hasDynamicIndexingMaps)(const Concept *impl, ::mlir::Operation *);
    LogicalResult (*verifyIndexingMapRequiredAttributes)(const Concept *impl, ::mlir::Operation *);
    ArrayAttr (*indexing_maps)(const Concept *impl, ::mlir::Operation *);
    SmallVector<AffineMap> (*getIndexingMaps)(const Concept *impl, ::mlir::Operation *);
    bool (*hasDynamicShape)(const Concept *impl, ::mlir::Operation *);
    bool (*hasBufferSemantics)(const Concept *impl, ::mlir::Operation *);
    bool (*hasTensorSemantics)(const Concept *impl, ::mlir::Operation *);
    std::string (*getLibraryCallName)(const Concept *impl, ::mlir::Operation *);
    bool (*hasIndexSemantics)(const Concept *impl, ::mlir::Operation *);
    AffineMap (*getLoopsToShapesMap)(const Concept *impl, ::mlir::Operation *);
    AffineMap (*getShapesToLoopsMap)(const Concept *impl, ::mlir::Operation *);
    bool (*canOpOperandsBeDropped)(const Concept *impl, ::mlir::Operation *, ArrayRef<OpOperand *>);
    std::pair<int64_t, int64_t> (*getResultsPositionInLoopsToShapeMap)(const Concept *impl, ::mlir::Operation *);
    SmallVector<int64_t> (*getStaticShape)(const Concept *impl, ::mlir::Operation *);
    SmallVector<int64_t, 4> (*getStaticLoopRanges)(const Concept *impl, ::mlir::Operation *);
    Operation *(*clone)(const Concept *impl, ::mlir::Operation *, OpBuilder &, Location, TypeRange, ValueRange);
    Operation *(*cloneWithMapper)(const Concept *impl, ::mlir::Operation *, OpBuilder &, Location, TypeRange, ValueRange, BlockAndValueMapping &);
    Operation *(*cloneWithoutRegions)(const Concept *impl, ::mlir::Operation *, OpBuilder &, Location, TypeRange, ValueRange);
    std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> (*getRegionBuilder)();
    bool (*hasOnlyProjectedPermutations)(const Concept *impl, ::mlir::Operation *);
  };
  template<typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = ::mlir::linalg::LinalgOp;
    Model() : Concept{getNumParallelLoops, getParallelDims, getNumReductionLoops, getReductionDims, getNumWindowLoops, getWindowDims, getNumLoops, hasSingleReductionLoop, inputs, getNumInputs, outputs, getNumOutputs, getNumInputsAndOutputs, getInputOperands, getInputOperand, getInputBufferOperands, getInputTensorOperands, getOutputOperands, getOutputOperand, setOutputOperand, getOutputBufferOperands, getOutputTensorOperands, getOutputBufferTypes, getOutputTensorTypes, getInputAndOutputOperands, payloadUsesValueFromOperand, isInputTensor, isOutputTensor, isInitTensor, getRank, getRegionOutputArgs, getShape, isScalar, getTiedBlockArgument, getTiedIndexingMap, getTiedIndexingMapForResult, getTiedOpResult, getTiedYieldValue, getBlock, iterator_types, hasDynamicIndexingMaps, verifyIndexingMapRequiredAttributes, indexing_maps, getIndexingMaps, hasDynamicShape, hasBufferSemantics, hasTensorSemantics, getLibraryCallName, hasIndexSemantics, getLoopsToShapesMap, getShapesToLoopsMap, canOpOperandsBeDropped, getResultsPositionInLoopsToShapeMap, getStaticShape, getStaticLoopRanges, clone, cloneWithMapper, cloneWithoutRegions, getRegionBuilder, hasOnlyProjectedPermutations} {}

    static inline unsigned getNumParallelLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getParallelDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumReductionLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getReductionDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumWindowLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getWindowDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasSingleReductionLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
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
    static inline bool payloadUsesValueFromOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isInputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isOutputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isInitTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline int64_t getRank(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline Block::BlockArgListType getRegionOutputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline bool isScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline BlockArgument getTiedBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline AffineMap getTiedIndexingMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline AffineMap getTiedIndexingMapForResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpResult result);
    static inline OpResult getTiedOpResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline OpOperand *getTiedYieldValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline Block*getBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayAttr iterator_types(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasDynamicIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult verifyIndexingMapRequiredAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayAttr indexing_maps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<AffineMap> getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasDynamicShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasBufferSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasTensorSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline std::string getLibraryCallName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasIndexSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getLoopsToShapesMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getShapesToLoopsMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool canOpOperandsBeDropped(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands);
    static inline std::pair<int64_t, int64_t> getResultsPositionInLoopsToShapeMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<int64_t> getStaticShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<int64_t, 4> getStaticLoopRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Operation *clone(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
    static inline Operation *cloneWithMapper(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands, BlockAndValueMapping & bvm);
    static inline Operation *cloneWithoutRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
    static inline std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> getRegionBuilder();
    static inline bool hasOnlyProjectedPermutations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteOp>
  class FallbackModel : public Concept {
  public:
    using Interface = ::mlir::linalg::LinalgOp;
    FallbackModel() : Concept{getNumParallelLoops, getParallelDims, getNumReductionLoops, getReductionDims, getNumWindowLoops, getWindowDims, getNumLoops, hasSingleReductionLoop, inputs, getNumInputs, outputs, getNumOutputs, getNumInputsAndOutputs, getInputOperands, getInputOperand, getInputBufferOperands, getInputTensorOperands, getOutputOperands, getOutputOperand, setOutputOperand, getOutputBufferOperands, getOutputTensorOperands, getOutputBufferTypes, getOutputTensorTypes, getInputAndOutputOperands, payloadUsesValueFromOperand, isInputTensor, isOutputTensor, isInitTensor, getRank, getRegionOutputArgs, getShape, isScalar, getTiedBlockArgument, getTiedIndexingMap, getTiedIndexingMapForResult, getTiedOpResult, getTiedYieldValue, getBlock, iterator_types, hasDynamicIndexingMaps, verifyIndexingMapRequiredAttributes, indexing_maps, getIndexingMaps, hasDynamicShape, hasBufferSemantics, hasTensorSemantics, getLibraryCallName, hasIndexSemantics, getLoopsToShapesMap, getShapesToLoopsMap, canOpOperandsBeDropped, getResultsPositionInLoopsToShapeMap, getStaticShape, getStaticLoopRanges, clone, cloneWithMapper, cloneWithoutRegions, getRegionBuilder, hasOnlyProjectedPermutations} {}

    static inline unsigned getNumParallelLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getParallelDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumReductionLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getReductionDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumWindowLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline void getWindowDims(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> & res);
    static inline unsigned getNumLoops(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasSingleReductionLoop(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
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
    static inline bool payloadUsesValueFromOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isInputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isOutputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline bool isInitTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline int64_t getRank(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline Block::BlockArgListType getRegionOutputArgs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayRef<int64_t> getShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline bool isScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline BlockArgument getTiedBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand);
    static inline AffineMap getTiedIndexingMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline AffineMap getTiedIndexingMapForResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpResult result);
    static inline OpResult getTiedOpResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline OpOperand *getTiedYieldValue(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand);
    static inline Block*getBlock(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayAttr iterator_types(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasDynamicIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline LogicalResult verifyIndexingMapRequiredAttributes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline ArrayAttr indexing_maps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<AffineMap> getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasDynamicShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasBufferSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasTensorSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline std::string getLibraryCallName(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool hasIndexSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getLoopsToShapesMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline AffineMap getShapesToLoopsMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline bool canOpOperandsBeDropped(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands);
    static inline std::pair<int64_t, int64_t> getResultsPositionInLoopsToShapeMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<int64_t> getStaticShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline SmallVector<int64_t, 4> getStaticLoopRanges(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
    static inline Operation *clone(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
    static inline Operation *cloneWithMapper(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands, BlockAndValueMapping & bvm);
    static inline Operation *cloneWithoutRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
    static inline std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> getRegionBuilder();
    static inline bool hasOnlyProjectedPermutations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val);
  };
  template<typename ConcreteModel, typename ConcreteOp>
  class ExternalModel : public FallbackModel<ConcreteModel> {
  public:
    unsigned getNumParallelLoops(::mlir::Operation *tablegen_opaque_val) const;
    void getParallelDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> &res) const;
    unsigned getNumReductionLoops(::mlir::Operation *tablegen_opaque_val) const;
    void getReductionDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> &res) const;
    unsigned getNumWindowLoops(::mlir::Operation *tablegen_opaque_val) const;
    void getWindowDims(::mlir::Operation *tablegen_opaque_val, SmallVectorImpl<unsigned> &res) const;
    unsigned getNumLoops(::mlir::Operation *tablegen_opaque_val) const;
    bool hasSingleReductionLoop(::mlir::Operation *tablegen_opaque_val) const;
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
    bool payloadUsesValueFromOperand(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const;
    bool isInputTensor(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const;
    bool isOutputTensor(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const;
    bool isInitTensor(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const;
    int64_t getRank(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    Block::BlockArgListType getRegionOutputArgs(::mlir::Operation *tablegen_opaque_val) const;
    ArrayRef<int64_t> getShape(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    bool isScalar(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    BlockArgument getTiedBlockArgument(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const;
    AffineMap getTiedIndexingMap(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    AffineMap getTiedIndexingMapForResult(::mlir::Operation *tablegen_opaque_val, OpResult result) const;
    OpResult getTiedOpResult(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    OpOperand *getTiedYieldValue(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const;
    Block*getBlock(::mlir::Operation *tablegen_opaque_val) const;
    ArrayAttr iterator_types(::mlir::Operation *tablegen_opaque_val) const;
    bool hasDynamicIndexingMaps(::mlir::Operation *tablegen_opaque_val) const;
    LogicalResult verifyIndexingMapRequiredAttributes(::mlir::Operation *tablegen_opaque_val) const;
    SmallVector<AffineMap> getIndexingMaps(::mlir::Operation *tablegen_opaque_val) const;
    bool hasDynamicShape(::mlir::Operation *tablegen_opaque_val) const;
    bool hasBufferSemantics(::mlir::Operation *tablegen_opaque_val) const;
    bool hasTensorSemantics(::mlir::Operation *tablegen_opaque_val) const;
    std::string getLibraryCallName(::mlir::Operation *tablegen_opaque_val) const;
    AffineMap getLoopsToShapesMap(::mlir::Operation *tablegen_opaque_val) const;
    AffineMap getShapesToLoopsMap(::mlir::Operation *tablegen_opaque_val) const;
    bool canOpOperandsBeDropped(::mlir::Operation *tablegen_opaque_val, ArrayRef<OpOperand *> droppedOperands) const;
    std::pair<int64_t, int64_t> getResultsPositionInLoopsToShapeMap(::mlir::Operation *tablegen_opaque_val) const;
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
  unsigned getNumParallelLoops();
  void getParallelDims(SmallVectorImpl<unsigned> & res);
  unsigned getNumReductionLoops();
  void getReductionDims(SmallVectorImpl<unsigned> & res);
  unsigned getNumWindowLoops();
  void getWindowDims(SmallVectorImpl<unsigned> & res);
  unsigned getNumLoops();
  bool hasSingleReductionLoop();
  ValueRange inputs();
  int64_t getNumInputs();
  ValueRange outputs();
  int64_t getNumOutputs();
  int64_t getNumInputsAndOutputs();
  OpOperandVector getInputOperands();
  OpOperand*getInputOperand(int64_t i);
  OpOperandVector getInputBufferOperands();
  OpOperandVector getInputTensorOperands();
  OpOperandVector getOutputOperands();
  OpOperand*getOutputOperand(int64_t i);
  void setOutputOperand(int64_t i, Value value);
  OpOperandVector getOutputBufferOperands();
  OpOperandVector getOutputTensorOperands();
  SmallVector<MemRefType> getOutputBufferTypes();
  SmallVector<RankedTensorType> getOutputTensorTypes();
  OpOperandVector getInputAndOutputOperands();
  bool payloadUsesValueFromOperand(OpOperand * opOperand);
  bool isInputTensor(OpOperand * opOperand);
  bool isOutputTensor(OpOperand * opOperand);
  bool isInitTensor(OpOperand * opOperand);
  int64_t getRank(OpOperand* opOperand);
  Block::BlockArgListType getRegionOutputArgs();
  ArrayRef<int64_t> getShape(OpOperand* opOperand);
  bool isScalar(OpOperand* opOperand);
  BlockArgument getTiedBlockArgument(OpOperand * opOperand);
  AffineMap getTiedIndexingMap(OpOperand* opOperand);
  AffineMap getTiedIndexingMapForResult(OpResult result);
  OpResult getTiedOpResult(OpOperand* opOperand);
  OpOperand *getTiedYieldValue(OpOperand* opOperand);
  Block*getBlock();
  ArrayAttr iterator_types();
  bool hasDynamicIndexingMaps();
  LogicalResult verifyIndexingMapRequiredAttributes();
  ArrayAttr indexing_maps();
  SmallVector<AffineMap> getIndexingMaps();
  bool hasDynamicShape();
  bool hasBufferSemantics();
  bool hasTensorSemantics();
  std::string getLibraryCallName();
  bool hasIndexSemantics();
  AffineMap getLoopsToShapesMap();
  AffineMap getShapesToLoopsMap();
  bool canOpOperandsBeDropped(ArrayRef<OpOperand *> droppedOperands);
  std::pair<int64_t, int64_t> getResultsPositionInLoopsToShapeMap();
  SmallVector<int64_t> getStaticShape();
  SmallVector<int64_t, 4> getStaticLoopRanges();
  Operation *clone(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
  Operation *cloneWithMapper(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands, BlockAndValueMapping & bvm);
  Operation *cloneWithoutRegions(OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands);
  std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> getRegionBuilder();
  bool hasOnlyProjectedPermutations();

    /// Return the flat list of all operand dimension sizes in the order they
    /// appear in the operands.
    SmallVector<Value, 4> createFlatListOfOperandDims(OpBuilder &, Location);

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
    unsigned getNumParallelLoops() {
      return getNumIterators(getParallelIteratorTypeName(),
                               (*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    void getParallelDims(SmallVectorImpl<unsigned> & res) {
      return getDimsOfType((*static_cast<ConcreteOp *>(this)), getParallelIteratorTypeName(), res);
    }
    unsigned getNumReductionLoops() {
      return getNumIterators(getReductionIteratorTypeName(),
                               (*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    void getReductionDims(SmallVectorImpl<unsigned> & res) {
      return getDimsOfType((*static_cast<ConcreteOp *>(this)), getReductionIteratorTypeName(), res);
    }
    unsigned getNumWindowLoops() {
      return getNumIterators(getWindowIteratorTypeName(),
                               (*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    void getWindowDims(SmallVectorImpl<unsigned> & res) {
      return getDimsOfType((*static_cast<ConcreteOp *>(this)).getOperation(), getWindowIteratorTypeName(), res);
    }
    unsigned getNumLoops() {
      return getNumIterators((*static_cast<ConcreteOp *>(this)).iterator_types());
    }
    bool hasSingleReductionLoop() {
      auto iters = (*static_cast<ConcreteOp *>(this)).iterator_types();
        return iters.size() == 1 &&
               getNumIterators(getReductionIteratorTypeName(), iters) == 1;
    }
    int64_t getNumInputs() {
      return (*static_cast<ConcreteOp *>(this)).inputs().size();
    }
    int64_t getNumOutputs() {
      return (*static_cast<ConcreteOp *>(this)).outputs().size();
    }
    int64_t getNumInputsAndOutputs() {
      return this->getOperation()->getNumOperands();
    }
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
    OpOperand*getInputOperand(int64_t i) {
      assert(i >= 0 && i < getNumInputs());
        return &this->getOperation()->getOpOperand(i);
    }
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
    OpOperand*getOutputOperand(int64_t i) {
      assert(i >= 0 && i < getNumOutputs());
        return &this->getOperation()->getOpOperand(getNumInputs() + i);
    }
    void setOutputOperand(int64_t i, Value value) {
      assert(i >= 0 && i < getNumOutputs());
        this->getOperation()->setOperand(getNumInputs() + i, value);
    }
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
    bool payloadUsesValueFromOperand(OpOperand * opOperand) {
      unsigned bbArgNumber = opOperand->getOperandNumber();
        // Init tensors have uses.
        return !getBlock()->getArgument(bbArgNumber).use_empty();
    }
    bool isInputTensor(OpOperand * opOperand) {
      if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() < (*static_cast<ConcreteOp *>(this)).getNumInputs())
          return true;
        return false;
    }
    bool isOutputTensor(OpOperand * opOperand) {
      if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() >= (*static_cast<ConcreteOp *>(this)).getNumInputs())
          return true;
        return false;
    }
    bool isInitTensor(OpOperand * opOperand) {
      if (!(*static_cast<ConcreteOp *>(this)).isOutputTensor(opOperand))
          return false;
        return payloadUsesValueFromOperand(opOperand);
    }
    int64_t getRank(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              opOperand->get().getType().template dyn_cast<ShapedType>())
          return shapedType.getRank();
        return 0;
    }
    Block::BlockArgListType getRegionOutputArgs() {
      return getBlock()->getArguments().take_back(this->getNumOutputs());
    }
    ArrayRef<int64_t> getShape(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        if (auto shapedType =
              opOperand->get().getType().template dyn_cast<ShapedType>())
          return shapedType.getShape();
        return {};
    }
    bool isScalar(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        return !opOperand->get().getType().template isa<ShapedType>();
    }
    BlockArgument getTiedBlockArgument(OpOperand * opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        return getBlock()->getArgument(opOperand->getOperandNumber());
    }
    AffineMap getTiedIndexingMap(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        auto indexingMaps =
          (*static_cast<ConcreteOp *>(this)).indexing_maps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + opOperand->getOperandNumber());
    }
    AffineMap getTiedIndexingMapForResult(OpResult result) {
      assert(result.getOwner() == this->getOperation());
        auto indexingMaps =
          (*static_cast<ConcreteOp *>(this)).indexing_maps().template getAsValueRange<AffineMapAttr>();
          return *(indexingMaps.begin() + getNumInputs() +
              result.getResultNumber());
    }
    OpResult getTiedOpResult(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        int64_t resultIndex = opOperand->getOperandNumber() - getNumInputs();
        assert(resultIndex >= 0 &&
               resultIndex < this->getOperation()->getNumResults() );
        return this->getOperation()->getResult(resultIndex);
    }
    OpOperand *getTiedYieldValue(OpOperand* opOperand) {
      assert(opOperand->getOwner() == this->getOperation());
        int64_t resultIndex = opOperand->getOperandNumber() - getNumInputs();
        assert(resultIndex >= 0 &&
               resultIndex < this->getOperation()->getNumResults());
        Operation *yieldOp = getBlock()->getTerminator();
        return &yieldOp->getOpOperand(resultIndex);
    }
    Block*getBlock() {
      // Assume the concrete operation implements the
        // SingleBlockImplicitTerminator trait.
        return (*static_cast<ConcreteOp *>(this)).getBody();
    }
    ArrayAttr iterator_types() {
      return (*static_cast<ConcreteOp *>(this)).iterator_types();
    }
    bool hasDynamicIndexingMaps() {
      return false;
    }
    LogicalResult verifyIndexingMapRequiredAttributes() {
      return success();
    }
    SmallVector<AffineMap> getIndexingMaps() {
      auto range = (*static_cast<ConcreteOp *>(this)).indexing_maps()
          .template getAsValueRange<AffineMapAttr>();
        return {range.begin(), range.end()};
    }
    bool hasDynamicShape() {
      return llvm::any_of(getStaticShape(), ShapedType::isDynamic);
    }
    bool hasBufferSemantics() {
      return this->getOperation()->getNumResults() == 0 &&
          llvm::all_of(this->getOperation()->getOpOperands(),
            [&](OpOperand &opOperand) {
              return isScalar(&opOperand) ||
                     opOperand.get().getType().template isa<MemRefType>();
            });
    }
    bool hasTensorSemantics() {
      return llvm::all_of(this->getOperation()->getOpOperands(),
          [&](OpOperand &opOperand) {
            return isScalar(&opOperand) ||
                   opOperand.get().getType().template isa<RankedTensorType>();
          });
    }
    std::string getLibraryCallName() {
      return (*static_cast<ConcreteOp *>(this)).getLibraryCallName();
    }
    AffineMap getLoopsToShapesMap() {
      auto r = (*static_cast<ConcreteOp *>(this)).indexing_maps().template getAsRange<AffineMapAttr>();
        auto maps = llvm::to_vector<8>(
            llvm::map_range(r, [](AffineMapAttr a) { return a.getValue(); }));
        return concatAffineMaps(maps);
    }
    AffineMap getShapesToLoopsMap() {
      return inversePermutation(getLoopsToShapesMap());
    }
    bool canOpOperandsBeDropped(ArrayRef<OpOperand *> droppedOperands) {
      return detail::canOpOperandsBeDroppedImpl((*static_cast<ConcreteOp *>(this)), droppedOperands);
    }
    std::pair<int64_t, int64_t> getResultsPositionInLoopsToShapeMap() {
      int64_t inputRankSum = 0;
        int64_t outputRankSum = 0;
        for(OpOperand *input : getInputOperands())
          inputRankSum += getRank(input);
        for(OpOperand *output : getOutputOperands())
          outputRankSum += getRank(output);
        return {inputRankSum, inputRankSum + outputRankSum};
    }
    SmallVector<int64_t> getStaticShape() {
      SmallVector<int64_t> res;
        for (OpOperand *opOperand : getInputAndOutputOperands())
          llvm::append_range(res, getShape(opOperand));
        return res;
    }
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
ValueRange detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::inputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inputs();
}
template<typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumInputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInputs();
}
template<typename ConcreteOp>
ValueRange detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::outputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).outputs();
}
template<typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumOutputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumOutputs();
}
template<typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getNumInputsAndOutputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInputsAndOutputs();
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputOperands();
}
template<typename ConcreteOp>
OpOperand*detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputOperand(i);
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputBufferOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputBufferOperands();
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputTensorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputTensorOperands();
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputOperands();
}
template<typename ConcreteOp>
OpOperand*detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputOperand(i);
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::setOutputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i, Value value) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).setOutputOperand(i, value);
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputBufferOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputBufferOperands();
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputTensorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputTensorOperands();
}
template<typename ConcreteOp>
SmallVector<MemRefType> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputBufferTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputBufferTypes();
}
template<typename ConcreteOp>
SmallVector<RankedTensorType> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getOutputTensorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getOutputTensorTypes();
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getInputAndOutputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getInputAndOutputOperands();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::payloadUsesValueFromOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).payloadUsesValueFromOperand(opOperand);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::isInputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isInputTensor(opOperand);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::isOutputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isOutputTensor(opOperand);
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
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::isScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).isScalar(opOperand);
}
template<typename ConcreteOp>
BlockArgument detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getTiedBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTiedBlockArgument(opOperand);
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
OpResult detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getTiedOpResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getTiedOpResult(opOperand);
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
ArrayAttr detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::indexing_maps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).indexing_maps();
}
template<typename ConcreteOp>
SmallVector<AffineMap> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasDynamicShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasDynamicShape();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasBufferSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasBufferSemantics();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasTensorSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).hasTensorSemantics();
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
std::pair<int64_t, int64_t> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getResultsPositionInLoopsToShapeMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getResultsPositionInLoopsToShapeMap();
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
Operation *detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::clone(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
  BlockAndValueMapping bvm;
        OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttrs());
        for (Region &r : (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getRegions())
          r.cloneInto(state.addRegion(), bvm);
        return b.create(state);
}
template<typename ConcreteOp>
Operation *detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::cloneWithMapper(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands, BlockAndValueMapping & bvm) {
  OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttrs());
        for (Region &r : (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getRegions())
          r.cloneInto(state.addRegion(), bvm);
        return b.create(state);
}
template<typename ConcreteOp>
Operation *detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::cloneWithoutRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
  OperationState state(
          loc, ConcreteOp::getOperationName(), operands, resultTypes,
          (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getAttrs());
        for (size_t cnt = 0, e = (llvm::cast<ConcreteOp>(tablegen_opaque_val))->getNumRegions(); cnt < e; ++cnt)
          state.addRegion();
        return b.create(state);
}
template<typename ConcreteOp>
std::function<void(ImplicitLocOpBuilder &, Block &, ArrayRef<NamedAttribute>)> detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::getRegionBuilder() {
  return ConcreteOp::getRegionBuilder();
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::Model<ConcreteOp>::hasOnlyProjectedPermutations(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return llvm::all_of((llvm::cast<ConcreteOp>(tablegen_opaque_val)).getIndexingMaps(),
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
ValueRange detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::inputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->inputs(tablegen_opaque_val);
}
template<typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getNumInputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumInputs(tablegen_opaque_val);
}
template<typename ConcreteOp>
ValueRange detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::outputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->outputs(tablegen_opaque_val);
}
template<typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getNumOutputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumOutputs(tablegen_opaque_val);
}
template<typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getNumInputsAndOutputs(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getNumInputsAndOutputs(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getInputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getInputOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperand*detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getInputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i) {
  return static_cast<const ConcreteOp *>(impl)->getInputOperand(tablegen_opaque_val, i);
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getInputBufferOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getInputBufferOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getInputTensorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getInputTensorOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getOutputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOutputOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperand*detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getOutputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i) {
  return static_cast<const ConcreteOp *>(impl)->getOutputOperand(tablegen_opaque_val, i);
}
template<typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::setOutputOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, int64_t i, Value value) {
  return static_cast<const ConcreteOp *>(impl)->setOutputOperand(tablegen_opaque_val, i, value);
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getOutputBufferOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOutputBufferOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getOutputTensorOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOutputTensorOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<MemRefType> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getOutputBufferTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOutputBufferTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<RankedTensorType> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getOutputTensorTypes(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getOutputTensorTypes(tablegen_opaque_val);
}
template<typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getInputAndOutputOperands(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getInputAndOutputOperands(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::payloadUsesValueFromOperand(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->payloadUsesValueFromOperand(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::isInputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->isInputTensor(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::isOutputTensor(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->isOutputTensor(tablegen_opaque_val, opOperand);
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
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::isScalar(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return static_cast<const ConcreteOp *>(impl)->isScalar(tablegen_opaque_val, opOperand);
}
template<typename ConcreteOp>
BlockArgument detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getTiedBlockArgument(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand * opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getTiedBlockArgument(tablegen_opaque_val, opOperand);
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
OpResult detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getTiedOpResult(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpOperand* opOperand) {
  return static_cast<const ConcreteOp *>(impl)->getTiedOpResult(tablegen_opaque_val, opOperand);
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
ArrayAttr detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::indexing_maps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->indexing_maps(tablegen_opaque_val);
}
template<typename ConcreteOp>
SmallVector<AffineMap> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getIndexingMaps(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getIndexingMaps(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::hasDynamicShape(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasDynamicShape(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::hasBufferSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasBufferSemantics(tablegen_opaque_val);
}
template<typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::hasTensorSemantics(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->hasTensorSemantics(tablegen_opaque_val);
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
std::pair<int64_t, int64_t> detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::getResultsPositionInLoopsToShapeMap(const Concept *impl, ::mlir::Operation *tablegen_opaque_val) {
  return static_cast<const ConcreteOp *>(impl)->getResultsPositionInLoopsToShapeMap(tablegen_opaque_val);
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
Operation *detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::clone(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
  return static_cast<const ConcreteOp *>(impl)->clone(tablegen_opaque_val, b, loc, resultTypes, operands);
}
template<typename ConcreteOp>
Operation *detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::cloneWithMapper(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands, BlockAndValueMapping & bvm) {
  return static_cast<const ConcreteOp *>(impl)->cloneWithMapper(tablegen_opaque_val, b, loc, resultTypes, operands, bvm);
}
template<typename ConcreteOp>
Operation *detail::LinalgOpInterfaceTraits::FallbackModel<ConcreteOp>::cloneWithoutRegions(const Concept *impl, ::mlir::Operation *tablegen_opaque_val, OpBuilder & b, Location loc, TypeRange resultTypes, ValueRange operands) {
  return static_cast<const ConcreteOp *>(impl)->cloneWithoutRegions(tablegen_opaque_val, b, loc, resultTypes, operands);
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
int64_t detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumInputs(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).inputs().size();
}
template<typename ConcreteModel, typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumOutputs(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).outputs().size();
}
template<typename ConcreteModel, typename ConcreteOp>
int64_t detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getNumInputsAndOutputs(::mlir::Operation *tablegen_opaque_val) const {
return this->getOperation()->getNumOperands();
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getInputOperands(::mlir::Operation *tablegen_opaque_val) const {
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
OpOperand*detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getInputOperand(::mlir::Operation *tablegen_opaque_val, int64_t i) const {
assert(i >= 0 && i < getNumInputs());
        return &this->getOperation()->getOpOperand(i);
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getInputBufferOperands(::mlir::Operation *tablegen_opaque_val) const {
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
OpOperandVector detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getInputTensorOperands(::mlir::Operation *tablegen_opaque_val) const {
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
OpOperandVector detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOutputOperands(::mlir::Operation *tablegen_opaque_val) const {
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
OpOperand*detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOutputOperand(::mlir::Operation *tablegen_opaque_val, int64_t i) const {
assert(i >= 0 && i < getNumOutputs());
        return &this->getOperation()->getOpOperand(getNumInputs() + i);
}
template<typename ConcreteModel, typename ConcreteOp>
void detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::setOutputOperand(::mlir::Operation *tablegen_opaque_val, int64_t i, Value value) const {
assert(i >= 0 && i < getNumOutputs());
        this->getOperation()->setOperand(getNumInputs() + i, value);
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperandVector detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOutputBufferOperands(::mlir::Operation *tablegen_opaque_val) const {
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
OpOperandVector detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOutputTensorOperands(::mlir::Operation *tablegen_opaque_val) const {
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
SmallVector<MemRefType> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOutputBufferTypes(::mlir::Operation *tablegen_opaque_val) const {
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
SmallVector<RankedTensorType> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getOutputTensorTypes(::mlir::Operation *tablegen_opaque_val) const {
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
OpOperandVector detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getInputAndOutputOperands(::mlir::Operation *tablegen_opaque_val) const {
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
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::payloadUsesValueFromOperand(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const {
unsigned bbArgNumber = opOperand->getOperandNumber();
        // Init tensors have uses.
        return !getBlock()->getArgument(bbArgNumber).use_empty();
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isInputTensor(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const {
if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() < (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInputs())
          return true;
        return false;
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isOutputTensor(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const {
if (!opOperand->get().getType().template isa<RankedTensorType>())
          return false;
        if (opOperand->getOperandNumber() >= (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getNumInputs())
          return true;
        return false;
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
return getBlock()->getArguments().take_back(this->getNumOutputs());
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
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::isScalar(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        return !opOperand->get().getType().template isa<ShapedType>();
}
template<typename ConcreteModel, typename ConcreteOp>
BlockArgument detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiedBlockArgument(::mlir::Operation *tablegen_opaque_val, OpOperand *opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        return getBlock()->getArgument(opOperand->getOperandNumber());
}
template<typename ConcreteModel, typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiedIndexingMap(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        auto indexingMaps =
          (llvm::cast<ConcreteOp>(tablegen_opaque_val)).indexing_maps().template getAsValueRange<AffineMapAttr>();
        return *(indexingMaps.begin() + opOperand->getOperandNumber());
}
template<typename ConcreteModel, typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiedIndexingMapForResult(::mlir::Operation *tablegen_opaque_val, OpResult result) const {
assert(result.getOwner() == this->getOperation());
        auto indexingMaps =
          (llvm::cast<ConcreteOp>(tablegen_opaque_val)).indexing_maps().template getAsValueRange<AffineMapAttr>();
          return *(indexingMaps.begin() + getNumInputs() +
              result.getResultNumber());
}
template<typename ConcreteModel, typename ConcreteOp>
OpResult detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiedOpResult(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        int64_t resultIndex = opOperand->getOperandNumber() - getNumInputs();
        assert(resultIndex >= 0 &&
               resultIndex < this->getOperation()->getNumResults() );
        return this->getOperation()->getResult(resultIndex);
}
template<typename ConcreteModel, typename ConcreteOp>
OpOperand *detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getTiedYieldValue(::mlir::Operation *tablegen_opaque_val, OpOperand*opOperand) const {
assert(opOperand->getOwner() == this->getOperation());
        int64_t resultIndex = opOperand->getOperandNumber() - getNumInputs();
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
SmallVector<AffineMap> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getIndexingMaps(::mlir::Operation *tablegen_opaque_val) const {
auto range = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).indexing_maps()
          .template getAsValueRange<AffineMapAttr>();
        return {range.begin(), range.end()};
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasDynamicShape(::mlir::Operation *tablegen_opaque_val) const {
return llvm::any_of(getStaticShape(), ShapedType::isDynamic);
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasBufferSemantics(::mlir::Operation *tablegen_opaque_val) const {
return this->getOperation()->getNumResults() == 0 &&
          llvm::all_of(this->getOperation()->getOpOperands(),
            [&](OpOperand &opOperand) {
              return isScalar(&opOperand) ||
                     opOperand.get().getType().template isa<MemRefType>();
            });
}
template<typename ConcreteModel, typename ConcreteOp>
bool detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::hasTensorSemantics(::mlir::Operation *tablegen_opaque_val) const {
return llvm::all_of(this->getOperation()->getOpOperands(),
          [&](OpOperand &opOperand) {
            return isScalar(&opOperand) ||
                   opOperand.get().getType().template isa<RankedTensorType>();
          });
}
template<typename ConcreteModel, typename ConcreteOp>
std::string detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLibraryCallName(::mlir::Operation *tablegen_opaque_val) const {
return (llvm::cast<ConcreteOp>(tablegen_opaque_val)).getLibraryCallName();
}
template<typename ConcreteModel, typename ConcreteOp>
AffineMap detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getLoopsToShapesMap(::mlir::Operation *tablegen_opaque_val) const {
auto r = (llvm::cast<ConcreteOp>(tablegen_opaque_val)).indexing_maps().template getAsRange<AffineMapAttr>();
        auto maps = llvm::to_vector<8>(
            llvm::map_range(r, [](AffineMapAttr a) { return a.getValue(); }));
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
std::pair<int64_t, int64_t> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getResultsPositionInLoopsToShapeMap(::mlir::Operation *tablegen_opaque_val) const {
int64_t inputRankSum = 0;
        int64_t outputRankSum = 0;
        for(OpOperand *input : getInputOperands())
          inputRankSum += getRank(input);
        for(OpOperand *output : getOutputOperands())
          outputRankSum += getRank(output);
        return {inputRankSum, inputRankSum + outputRankSum};
}
template<typename ConcreteModel, typename ConcreteOp>
SmallVector<int64_t> detail::LinalgOpInterfaceTraits::ExternalModel<ConcreteModel, ConcreteOp>::getStaticShape(::mlir::Operation *tablegen_opaque_val) const {
SmallVector<int64_t> res;
        for (OpOperand *opOperand : getInputAndOutputOperands())
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
