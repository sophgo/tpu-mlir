/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace bufferization {
class AllocTensorOp;
} // namespace bufferization
} // namespace mlir
namespace mlir {
namespace bufferization {
class CloneOp;
} // namespace bufferization
} // namespace mlir
namespace mlir {
namespace bufferization {
class DeallocTensorOp;
} // namespace bufferization
} // namespace mlir
namespace mlir {
namespace bufferization {
class ToMemrefOp;
} // namespace bufferization
} // namespace mlir
namespace mlir {
namespace bufferization {
class ToTensorOp;
} // namespace bufferization
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace bufferization {

//===----------------------------------------------------------------------===//
// ::mlir::bufferization::AllocTensorOp declarations
//===----------------------------------------------------------------------===//

class AllocTensorOpAdaptor {
public:
  AllocTensorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AllocTensorOpAdaptor(AllocTensorOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getDynamicSizes();
  ::mlir::Value getCopy();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getMemorySpaceAttr();
  ::llvm::Optional<uint64_t> getMemorySpace();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class AllocTensorOp : public ::mlir::Op<AllocTensorOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::bufferization::BufferizableOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocTensorOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("memory_space"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getMemorySpaceAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMemorySpaceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("bufferization.alloc_tensor");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getDynamicSizes();
  ::mlir::TypedValue<::mlir::TensorType> getCopy();
  ::mlir::MutableOperandRange getDynamicSizesMutable();
  ::mlir::MutableOperandRange getCopyMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getResult();
  ::mlir::IntegerAttr getMemorySpaceAttr();
  ::llvm::Optional<uint64_t> getMemorySpace();
  void setMemorySpaceAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeMemory_spaceAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, RankedTensorType type, ValueRange dynamicSizes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, RankedTensorType type, ValueRange dynamicSizes, Value copy);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange dynamic_sizes, /*optional*/::mlir::Value copy, /*optional*/::mlir::IntegerAttr memory_space);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamic_sizes, /*optional*/::mlir::Value copy, /*optional*/::mlir::IntegerAttr memory_space);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult reifyResultShapes(::mlir::OpBuilder &builder, ::mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  LogicalResult bufferize(RewriterBase &rewriter,
                          const BufferizationOptions &options);

  bool isMemoryWrite(OpResult opResult, const AnalysisState &state);

  bool bufferizesToAllocation(OpResult opResult) { return true; }

  bool bufferizesToMemoryRead(OpOperand &opOperand,
                              const AnalysisState &state);

  bool bufferizesToMemoryWrite(OpOperand &opOperand,
                               const AnalysisState &state);

  SmallVector<OpResult> getAliasingOpResult(
      OpOperand &opOperand, const AnalysisState &state);

  FailureOr<BaseMemRefType> getBufferType(
      Value value, const BufferizationOptions &options,
      const DenseMap<Value, BaseMemRefType> &fixedTypes);

  RankedTensorType getType() {
    return getResult().getType().cast<RankedTensorType>();
  }

  // Return true if the size of the tensor is dynamic at `idx`
  bool isDynamicDim(unsigned idx) {
    return getType().isDynamicDim(idx);
  }

  // Return the argument position that contains the dynamic size of
  // the tensor at dimension `idx`. Asserts that the shape is
  // dynamic at that `idx`.
  unsigned getIndexOfDynamicSize(unsigned idx) {
    assert(!getCopy() && "no dim sizes specified when copying a tensor");
    assert(isDynamicDim(idx) && "expected dynamic size");
    ArrayRef<int64_t> shape = getType().getShape();
    return std::count_if(
        shape.begin(), shape.begin() + idx,
        [&](int64_t size) { return ShapedType::isDynamic(size); });
  }

  // Return the Value of the dynamic size of the tensor at dimension
  // `idx`. Asserts that the shape is dynamic at that `idx.
  Value getDynamicSize(OpBuilder &b, unsigned idx);

  // Assert that the size of the result tensor is static at `idx`
  // and return the shape.
  int64_t getStaticSize(unsigned idx) {
    assert(!isDynamicDim(idx) && "expected static size");
    return getType().getShape()[idx];
  }
};
} // namespace bufferization
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::bufferization::AllocTensorOp)

namespace mlir {
namespace bufferization {

//===----------------------------------------------------------------------===//
// ::mlir::bufferization::CloneOp declarations
//===----------------------------------------------------------------------===//

class CloneOpAdaptor {
public:
  CloneOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CloneOpAdaptor(CloneOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getInput();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class CloneOp : public ::mlir::Op<CloneOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::CopyOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::bufferization::AllocationOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CloneOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("bufferization.clone");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getInput();
  ::mlir::MutableOperandRange getInputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getOutput();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type output, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::Optional<::mlir::Operation*> buildDealloc(::mlir::OpBuilder&builder, ::mlir::Value alloc);
  static ::mlir::Optional<::mlir::Value> buildClone(::mlir::OpBuilder&builder, ::mlir::Value alloc);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
  Value getSource() { return getInput(); }
  Value getTarget() { return getOutput(); }
};
} // namespace bufferization
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::bufferization::CloneOp)

namespace mlir {
namespace bufferization {

//===----------------------------------------------------------------------===//
// ::mlir::bufferization::DeallocTensorOp declarations
//===----------------------------------------------------------------------===//

class DeallocTensorOpAdaptor {
public:
  DeallocTensorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  DeallocTensorOpAdaptor(DeallocTensorOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTensor();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class DeallocTensorOp : public ::mlir::Op<DeallocTensorOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::bufferization::BufferizableOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DeallocTensorOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("bufferization.dealloc_tensor");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getTensor();
  ::mlir::MutableOperandRange getTensorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
  bool bufferizesToMemoryRead(OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  SmallVector<OpResult> getAliasingOpResult(
      OpOperand &opOperand, const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(RewriterBase &rewriter,
                          const BufferizationOptions &options);
};
} // namespace bufferization
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::bufferization::DeallocTensorOp)

namespace mlir {
namespace bufferization {

//===----------------------------------------------------------------------===//
// ::mlir::bufferization::ToMemrefOp declarations
//===----------------------------------------------------------------------===//

class ToMemrefOpAdaptor {
public:
  ToMemrefOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ToMemrefOpAdaptor(ToMemrefOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTensor();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ToMemrefOp : public ::mlir::Op<ToMemrefOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::bufferization::BufferizableOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultShape, ::mlir::OpTrait::SameOperandsAndResultElementType, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ToMemrefOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("bufferization.to_memref");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getTensor();
  ::mlir::MutableOperandRange getTensorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getMemref();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type memref, ::mlir::Value tensor);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  //===------------------------------------------------------------------===//
  // BufferizableOpInterface implementation
  //===------------------------------------------------------------------===//

  // Note: ToMemrefOp / ToTensorOp are temporary ops that are inserted at the
  // bufferization boundary. When One-Shot bufferization is complete, there
  // should be no such ops left over. If `allowUnknownOps` (or after running a
  // partial bufferization pass), such ops may be part of the resulting IR,
  // but such IR may no longer be analyzable by One-Shot analysis.

  bool bufferizesToMemoryRead(OpOperand &opOperand,
                              const AnalysisState &state) const {
    // It is unknown whether the resulting memref will be read or not.
    return true;
  }

  bool bufferizesToMemoryWrite(OpOperand &opOperand,
                               const AnalysisState &state) const {
    // It is unknown whether the resulting MemRef will be written or not.
    return true;
  }

  bool mustBufferizeInPlace(OpOperand &opOperand,
                            const AnalysisState &state) const {
    // ToMemrefOps always bufferize inplace.
    return true;
  }

  SmallVector<OpResult> getAliasingOpResult(
      OpOperand &opOperand, const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(RewriterBase &rewriter,
                          const BufferizationOptions &options);
};
} // namespace bufferization
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::bufferization::ToMemrefOp)

namespace mlir {
namespace bufferization {

//===----------------------------------------------------------------------===//
// ::mlir::bufferization::ToTensorOp declarations
//===----------------------------------------------------------------------===//

class ToTensorOpAdaptor {
public:
  ToTensorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ToTensorOpAdaptor(ToTensorOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getMemref();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ToTensorOp : public ::mlir::Op<ToTensorOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::bufferization::BufferizableOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultShape, ::mlir::OpTrait::SameOperandsAndResultElementType, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ToTensorOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("bufferization.to_tensor");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getMemref();
  ::mlir::MutableOperandRange getMemrefMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(::llvm::ArrayRef<::mlir::Attribute> operands);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  /// The result of a to_tensor is always a tensor.
  TensorType getType() {
    Type resultType = getResult().getType();
    if (resultType.isa<TensorType>())
      return resultType.cast<TensorType>();
    return {};
  }

  //===------------------------------------------------------------------===//
  // BufferizableOpInterface implementation
  //===------------------------------------------------------------------===//

  // ToTensorOp conceptually loads a tensor from a memory location. The
  // One-Shot analysis has no information about the memref that is loaded from
  // by ToTensorOp. We have to assume that the loaded tensor may after
  // bufferization potentially alias with any other bufferized tensor. Since
  // ToTensorOp and ToMemrefOp have no aliasing OpOperand/OpResult pairs, this
  // cannot be encoded directly in the analysis. However, declaring ToTensorOp
  // results as not writable enforces a buffer copy and has the same effect.

  LogicalResult bufferize(RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    // to_tensor cannot be bufferized. However, other ops that are using
    // to_tensor's result will eventually be bufferized. At that point, they
    // will start using to_tensor's memref operand. Once all users of
    // to_tensor are bufferized, the op will not have any users anymore and
    // DCE away. In case of partial bufferization, to_memref(to_tensor(x))
    // constructs may be left over. These are folded by the canonicalizer or
    // FinalizingBufferize.
    return success();
  }

  bool isWritable(Value value, const AnalysisState &state) const {
    // It is unknown whether the memref operand is writable or not.
    return false;
  }

  FailureOr<BaseMemRefType> getBufferType(
      Value value, const BufferizationOptions &options,
      const DenseMap<Value, BaseMemRefType> &fixedTypes) {
    return getMemref().getType().cast<BaseMemRefType>();
  }
};
} // namespace bufferization
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::bufferization::ToTensorOp)


#endif  // GET_OP_CLASSES

