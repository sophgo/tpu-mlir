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
namespace linalg {
class BatchMatmulOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class BatchMatvecOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv1DNwcWcfOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv1DOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv2DNchwFchwOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv2DNhwcHwcfOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv2DNhwcHwcfQOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv2DOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv3DNdhwcDhwcfOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Conv3DOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class CopyOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class DepthwiseConv1DNwcWcOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class DepthwiseConv2DNhwcHwcOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class DepthwiseConv2DNhwcHwcQOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class DepthwiseConv2DNhwcHwcmOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class DepthwiseConv2DNhwcHwcmQOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class DotOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ElemwiseBinaryOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class ElemwiseUnaryOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class FillOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class FillRng2DOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class GenericOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class MatmulOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class MatmulUnsignedOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class MatvecOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class Mmt4DOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNchwMaxOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNchwSumOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNdhwcMaxOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNdhwcMinOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNdhwcSumOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNhwcMaxOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNhwcMaxUnsignedOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNhwcMinOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNhwcMinUnsignedOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class PoolingNhwcSumOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class QuantizedBatchMatmulOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class QuantizedMatmulOp;
} // namespace linalg
} // namespace mlir
namespace mlir {
namespace linalg {
class VecmatOp;
} // namespace linalg
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatmulOp declarations
//===----------------------------------------------------------------------===//

class BatchMatmulOpAdaptor {
public:
  BatchMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  BatchMatmulOpAdaptor(BatchMatmulOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class BatchMatmulOp : public ::mlir::Op<BatchMatmulOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BatchMatmulOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.batch_matmul");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::BatchMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::BatchMatvecOp declarations
//===----------------------------------------------------------------------===//

class BatchMatvecOpAdaptor {
public:
  BatchMatvecOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  BatchMatvecOpAdaptor(BatchMatvecOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class BatchMatvecOp : public ::mlir::Op<BatchMatvecOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BatchMatvecOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.batch_matvec");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::BatchMatvecOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv1DNwcWcfOp declarations
//===----------------------------------------------------------------------===//

class Conv1DNwcWcfOpAdaptor {
public:
  Conv1DNwcWcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  Conv1DNwcWcfOpAdaptor(Conv1DNwcWcfOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class Conv1DNwcWcfOp : public ::mlir::Op<Conv1DNwcWcfOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv1DNwcWcfOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_1d_nwc_wcf");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv1DNwcWcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv1DOp declarations
//===----------------------------------------------------------------------===//

class Conv1DOpAdaptor {
public:
  Conv1DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  Conv1DOpAdaptor(Conv1DOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class Conv1DOp : public ::mlir::Op<Conv1DOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv1DOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_1d");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv1DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNchwFchwOp declarations
//===----------------------------------------------------------------------===//

class Conv2DNchwFchwOpAdaptor {
public:
  Conv2DNchwFchwOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  Conv2DNchwFchwOpAdaptor(Conv2DNchwFchwOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class Conv2DNchwFchwOp : public ::mlir::Op<Conv2DNchwFchwOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv2DNchwFchwOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_2d_nchw_fchw");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNchwFchwOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNhwcHwcfOp declarations
//===----------------------------------------------------------------------===//

class Conv2DNhwcHwcfOpAdaptor {
public:
  Conv2DNhwcHwcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  Conv2DNhwcHwcfOpAdaptor(Conv2DNhwcHwcfOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class Conv2DNhwcHwcfOp : public ::mlir::Op<Conv2DNhwcHwcfOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv2DNhwcHwcfOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_2d_nhwc_hwcf");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNhwcHwcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DNhwcHwcfQOp declarations
//===----------------------------------------------------------------------===//

class Conv2DNhwcHwcfQOpAdaptor {
public:
  Conv2DNhwcHwcfQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  Conv2DNhwcHwcfQOpAdaptor(Conv2DNhwcHwcfQOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class Conv2DNhwcHwcfQOp : public ::mlir::Op<Conv2DNhwcHwcfQOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv2DNhwcHwcfQOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_2d_nhwc_hwcf_q");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DNhwcHwcfQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv2DOp declarations
//===----------------------------------------------------------------------===//

class Conv2DOpAdaptor {
public:
  Conv2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  Conv2DOpAdaptor(Conv2DOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class Conv2DOp : public ::mlir::Op<Conv2DOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv2DOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_2d");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv2DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv3DNdhwcDhwcfOp declarations
//===----------------------------------------------------------------------===//

class Conv3DNdhwcDhwcfOpAdaptor {
public:
  Conv3DNdhwcDhwcfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  Conv3DNdhwcDhwcfOpAdaptor(Conv3DNdhwcDhwcfOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class Conv3DNdhwcDhwcfOp : public ::mlir::Op<Conv3DNdhwcDhwcfOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv3DNdhwcDhwcfOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_3d_ndhwc_dhwcf");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv3DNdhwcDhwcfOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Conv3DOp declarations
//===----------------------------------------------------------------------===//

class Conv3DOpAdaptor {
public:
  Conv3DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  Conv3DOpAdaptor(Conv3DOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class Conv3DOp : public ::mlir::Op<Conv3DOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Conv3DOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.conv_3d");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Conv3DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::CopyOp declarations
//===----------------------------------------------------------------------===//

class CopyOpAdaptor {
public:
  CopyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CopyOpAdaptor(CopyOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::linalg::TypeFnAttr castAttr();
  ::mlir::linalg::TypeFn cast();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class CopyOp : public ::mlir::Op<CopyOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CopyOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("cast"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr castAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr castAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.copy");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::linalg::TypeFnAttr castAttr();
  ::mlir::linalg::TypeFn cast();
  void castAttr(::mlir::linalg::TypeFnAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute cast, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::CopyOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv1DNwcWcOp declarations
//===----------------------------------------------------------------------===//

class DepthwiseConv1DNwcWcOpAdaptor {
public:
  DepthwiseConv1DNwcWcOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  DepthwiseConv1DNwcWcOpAdaptor(DepthwiseConv1DNwcWcOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class DepthwiseConv1DNwcWcOp : public ::mlir::Op<DepthwiseConv1DNwcWcOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DepthwiseConv1DNwcWcOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.depthwise_conv_1d_nwc_wc");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv1DNwcWcOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcHwcOp declarations
//===----------------------------------------------------------------------===//

class DepthwiseConv2DNhwcHwcOpAdaptor {
public:
  DepthwiseConv2DNhwcHwcOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  DepthwiseConv2DNhwcHwcOpAdaptor(DepthwiseConv2DNhwcHwcOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class DepthwiseConv2DNhwcHwcOp : public ::mlir::Op<DepthwiseConv2DNhwcHwcOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DepthwiseConv2DNhwcHwcOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.depthwise_conv_2d_nhwc_hwc");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcHwcOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcHwcQOp declarations
//===----------------------------------------------------------------------===//

class DepthwiseConv2DNhwcHwcQOpAdaptor {
public:
  DepthwiseConv2DNhwcHwcQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  DepthwiseConv2DNhwcHwcQOpAdaptor(DepthwiseConv2DNhwcHwcQOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class DepthwiseConv2DNhwcHwcQOp : public ::mlir::Op<DepthwiseConv2DNhwcHwcQOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DepthwiseConv2DNhwcHwcQOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.depthwise_conv_2d_nhwc_hwc_q");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcHwcQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcHwcmOp declarations
//===----------------------------------------------------------------------===//

class DepthwiseConv2DNhwcHwcmOpAdaptor {
public:
  DepthwiseConv2DNhwcHwcmOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  DepthwiseConv2DNhwcHwcmOpAdaptor(DepthwiseConv2DNhwcHwcmOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class DepthwiseConv2DNhwcHwcmOp : public ::mlir::Op<DepthwiseConv2DNhwcHwcmOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DepthwiseConv2DNhwcHwcmOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.depthwise_conv_2d_nhwc_hwcm");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcHwcmOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DepthwiseConv2DNhwcHwcmQOp declarations
//===----------------------------------------------------------------------===//

class DepthwiseConv2DNhwcHwcmQOpAdaptor {
public:
  DepthwiseConv2DNhwcHwcmQOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  DepthwiseConv2DNhwcHwcmQOpAdaptor(DepthwiseConv2DNhwcHwcmQOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class DepthwiseConv2DNhwcHwcmQOp : public ::mlir::Op<DepthwiseConv2DNhwcHwcmQOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DepthwiseConv2DNhwcHwcmQOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.depthwise_conv_2d_nhwc_hwcm_q");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::DepthwiseConv2DNhwcHwcmQOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::DotOp declarations
//===----------------------------------------------------------------------===//

class DotOpAdaptor {
public:
  DotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  DotOpAdaptor(DotOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class DotOp : public ::mlir::Op<DotOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DotOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.dot");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::DotOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ElemwiseBinaryOp declarations
//===----------------------------------------------------------------------===//

class ElemwiseBinaryOpAdaptor {
public:
  ElemwiseBinaryOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ElemwiseBinaryOpAdaptor(ElemwiseBinaryOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::linalg::BinaryFnAttr funAttr();
  ::mlir::linalg::BinaryFn fun();
  ::mlir::linalg::TypeFnAttr castAttr();
  ::mlir::linalg::TypeFn cast();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ElemwiseBinaryOp : public ::mlir::Op<ElemwiseBinaryOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ElemwiseBinaryOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("cast"), ::llvm::StringRef("fun"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr castAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr castAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr funAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr funAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.elemwise_binary");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::linalg::BinaryFnAttr funAttr();
  ::mlir::linalg::BinaryFn fun();
  ::mlir::linalg::TypeFnAttr castAttr();
  ::mlir::linalg::TypeFn cast();
  void funAttr(::mlir::linalg::BinaryFnAttr attr);
  void castAttr(::mlir::linalg::TypeFnAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute fun, Attribute cast, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::ElemwiseBinaryOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::ElemwiseUnaryOp declarations
//===----------------------------------------------------------------------===//

class ElemwiseUnaryOpAdaptor {
public:
  ElemwiseUnaryOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ElemwiseUnaryOpAdaptor(ElemwiseUnaryOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::linalg::UnaryFnAttr funAttr();
  ::mlir::linalg::UnaryFn fun();
  ::mlir::linalg::TypeFnAttr castAttr();
  ::mlir::linalg::TypeFn cast();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ElemwiseUnaryOp : public ::mlir::Op<ElemwiseUnaryOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ElemwiseUnaryOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("cast"), ::llvm::StringRef("fun"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr castAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr castAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr funAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr funAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.elemwise_unary");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::linalg::UnaryFnAttr funAttr();
  ::mlir::linalg::UnaryFn fun();
  ::mlir::linalg::TypeFnAttr castAttr();
  ::mlir::linalg::TypeFn cast();
  void funAttr(::mlir::linalg::UnaryFnAttr attr);
  void castAttr(::mlir::linalg::TypeFnAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute fun, Attribute cast, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::ElemwiseUnaryOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillOp declarations
//===----------------------------------------------------------------------===//

class FillOpAdaptor {
public:
  FillOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  FillOpAdaptor(FillOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class FillOp : public ::mlir::Op<FillOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::FillOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FillOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.fill");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::FillOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::FillRng2DOp declarations
//===----------------------------------------------------------------------===//

class FillRng2DOpAdaptor {
public:
  FillRng2DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  FillRng2DOpAdaptor(FillRng2DOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class FillRng2DOp : public ::mlir::Op<FillRng2DOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FillRng2DOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.fill_rng_2d");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::FillRng2DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::GenericOp declarations
//===----------------------------------------------------------------------===//

class GenericOpAdaptor {
public:
  GenericOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  GenericOpAdaptor(GenericOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr indexing_mapsAttr();
  ::mlir::ArrayAttr indexing_maps();
  ::mlir::ArrayAttr iterator_typesAttr();
  ::mlir::ArrayAttr iterator_types();
  ::mlir::StringAttr docAttr();
  ::llvm::Optional< ::llvm::StringRef > doc();
  ::mlir::StringAttr library_callAttr();
  ::llvm::Optional< ::llvm::StringRef > library_call();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class GenericOp : public ::mlir::Op<GenericOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GenericOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("doc"), ::llvm::StringRef("indexing_maps"), ::llvm::StringRef("iterator_types"), ::llvm::StringRef("library_call"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr docAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr docAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr indexing_mapsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr indexing_mapsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr iterator_typesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr iterator_typesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr library_callAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr library_callAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.generic");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::ArrayAttr indexing_mapsAttr();
  ::mlir::ArrayAttr indexing_maps();
  ::mlir::ArrayAttr iterator_typesAttr();
  ::mlir::ArrayAttr iterator_types();
  ::mlir::StringAttr docAttr();
  ::llvm::Optional< ::llvm::StringRef > doc();
  ::mlir::StringAttr library_callAttr();
  ::llvm::Optional< ::llvm::StringRef > library_call();
  void indexing_mapsAttr(::mlir::ArrayAttr attr);
  void iterator_typesAttr(::mlir::ArrayAttr attr);
  void docAttr(::mlir::StringAttr attr);
  void library_callAttr(::mlir::StringAttr attr);
  ::mlir::Attribute removeDocAttr();
  ::mlir::Attribute removeLibrary_callAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayAttr indexingMaps, ArrayAttr iteratorTypes, StringAttr doc, StringAttr libraryCall, function_ref<void(OpBuilder &, Location, ValueRange)> odsArg7, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall, function_ref<void(OpBuilder &, Location, ValueRange)> odsArg7 = nullptr, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, StringRef doc, StringRef libraryCall, function_ref<void(OpBuilder &, Location, ValueRange)> odsArg6 = nullptr, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, function_ref<void(OpBuilder &, Location, ValueRange)> odsArg5 = nullptr, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputBuffers, ArrayRef<AffineMap> indexingMaps, ArrayRef<StringRef> iteratorTypes, function_ref<void(OpBuilder &, Location, ValueRange)> odsArg4 = nullptr, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange result_tensors, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, /*optional*/::mlir::StringAttr doc, /*optional*/::mlir::StringAttr library_call);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

  SmallVector<StringRef, 8> linalgTraitAttrNames() {
    return SmallVector<StringRef, 8>{
      getDocAttrName(),
      getIndexingMapsAttrName(), getLibraryCallAttrName(),
      getIteratorTypesAttrName(),
    };
  }
  std::string getLibraryCallName() {
    return library_call().hasValue() ?
      library_call()->str() : "op_has_no_registered_library_name";
  }

  static std::function<void(ImplicitLocOpBuilder &,
                            Block &, ArrayRef<NamedAttribute>)>
  getRegionBuilder() {
    return nullptr;
  }
};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::GenericOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulOp declarations
//===----------------------------------------------------------------------===//

class MatmulOpAdaptor {
public:
  MatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  MatmulOpAdaptor(MatmulOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::linalg::TypeFnAttr castAttr();
  ::mlir::linalg::TypeFn cast();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class MatmulOp : public ::mlir::Op<MatmulOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MatmulOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("cast"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr castAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr castAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.matmul");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::linalg::TypeFnAttr castAttr();
  ::mlir::linalg::TypeFn cast();
  void castAttr(::mlir::linalg::TypeFnAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute cast, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::MatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatmulUnsignedOp declarations
//===----------------------------------------------------------------------===//

class MatmulUnsignedOpAdaptor {
public:
  MatmulUnsignedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  MatmulUnsignedOpAdaptor(MatmulUnsignedOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class MatmulUnsignedOp : public ::mlir::Op<MatmulUnsignedOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MatmulUnsignedOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.matmul_unsigned");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::MatmulUnsignedOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::MatvecOp declarations
//===----------------------------------------------------------------------===//

class MatvecOpAdaptor {
public:
  MatvecOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  MatvecOpAdaptor(MatvecOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class MatvecOp : public ::mlir::Op<MatvecOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MatvecOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.matvec");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::MatvecOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::Mmt4DOp declarations
//===----------------------------------------------------------------------===//

class Mmt4DOpAdaptor {
public:
  Mmt4DOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  Mmt4DOpAdaptor(Mmt4DOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class Mmt4DOp : public ::mlir::Op<Mmt4DOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Mmt4DOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.mmt4d");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::Mmt4DOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNchwMaxOp declarations
//===----------------------------------------------------------------------===//

class PoolingNchwMaxOpAdaptor {
public:
  PoolingNchwMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  PoolingNchwMaxOpAdaptor(PoolingNchwMaxOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class PoolingNchwMaxOp : public ::mlir::Op<PoolingNchwMaxOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNchwMaxOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_nchw_max");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNchwMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNchwSumOp declarations
//===----------------------------------------------------------------------===//

class PoolingNchwSumOpAdaptor {
public:
  PoolingNchwSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  PoolingNchwSumOpAdaptor(PoolingNchwSumOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class PoolingNchwSumOp : public ::mlir::Op<PoolingNchwSumOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNchwSumOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_nchw_sum");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNchwSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcMaxOp declarations
//===----------------------------------------------------------------------===//

class PoolingNdhwcMaxOpAdaptor {
public:
  PoolingNdhwcMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  PoolingNdhwcMaxOpAdaptor(PoolingNdhwcMaxOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class PoolingNdhwcMaxOp : public ::mlir::Op<PoolingNdhwcMaxOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNdhwcMaxOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_ndhwc_max");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcMinOp declarations
//===----------------------------------------------------------------------===//

class PoolingNdhwcMinOpAdaptor {
public:
  PoolingNdhwcMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  PoolingNdhwcMinOpAdaptor(PoolingNdhwcMinOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class PoolingNdhwcMinOp : public ::mlir::Op<PoolingNdhwcMinOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNdhwcMinOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_ndhwc_min");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcMinOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNdhwcSumOp declarations
//===----------------------------------------------------------------------===//

class PoolingNdhwcSumOpAdaptor {
public:
  PoolingNdhwcSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  PoolingNdhwcSumOpAdaptor(PoolingNdhwcSumOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class PoolingNdhwcSumOp : public ::mlir::Op<PoolingNdhwcSumOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNdhwcSumOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_ndhwc_sum");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNdhwcSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMaxOp declarations
//===----------------------------------------------------------------------===//

class PoolingNhwcMaxOpAdaptor {
public:
  PoolingNhwcMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  PoolingNhwcMaxOpAdaptor(PoolingNhwcMaxOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class PoolingNhwcMaxOp : public ::mlir::Op<PoolingNhwcMaxOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNhwcMaxOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_nhwc_max");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMaxOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMaxUnsignedOp declarations
//===----------------------------------------------------------------------===//

class PoolingNhwcMaxUnsignedOpAdaptor {
public:
  PoolingNhwcMaxUnsignedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  PoolingNhwcMaxUnsignedOpAdaptor(PoolingNhwcMaxUnsignedOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class PoolingNhwcMaxUnsignedOp : public ::mlir::Op<PoolingNhwcMaxUnsignedOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNhwcMaxUnsignedOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_nhwc_max_unsigned");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMaxUnsignedOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMinOp declarations
//===----------------------------------------------------------------------===//

class PoolingNhwcMinOpAdaptor {
public:
  PoolingNhwcMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  PoolingNhwcMinOpAdaptor(PoolingNhwcMinOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class PoolingNhwcMinOp : public ::mlir::Op<PoolingNhwcMinOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNhwcMinOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_nhwc_min");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMinOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcMinUnsignedOp declarations
//===----------------------------------------------------------------------===//

class PoolingNhwcMinUnsignedOpAdaptor {
public:
  PoolingNhwcMinUnsignedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  PoolingNhwcMinUnsignedOpAdaptor(PoolingNhwcMinUnsignedOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class PoolingNhwcMinUnsignedOp : public ::mlir::Op<PoolingNhwcMinUnsignedOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNhwcMinUnsignedOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_nhwc_min_unsigned");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcMinUnsignedOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::PoolingNhwcSumOp declarations
//===----------------------------------------------------------------------===//

class PoolingNhwcSumOpAdaptor {
public:
  PoolingNhwcSumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  PoolingNhwcSumOpAdaptor(PoolingNhwcSumOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class PoolingNhwcSumOp : public ::mlir::Op<PoolingNhwcSumOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ConvolutionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PoolingNhwcSumOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dilations"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("strides")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr dilationsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr dilationsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr stridesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr stridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.pooling_nhwc_sum");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  ::mlir::DenseIntElementsAttr stridesAttr();
  ::mlir::DenseIntElementsAttr strides();
  ::mlir::DenseIntElementsAttr dilationsAttr();
  ::mlir::DenseIntElementsAttr dilations();
  void stridesAttr(::mlir::DenseIntElementsAttr attr);
  void dilationsAttr(::mlir::DenseIntElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, Attribute strides, Attribute dilations, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

      bool hasDynamicIndexingMaps();
      LogicalResult verifyIndexingMapRequiredAttributes();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::PoolingNhwcSumOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::QuantizedBatchMatmulOp declarations
//===----------------------------------------------------------------------===//

class QuantizedBatchMatmulOpAdaptor {
public:
  QuantizedBatchMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  QuantizedBatchMatmulOpAdaptor(QuantizedBatchMatmulOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class QuantizedBatchMatmulOp : public ::mlir::Op<QuantizedBatchMatmulOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = QuantizedBatchMatmulOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.quantized_batch_matmul");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::QuantizedBatchMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::QuantizedMatmulOp declarations
//===----------------------------------------------------------------------===//

class QuantizedMatmulOpAdaptor {
public:
  QuantizedMatmulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  QuantizedMatmulOpAdaptor(QuantizedMatmulOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class QuantizedMatmulOp : public ::mlir::Op<QuantizedMatmulOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = QuantizedMatmulOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.quantized_matmul");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::QuantizedMatmulOp)

namespace mlir {
namespace linalg {

//===----------------------------------------------------------------------===//
// ::mlir::linalg::VecmatOp declarations
//===----------------------------------------------------------------------===//

class VecmatOpAdaptor {
public:
  VecmatOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  VecmatOpAdaptor(VecmatOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange inputs();
  ::mlir::ValueRange outputs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class VecmatOp : public ::mlir::Op<VecmatOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::linalg::LinalgOp::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::linalg::ContractionOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = VecmatOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("linalg.vecmat");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range inputs();
  ::mlir::Operation::operand_range outputs();
  ::mlir::MutableOperandRange inputsMutable();
  ::mlir::MutableOperandRange outputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range result_tensors();
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTensorTypes, ValueRange operands, ArrayRef<NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  // Return whether the op accesses the iteration indices.
  bool hasIndexSemantics() {
    return !this->getBody()->getOps<IndexOp>().empty();
  }

  LogicalResult reifyResultShapes(OpBuilder &b,
      ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
    return llvm::cast<LinalgOp>(getOperation()).reifyResultShapes(b,
        reifiedReturnShapes);
  }

  void getSuccessorRegions(
      Optional<unsigned> index, ArrayRef<Attribute> operands,
      SmallVectorImpl<RegionSuccessor> &regions) {
    // Op has a region, but conceptually the control flow does not enter the
    // region.
  }

    // Auto-generated.
    ArrayAttr iterator_types();
    ArrayAttr indexing_maps();
    static void regionBuilder(ImplicitLocOpBuilder &b,
                              Block &block, ArrayRef<NamedAttribute> attrs);
    static std::function<void(ImplicitLocOpBuilder &,
                              Block &, ArrayRef<NamedAttribute>)>
    getRegionBuilder() {
      return regionBuilder;
    }

    // Generic methods.
    static unsigned getNumRegionArgs();
    std::string getLibraryCallName();

};
} // namespace linalg
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::linalg::VecmatOp)


#endif  // GET_OP_CLASSES

