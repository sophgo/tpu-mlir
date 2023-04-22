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
namespace tensor {
class CastOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class CollapseShapeOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class DimOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class EmptyOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class ExpandShapeOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class ExtractOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class ExtractSliceOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class FromElementsOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class GatherOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class GenerateOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class InsertOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class InsertSliceOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class PackOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class PadOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class ParallelInsertSliceOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class RankOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class ReshapeOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class ScatterOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class SplatOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class UnPackOp;
} // namespace tensor
} // namespace mlir
namespace mlir {
namespace tensor {
class YieldOp;
} // namespace tensor
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::CastOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CastOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CastOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class CastOpGenericAdaptor : public detail::CastOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CastOpGenericAdaptorBase;
public:
  CastOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSource() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CastOpAdaptor : public CastOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CastOpGenericAdaptor::CastOpGenericAdaptor;
  CastOpAdaptor(CastOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CastOp : public ::mlir::Op<CastOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::CastOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CastOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CastOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.cast");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getSource();
  ::mlir::MutableOperandRange getSourceMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getDest();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::Value source);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static bool areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::CastOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::CollapseShapeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CollapseShapeOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CollapseShapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getReassociationAttr();
  ::mlir::ArrayAttr getReassociation();
};
} // namespace detail
template <typename RangeT>
class CollapseShapeOpGenericAdaptor : public detail::CollapseShapeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CollapseShapeOpGenericAdaptorBase;
public:
  CollapseShapeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSrc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CollapseShapeOpAdaptor : public CollapseShapeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CollapseShapeOpGenericAdaptor::CollapseShapeOpGenericAdaptor;
  CollapseShapeOpAdaptor(CollapseShapeOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CollapseShapeOp : public ::mlir::Op<CollapseShapeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CollapseShapeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CollapseShapeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("reassociation")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getReassociationAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getReassociationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.collapse_shape");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getSrc();
  ::mlir::MutableOperandRange getSrcMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getResult();
  ::mlir::ArrayAttr getReassociationAttr();
  ::mlir::ArrayAttr getReassociation();
  void setReassociationAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value src, ArrayRef<ReassociationIndices> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationIndices> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::ArrayAttr reassociation);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ArrayAttr reassociation);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  static StringRef getReassociationAttrStrName() { return "reassociation"; }
  SmallVector<AffineMap, 4> getReassociationMaps();
  SmallVector<ReassociationExprs, 4> getReassociationExprs();
  SmallVector<ReassociationIndices, 4> getReassociationIndices() {
    SmallVector<ReassociationIndices, 4> reassociationIndices;
    for (auto attr : getReassociation())
      reassociationIndices.push_back(llvm::to_vector<2>(
          llvm::map_range(attr.cast<ArrayAttr>(), [&](Attribute indexAttr) {
            return indexAttr.cast<IntegerAttr>().getInt();
          })));
    return reassociationIndices;
  }
  RankedTensorType getSrcType() {
    return getSrc().getType().cast<RankedTensorType>();
  }
  RankedTensorType getResultType() {
    return getResult().getType().cast<RankedTensorType>();
  }

  static RankedTensorType
  inferCollapsedType(RankedTensorType type, ArrayRef<AffineMap> reassociation);
  static RankedTensorType
  inferCollapsedType(RankedTensorType type, 
                     SmallVector<ReassociationIndices> reassociation);
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::CollapseShapeOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::DimOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DimOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  DimOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class DimOpGenericAdaptor : public detail::DimOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DimOpGenericAdaptorBase;
public:
  DimOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSource() {
    return (*getODSOperands(0).begin());
  }

  ValueT getIndex() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DimOpAdaptor : public DimOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DimOpGenericAdaptor::DimOpGenericAdaptor;
  DimOpAdaptor(DimOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DimOp : public ::mlir::Op<DimOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::ShapedDimOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DimOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DimOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.dim");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getSource();
  ::mlir::TypedValue<::mlir::IndexType> getIndex();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getIndexMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::IndexType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, int64_t index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value index);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  /// Helper function to get the index as a simple integer if it is constant.
  std::optional<int64_t> getConstantIndex();

  /// Interface method of ShapedDimOpInterface: Return the source tensor.
  Value getShapedValue() { return getSource(); }

  /// Interface method of ShapedDimOpInterface: Return the dimension.
  OpFoldResult getDimension() { return getIndex(); }

  /// Interface method for ConditionallySpeculatable.
  Speculation::Speculatability getSpeculatability();
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::DimOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::EmptyOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class EmptyOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  EmptyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class EmptyOpGenericAdaptor : public detail::EmptyOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::EmptyOpGenericAdaptorBase;
public:
  EmptyOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getDynamicSizes() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class EmptyOpAdaptor : public EmptyOpGenericAdaptor<::mlir::ValueRange> {
public:
  using EmptyOpGenericAdaptor::EmptyOpGenericAdaptor;
  EmptyOpAdaptor(EmptyOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class EmptyOp : public ::mlir::Op<EmptyOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = EmptyOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = EmptyOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.empty");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getDynamicSizes();
  ::mlir::MutableOperandRange getDynamicSizesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<int64_t> staticShape, Type elementType, Attribute encoding = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<int64_t> staticShape, Type elementType, ValueRange dynamicSizes, Attribute encoding = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<OpFoldResult> sizes, Type elementType, Attribute encoding = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange dynamicSizes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult reifyResultShapes(::mlir::OpBuilder &builder, ::mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  RankedTensorType getType() {
    return getResult().getType().cast<RankedTensorType>();
  }

  // Return both static and dynamic sizes as a list of `OpFoldResult`.
  SmallVector<OpFoldResult> getMixedSizes();

  // Return the Value of the dynamic size of the tensor at dimension `idx`.
  // Asserts that the shape is dynamic at that `idx`.
  Value getDynamicSize(unsigned idx);
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::EmptyOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ExpandShapeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ExpandShapeOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ExpandShapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getReassociationAttr();
  ::mlir::ArrayAttr getReassociation();
};
} // namespace detail
template <typename RangeT>
class ExpandShapeOpGenericAdaptor : public detail::ExpandShapeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExpandShapeOpGenericAdaptorBase;
public:
  ExpandShapeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSrc() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExpandShapeOpAdaptor : public ExpandShapeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExpandShapeOpGenericAdaptor::ExpandShapeOpGenericAdaptor;
  ExpandShapeOpAdaptor(ExpandShapeOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ExpandShapeOp : public ::mlir::Op<ExpandShapeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExpandShapeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExpandShapeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("reassociation")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getReassociationAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getReassociationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.expand_shape");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getSrc();
  ::mlir::MutableOperandRange getSrcMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getResult();
  ::mlir::ArrayAttr getReassociationAttr();
  ::mlir::ArrayAttr getReassociation();
  void setReassociationAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationIndices> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value src, ::mlir::ArrayAttr reassociation);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::ArrayAttr reassociation);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  static StringRef getReassociationAttrStrName() { return "reassociation"; }
  SmallVector<AffineMap, 4> getReassociationMaps();
  SmallVector<ReassociationExprs, 4> getReassociationExprs();
  SmallVector<ReassociationIndices, 4> getReassociationIndices() {
    SmallVector<ReassociationIndices, 4> reassociationIndices;
    for (auto attr : getReassociation())
      reassociationIndices.push_back(llvm::to_vector<2>(
          llvm::map_range(attr.cast<ArrayAttr>(), [&](Attribute indexAttr) {
            return indexAttr.cast<IntegerAttr>().getInt();
          })));
    return reassociationIndices;
  }
  RankedTensorType getSrcType() {
    return getSrc().getType().cast<RankedTensorType>();
  }
  RankedTensorType getResultType() {
    return getResult().getType().cast<RankedTensorType>();
  }

  int64_t getCorrespondingSourceDim(int64_t resultDim);
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::ExpandShapeOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ExtractOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ExtractOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ExtractOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ExtractOpGenericAdaptor : public detail::ExtractOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExtractOpGenericAdaptorBase;
public:
  ExtractOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getTensor() {
    return (*getODSOperands(0).begin());
  }

  RangeT getIndices() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExtractOpAdaptor : public ExtractOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExtractOpGenericAdaptor::ExtractOpGenericAdaptor;
  ExtractOpAdaptor(ExtractOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ExtractOp : public ::mlir::Op<ExtractOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExtractOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExtractOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.extract");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getTensor();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::MutableOperandRange getTensorMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value tensor, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::ExtractOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ExtractSliceOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ExtractSliceOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ExtractSliceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI64ArrayAttr getStaticOffsetsAttr();
  ::llvm::ArrayRef<int64_t> getStaticOffsets();
  ::mlir::DenseI64ArrayAttr getStaticSizesAttr();
  ::llvm::ArrayRef<int64_t> getStaticSizes();
  ::mlir::DenseI64ArrayAttr getStaticStridesAttr();
  ::llvm::ArrayRef<int64_t> getStaticStrides();
};
} // namespace detail
template <typename RangeT>
class ExtractSliceOpGenericAdaptor : public detail::ExtractSliceOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExtractSliceOpGenericAdaptorBase;
public:
  ExtractSliceOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSource() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOffsets() {
    return getODSOperands(1);
  }

  RangeT getSizes() {
    return getODSOperands(2);
  }

  RangeT getStrides() {
    return getODSOperands(3);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ExtractSliceOpAdaptor : public ExtractSliceOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExtractSliceOpGenericAdaptor::ExtractSliceOpGenericAdaptor;
  ExtractSliceOpAdaptor(ExtractSliceOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ExtractSliceOp : public ::mlir::Op<ExtractSliceOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OffsetSizeAndStrideOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExtractSliceOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExtractSliceOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("static_offsets"), ::llvm::StringRef("static_sizes"), ::llvm::StringRef("static_strides")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getStaticOffsetsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getStaticOffsetsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getStaticSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getStaticSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getStaticStridesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getStaticStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.extract_slice");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getSource();
  ::mlir::Operation::operand_range getOffsets();
  ::mlir::Operation::operand_range getSizes();
  ::mlir::Operation::operand_range getStrides();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getOffsetsMutable();
  ::mlir::MutableOperandRange getSizesMutable();
  ::mlir::MutableOperandRange getStridesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getResult();
  ::mlir::DenseI64ArrayAttr getStaticOffsetsAttr();
  ::llvm::ArrayRef<int64_t> getStaticOffsets();
  ::mlir::DenseI64ArrayAttr getStaticSizesAttr();
  ::llvm::ArrayRef<int64_t> getStaticSizes();
  ::mlir::DenseI64ArrayAttr getStaticStridesAttr();
  ::llvm::ArrayRef<int64_t> getStaticStrides();
  void setStaticOffsetsAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticOffsets(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticSizesAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticSizes(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticStridesAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticStrides(::llvm::ArrayRef<int64_t> attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, RankedTensorType resultType, Value source, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ValueRange offsets, ValueRange sizes, ValueRange strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, RankedTensorType resultType, Value source, ValueRange offsets, ValueRange sizes, ValueRange strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ArrayRef<Range> ranges, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  ::mlir::LogicalResult reifyResultShapes(::mlir::OpBuilder &builder, ::mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// Return the type of the base tensor operand.
  ::mlir::RankedTensorType getSourceType() { 
    return getSource().getType().cast<RankedTensorType>();
  }

  /// Return the type of the result tensor.
  ::mlir::RankedTensorType getResultType() { 
    return getResult().getType().cast<RankedTensorType>();
  }

  /// Return the dynamic sizes for this subview operation if specified.
  ::mlir::Operation::operand_range getDynamicSizes() { return getSizes(); }

  /// Return the list of Range (i.e. offset, size, stride). Each
  /// Range entry contains either the dynamic value or a ConstantIndexOp
  /// constructed with `b` at location `loc`.
  ::mlir::SmallVector<::mlir::Range, 8> getOrCreateRanges(
      ::mlir::OpBuilder &b, ::mlir::Location loc) {
    return ::mlir::getOrCreateRanges(*this, b, loc);
  }

  /// The result of an extract_slice is always a tensor.
  // TODO: deprecate
  RankedTensorType getType() {
    return getResultType();
  }

  /// Compute the rank-reduction mask that can be applied to map the source
  /// tensor type to the result tensor type by dropping unit dims.
  std::optional<llvm::SmallDenseSet<unsigned>>
  computeRankReductionMask() {
    return ::mlir::computeRankReductionMask(getSourceType().getShape(),
                                            getType().getShape());
  };

  /// An extract_slice result type can be inferred, when it is not
  /// rank-reduced, from the source type and the static representation of
  /// offsets, sizes and strides. Special sentinels encode the dynamic case.
  static RankedTensorType inferResultType(
    RankedTensorType sourceTensorType,
    ArrayRef<int64_t> staticOffsets,
    ArrayRef<int64_t> staticSizes,
    ArrayRef<int64_t> staticStrides);
  static RankedTensorType inferResultType(
    RankedTensorType sourceTensorType,
    ArrayRef<OpFoldResult> staticOffsets,
    ArrayRef<OpFoldResult> staticSizes,
    ArrayRef<OpFoldResult> staticStrides);

  /// If the rank is reduced (i.e. the desiredResultRank is smaller than the
  /// number of sizes), drop as many size 1 as needed to produce an inferred type
  /// with the desired rank.
  ///
  /// Note that there may be multiple ways to compute this rank-reduced type:
  ///   e.g. 1x6x1 can rank-reduce to either 1x6 or 6x1 2-D tensors.
  ///
  /// To disambiguate, this function always drops the first 1 sizes occurrences.
  static RankedTensorType inferCanonicalRankReducedResultType(
    unsigned resultRank,
    RankedTensorType sourceRankedTensorType,
    ArrayRef<int64_t> staticOffsets,
    ArrayRef<int64_t> staticSizes,
    ArrayRef<int64_t> staticStrides);
  static RankedTensorType inferCanonicalRankReducedResultType(
    unsigned resultRank,
    RankedTensorType sourceRankedTensorType,
    ArrayRef<OpFoldResult> staticOffsets,
    ArrayRef<OpFoldResult> staticSizes,
    ArrayRef<OpFoldResult> staticStrides);

  /// Return the expected rank of each of the`static_offsets`, `static_sizes`
  /// and `static_strides` attributes.
  std::array<unsigned, 3> getArrayAttrMaxRanks() {
    unsigned rank = getSourceType().getRank();
    return {rank, rank, rank};
  }

  /// Return the number of leading operands before the `offsets`, `sizes` and
  /// and `strides` operands.
  static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

  /// Return the dimensions of the source that are dropped in the
  /// result when the result is rank-reduced.
  llvm::SmallBitVector getDroppedDims();

  /// Given a `value`, asserted to be of RankedTensorType, build an
  /// ExtractSliceOp that results in a rank-reducing extract to the desired
  /// tensor shape and return the new value created.
  /// If the shape of `value` is already the `desiredShape`, just return
  /// `value`.
  /// If the shape of `value` cannot be rank-reduced to `desiredShape`, fail.
  static FailureOr<Value> rankReduceIfNeeded(
    OpBuilder &b, Location loc, Value value, ArrayRef<int64_t> desiredShape);
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::ExtractSliceOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::FromElementsOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class FromElementsOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  FromElementsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class FromElementsOpGenericAdaptor : public detail::FromElementsOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::FromElementsOpGenericAdaptorBase;
public:
  FromElementsOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getElements() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class FromElementsOpAdaptor : public FromElementsOpGenericAdaptor<::mlir::ValueRange> {
public:
  using FromElementsOpGenericAdaptor::FromElementsOpGenericAdaptor;
  FromElementsOpAdaptor(FromElementsOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class FromElementsOp : public ::mlir::Op<FromElementsOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FromElementsOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = FromElementsOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.from_elements");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getElements();
  ::mlir::MutableOperandRange getElementsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange elements);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange elements);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::FromElementsOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::GatherOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GatherOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  GatherOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI64ArrayAttr getGatherDimsAttr();
  ::llvm::ArrayRef<int64_t> getGatherDims();
  ::mlir::UnitAttr getUniqueAttr();
  bool getUnique();
};
} // namespace detail
template <typename RangeT>
class GatherOpGenericAdaptor : public detail::GatherOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GatherOpGenericAdaptorBase;
public:
  GatherOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSource() {
    return (*getODSOperands(0).begin());
  }

  ValueT getIndices() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GatherOpAdaptor : public GatherOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GatherOpGenericAdaptor::GatherOpGenericAdaptor;
  GatherOpAdaptor(GatherOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class GatherOp : public ::mlir::Op<GatherOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GatherOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GatherOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("gather_dims"), ::llvm::StringRef("unique")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getGatherDimsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getGatherDimsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getUniqueAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getUniqueAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.gather");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getSource();
  ::mlir::TypedValue<::mlir::RankedTensorType> getIndices();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getResult();
  ::mlir::DenseI64ArrayAttr getGatherDimsAttr();
  ::llvm::ArrayRef<int64_t> getGatherDims();
  ::mlir::UnitAttr getUniqueAttr();
  bool getUnique();
  void setGatherDimsAttr(::mlir::DenseI64ArrayAttr attr);
  void setGatherDims(::llvm::ArrayRef<int64_t> attrValue);
  void setUniqueAttr(::mlir::UnitAttr attr);
  void setUnique(bool attrValue);
  ::mlir::Attribute removeUniqueAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value indices, ::mlir::DenseI64ArrayAttr gather_dims, /*optional*/::mlir::UnitAttr unique);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value indices, ::mlir::DenseI64ArrayAttr gather_dims, /*optional*/::mlir::UnitAttr unique);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value indices, ::llvm::ArrayRef<int64_t> gather_dims, /*optional*/bool unique = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value indices, ::llvm::ArrayRef<int64_t> gather_dims, /*optional*/bool unique = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  // TODO: InferTypeOpInterface once enough confidence is built with
  // tensor<tensor> and its lwoering to memref<memref>.
  static RankedTensorType inferResultType(RankedTensorType sourceType,
                                          RankedTensorType indicesType,
                                          ArrayRef<int64_t> gatherDims,
                                          bool rankReduced);
  RankedTensorType getIndicesType() {
    return getIndices().getType().cast<RankedTensorType>();
  }
  RankedTensorType getSourceType() {
    return getSource().getType().cast<RankedTensorType>();
  }
  RankedTensorType getResultType() {
    return getResult().getType().cast<RankedTensorType>();
  }
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::GatherOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::GenerateOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GenerateOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  GenerateOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getBody();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class GenerateOpGenericAdaptor : public detail::GenerateOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GenerateOpGenericAdaptorBase;
public:
  GenerateOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getDynamicExtents() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GenerateOpAdaptor : public GenerateOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GenerateOpGenericAdaptor::GenerateOpGenericAdaptor;
  GenerateOpAdaptor(GenerateOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class GenerateOp : public ::mlir::Op<GenerateOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<mlir::tensor::YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::ReifyRankedShapedTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GenerateOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GenerateOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.generate");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getDynamicExtents();
  ::mlir::MutableOperandRange getDynamicExtentsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getResult();
  ::mlir::Region &getBody();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultTy, ValueRange dynamicExtents, function_ref<void(OpBuilder &, Location, ValueRange)> odsArg2);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange dynamicExtents);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult verifyRegions();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  ::mlir::LogicalResult reifyResultShapes(::mlir::OpBuilder &builder, ::mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::GenerateOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::InsertOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class InsertOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  InsertOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class InsertOpGenericAdaptor : public detail::InsertOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::InsertOpGenericAdaptorBase;
public:
  InsertOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getScalar() {
    return (*getODSOperands(0).begin());
  }

  ValueT getDest() {
    return (*getODSOperands(1).begin());
  }

  RangeT getIndices() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class InsertOpAdaptor : public InsertOpGenericAdaptor<::mlir::ValueRange> {
public:
  using InsertOpGenericAdaptor::InsertOpGenericAdaptor;
  InsertOpAdaptor(InsertOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class InsertOp : public ::mlir::Op<InsertOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::DestinationStyleOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InsertOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = InsertOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.insert");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getScalar();
  ::mlir::TypedValue<::mlir::RankedTensorType> getDest();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::MutableOperandRange getScalarMutable();
  ::mlir::MutableOperandRange getDestMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value scalar, ::mlir::Value dest, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value scalar, ::mlir::Value dest, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value scalar, ::mlir::Value dest, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
    return {1, 2};  // `dest` operand
  }
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::InsertOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::InsertSliceOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class InsertSliceOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  InsertSliceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI64ArrayAttr getStaticOffsetsAttr();
  ::llvm::ArrayRef<int64_t> getStaticOffsets();
  ::mlir::DenseI64ArrayAttr getStaticSizesAttr();
  ::llvm::ArrayRef<int64_t> getStaticSizes();
  ::mlir::DenseI64ArrayAttr getStaticStridesAttr();
  ::llvm::ArrayRef<int64_t> getStaticStrides();
};
} // namespace detail
template <typename RangeT>
class InsertSliceOpGenericAdaptor : public detail::InsertSliceOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::InsertSliceOpGenericAdaptorBase;
public:
  InsertSliceOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSource() {
    return (*getODSOperands(0).begin());
  }

  ValueT getDest() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOffsets() {
    return getODSOperands(2);
  }

  RangeT getSizes() {
    return getODSOperands(3);
  }

  RangeT getStrides() {
    return getODSOperands(4);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class InsertSliceOpAdaptor : public InsertSliceOpGenericAdaptor<::mlir::ValueRange> {
public:
  using InsertSliceOpGenericAdaptor::InsertSliceOpGenericAdaptor;
  InsertSliceOpAdaptor(InsertSliceOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class InsertSliceOp : public ::mlir::Op<InsertSliceOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::DestinationStyleOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OffsetSizeAndStrideOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InsertSliceOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = InsertSliceOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("static_offsets"), ::llvm::StringRef("static_sizes"), ::llvm::StringRef("static_strides")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getStaticOffsetsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getStaticOffsetsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getStaticSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getStaticSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getStaticStridesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getStaticStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.insert_slice");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getSource();
  ::mlir::TypedValue<::mlir::RankedTensorType> getDest();
  ::mlir::Operation::operand_range getOffsets();
  ::mlir::Operation::operand_range getSizes();
  ::mlir::Operation::operand_range getStrides();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getDestMutable();
  ::mlir::MutableOperandRange getOffsetsMutable();
  ::mlir::MutableOperandRange getSizesMutable();
  ::mlir::MutableOperandRange getStridesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getResult();
  ::mlir::DenseI64ArrayAttr getStaticOffsetsAttr();
  ::llvm::ArrayRef<int64_t> getStaticOffsets();
  ::mlir::DenseI64ArrayAttr getStaticSizesAttr();
  ::llvm::ArrayRef<int64_t> getStaticSizes();
  ::mlir::DenseI64ArrayAttr getStaticStridesAttr();
  ::llvm::ArrayRef<int64_t> getStaticStrides();
  void setStaticOffsetsAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticOffsets(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticSizesAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticSizes(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticStridesAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticStrides(::llvm::ArrayRef<int64_t> attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, ValueRange offsets, ValueRange sizes, ValueRange strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, ArrayRef<Range> ranges, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  ::mlir::LogicalResult reifyResultShapes(::mlir::OpBuilder &builder, ::mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  /// Return the type of the base tensor operand.
  ::mlir::RankedTensorType getSourceType() { 
    return getSource().getType().cast<RankedTensorType>();
  }

  /// Return the type of the result tensor.
  ::mlir::RankedTensorType getResultType() { 
    return getResult().getType().cast<RankedTensorType>();
  }

  /// Return the dynamic sizes for this subview operation if specified.
  ::mlir::Operation::operand_range getDynamicSizes() { return getSizes(); }

  /// Return the list of Range (i.e. offset, size, stride). Each
  /// Range entry contains either the dynamic value or a ConstantIndexOp
  /// constructed with `b` at location `loc`.
  ::mlir::SmallVector<::mlir::Range, 8> getOrCreateRanges(
      ::mlir::OpBuilder &b, ::mlir::Location loc) {
    return ::mlir::getOrCreateRanges(*this, b, loc);
  }

  /// The result of a insert_slice is always a tensor.
  // TODO: Deprecate this method.
  RankedTensorType getType() {
    return getResultType();
  }

  /// The `dest` type is the same as the result type.
  RankedTensorType getDestType() {
    return getResultType();
  }

  /// Return the expected rank of each of the`static_offsets`, `static_sizes`
  /// and `static_strides` attributes.
  std::array<unsigned, 3> getArrayAttrMaxRanks() {
    unsigned rank = getResultType().getRank();
    return {rank, rank, rank};
  }

  /// Return the dimensions of the dest that are omitted to insert a source
  /// when the result is rank-extended.
  llvm::SmallBitVector getDroppedDims();

  /// Return the number of leading operands before the `offsets`, `sizes` and
  /// and `strides` operands.
  static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }

  std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
    return {1, 2};  // `dest` operand
  }
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::InsertSliceOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::PackOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class PackOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  PackOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI64ArrayAttr getOuterDimsPermAttr();
  ::llvm::ArrayRef<int64_t> getOuterDimsPerm();
  ::mlir::DenseI64ArrayAttr getInnerDimsPosAttr();
  ::llvm::ArrayRef<int64_t> getInnerDimsPos();
  ::mlir::DenseI64ArrayAttr getStaticInnerTilesAttr();
  ::llvm::ArrayRef<int64_t> getStaticInnerTiles();
};
} // namespace detail
template <typename RangeT>
class PackOpGenericAdaptor : public detail::PackOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PackOpGenericAdaptorBase;
public:
  PackOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSource() {
    return (*getODSOperands(0).begin());
  }

  ValueT getDest() {
    return (*getODSOperands(1).begin());
  }

  ValueT getPaddingValue() {
    auto operands = getODSOperands(2);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getInnerTiles() {
    return getODSOperands(3);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class PackOpAdaptor : public PackOpGenericAdaptor<::mlir::ValueRange> {
public:
  using PackOpGenericAdaptor::PackOpGenericAdaptor;
  PackOpAdaptor(PackOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class PackOp : public ::mlir::Op<PackOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::DestinationStyleOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PackOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PackOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("inner_dims_pos"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("outer_dims_perm"), ::llvm::StringRef("static_inner_tiles")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getInnerDimsPosAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInnerDimsPosAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOuterDimsPermAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOuterDimsPermAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getStaticInnerTilesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getStaticInnerTilesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.pack");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getSource();
  ::mlir::TypedValue<::mlir::RankedTensorType> getDest();
  ::mlir::Value getPaddingValue();
  ::mlir::Operation::operand_range getInnerTiles();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getDestMutable();
  ::mlir::MutableOperandRange getPaddingValueMutable();
  ::mlir::MutableOperandRange getInnerTilesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getResult();
  ::mlir::DenseI64ArrayAttr getOuterDimsPermAttr();
  ::llvm::ArrayRef<int64_t> getOuterDimsPerm();
  ::mlir::DenseI64ArrayAttr getInnerDimsPosAttr();
  ::llvm::ArrayRef<int64_t> getInnerDimsPos();
  ::mlir::DenseI64ArrayAttr getStaticInnerTilesAttr();
  ::llvm::ArrayRef<int64_t> getStaticInnerTiles();
  void setOuterDimsPermAttr(::mlir::DenseI64ArrayAttr attr);
  void setOuterDimsPerm(::std::optional<::llvm::ArrayRef<int64_t>> attrValue);
  void setInnerDimsPosAttr(::mlir::DenseI64ArrayAttr attr);
  void setInnerDimsPos(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticInnerTilesAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticInnerTiles(::llvm::ArrayRef<int64_t> attrValue);
  ::mlir::Attribute removeOuterDimsPermAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, ArrayRef<int64_t> innerDimsPos, ArrayRef<OpFoldResult> innerTiles, std::optional<Value> paddingValue = std::nullopt, ArrayRef<int64_t> outerDimsPerm = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value padding_value, /*optional*/::mlir::DenseI64ArrayAttr outer_dims_perm, ::mlir::DenseI64ArrayAttr inner_dims_pos, ::mlir::ValueRange inner_tiles, ::mlir::DenseI64ArrayAttr static_inner_tiles);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value padding_value, /*optional*/::mlir::DenseI64ArrayAttr outer_dims_perm, ::mlir::DenseI64ArrayAttr inner_dims_pos, ::mlir::ValueRange inner_tiles, ::mlir::DenseI64ArrayAttr static_inner_tiles);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value padding_value, /*optional*/::mlir::DenseI64ArrayAttr outer_dims_perm, ::mlir::DenseI64ArrayAttr inner_dims_pos, ::mlir::ValueRange inner_tiles, ::mlir::DenseI64ArrayAttr static_inner_tiles);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value padding_value, /*optional*/::llvm::ArrayRef<int64_t> outer_dims_perm, ::llvm::ArrayRef<int64_t> inner_dims_pos, ::mlir::ValueRange inner_tiles, ::llvm::ArrayRef<int64_t> static_inner_tiles);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value padding_value, /*optional*/::llvm::ArrayRef<int64_t> outer_dims_perm, ::llvm::ArrayRef<int64_t> inner_dims_pos, ::mlir::ValueRange inner_tiles, ::llvm::ArrayRef<int64_t> static_inner_tiles);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::Value padding_value, /*optional*/::llvm::ArrayRef<int64_t> outer_dims_perm, ::llvm::ArrayRef<int64_t> inner_dims_pos, ::mlir::ValueRange inner_tiles, ::llvm::ArrayRef<int64_t> static_inner_tiles);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::LogicalResult canonicalize(PackOp op, ::mlir::PatternRewriter &rewriter);
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  ::mlir::LogicalResult reifyResultShapes(::mlir::OpBuilder &builder, ::mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  size_t getSourceRank() { return getSourceType().getRank(); };
  size_t getDestRank() { return getDestType().getRank(); };
  RankedTensorType getSourceType() {
    return getSource().getType().cast<RankedTensorType>(); };
  RankedTensorType getDestType() {
    return getDest().getType().cast<RankedTensorType>(); };

  /// Return position for init operand. Init operand is `dest`.
  std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
    return {1, 2}; // `dest` operand
  }

  /// Interface method for ConditionallySpeculatable.
  Speculation::Speculatability getSpeculatability();

  /// Return a mapping from positions `inner_dims_pos` to their
  /// tile factors.
  DenseMap<int64_t, OpFoldResult> getDimAndTileMapping();

  /// Return the tile sizes as OpFoldResult.
  SmallVector<OpFoldResult> getMixedTiles();

  /// Return the tile sizes as `int64_t`. If a tile size is dynamic
  /// a sentinel `kDynamic` is introduced at that position in
  /// the returned vector.
  SmallVector<int64_t> getStaticTiles();

  // Method to get the shape of the result as `SmallVector<OpFoldResult>`.
  // This is a static method to allow getting the shape of the destination
  // expected while creating a `pack` op.
  static SmallVector<OpFoldResult> getResultShape(OpBuilder &builder,
      Location loc, ArrayRef<OpFoldResult> sourceDims,
      ArrayRef<OpFoldResult> innerTileDims, ArrayRef<int64_t> innerDimsPos,
      ArrayRef<int64_t> outerDimsPerm = {});

  // Method to get the `RankedTensorType` of the result based on the inner
  // tiles, position of the inner tiles (innerDimsPos)  and interchange vector
  // of outer loops (outerDimsPerm).
  static RankedTensorType inferPackedType(RankedTensorType sourceType,
      ArrayRef<int64_t> innerTileSizes, ArrayRef<int64_t> innerDimsPos,
      ArrayRef<int64_t> outerDimsPerm = {});

  // Returns true if we have enough static information to catch undefined
  // behavior when the tile size does not divide perfectly the dimension of
  // the input tensor.
  static bool requirePaddingValue(ArrayRef<int64_t> inputShape,
                                  ArrayRef<int64_t> innerDimsPos,
                                  ArrayRef<OpFoldResult> innerTiles);

  static Value createDestinationTensor(OpBuilder &b, Location loc,
      Value source, ArrayRef<OpFoldResult> innerTileSizes,
      ArrayRef<int64_t> innerDimsPos, ArrayRef<int64_t> outerDimsPerm);

  /// Build and return a new PackOp that is a clone of the current PackOp with
  /// (innerDimsPos, innerTiles) (resp. outerDimsPerm) are permuted by 
  /// innerPermutation (resp. outerPermutation).
  /// A new `tensor.empty` of the proper shape is built in the process.
  /// Asserts that:
  ///   - At least one of innerPermutation or outerPermutation is non-empty.
  ///   - If not empty, innerPermutation is a valid permutation of size
  ///     matching innerDimPos.
  ///   - If not empty, outerPermutation is a valid permutation of size 
  ///     matching outerDimsPerm.
  PackOp createTransposedClone(OpBuilder &b,
                               Location loc,
                               ArrayRef<int64_t> innerPermutation,
                               ArrayRef<int64_t> outerPermutation);

  /// Check if this PackOp is like a simple pad operation.
  /// In other words, this operation:
  /// 1. adds useless dimensions (dimension of size 1),
  /// 2. pads the other ones, and
  /// 3. doesn't shuffle the dimensions
  bool isLikePad();
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::PackOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::PadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class PadOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  PadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI64ArrayAttr getStaticLowAttr();
  ::llvm::ArrayRef<int64_t> getStaticLow();
  ::mlir::DenseI64ArrayAttr getStaticHighAttr();
  ::llvm::ArrayRef<int64_t> getStaticHigh();
  ::mlir::UnitAttr getNofoldAttr();
  bool getNofold();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class PadOpGenericAdaptor : public detail::PadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PadOpGenericAdaptorBase;
public:
  PadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSource() {
    return (*getODSOperands(0).begin());
  }

  RangeT getLow() {
    return getODSOperands(1);
  }

  RangeT getHigh() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class PadOpAdaptor : public PadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using PadOpGenericAdaptor::PadOpGenericAdaptor;
  PadOpAdaptor(PadOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class PadOp : public ::mlir::Op<PadOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<mlir::tensor::YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("nofold"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("static_high"), ::llvm::StringRef("static_low")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNofoldAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNofoldAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getStaticHighAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getStaticHighAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getStaticLowAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getStaticLowAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.pad");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getSource();
  ::mlir::Operation::operand_range getLow();
  ::mlir::Operation::operand_range getHigh();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getLowMutable();
  ::mlir::MutableOperandRange getHighMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getResult();
  ::mlir::Region &getRegion();
  ::mlir::DenseI64ArrayAttr getStaticLowAttr();
  ::llvm::ArrayRef<int64_t> getStaticLow();
  ::mlir::DenseI64ArrayAttr getStaticHighAttr();
  ::llvm::ArrayRef<int64_t> getStaticHigh();
  ::mlir::UnitAttr getNofoldAttr();
  bool getNofold();
  void setStaticLowAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticLow(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticHighAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticHigh(::llvm::ArrayRef<int64_t> attrValue);
  void setNofoldAttr(::mlir::UnitAttr attr);
  void setNofold(bool attrValue);
  ::mlir::Attribute removeNofoldAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value source, ArrayRef<int64_t> staticLow, ArrayRef<int64_t> staticHigh, ValueRange low, ValueRange high, bool nofold = false, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value source, ValueRange low, ValueRange high, bool nofold = false, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value source, ArrayRef<OpFoldResult> low, ArrayRef<OpFoldResult> high, bool nofold = false, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value source, ArrayRef<OpFoldResult> low, ArrayRef<OpFoldResult> high, Value constantPadValue, bool nofold = false, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange low, ::mlir::ValueRange high, ::mlir::DenseI64ArrayAttr static_low, ::mlir::DenseI64ArrayAttr static_high, /*optional*/::mlir::UnitAttr nofold);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange low, ::mlir::ValueRange high, ::mlir::DenseI64ArrayAttr static_low, ::mlir::DenseI64ArrayAttr static_high, /*optional*/::mlir::UnitAttr nofold);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::ValueRange low, ::mlir::ValueRange high, ::llvm::ArrayRef<int64_t> static_low, ::llvm::ArrayRef<int64_t> static_high, /*optional*/bool nofold = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::ValueRange low, ::mlir::ValueRange high, ::llvm::ArrayRef<int64_t> static_low, ::llvm::ArrayRef<int64_t> static_high, /*optional*/bool nofold = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult verifyRegions();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  static StringRef getStaticLowAttrStrName() {
    return "static_low";
  }

  static StringRef getStaticHighAttrStrName() {
    return "static_high";
  }

  RankedTensorType getSourceType() {
    return getSource().getType().cast<RankedTensorType>();
  }
  RankedTensorType getResultType() {
    return getResult().getType().cast<RankedTensorType>();
  }

  // Infer the shape of the result tensor given the type of the source tensor
  // and paddings. Known result dimensions that cannot necessarily be inferred
  // from low/high padding sizes can be optionally specified. Those will be
  // considered when computing the result type.
  static RankedTensorType inferResultType(
                              RankedTensorType sourceType,
                              ArrayRef<int64_t> staticLow,
                              ArrayRef<int64_t> staticHigh,
                              ArrayRef<int64_t> resultShape = {});

  // Return the pad value if it is a constant. Return null value otherwise.
  Value getConstantPaddingValue();

  // Return a vector of all the static or dynamic values (low/high padding) of
  // the op.
  inline SmallVector<OpFoldResult> getMixedPadImpl(ArrayRef<int64_t> staticAttrs,
                                                   ValueRange values) {
    Builder builder(*this);
    SmallVector<OpFoldResult> res;
    unsigned numDynamic = 0;
    unsigned count = staticAttrs.size();
    for (unsigned idx = 0; idx < count; ++idx) {
      if (ShapedType::isDynamic(staticAttrs[idx]))
        res.push_back(values[numDynamic++]);
      else
        res.push_back(builder.getI64IntegerAttr(staticAttrs[idx]));
    }
    return res;
  }
  SmallVector<OpFoldResult> getMixedLowPad() {
    return getMixedPadImpl(getStaticLow(), getLow());
  }
  SmallVector<OpFoldResult> getMixedHighPad() {
    return getMixedPadImpl(getStaticHigh(), getHigh());
  }
  // Return true if low padding is guaranteed to be 0.
  bool hasZeroLowPad() {
    return llvm::all_of(getMixedLowPad(), [](OpFoldResult ofr) {
      return getConstantIntValue(ofr) == static_cast<int64_t>(0);
    });
  }
  // Return true if high padding is guaranteed to be 0.
  bool hasZeroHighPad() {
    return llvm::all_of(getMixedHighPad(), [](OpFoldResult ofr) {
      return getConstantIntValue(ofr) == static_cast<int64_t>(0);
    });
  }
  /// Return the dimensions with a non-zero low or high padding.
  llvm::SmallBitVector getPaddedDims();
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::PadOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ParallelInsertSliceOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ParallelInsertSliceOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ParallelInsertSliceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI64ArrayAttr getStaticOffsetsAttr();
  ::llvm::ArrayRef<int64_t> getStaticOffsets();
  ::mlir::DenseI64ArrayAttr getStaticSizesAttr();
  ::llvm::ArrayRef<int64_t> getStaticSizes();
  ::mlir::DenseI64ArrayAttr getStaticStridesAttr();
  ::llvm::ArrayRef<int64_t> getStaticStrides();
};
} // namespace detail
template <typename RangeT>
class ParallelInsertSliceOpGenericAdaptor : public detail::ParallelInsertSliceOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ParallelInsertSliceOpGenericAdaptorBase;
public:
  ParallelInsertSliceOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSource() {
    return (*getODSOperands(0).begin());
  }

  ValueT getDest() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOffsets() {
    return getODSOperands(2);
  }

  RangeT getSizes() {
    return getODSOperands(3);
  }

  RangeT getStrides() {
    return getODSOperands(4);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ParallelInsertSliceOpAdaptor : public ParallelInsertSliceOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ParallelInsertSliceOpGenericAdaptor::ParallelInsertSliceOpGenericAdaptor;
  ParallelInsertSliceOpAdaptor(ParallelInsertSliceOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ParallelInsertSliceOp : public ::mlir::Op<ParallelInsertSliceOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OffsetSizeAndStrideOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelInsertSliceOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ParallelInsertSliceOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("static_offsets"), ::llvm::StringRef("static_sizes"), ::llvm::StringRef("static_strides")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getStaticOffsetsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getStaticOffsetsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getStaticSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getStaticSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getStaticStridesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getStaticStridesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.parallel_insert_slice");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getSource();
  ::mlir::TypedValue<::mlir::RankedTensorType> getDest();
  ::mlir::Operation::operand_range getOffsets();
  ::mlir::Operation::operand_range getSizes();
  ::mlir::Operation::operand_range getStrides();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getDestMutable();
  ::mlir::MutableOperandRange getOffsetsMutable();
  ::mlir::MutableOperandRange getSizesMutable();
  ::mlir::MutableOperandRange getStridesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::DenseI64ArrayAttr getStaticOffsetsAttr();
  ::llvm::ArrayRef<int64_t> getStaticOffsets();
  ::mlir::DenseI64ArrayAttr getStaticSizesAttr();
  ::llvm::ArrayRef<int64_t> getStaticSizes();
  ::mlir::DenseI64ArrayAttr getStaticStridesAttr();
  ::llvm::ArrayRef<int64_t> getStaticStrides();
  void setStaticOffsetsAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticOffsets(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticSizesAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticSizes(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticStridesAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticStrides(::llvm::ArrayRef<int64_t> attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, ArrayRef<Range> ranges, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, ValueRange offsets, ValueRange sizes, ValueRange strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::DenseI64ArrayAttr static_offsets, ::mlir::DenseI64ArrayAttr static_sizes, ::mlir::DenseI64ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::llvm::ArrayRef<int64_t> static_offsets, ::llvm::ArrayRef<int64_t> static_sizes, ::llvm::ArrayRef<int64_t> static_strides);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  Type yieldedType() { return getDest().getType(); }

  RankedTensorType getSourceType() {
    return getSource().getType().cast<RankedTensorType>();
  }

  RankedTensorType getDestType() {
    return getDest().getType().cast<RankedTensorType>();
  }

  ParallelCombiningOpInterface getParallelCombiningParent() {
    return dyn_cast<ParallelCombiningOpInterface>(
      getOperation()->getParentOp());
  }

  /// Return the expected rank of each of the `static_offsets`, `static_sizes`
  /// and `static_strides` attributes.
  std::array<unsigned, 3> getArrayAttrMaxRanks() {
    unsigned rank = getDestType().getRank();
    return {rank, rank, rank};
  }

  /// Return the number of leading operands before `offsets`, `sizes` and
  /// `strides` operands.
  static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

  /// Return the OpResult of the enclosing ForallOp that is
  /// corresponding to this ParallelInsertSliceOp.
  OpResult getTiedOpResult();

  /// Return the dimensions of the dest that are omitted to insert a source
  /// when the result is rank-extended.
  llvm::SmallBitVector getDroppedDims();
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::ParallelInsertSliceOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::RankOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class RankOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  RankOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class RankOpGenericAdaptor : public detail::RankOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::RankOpGenericAdaptorBase;
public:
  RankOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getTensor() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class RankOpAdaptor : public RankOpGenericAdaptor<::mlir::ValueRange> {
public:
  using RankOpGenericAdaptor::RankOpGenericAdaptor;
  RankOpAdaptor(RankOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class RankOp : public ::mlir::Op<RankOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RankOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = RankOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.rank");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getTensor();
  ::mlir::MutableOperandRange getTensorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value tensor);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::RankOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ReshapeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ReshapeOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ReshapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ReshapeOpGenericAdaptor : public detail::ReshapeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReshapeOpGenericAdaptorBase;
public:
  ReshapeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSource() {
    return (*getODSOperands(0).begin());
  }

  ValueT getShape() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReshapeOpAdaptor : public ReshapeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReshapeOpGenericAdaptor::ReshapeOpGenericAdaptor;
  ReshapeOpAdaptor(ReshapeOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ReshapeOp : public ::mlir::Op<ReshapeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReshapeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReshapeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.reshape");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getSource();
  ::mlir::TypedValue<::mlir::RankedTensorType> getShape();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getShapeMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TensorType resultType, Value operand, Value shape);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value shape);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value shape);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  TensorType getResultType() { return getResult().getType().cast<TensorType>(); }
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::ReshapeOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::ScatterOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ScatterOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ScatterOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI64ArrayAttr getScatterDimsAttr();
  ::llvm::ArrayRef<int64_t> getScatterDims();
  ::mlir::UnitAttr getUniqueAttr();
  bool getUnique();
};
} // namespace detail
template <typename RangeT>
class ScatterOpGenericAdaptor : public detail::ScatterOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ScatterOpGenericAdaptorBase;
public:
  ScatterOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSource() {
    return (*getODSOperands(0).begin());
  }

  ValueT getDest() {
    return (*getODSOperands(1).begin());
  }

  ValueT getIndices() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ScatterOpAdaptor : public ScatterOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ScatterOpGenericAdaptor::ScatterOpGenericAdaptor;
  ScatterOpAdaptor(ScatterOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ScatterOp : public ::mlir::Op<ScatterOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScatterOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ScatterOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("scatter_dims"), ::llvm::StringRef("unique")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getScatterDimsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getScatterDimsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getUniqueAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getUniqueAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.scatter");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getSource();
  ::mlir::TypedValue<::mlir::RankedTensorType> getDest();
  ::mlir::TypedValue<::mlir::RankedTensorType> getIndices();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getDestMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getResult();
  ::mlir::DenseI64ArrayAttr getScatterDimsAttr();
  ::llvm::ArrayRef<int64_t> getScatterDims();
  ::mlir::UnitAttr getUniqueAttr();
  bool getUnique();
  void setScatterDimsAttr(::mlir::DenseI64ArrayAttr attr);
  void setScatterDims(::llvm::ArrayRef<int64_t> attrValue);
  void setUniqueAttr(::mlir::UnitAttr attr);
  void setUnique(bool attrValue);
  ::mlir::Attribute removeUniqueAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, ::mlir::Value indices, ::mlir::DenseI64ArrayAttr scatter_dims, /*optional*/::mlir::UnitAttr unique);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::Value indices, ::mlir::DenseI64ArrayAttr scatter_dims, /*optional*/::mlir::UnitAttr unique);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, ::mlir::Value indices, ::llvm::ArrayRef<int64_t> scatter_dims, /*optional*/bool unique = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::Value indices, ::llvm::ArrayRef<int64_t> scatter_dims, /*optional*/bool unique = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  RankedTensorType getDestType() {
    return getDest().getType().cast<RankedTensorType>();
  }
  RankedTensorType getIndicesType() {
    return getIndices().getType().cast<RankedTensorType>();
  }
  RankedTensorType getSourceType() {
    return getSource().getType().cast<RankedTensorType>();
  }
  RankedTensorType getResultType() {
    return getResult().getType().cast<RankedTensorType>();
  }
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::ScatterOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::SplatOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SplatOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  SplatOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class SplatOpGenericAdaptor : public detail::SplatOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SplatOpGenericAdaptorBase;
public:
  SplatOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class SplatOpAdaptor : public SplatOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SplatOpGenericAdaptor::SplatOpGenericAdaptor;
  SplatOpAdaptor(SplatOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class SplatOp : public ::mlir::Op<SplatOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SplatOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SplatOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.splat");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getInput();
  ::mlir::MutableOperandRange getInputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getAggregate();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value element, Type aggregateType);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type aggregate, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::SplatOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::UnPackOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class UnPackOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  UnPackOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI64ArrayAttr getOuterDimsPermAttr();
  ::llvm::ArrayRef<int64_t> getOuterDimsPerm();
  ::mlir::DenseI64ArrayAttr getInnerDimsPosAttr();
  ::llvm::ArrayRef<int64_t> getInnerDimsPos();
  ::mlir::DenseI64ArrayAttr getStaticInnerTilesAttr();
  ::llvm::ArrayRef<int64_t> getStaticInnerTiles();
};
} // namespace detail
template <typename RangeT>
class UnPackOpGenericAdaptor : public detail::UnPackOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::UnPackOpGenericAdaptorBase;
public:
  UnPackOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getSource() {
    return (*getODSOperands(0).begin());
  }

  ValueT getDest() {
    return (*getODSOperands(1).begin());
  }

  RangeT getInnerTiles() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class UnPackOpAdaptor : public UnPackOpGenericAdaptor<::mlir::ValueRange> {
public:
  using UnPackOpGenericAdaptor::UnPackOpGenericAdaptor;
  UnPackOpAdaptor(UnPackOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class UnPackOp : public ::mlir::Op<UnPackOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::DestinationStyleOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::ReifyRankedShapedTypeOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = UnPackOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = UnPackOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("inner_dims_pos"), ::llvm::StringRef("outer_dims_perm"), ::llvm::StringRef("static_inner_tiles")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getInnerDimsPosAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInnerDimsPosAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOuterDimsPermAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOuterDimsPermAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getStaticInnerTilesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getStaticInnerTilesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.unpack");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getSource();
  ::mlir::TypedValue<::mlir::RankedTensorType> getDest();
  ::mlir::Operation::operand_range getInnerTiles();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getDestMutable();
  ::mlir::MutableOperandRange getInnerTilesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::RankedTensorType> getResult();
  ::mlir::DenseI64ArrayAttr getOuterDimsPermAttr();
  ::llvm::ArrayRef<int64_t> getOuterDimsPerm();
  ::mlir::DenseI64ArrayAttr getInnerDimsPosAttr();
  ::llvm::ArrayRef<int64_t> getInnerDimsPos();
  ::mlir::DenseI64ArrayAttr getStaticInnerTilesAttr();
  ::llvm::ArrayRef<int64_t> getStaticInnerTiles();
  void setOuterDimsPermAttr(::mlir::DenseI64ArrayAttr attr);
  void setOuterDimsPerm(::std::optional<::llvm::ArrayRef<int64_t>> attrValue);
  void setInnerDimsPosAttr(::mlir::DenseI64ArrayAttr attr);
  void setInnerDimsPos(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticInnerTilesAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticInnerTiles(::llvm::ArrayRef<int64_t> attrValue);
  ::mlir::Attribute removeOuterDimsPermAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, ArrayRef<int64_t> innerDimsPos, ArrayRef<OpFoldResult> innerTiles, ArrayRef<int64_t> outerDimsPerm = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::DenseI64ArrayAttr outer_dims_perm, ::mlir::DenseI64ArrayAttr inner_dims_pos, ::mlir::ValueRange inner_tiles, ::mlir::DenseI64ArrayAttr static_inner_tiles);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::DenseI64ArrayAttr outer_dims_perm, ::mlir::DenseI64ArrayAttr inner_dims_pos, ::mlir::ValueRange inner_tiles, ::mlir::DenseI64ArrayAttr static_inner_tiles);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, /*optional*/::mlir::DenseI64ArrayAttr outer_dims_perm, ::mlir::DenseI64ArrayAttr inner_dims_pos, ::mlir::ValueRange inner_tiles, ::mlir::DenseI64ArrayAttr static_inner_tiles);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value source, ::mlir::Value dest, /*optional*/::llvm::ArrayRef<int64_t> outer_dims_perm, ::llvm::ArrayRef<int64_t> inner_dims_pos, ::mlir::ValueRange inner_tiles, ::llvm::ArrayRef<int64_t> static_inner_tiles);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, /*optional*/::llvm::ArrayRef<int64_t> outer_dims_perm, ::llvm::ArrayRef<int64_t> inner_dims_pos, ::mlir::ValueRange inner_tiles, ::llvm::ArrayRef<int64_t> static_inner_tiles);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, /*optional*/::llvm::ArrayRef<int64_t> outer_dims_perm, ::llvm::ArrayRef<int64_t> inner_dims_pos, ::mlir::ValueRange inner_tiles, ::llvm::ArrayRef<int64_t> static_inner_tiles);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::LogicalResult canonicalize(UnPackOp op, ::mlir::PatternRewriter &rewriter);
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  ::mlir::LogicalResult reifyResultShapes(::mlir::OpBuilder &builder, ::mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  size_t getSourceRank() { return getSourceType().getRank(); };
  size_t getDestRank() { return getDestType().getRank(); };
  RankedTensorType getSourceType() {
    return getSource().getType().cast<RankedTensorType>(); };
  RankedTensorType getDestType() {
    return getDest().getType().cast<RankedTensorType>(); };

  /// Return position for init operand. Init operand is `dest`.
  std::pair<int64_t, int64_t> getDpsInitsPositionRange() {
    return {1, 2}; // `dest` operand
  }

  /// Interface method for ConditionallySpeculatable.
  Speculation::Speculatability getSpeculatability();

  /// Return a mapping from positions `inner_dims_pos` to their
  /// tile factors.
  DenseMap<int64_t, OpFoldResult> getDimAndTileMapping();

  /// Return the tile sizes as OpFoldResult.
  SmallVector<OpFoldResult> getMixedTiles();

  /// Return the tile sizes as `int64_t`. If a tile size is dynamic
  /// a sentinel `kDynamic` is introduced at that position in
  /// the returned vector.
  SmallVector<int64_t> getStaticTiles();

  static Value createDestinationTensor(OpBuilder &b, Location loc,
      Value source, ArrayRef<OpFoldResult> innerTileSizes,
      ArrayRef<int64_t> innerDimsPos, ArrayRef<int64_t> outerDimsPerm);

  /// Build and return a new UnPackOp that is a clone of the current UnPackOp
  /// with (innerDimsPos, innerTiles) (resp. outerDimsPerm) are permuted by 
  /// innerPermutation (resp. outerPermutation).
  /// Asserts that:
  ///   - At least one of innerPermutation or outerPermutation is non-empty.
  ///   - If not empty, innerPermutation is a valid permutation of size
  ///     matching innerDimPos.
  ///   - If not empty, outerPermutation is a valid permutation of size 
  ///     matching outerDimsPerm.
  UnPackOp createTransposedClone(OpBuilder &b,
                                 Location loc,
                                 Value transposedSource,
                                 ArrayRef<int64_t> innerPermutation,
                                 ArrayRef<int64_t> outerPermutation);

  /// Check if this UnPackOp is like a simple unpad operation.
  /// In other words, this operation:
  /// 1. drops useless dimensions (dimension of size 1), and
  /// 2. reduces dimensions in place (i.e., no tranpose.)
  bool isLikeUnPad();
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::UnPackOp)

namespace mlir {
namespace tensor {

//===----------------------------------------------------------------------===//
// ::mlir::tensor::YieldOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class YieldOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  YieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class YieldOpGenericAdaptor : public detail::YieldOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::YieldOpGenericAdaptorBase;
public:
  YieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getValue() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class YieldOpAdaptor : public YieldOpGenericAdaptor<::mlir::ValueRange> {
public:
  using YieldOpGenericAdaptor::YieldOpGenericAdaptor;
  YieldOpAdaptor(YieldOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::HasParent<::mlir::tensor::GenerateOp, ::mlir::tensor::PadOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YieldOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = YieldOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("tensor.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getValue();
  ::mlir::MutableOperandRange getValueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace tensor
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tensor::YieldOp)


#endif  // GET_OP_CLASSES

