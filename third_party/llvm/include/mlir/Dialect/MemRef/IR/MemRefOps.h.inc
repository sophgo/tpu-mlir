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
namespace memref {
class AssumeAlignmentOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class AtomicRMWOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class AtomicYieldOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class CopyOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class GenericAtomicRMWOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class LoadOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class AllocOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class AllocaOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class AllocaScopeOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class AllocaScopeReturnOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class CastOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class CollapseShapeOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class DeallocOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class DimOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class DmaStartOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class DmaWaitOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class ExpandShapeOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class ExtractAlignedPointerAsIndexOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class ExtractStridedMetadataOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class GetGlobalOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class GlobalOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class MemorySpaceCastOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class PrefetchOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class RankOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class ReallocOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class ReinterpretCastOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class ReshapeOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class StoreOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class TransposeOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class ViewOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class SubViewOp;
} // namespace memref
} // namespace mlir
namespace mlir {
namespace memref {
class TensorStoreOp;
} // namespace memref
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AssumeAlignmentOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AssumeAlignmentOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AssumeAlignmentOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getAlignmentAttr();
  uint32_t getAlignment();
};
} // namespace detail
template <typename RangeT>
class AssumeAlignmentOpGenericAdaptor : public detail::AssumeAlignmentOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AssumeAlignmentOpGenericAdaptorBase;
public:
  AssumeAlignmentOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemref() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AssumeAlignmentOpAdaptor : public AssumeAlignmentOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AssumeAlignmentOpGenericAdaptor::AssumeAlignmentOpGenericAdaptor;
  AssumeAlignmentOpAdaptor(AssumeAlignmentOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AssumeAlignmentOp : public ::mlir::Op<AssumeAlignmentOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AssumeAlignmentOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AssumeAlignmentOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("alignment")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAlignmentAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAlignmentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.assume_alignment");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::MutableOperandRange getMemrefMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr getAlignmentAttr();
  uint32_t getAlignment();
  void setAlignmentAttr(::mlir::IntegerAttr attr);
  void setAlignment(uint32_t attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, uint32_t alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, uint32_t alignment);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::AssumeAlignmentOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AtomicRMWOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AtomicRMWOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AtomicRMWOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::arith::AtomicRMWKindAttr getKindAttr();
  ::mlir::arith::AtomicRMWKind getKind();
};
} // namespace detail
template <typename RangeT>
class AtomicRMWOpGenericAdaptor : public detail::AtomicRMWOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AtomicRMWOpGenericAdaptorBase;
public:
  AtomicRMWOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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

  ValueT getMemref() {
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
class AtomicRMWOpAdaptor : public AtomicRMWOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AtomicRMWOpGenericAdaptor::AtomicRMWOpGenericAdaptor;
  AtomicRMWOpAdaptor(AtomicRMWOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AtomicRMWOp : public ::mlir::Op<AtomicRMWOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicRMWOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AtomicRMWOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("kind")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getKindAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getKindAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.atomic_rmw");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getValue();
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::MutableOperandRange getValueMutable();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getResult();
  ::mlir::arith::AtomicRMWKindAttr getKindAttr();
  ::mlir::arith::AtomicRMWKind getKind();
  void setKindAttr(::mlir::arith::AtomicRMWKindAttr attr);
  void setKind(::mlir::arith::AtomicRMWKind attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::arith::AtomicRMWKindAttr kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::arith::AtomicRMWKindAttr kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::arith::AtomicRMWKindAttr kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::arith::AtomicRMWKind kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::arith::AtomicRMWKind kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::arith::AtomicRMWKind kind, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
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
  MemRefType getMemRefType() {
    return getMemref().getType().cast<MemRefType>();
  }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::AtomicRMWOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AtomicYieldOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AtomicYieldOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AtomicYieldOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class AtomicYieldOpGenericAdaptor : public detail::AtomicYieldOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AtomicYieldOpGenericAdaptorBase;
public:
  AtomicYieldOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getResult() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AtomicYieldOpAdaptor : public AtomicYieldOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AtomicYieldOpGenericAdaptor::AtomicYieldOpGenericAdaptor;
  AtomicYieldOpAdaptor(AtomicYieldOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AtomicYieldOp : public ::mlir::Op<AtomicYieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::HasParent<GenericAtomicRMWOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicYieldOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AtomicYieldOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.atomic_yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getResult();
  ::mlir::MutableOperandRange getResultMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value result);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value result);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::AtomicYieldOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::CopyOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class CopyOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  CopyOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class CopyOpGenericAdaptor : public detail::CopyOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::CopyOpGenericAdaptorBase;
public:
  CopyOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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

  ValueT getTarget() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class CopyOpAdaptor : public CopyOpGenericAdaptor<::mlir::ValueRange> {
public:
  using CopyOpGenericAdaptor::CopyOpGenericAdaptor;
  CopyOpAdaptor(CopyOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class CopyOp : public ::mlir::Op<CopyOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::CopyOpInterface::Trait, ::mlir::OpTrait::SameOperandsElementType, ::mlir::OpTrait::SameOperandsShape, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CopyOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = CopyOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.copy");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::BaseMemRefType> getSource();
  ::mlir::TypedValue<::mlir::BaseMemRefType> getTarget();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value target);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value target);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::CopyOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::GenericAtomicRMWOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GenericAtomicRMWOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  GenericAtomicRMWOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getAtomicBody();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class GenericAtomicRMWOpGenericAdaptor : public detail::GenericAtomicRMWOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GenericAtomicRMWOpGenericAdaptorBase;
public:
  GenericAtomicRMWOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemref() {
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
class GenericAtomicRMWOpAdaptor : public GenericAtomicRMWOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GenericAtomicRMWOpGenericAdaptor::GenericAtomicRMWOpGenericAdaptor;
  GenericAtomicRMWOpAdaptor(GenericAtomicRMWOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class GenericAtomicRMWOp : public ::mlir::Op<GenericAtomicRMWOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::SingleBlockImplicitTerminator<AtomicYieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GenericAtomicRMWOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GenericAtomicRMWOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.generic_atomic_rmw");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getResult();
  ::mlir::Region &getAtomicBody();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value memref, ValueRange ivs);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
public:
  // TODO: remove post migrating callers.
  Region &body() { return getRegion(); }

  // The value stored in memref[ivs].
  Value getCurrentValue() {
    return getRegion().getArgument(0);
  }
  MemRefType getMemRefType() {
    return getMemref().getType().cast<MemRefType>();
  }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::GenericAtomicRMWOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::LoadOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class LoadOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  LoadOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getNontemporalAttr();
  bool getNontemporal();
};
} // namespace detail
template <typename RangeT>
class LoadOpGenericAdaptor : public detail::LoadOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::LoadOpGenericAdaptorBase;
public:
  LoadOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemref() {
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
class LoadOpAdaptor : public LoadOpGenericAdaptor<::mlir::ValueRange> {
public:
  using LoadOpGenericAdaptor::LoadOpGenericAdaptor;
  LoadOpAdaptor(LoadOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class LoadOp : public ::mlir::Op<LoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::InferTypeOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoadOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = LoadOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("nontemporal")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNontemporalAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNontemporalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.load");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getResult();
  ::mlir::BoolAttr getNontemporalAttr();
  bool getNontemporal();
  void setNontemporalAttr(::mlir::BoolAttr attr);
  void setNontemporal(::std::optional<bool> attrValue);
  ::mlir::Attribute removeNontemporalAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/::mlir::BoolAttr nontemporal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/::mlir::BoolAttr nontemporal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/::mlir::BoolAttr nontemporal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/bool nontemporal = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/bool nontemporal = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/bool nontemporal = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
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
  Value getMemRef() { return getOperand(0); }
  void setMemRef(Value value) { setOperand(0, value); }
  MemRefType getMemRefType() {
    return getMemRef().getType().cast<MemRefType>();
  }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::LoadOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AllocOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AllocOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AllocOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getAlignmentAttr();
  ::std::optional<uint64_t> getAlignment();
};
} // namespace detail
template <typename RangeT>
class AllocOpGenericAdaptor : public detail::AllocOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllocOpGenericAdaptorBase;
public:
  AllocOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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

  RangeT getSymbolOperands() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllocOpAdaptor : public AllocOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllocOpGenericAdaptor::AllocOpGenericAdaptor;
  AllocOpAdaptor(AllocOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AllocOp : public ::mlir::Op<AllocOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllocOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("alignment"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAlignmentAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAlignmentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.alloc");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getDynamicSizes();
  ::mlir::Operation::operand_range getSymbolOperands();
  ::mlir::MutableOperandRange getDynamicSizesMutable();
  ::mlir::MutableOperandRange getSymbolOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::IntegerAttr getAlignmentAttr();
  ::std::optional<uint64_t> getAlignment();
  void setAlignmentAttr(::mlir::IntegerAttr attr);
  void setAlignment(::std::optional<uint64_t> attrValue);
  ::mlir::Attribute removeAlignmentAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, IntegerAttr alignment = IntegerAttr());
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange dynamicSizes, IntegerAttr alignment = IntegerAttr());
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange dynamicSizes, ValueRange symbolOperands, IntegerAttr alignment = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type memref, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
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
  static StringRef getAlignmentAttrStrName() { return "alignment"; }

  MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

  SmallVector<OpFoldResult> getMixedSizes() {
    SmallVector<OpFoldResult> result;
    unsigned ctr = 0;
    OpBuilder b(getContext());
    for (int64_t i = 0, e = getType().getRank(); i < e; ++i) {
      if (getType().isDynamicDim(i)) {
        result.push_back(getDynamicSizes()[ctr++]);
      } else {
        result.push_back(b.getIndexAttr(getType().getShape()[i]));
      }
    }
    return result;
  }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::AllocOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AllocaOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AllocaOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AllocaOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getAlignmentAttr();
  ::std::optional<uint64_t> getAlignment();
};
} // namespace detail
template <typename RangeT>
class AllocaOpGenericAdaptor : public detail::AllocaOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllocaOpGenericAdaptorBase;
public:
  AllocaOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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

  RangeT getSymbolOperands() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllocaOpAdaptor : public AllocaOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllocaOpGenericAdaptor::AllocaOpGenericAdaptor;
  AllocaOpAdaptor(AllocaOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AllocaOp : public ::mlir::Op<AllocaOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocaOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllocaOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("alignment"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAlignmentAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAlignmentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.alloca");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getDynamicSizes();
  ::mlir::Operation::operand_range getSymbolOperands();
  ::mlir::MutableOperandRange getDynamicSizesMutable();
  ::mlir::MutableOperandRange getSymbolOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::IntegerAttr getAlignmentAttr();
  ::std::optional<uint64_t> getAlignment();
  void setAlignmentAttr(::mlir::IntegerAttr attr);
  void setAlignment(::std::optional<uint64_t> attrValue);
  ::mlir::Attribute removeAlignmentAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, IntegerAttr alignment = IntegerAttr());
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange dynamicSizes, IntegerAttr alignment = IntegerAttr());
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType memrefType, ValueRange dynamicSizes, ValueRange symbolOperands, IntegerAttr alignment = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type memref, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange dynamicSizes, ::mlir::ValueRange symbolOperands, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
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
  static StringRef getAlignmentAttrStrName() { return "alignment"; }

  MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

  SmallVector<OpFoldResult> getMixedSizes() {
    SmallVector<OpFoldResult> result;
    unsigned ctr = 0;
    OpBuilder b(getContext());
    for (int64_t i = 0, e = getType().getRank(); i < e; ++i) {
      if (getType().isDynamicDim(i)) {
        result.push_back(getDynamicSizes()[ctr++]);
      } else {
        result.push_back(b.getIndexAttr(getType().getShape()[i]));
      }
    }
    return result;
  }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::AllocaOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AllocaScopeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AllocaScopeOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AllocaScopeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getBodyRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class AllocaScopeOpGenericAdaptor : public detail::AllocaScopeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllocaScopeOpGenericAdaptorBase;
public:
  AllocaScopeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllocaScopeOpAdaptor : public AllocaScopeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllocaScopeOpGenericAdaptor::AllocaScopeOpGenericAdaptor;
  AllocaScopeOpAdaptor(AllocaScopeOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AllocaScopeOp : public ::mlir::Op<AllocaScopeOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<AllocaScopeReturnOp>::Impl, ::mlir::OpTrait::NoRegionArguments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::RegionBranchOpInterface::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocaScopeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllocaScopeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.alloca_scope");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getBodyRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getSuccessorRegions(::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
public:
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::AllocaScopeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::AllocaScopeReturnOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AllocaScopeReturnOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  AllocaScopeReturnOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class AllocaScopeReturnOpGenericAdaptor : public detail::AllocaScopeReturnOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AllocaScopeReturnOpGenericAdaptorBase;
public:
  AllocaScopeReturnOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getResults() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AllocaScopeReturnOpAdaptor : public AllocaScopeReturnOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AllocaScopeReturnOpGenericAdaptor::AllocaScopeReturnOpGenericAdaptor;
  AllocaScopeReturnOpAdaptor(AllocaScopeReturnOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AllocaScopeReturnOp : public ::mlir::Op<AllocaScopeReturnOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<AllocaScopeOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AllocaScopeReturnOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AllocaScopeReturnOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.alloca_scope.return");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getResults();
  ::mlir::MutableOperandRange getResultsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::AllocaScopeReturnOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::CastOp declarations
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
class CastOp : public ::mlir::Op<CastOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::BaseMemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::CastOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultShape, ::mlir::ViewLikeOpInterface::Trait> {
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
    return ::llvm::StringLiteral("memref.cast");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::BaseMemRefType> getSource();
  ::mlir::MutableOperandRange getSourceMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::BaseMemRefType> getDest();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::Value source);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static bool areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  /// Fold the given CastOp into consumer op.
  static bool canFoldIntoConsumerOp(CastOp castOp);

  Value getViewSource() { return getSource(); }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::CastOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::CollapseShapeOp declarations
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
class CollapseShapeOp : public ::mlir::Op<CollapseShapeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::ViewLikeOpInterface::Trait> {
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
    return ::llvm::StringLiteral("memref.collapse_shape");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getSrc();
  ::mlir::MutableOperandRange getSrcMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getResult();
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
  };

  MemRefType getSrcType() { return getSrc().getType().cast<MemRefType>(); }

  MemRefType getResultType() { return getResult().getType().cast<MemRefType>(); }

  Value getViewSource() { return getSrc(); }

  /// Return `true` if this source MemRef type is guaranteed to be collapsible
  /// according to the given reassociation indices. In the presence of dynamic
  /// strides this is usually not the case.
  static bool isGuaranteedCollapsible(
      MemRefType srcType, ArrayRef<ReassociationIndices> reassociation);

  static MemRefType computeCollapsedType(
      MemRefType srcType, ArrayRef<ReassociationIndices> reassociation);
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::CollapseShapeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::DeallocOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DeallocOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  DeallocOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class DeallocOpGenericAdaptor : public detail::DeallocOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DeallocOpGenericAdaptorBase;
public:
  DeallocOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemref() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DeallocOpAdaptor : public DeallocOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DeallocOpGenericAdaptor::DeallocOpGenericAdaptor;
  DeallocOpAdaptor(DeallocOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DeallocOp : public ::mlir::Op<DeallocOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DeallocOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DeallocOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.dealloc");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::BaseMemRefType> getMemref();
  ::mlir::MutableOperandRange getMemrefMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::DeallocOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::DimOp declarations
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
class DimOp : public ::mlir::Op<DimOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::ConditionallySpeculatable::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::ShapedDimOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
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
    return ::llvm::StringLiteral("memref.dim");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getSource();
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

  /// Interface method of ShapedDimOpInterface: Return the source memref.
  Value getShapedValue() { return getSource(); }

  /// Interface method of ShapedDimOpInterface: Return the dimension.
  OpFoldResult getDimension() { return getIndex(); }

  /// Interface method for ConditionallySpeculatable.
  Speculation::Speculatability getSpeculatability();
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::DimOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::DmaStartOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DmaStartOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  DmaStartOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class DmaStartOpGenericAdaptor : public detail::DmaStartOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DmaStartOpGenericAdaptorBase;
public:
  DmaStartOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return getODSOperands(0);
  }

private:
  RangeT odsOperands;
};
class DmaStartOpAdaptor : public DmaStartOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DmaStartOpGenericAdaptor::DmaStartOpGenericAdaptor;
  DmaStartOpAdaptor(DmaStartOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DmaStartOp : public ::mlir::Op<DmaStartOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DmaStartOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DmaStartOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.dma_start");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getOperands();
  ::mlir::MutableOperandRange getOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value srcMemRef, ValueRange srcIndices, Value destMemRef, ValueRange destIndices, Value numElements, Value tagMemRef, ValueRange tagIndices, Value stride = {}, Value elementsPerStride = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
public:
  // Returns the source MemRefType for this DMA operation.
  Value getSrcMemRef() { return getOperand(0); }
  // Returns the rank (number of indices) of the source MemRefType.
  unsigned getSrcMemRefRank() {
    return getSrcMemRef().getType().cast<MemRefType>().getRank();
  }
  // Returns the source memref indices for this DMA operation.
  operand_range getSrcIndices() {
    return {(*this)->operand_begin() + 1,
            (*this)->operand_begin() + 1 + getSrcMemRefRank()};
  }

  // Returns the destination MemRefType for this DMA operations.
  Value getDstMemRef() { return getOperand(1 + getSrcMemRefRank()); }
  // Returns the rank (number of indices) of the destination MemRefType.
  unsigned getDstMemRefRank() {
    return getDstMemRef().getType().cast<MemRefType>().getRank();
  }
  unsigned getSrcMemorySpace() {
    return getSrcMemRef().getType().cast<MemRefType>().getMemorySpaceAsInt();
  }
  unsigned getDstMemorySpace() {
    return getDstMemRef().getType().cast<MemRefType>().getMemorySpaceAsInt();
  }

  // Returns the destination memref indices for this DMA operation.
  operand_range getDstIndices() {
    return {(*this)->operand_begin() + 1 + getSrcMemRefRank() + 1,
            (*this)->operand_begin() + 1 + getSrcMemRefRank() + 1 +
                getDstMemRefRank()};
  }

  // Returns the number of elements being transferred by this DMA operation.
  Value getNumElements() {
    return getOperand(1 + getSrcMemRefRank() + 1 + getDstMemRefRank());
  }

  // Returns the Tag MemRef for this DMA operation.
  Value getTagMemRef() {
    return getOperand(1 + getSrcMemRefRank() + 1 + getDstMemRefRank() + 1);
  }

  // Returns the rank (number of indices) of the tag MemRefType.
  unsigned getTagMemRefRank() {
    return getTagMemRef().getType().cast<MemRefType>().getRank();
  }

  // Returns the tag memref index for this DMA operation.
  operand_range getTagIndices() {
    unsigned tagIndexStartPos =
        1 + getSrcMemRefRank() + 1 + getDstMemRefRank() + 1 + 1;
    return {(*this)->operand_begin() + tagIndexStartPos,
            (*this)->operand_begin() + tagIndexStartPos + getTagMemRefRank()};
  }

  /// Returns true if this is a DMA from a faster memory space to a slower
  /// one.
  bool isDestMemorySpaceFaster() {
    return (getSrcMemorySpace() < getDstMemorySpace());
  }

  /// Returns true if this is a DMA from a slower memory space to a faster
  /// one.
  bool isSrcMemorySpaceFaster() {
    // Assumes that a lower number is for a slower memory space.
    return (getDstMemorySpace() < getSrcMemorySpace());
  }

  /// Given a DMA start operation, returns the operand position of either the
  /// source or destination memref depending on the one that is at the higher
  /// level of the memory hierarchy. Asserts failure if neither is true.
  unsigned getFasterMemPos() {
    assert(isSrcMemorySpaceFaster() || isDestMemorySpaceFaster());
    return isSrcMemorySpaceFaster() ? 0 : getSrcMemRefRank() + 1;
  }

  bool isStrided() {
    return getNumOperands() != 1 + getSrcMemRefRank() + 1 +
                               getDstMemRefRank() + 1 + 1 +
                               getTagMemRefRank();
  }

  Value getStride() {
    if (!isStrided())
      return nullptr;
    return getOperand(getNumOperands() - 1 - 1);
  }

  Value getNumElementsPerStride() {
    if (!isStrided())
      return nullptr;
    return getOperand(getNumOperands() - 1);
  }

  void getEffects(
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
      effects) {
    effects.emplace_back(MemoryEffects::Read::get(), getSrcMemRef(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), getDstMemRef(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Read::get(), getTagMemRef(),
                         SideEffects::DefaultResource::get());
  }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::DmaStartOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::DmaWaitOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class DmaWaitOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  DmaWaitOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class DmaWaitOpGenericAdaptor : public detail::DmaWaitOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::DmaWaitOpGenericAdaptorBase;
public:
  DmaWaitOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getTagMemRef() {
    return (*getODSOperands(0).begin());
  }

  RangeT getTagIndices() {
    return getODSOperands(1);
  }

  ValueT getNumElements() {
    return (*getODSOperands(2).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class DmaWaitOpAdaptor : public DmaWaitOpGenericAdaptor<::mlir::ValueRange> {
public:
  using DmaWaitOpGenericAdaptor::DmaWaitOpGenericAdaptor;
  DmaWaitOpAdaptor(DmaWaitOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class DmaWaitOp : public ::mlir::Op<DmaWaitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DmaWaitOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = DmaWaitOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.dma_wait");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getTagMemRef();
  ::mlir::Operation::operand_range getTagIndices();
  ::mlir::TypedValue<::mlir::IndexType> getNumElements();
  ::mlir::MutableOperandRange getTagMemRefMutable();
  ::mlir::MutableOperandRange getTagIndicesMutable();
  ::mlir::MutableOperandRange getNumElementsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tagMemRef, ::mlir::ValueRange tagIndices, ::mlir::Value numElements);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tagMemRef, ::mlir::ValueRange tagIndices, ::mlir::Value numElements);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
  /// Returns the rank (number of indices) of the tag memref.
  unsigned getTagMemRefRank() {
    return getTagMemRef().getType().cast<MemRefType>().getRank();
  }
  void getEffects(
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &
      effects) {
    effects.emplace_back(MemoryEffects::Read::get(), getTagMemRef(),
                         SideEffects::DefaultResource::get());
  }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::DmaWaitOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ExpandShapeOp declarations
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
class ExpandShapeOp : public ::mlir::Op<ExpandShapeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::ViewLikeOpInterface::Trait> {
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
    return ::llvm::StringLiteral("memref.expand_shape");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getSrc();
  ::mlir::MutableOperandRange getSrcMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getResult();
  ::mlir::ArrayAttr getReassociationAttr();
  ::mlir::ArrayAttr getReassociation();
  void setReassociationAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationIndices> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value src, ArrayRef<ReassociationExprs> reassociation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<int64_t> resultShape, Value src, ArrayRef<ReassociationIndices> reassociation);
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
  };

  MemRefType getSrcType() { return getSrc().getType().cast<MemRefType>(); }

  MemRefType getResultType() { return getResult().getType().cast<MemRefType>(); }

  Value getViewSource() { return getSrc(); }

  static FailureOr<MemRefType> computeExpandedType(
      MemRefType srcType, ArrayRef<int64_t> resultShape,
      ArrayRef<ReassociationIndices> reassociation);
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::ExpandShapeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ExtractAlignedPointerAsIndexOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ExtractAlignedPointerAsIndexOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ExtractAlignedPointerAsIndexOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ExtractAlignedPointerAsIndexOpGenericAdaptor : public detail::ExtractAlignedPointerAsIndexOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExtractAlignedPointerAsIndexOpGenericAdaptorBase;
public:
  ExtractAlignedPointerAsIndexOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class ExtractAlignedPointerAsIndexOpAdaptor : public ExtractAlignedPointerAsIndexOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExtractAlignedPointerAsIndexOpGenericAdaptor::ExtractAlignedPointerAsIndexOpGenericAdaptor;
  ExtractAlignedPointerAsIndexOpAdaptor(ExtractAlignedPointerAsIndexOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ExtractAlignedPointerAsIndexOp : public ::mlir::Op<ExtractAlignedPointerAsIndexOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExtractAlignedPointerAsIndexOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExtractAlignedPointerAsIndexOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.extract_aligned_pointer_as_index");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getSource();
  ::mlir::MutableOperandRange getSourceMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::IndexType> getAlignedPointer();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type aligned_pointer, ::mlir::Value source);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::ExtractAlignedPointerAsIndexOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ExtractStridedMetadataOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ExtractStridedMetadataOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ExtractStridedMetadataOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ExtractStridedMetadataOpGenericAdaptor : public detail::ExtractStridedMetadataOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExtractStridedMetadataOpGenericAdaptorBase;
public:
  ExtractStridedMetadataOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class ExtractStridedMetadataOpAdaptor : public ExtractStridedMetadataOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExtractStridedMetadataOpGenericAdaptor::ExtractStridedMetadataOpGenericAdaptor;
  ExtractStridedMetadataOpAdaptor(ExtractStridedMetadataOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ExtractStridedMetadataOp : public ::mlir::Op<ExtractStridedMetadataOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<2>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExtractStridedMetadataOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExtractStridedMetadataOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.extract_strided_metadata");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getSource();
  ::mlir::MutableOperandRange getSourceMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getBaseBuffer();
  ::mlir::TypedValue<::mlir::IndexType> getOffset();
  ::mlir::Operation::result_range getSizes();
  ::mlir::Operation::result_range getStrides();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type base_buffer, ::mlir::Type offset, ::mlir::TypeRange sizes, ::mlir::TypeRange strides, ::mlir::Value source);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  /// Return a vector of all the static or dynamic sizes of the op, while
  /// statically inferring the sizes of the dynamic sizes, when possible.
  /// This is best effort.
  /// E.g., if `getSizes` returns `[%dyn_size0, %dyn_size1]`, but the
  /// source memref type is `memref<2x8xi16>`, this method will
  /// return `[2, 8]`.
  /// Similarly if the resulting memref type is `memref<2x?xi16>`, but
  /// `%dyn_size1` can statically be pinned to a constant value, this
  /// constant value is returned instead of `%dyn_size`.
  SmallVector<OpFoldResult> getConstifiedMixedSizes();
  /// Similar to `getConstifiedMixedSizes` but for strides.
  SmallVector<OpFoldResult> getConstifiedMixedStrides();
  /// Similar to `getConstifiedMixedSizes` but for the offset.
  OpFoldResult getConstifiedMixedOffset();
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::ExtractStridedMetadataOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::GetGlobalOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GetGlobalOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  GetGlobalOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FlatSymbolRefAttr getNameAttr();
  ::llvm::StringRef getName();
};
} // namespace detail
template <typename RangeT>
class GetGlobalOpGenericAdaptor : public detail::GetGlobalOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GetGlobalOpGenericAdaptorBase;
public:
  GetGlobalOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GetGlobalOpAdaptor : public GetGlobalOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GetGlobalOpGenericAdaptor::GetGlobalOpGenericAdaptor;
  GetGlobalOpAdaptor(GetGlobalOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class GetGlobalOp : public ::mlir::Op<GetGlobalOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::SymbolUserOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GetGlobalOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GetGlobalOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("name")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.get_global");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getResult();
  ::mlir::FlatSymbolRefAttr getNameAttr();
  ::llvm::StringRef getName();
  void setNameAttr(::mlir::FlatSymbolRefAttr attr);
  void setName(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::FlatSymbolRefAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::FlatSymbolRefAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::llvm::StringRef name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifySymbolUses(::mlir::SymbolTableCollection &symbolTable);
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
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::GetGlobalOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::GlobalOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GlobalOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  GlobalOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::StringAttr getSymVisibilityAttr();
  ::std::optional< ::llvm::StringRef > getSymVisibility();
  ::mlir::TypeAttr getTypeAttr();
  ::mlir::MemRefType getType();
  ::mlir::Attribute getInitialValueAttr();
  ::std::optional<::mlir::Attribute> getInitialValue();
  ::mlir::UnitAttr getConstantAttr();
  bool getConstant();
  ::mlir::IntegerAttr getAlignmentAttr();
  ::std::optional<uint64_t> getAlignment();
};
} // namespace detail
template <typename RangeT>
class GlobalOpGenericAdaptor : public detail::GlobalOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GlobalOpGenericAdaptorBase;
public:
  GlobalOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GlobalOpAdaptor : public GlobalOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GlobalOpGenericAdaptor::GlobalOpGenericAdaptor;
  GlobalOpAdaptor(GlobalOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class GlobalOp : public ::mlir::Op<GlobalOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GlobalOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GlobalOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("alignment"), ::llvm::StringRef("constant"), ::llvm::StringRef("initial_value"), ::llvm::StringRef("sym_name"), ::llvm::StringRef("sym_visibility"), ::llvm::StringRef("type")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAlignmentAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAlignmentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getConstantAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getConstantAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getInitialValueAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getInitialValueAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getSymNameAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getSymNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getSymVisibilityAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getSymVisibilityAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getTypeAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.global");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::StringAttr getSymVisibilityAttr();
  ::std::optional< ::llvm::StringRef > getSymVisibility();
  ::mlir::TypeAttr getTypeAttr();
  ::mlir::MemRefType getType();
  ::mlir::Attribute getInitialValueAttr();
  ::std::optional<::mlir::Attribute> getInitialValue();
  ::mlir::UnitAttr getConstantAttr();
  bool getConstant();
  ::mlir::IntegerAttr getAlignmentAttr();
  ::std::optional<uint64_t> getAlignment();
  void setSymNameAttr(::mlir::StringAttr attr);
  void setSymName(::llvm::StringRef attrValue);
  void setSymVisibilityAttr(::mlir::StringAttr attr);
  void setSymVisibility(::std::optional<::llvm::StringRef> attrValue);
  void setTypeAttr(::mlir::TypeAttr attr);
  void setType(::mlir::MemRefType attrValue);
  void setInitialValueAttr(::mlir::Attribute attr);
  void setConstantAttr(::mlir::UnitAttr attr);
  void setConstant(bool attrValue);
  void setAlignmentAttr(::mlir::IntegerAttr attr);
  void setAlignment(::std::optional<uint64_t> attrValue);
  ::mlir::Attribute removeSymVisibilityAttr();
  ::mlir::Attribute removeInitialValueAttr();
  ::mlir::Attribute removeConstantAttr();
  ::mlir::Attribute removeAlignmentAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::TypeAttr type, /*optional*/::mlir::Attribute initial_value, /*optional*/::mlir::UnitAttr constant, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::TypeAttr type, /*optional*/::mlir::Attribute initial_value, /*optional*/::mlir::UnitAttr constant, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::MemRefType type, /*optional*/::mlir::Attribute initial_value, /*optional*/bool constant, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr sym_visibility, ::mlir::MemRefType type, /*optional*/::mlir::Attribute initial_value, /*optional*/bool constant, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  bool isExternal() { return !getInitialValue(); }
  bool isUninitialized() {
    return !isExternal() && getInitialValue()->isa<UnitAttr>();
  }
  /// Returns the constant initial value if the memref.global is a constant,
  /// or null otherwise.
  ElementsAttr getConstantInitValue();
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::GlobalOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::MemorySpaceCastOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class MemorySpaceCastOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  MemorySpaceCastOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class MemorySpaceCastOpGenericAdaptor : public detail::MemorySpaceCastOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MemorySpaceCastOpGenericAdaptorBase;
public:
  MemorySpaceCastOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class MemorySpaceCastOpAdaptor : public MemorySpaceCastOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MemorySpaceCastOpGenericAdaptor::MemorySpaceCastOpGenericAdaptor;
  MemorySpaceCastOpAdaptor(MemorySpaceCastOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class MemorySpaceCastOp : public ::mlir::Op<MemorySpaceCastOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::BaseMemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::CastOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::SameOperandsAndResultElementType, ::mlir::OpTrait::SameOperandsAndResultShape, ::mlir::ViewLikeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MemorySpaceCastOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MemorySpaceCastOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.memory_space_cast");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::BaseMemRefType> getSource();
  ::mlir::MutableOperandRange getSourceMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::BaseMemRefType> getDest();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type dest, ::mlir::Value source);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static bool areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  Value getViewSource() { return getSource(); }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::MemorySpaceCastOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::PrefetchOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class PrefetchOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  PrefetchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getIsWriteAttr();
  bool getIsWrite();
  ::mlir::IntegerAttr getLocalityHintAttr();
  uint32_t getLocalityHint();
  ::mlir::BoolAttr getIsDataCacheAttr();
  bool getIsDataCache();
};
} // namespace detail
template <typename RangeT>
class PrefetchOpGenericAdaptor : public detail::PrefetchOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PrefetchOpGenericAdaptorBase;
public:
  PrefetchOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getMemref() {
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
class PrefetchOpAdaptor : public PrefetchOpGenericAdaptor<::mlir::ValueRange> {
public:
  using PrefetchOpGenericAdaptor::PrefetchOpGenericAdaptor;
  PrefetchOpAdaptor(PrefetchOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class PrefetchOp : public ::mlir::Op<PrefetchOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PrefetchOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PrefetchOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("isDataCache"), ::llvm::StringRef("isWrite"), ::llvm::StringRef("localityHint")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getIsDataCacheAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIsDataCacheAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIsWriteAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIsWriteAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getLocalityHintAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getLocalityHintAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.prefetch");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr getIsWriteAttr();
  bool getIsWrite();
  ::mlir::IntegerAttr getLocalityHintAttr();
  uint32_t getLocalityHint();
  ::mlir::BoolAttr getIsDataCacheAttr();
  bool getIsDataCache();
  void setIsWriteAttr(::mlir::BoolAttr attr);
  void setIsWrite(bool attrValue);
  void setLocalityHintAttr(::mlir::IntegerAttr attr);
  void setLocalityHint(uint32_t attrValue);
  void setIsDataCacheAttr(::mlir::BoolAttr attr);
  void setIsDataCache(bool attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr isWrite, ::mlir::IntegerAttr localityHint, ::mlir::BoolAttr isDataCache);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, bool isWrite, uint32_t localityHint, bool isDataCache);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
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
  MemRefType getMemRefType() {
    return getMemref().getType().cast<MemRefType>();
  }
  static StringRef getLocalityHintAttrStrName() { return "localityHint"; }
  static StringRef getIsWriteAttrStrName() { return "isWrite"; }
  static StringRef getIsDataCacheAttrStrName() { return "isDataCache"; }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::PrefetchOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::RankOp declarations
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

  ValueT getMemref() {
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
class RankOp : public ::mlir::Op<RankOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::IndexType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait> {
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
    return ::llvm::StringLiteral("memref.rank");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::BaseMemRefType> getMemref();
  ::mlir::MutableOperandRange getMemrefMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::RankOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ReallocOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ReallocOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ReallocOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getAlignmentAttr();
  ::std::optional<uint64_t> getAlignment();
};
} // namespace detail
template <typename RangeT>
class ReallocOpGenericAdaptor : public detail::ReallocOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReallocOpGenericAdaptorBase;
public:
  ReallocOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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

  ValueT getDynamicResultSize() {
    auto operands = getODSOperands(1);
    return operands.empty() ? ValueT{} : (*operands.begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReallocOpAdaptor : public ReallocOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReallocOpGenericAdaptor::ReallocOpGenericAdaptor;
  ReallocOpAdaptor(ReallocOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ReallocOp : public ::mlir::Op<ReallocOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReallocOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReallocOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("alignment")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getAlignmentAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAlignmentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.realloc");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getSource();
  ::mlir::TypedValue<::mlir::IndexType> getDynamicResultSize();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getDynamicResultSizeMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr getAlignmentAttr();
  ::std::optional<uint64_t> getAlignment();
  void setAlignmentAttr(::mlir::IntegerAttr attr);
  void setAlignment(::std::optional<uint64_t> attrValue);
  ::mlir::Attribute removeAlignmentAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, Value dynamicResultSize = Value());
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value source, /*optional*/::mlir::Value dynamicResultSize, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, /*optional*/::mlir::Value dynamicResultSize, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
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
  /// The result of a realloc is always a memref.
  MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::ReallocOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ReinterpretCastOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ReinterpretCastOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ReinterpretCastOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

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
class ReinterpretCastOpGenericAdaptor : public detail::ReinterpretCastOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReinterpretCastOpGenericAdaptorBase;
public:
  ReinterpretCastOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class ReinterpretCastOpAdaptor : public ReinterpretCastOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReinterpretCastOpGenericAdaptor::ReinterpretCastOpGenericAdaptor;
  ReinterpretCastOpAdaptor(ReinterpretCastOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ReinterpretCastOp : public ::mlir::Op<ReinterpretCastOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OffsetSizeAndStrideOpInterface::Trait, ::mlir::ViewLikeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReinterpretCastOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReinterpretCastOpGenericAdaptor<RangeT>;
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
    return ::llvm::StringLiteral("memref.reinterpret_cast");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::BaseMemRefType> getSource();
  ::mlir::Operation::operand_range getOffsets();
  ::mlir::Operation::operand_range getSizes();
  ::mlir::Operation::operand_range getStrides();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getOffsetsMutable();
  ::mlir::MutableOperandRange getSizesMutable();
  ::mlir::MutableOperandRange getStridesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getResult();
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, OpFoldResult offset, ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, int64_t offset, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, Value offset, ValueRange sizes, ValueRange strides, ArrayRef<NamedAttribute> attrs = {});
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
  /// Returns the dynamic sizes for this subview operation if specified.
  ::mlir::Operation::operand_range getDynamicSizes() { return getSizes(); }

  /// Return the list of Range (i.e. offset, size, stride). Each
  /// Range entry contains either the dynamic value or a ConstantIndexOp
  /// constructed with `b` at location `loc`.
  ::mlir::SmallVector<::mlir::Range, 8> getOrCreateRanges(
      ::mlir::OpBuilder &b, ::mlir::Location loc) {
    return ::mlir::getOrCreateRanges(*this, b, loc);
  }

  // The result of the op is always a ranked memref.
  MemRefType getType() { return getResult().getType(); }
  Value getViewSource() { return getSource(); }

  /// Return the rank of the result type.
  unsigned getResultRank() { return getType().getRank(); }

  /// Return the expected rank of each of the`static_offsets`, `static_sizes`
  /// and `static_strides` attributes.
  std::array<unsigned, 3> getArrayAttrMaxRanks() {
    unsigned resultRank = getType().getRank();
    return {1, resultRank, resultRank};
  }

  /// Return the number of leading operands before the `offsets`, `sizes` and
  /// and `strides` operands.
  static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

  /// Return a vector of all the static or dynamic sizes of the op, while
  /// statically inferring the sizes of the dynamic sizes, when possible.
  /// This is best effort.
  /// E.g., if `getMixedSizes` returns `[2, %dyn_size]`, but the resulting
  /// memref type is `memref<2x8xi16>`, this method will return `[2, 8]`.
  /// Similarly if the resulting memref type is `memref<2x?xi16>`, but
  /// `%dyn_size` can statically be pinned to a constant value, this
  /// constant value is returned instead of `%dyn_size`.
  SmallVector<OpFoldResult> getConstifiedMixedSizes();
  /// Similar to `getConstifiedMixedSizes` but for strides.
  SmallVector<OpFoldResult> getConstifiedMixedStrides();
  /// Similar to `getConstifiedMixedSizes` but for the offset.
  OpFoldResult getConstifiedMixedOffset();
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::ReinterpretCastOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ReshapeOp declarations
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
class ReshapeOp : public ::mlir::Op<ReshapeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::BaseMemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::ViewLikeOpInterface::Trait> {
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
    return ::llvm::StringLiteral("memref.reshape");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::BaseMemRefType> getSource();
  ::mlir::TypedValue<::mlir::MemRefType> getShape();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getShapeMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::BaseMemRefType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value operand, Value shape);
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
  MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  Value getViewSource() { return getSource(); }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::ReshapeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::StoreOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class StoreOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  StoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getNontemporalAttr();
  bool getNontemporal();
};
} // namespace detail
template <typename RangeT>
class StoreOpGenericAdaptor : public detail::StoreOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::StoreOpGenericAdaptorBase;
public:
  StoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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

  ValueT getMemref() {
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
class StoreOpAdaptor : public StoreOpGenericAdaptor<::mlir::ValueRange> {
public:
  using StoreOpGenericAdaptor::StoreOpGenericAdaptor;
  StoreOpAdaptor(StoreOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class StoreOp : public ::mlir::Op<StoreOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::MemRefsNormalizable, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = StoreOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = StoreOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("nontemporal")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getNontemporalAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNontemporalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.store");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getValue();
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::MutableOperandRange getValueMutable();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr getNontemporalAttr();
  bool getNontemporal();
  void setNontemporalAttr(::mlir::BoolAttr attr);
  void setNontemporal(::std::optional<bool> attrValue);
  ::mlir::Attribute removeNontemporalAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value valueToStore, Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/::mlir::BoolAttr nontemporal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/::mlir::BoolAttr nontemporal);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/bool nontemporal = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, /*optional*/bool nontemporal = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
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
  Value getValueToStore() { return getOperand(0); }

  Value getMemRef() { return getOperand(1); }
  void setMemRef(Value value) { setOperand(1, value); }
  MemRefType getMemRefType() {
    return getMemRef().getType().cast<MemRefType>();
  }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::StoreOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::TransposeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TransposeOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  TransposeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::AffineMapAttr getPermutationAttr();
  ::mlir::AffineMap getPermutation();
};
} // namespace detail
template <typename RangeT>
class TransposeOpGenericAdaptor : public detail::TransposeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TransposeOpGenericAdaptorBase;
public:
  TransposeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getIn() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TransposeOpAdaptor : public TransposeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TransposeOpGenericAdaptor::TransposeOpGenericAdaptor;
  TransposeOpAdaptor(TransposeOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TransposeOp : public ::mlir::Op<TransposeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TransposeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TransposeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("permutation")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getPermutationAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getPermutationAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.transpose");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getIn();
  ::mlir::MutableOperandRange getInMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::AffineMapAttr getPermutationAttr();
  ::mlir::AffineMap getPermutation();
  void setPermutationAttr(::mlir::AffineMapAttr attr);
  void setPermutation(::mlir::AffineMap attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value in, AffineMapAttr permutation, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in, ::mlir::AffineMapAttr permutation);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in, ::mlir::AffineMapAttr permutation);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value in, ::mlir::AffineMap permutation);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in, ::mlir::AffineMap permutation);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::OpFoldResult fold(FoldAdaptor adaptor);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
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
  static StringRef getPermutationAttrStrName() { return "permutation"; }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::TransposeOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::ViewOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ViewOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ViewOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ViewOpGenericAdaptor : public detail::ViewOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ViewOpGenericAdaptorBase;
public:
  ViewOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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

  ValueT getByteShift() {
    return (*getODSOperands(1).begin());
  }

  RangeT getSizes() {
    return getODSOperands(2);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ViewOpAdaptor : public ViewOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ViewOpGenericAdaptor::ViewOpGenericAdaptor;
  ViewOpAdaptor(ViewOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ViewOp : public ::mlir::Op<ViewOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ViewLikeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ViewOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ViewOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.view");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getSource();
  ::mlir::TypedValue<::mlir::IndexType> getByteShift();
  ::mlir::Operation::operand_range getSizes();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getByteShiftMutable();
  ::mlir::MutableOperandRange getSizesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value source, ::mlir::Value byte_shift, ::mlir::ValueRange sizes);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value byte_shift, ::mlir::ValueRange sizes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  ::mlir::Value getViewSource();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  /// The result of a view is always a memref.
  MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

  /// Returns the dynamic sizes for this view operation. This is redundant
  /// with `sizes` but needed in template implementations. More specifically:
  /// ```
  /// template <typename AnyMemRefDefOp>
  /// bool isMemRefSizeValidSymbol(AnyMemRefDefOp memrefDefOp, unsigned index,
  ///                              Region *region)
  /// ```
  operand_range getDynamicSizes() {
    return {getSizes().begin(), getSizes().end()};
  }
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::ViewOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::SubViewOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class SubViewOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  SubViewOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

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
class SubViewOpGenericAdaptor : public detail::SubViewOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::SubViewOpGenericAdaptorBase;
public:
  SubViewOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class SubViewOpAdaptor : public SubViewOpGenericAdaptor<::mlir::ValueRange> {
public:
  using SubViewOpGenericAdaptor::SubViewOpGenericAdaptor;
  SubViewOpAdaptor(SubViewOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class SubViewOp : public ::mlir::Op<SubViewOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::MemRefType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpAsmOpInterface::Trait, ::mlir::ViewLikeOpInterface::Trait, ::mlir::OffsetSizeAndStrideOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SubViewOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = SubViewOpGenericAdaptor<RangeT>;
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
    return ::llvm::StringLiteral("memref.subview");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getSource();
  ::mlir::Operation::operand_range getOffsets();
  ::mlir::Operation::operand_range getSizes();
  ::mlir::Operation::operand_range getStrides();
  ::mlir::MutableOperandRange getSourceMutable();
  ::mlir::MutableOperandRange getOffsetsMutable();
  ::mlir::MutableOperandRange getSizesMutable();
  ::mlir::MutableOperandRange getStridesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getResult();
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, ArrayRef<int64_t> offsets, ArrayRef<int64_t> sizes, ArrayRef<int64_t> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, ValueRange offsets, ValueRange sizes, ValueRange strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, MemRefType resultType, Value source, ValueRange offsets, ValueRange sizes, ValueRange strides, ArrayRef<NamedAttribute> attrs = {});
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
  ::mlir::Value getViewSource();
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
  /// Returns the dynamic sizes for this subview operation if specified.
  ::mlir::Operation::operand_range getDynamicSizes() { return getSizes(); }

  /// Return the list of Range (i.e. offset, size, stride). Each
  /// Range entry contains either the dynamic value or a ConstantIndexOp
  /// constructed with `b` at location `loc`.
  ::mlir::SmallVector<::mlir::Range, 8> getOrCreateRanges(
      ::mlir::OpBuilder &b, ::mlir::Location loc) {
    return ::mlir::getOrCreateRanges(*this, b, loc);
  }

  /// Returns the type of the base memref operand.
  MemRefType getSourceType() {
    return getSource().getType().cast<MemRefType>();
  }

  /// The result of a subview is always a memref.
  MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

  /// A subview result type can be fully inferred from the source type and the
  /// static representation of offsets, sizes and strides. Special sentinels
  /// encode the dynamic case.
  static Type inferResultType(MemRefType sourceMemRefType,
                              ArrayRef<int64_t> staticOffsets,
                              ArrayRef<int64_t> staticSizes,
                              ArrayRef<int64_t> staticStrides);
  static Type inferResultType(MemRefType sourceMemRefType,
                              ArrayRef<OpFoldResult> staticOffsets,
                              ArrayRef<OpFoldResult> staticSizes,
                              ArrayRef<OpFoldResult> staticStrides);

  /// A rank-reducing result type can be inferred from the desired result
  /// shape. Only the layout map is inferred.
  ///
  /// Note: The result shape cannot be inferred with just the result rank and
  /// and the desired sizes. In case there are more "ones" among the sizes
  /// than the difference in source/result rank, it is not clear which dims of
  /// size one should be dropped.
  static Type inferRankReducedResultType(ArrayRef<int64_t> resultShape,
                                         MemRefType sourceMemRefType,
                                         ArrayRef<int64_t> staticOffsets,
                                         ArrayRef<int64_t> staticSizes,
                                         ArrayRef<int64_t> staticStrides);
  static Type inferRankReducedResultType(ArrayRef<int64_t> resultShape,
                                         MemRefType sourceMemRefType,
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

  /// Return the dimensions of the source type that are dropped when
  /// the result is rank-reduced.
  llvm::SmallBitVector getDroppedDims();

  /// Given a `value`, asserted to be of MemRefType, build a SubViewOp that
  /// results in a rank reduction to the desired memref shape and return the
  /// new value created.
  /// If the shape of `value` is already the `desiredShape`, just return
  /// `value`.
  /// If the shape of `value` cannot be rank-reduced to `desiredShape`, fail.
  static FailureOr<Value> rankReduceIfNeeded(
    OpBuilder &b, Location loc, Value value, ArrayRef<int64_t> desiredShape);
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::SubViewOp)

namespace mlir {
namespace memref {

//===----------------------------------------------------------------------===//
// ::mlir::memref::TensorStoreOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TensorStoreOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  TensorStoreOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class TensorStoreOpGenericAdaptor : public detail::TensorStoreOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TensorStoreOpGenericAdaptorBase;
public:
  TensorStoreOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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

  ValueT getMemref() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TensorStoreOpAdaptor : public TensorStoreOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TensorStoreOpGenericAdaptor::TensorStoreOpGenericAdaptor;
  TensorStoreOpAdaptor(TensorStoreOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TensorStoreOp : public ::mlir::Op<TensorStoreOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::SameOperandsShape, ::mlir::OpTrait::SameOperandsElementType, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TensorStoreOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TensorStoreOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("memref.tensor_store");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getTensor();
  ::mlir::TypedValue<::mlir::BaseMemRefType> getMemref();
  ::mlir::MutableOperandRange getTensorMutable();
  ::mlir::MutableOperandRange getMemrefMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value tensor, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value tensor, ::mlir::Value memref);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace memref
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::memref::TensorStoreOp)


#endif  // GET_OP_CLASSES

