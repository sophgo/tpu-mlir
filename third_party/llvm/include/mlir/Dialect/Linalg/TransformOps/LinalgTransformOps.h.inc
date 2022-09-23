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
namespace transform {
class DecomposeOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class FuseIntoContainingOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class FuseOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class GeneralizeOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class InterchangeOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class MatchOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class MultiTileSizesOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class PadOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class PromoteOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class ScalarizeOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class SplitOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class SplitReductionOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class TileOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class TileToForeachThreadOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class VectorizeOp;
} // namespace transform
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::DecomposeOp declarations
//===----------------------------------------------------------------------===//

class DecomposeOpAdaptor {
public:
  DecomposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  DecomposeOpAdaptor(DecomposeOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class DecomposeOp : public ::mlir::Op<DecomposeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::pdl::OperationType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::FunctionalStyleTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::transform::TransformEachOpTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DecomposeOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.decompose");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTransformed();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
  ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::linalg::LinalgOp target, 
      ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
      ::mlir::transform::TransformState &state);
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::DecomposeOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::FuseIntoContainingOp declarations
//===----------------------------------------------------------------------===//

class FuseIntoContainingOpAdaptor {
public:
  FuseIntoContainingOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  FuseIntoContainingOpAdaptor(FuseIntoContainingOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getProducerOp();
  ::mlir::Value getContainingOp();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class FuseIntoContainingOp : public ::mlir::Op<FuseIntoContainingOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::pdl::OperationType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::transform::TransformOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FuseIntoContainingOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.fuse_into_containing_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getProducerOp();
  ::mlir::TypedValue<::mlir::pdl::OperationType> getContainingOp();
  ::mlir::MutableOperandRange getProducerOpMutable();
  ::mlir::MutableOperandRange getContainingOpMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getFusedOp();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type fused_op, ::mlir::Value producer_op, ::mlir::Value containing_op);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value producer_op, ::mlir::Value containing_op);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformResults &transformResults, ::mlir::transform::TransformState &state);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::FuseIntoContainingOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::FuseOp declarations
//===----------------------------------------------------------------------===//

class FuseOpAdaptor {
public:
  FuseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  FuseOpAdaptor(FuseOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getTileSizesAttr();
  ::mlir::ArrayAttr getTileSizes();
  ::mlir::ArrayAttr getTileInterchangeAttr();
  ::mlir::ArrayAttr getTileInterchange();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class FuseOp : public ::mlir::Op<FuseOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::FunctionalStyleTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FuseOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("tile_interchange"), ::llvm::StringRef("tile_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getTileInterchangeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getTileInterchangeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getTileSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getTileSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.fuse");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTransformed();
  ::mlir::Operation::result_range getLoops();
  ::mlir::ArrayAttr getTileSizesAttr();
  ::mlir::ArrayAttr getTileSizes();
  ::mlir::ArrayAttr getTileInterchangeAttr();
  ::mlir::ArrayAttr getTileInterchange();
  void setTileSizesAttr(::mlir::ArrayAttr attr);
  void setTileInterchangeAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::TypeRange loops, ::mlir::Value target, ::mlir::ArrayAttr tile_sizes, ::mlir::ArrayAttr tile_interchange);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr tile_sizes, ::mlir::ArrayAttr tile_interchange);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformResults &transformResults, ::mlir::transform::TransformState &state);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::FuseOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::GeneralizeOp declarations
//===----------------------------------------------------------------------===//

class GeneralizeOpAdaptor {
public:
  GeneralizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  GeneralizeOpAdaptor(GeneralizeOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class GeneralizeOp : public ::mlir::Op<GeneralizeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::pdl::OperationType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::FunctionalStyleTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::transform::TransformEachOpTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GeneralizeOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.generalize");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTransformed();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
  ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::linalg::LinalgOp target, 
      ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
      ::mlir::transform::TransformState &state);
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::GeneralizeOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::InterchangeOp declarations
//===----------------------------------------------------------------------===//

class InterchangeOpAdaptor {
public:
  InterchangeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  InterchangeOpAdaptor(InterchangeOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getIteratorInterchangeAttr();
  ::mlir::ArrayAttr getIteratorInterchange();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class InterchangeOp : public ::mlir::Op<InterchangeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::pdl::OperationType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::FunctionalStyleTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::transform::TransformEachOpTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InterchangeOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("iterator_interchange")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getIteratorInterchangeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIteratorInterchangeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.interchange");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTransformed();
  ::mlir::ArrayAttr getIteratorInterchangeAttr();
  ::mlir::ArrayAttr getIteratorInterchange();
  void setIteratorInterchangeAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::ArrayAttr iterator_interchange);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr iterator_interchange);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
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
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::linalg::GenericOp target, 
      ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
      ::mlir::transform::TransformState &state);
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::InterchangeOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::MatchOp declarations
//===----------------------------------------------------------------------===//

class MatchOpAdaptor {
public:
  MatchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  MatchOpAdaptor(MatchOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getOpsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOps();
  mlir::transform::MatchInterfaceEnumAttr getInterfaceAttr();
  ::llvm::Optional<mlir::transform::MatchInterfaceEnum> getInterface();
  ::mlir::DictionaryAttr getOpAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getOpAttrs();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class MatchOp : public ::mlir::Op<MatchOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::pdl::OperationType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::NavigationTransformOpTrait, ::mlir::transform::TransformOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MatchOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("interface"), ::llvm::StringRef("op_attrs"), ::llvm::StringRef("ops")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getInterfaceAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInterfaceAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOpAttrsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOpAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOpsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOpsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.match");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getResults();
  ::mlir::ArrayAttr getOpsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getOps();
  mlir::transform::MatchInterfaceEnumAttr getInterfaceAttr();
  ::llvm::Optional<mlir::transform::MatchInterfaceEnum> getInterface();
  ::mlir::DictionaryAttr getOpAttrsAttr();
  ::llvm::Optional< ::mlir::DictionaryAttr > getOpAttrs();
  void setOpsAttr(::mlir::ArrayAttr attr);
  void setInterfaceAttr(mlir::transform::MatchInterfaceEnumAttr attr);
  void setOpAttrsAttr(::mlir::DictionaryAttr attr);
  ::mlir::Attribute removeOpsAttr();
  ::mlir::Attribute removeInterfaceAttr();
  ::mlir::Attribute removeOp_attrsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type results, ::mlir::Value target, /*optional*/::mlir::ArrayAttr ops, /*optional*/mlir::transform::MatchInterfaceEnumAttr interface, /*optional*/::mlir::DictionaryAttr op_attrs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, /*optional*/::mlir::ArrayAttr ops, /*optional*/mlir::transform::MatchInterfaceEnumAttr interface, /*optional*/::mlir::DictionaryAttr op_attrs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformResults &transformResults, ::mlir::transform::TransformState &state);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::MatchOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::MultiTileSizesOp declarations
//===----------------------------------------------------------------------===//

class MultiTileSizesOpAdaptor {
public:
  MultiTileSizesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  MultiTileSizesOpAdaptor(MultiTileSizesOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getDimensionAttr();
  uint64_t getDimension();
  ::mlir::IntegerAttr getTargetSizeAttr();
  uint64_t getTargetSize();
  ::mlir::IntegerAttr getDivisorAttr();
  uint64_t getDivisor();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class MultiTileSizesOp : public ::mlir::Op<MultiTileSizesOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::NResults<3>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::transform::TransformEachOpTrait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MultiTileSizesOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dimension"), ::llvm::StringRef("divisor"), ::llvm::StringRef("target_size")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getDimensionAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getDivisorAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getDivisorAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getTargetSizeAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getTargetSizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.multitile_sizes");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getLowSize();
  ::mlir::TypedValue<::mlir::pdl::OperationType> getHighSize();
  ::mlir::TypedValue<::mlir::pdl::OperationType> getSplitPoint();
  ::mlir::IntegerAttr getDimensionAttr();
  uint64_t getDimension();
  ::mlir::IntegerAttr getTargetSizeAttr();
  uint64_t getTargetSize();
  ::mlir::IntegerAttr getDivisorAttr();
  uint64_t getDivisor();
  void setDimensionAttr(::mlir::IntegerAttr attr);
  void setTargetSizeAttr(::mlir::IntegerAttr attr);
  void setDivisorAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type low_size, ::mlir::Type high_size, ::mlir::Type split_point, ::mlir::Value target, ::mlir::IntegerAttr dimension, ::mlir::IntegerAttr target_size, ::mlir::IntegerAttr divisor);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr dimension, ::mlir::IntegerAttr target_size, ::mlir::IntegerAttr divisor);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type low_size, ::mlir::Type high_size, ::mlir::Type split_point, ::mlir::Value target, uint64_t dimension, uint64_t target_size, uint64_t divisor = 1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t dimension, uint64_t target_size, uint64_t divisor = 1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::linalg::LinalgOp target, 
      ::llvm::SmallVector<::mlir::Operation *> &results,
      TransformState &state);
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::MultiTileSizesOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::PadOp declarations
//===----------------------------------------------------------------------===//

class PadOpAdaptor {
public:
  PadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  PadOpAdaptor(PadOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getPaddingValuesAttr();
  ::mlir::ArrayAttr getPaddingValues();
  ::mlir::ArrayAttr getPaddingDimensionsAttr();
  ::mlir::ArrayAttr getPaddingDimensions();
  ::mlir::ArrayAttr getPackPaddingsAttr();
  ::mlir::ArrayAttr getPackPaddings();
  ::mlir::ArrayAttr getHoistPaddingsAttr();
  ::mlir::ArrayAttr getHoistPaddings();
  ::mlir::ArrayAttr getTransposePaddingsAttr();
  ::mlir::ArrayAttr getTransposePaddings();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class PadOp : public ::mlir::Op<PadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::pdl::OperationType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::FunctionalStyleTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::transform::TransformEachOpTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PadOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hoist_paddings"), ::llvm::StringRef("pack_paddings"), ::llvm::StringRef("padding_dimensions"), ::llvm::StringRef("padding_values"), ::llvm::StringRef("transpose_paddings")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getHoistPaddingsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getHoistPaddingsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getPackPaddingsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getPackPaddingsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getPaddingDimensionsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getPaddingDimensionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getPaddingValuesAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getPaddingValuesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getTransposePaddingsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getTransposePaddingsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.pad");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTransformed();
  ::mlir::ArrayAttr getPaddingValuesAttr();
  ::mlir::ArrayAttr getPaddingValues();
  ::mlir::ArrayAttr getPaddingDimensionsAttr();
  ::mlir::ArrayAttr getPaddingDimensions();
  ::mlir::ArrayAttr getPackPaddingsAttr();
  ::mlir::ArrayAttr getPackPaddings();
  ::mlir::ArrayAttr getHoistPaddingsAttr();
  ::mlir::ArrayAttr getHoistPaddings();
  ::mlir::ArrayAttr getTransposePaddingsAttr();
  ::mlir::ArrayAttr getTransposePaddings();
  void setPaddingValuesAttr(::mlir::ArrayAttr attr);
  void setPaddingDimensionsAttr(::mlir::ArrayAttr attr);
  void setPackPaddingsAttr(::mlir::ArrayAttr attr);
  void setHoistPaddingsAttr(::mlir::ArrayAttr attr);
  void setTransposePaddingsAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::ArrayAttr padding_values, ::mlir::ArrayAttr padding_dimensions, ::mlir::ArrayAttr pack_paddings, ::mlir::ArrayAttr hoist_paddings, ::mlir::ArrayAttr transpose_paddings);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr padding_values, ::mlir::ArrayAttr padding_dimensions, ::mlir::ArrayAttr pack_paddings, ::mlir::ArrayAttr hoist_paddings, ::mlir::ArrayAttr transpose_paddings);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
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
    assert(index < 5 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::linalg::LinalgOp target, 
      ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
      ::mlir::transform::TransformState &state);
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::PadOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::PromoteOp declarations
//===----------------------------------------------------------------------===//

class PromoteOpAdaptor {
public:
  PromoteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  PromoteOpAdaptor(PromoteOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getOperandsToPromoteAttr();
  ::mlir::ArrayAttr getOperandsToPromote();
  ::mlir::ArrayAttr getUseFullTileBuffersAttr();
  ::mlir::ArrayAttr getUseFullTileBuffers();
  ::mlir::UnitAttr getUseFullTilesByDefaultAttr();
  bool getUseFullTilesByDefault();
  ::mlir::UnitAttr getUseAllocaAttr();
  bool getUseAlloca();
  ::mlir::IntegerAttr getAlignmentAttr();
  ::llvm::Optional<uint64_t> getAlignment();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class PromoteOp : public ::mlir::Op<PromoteOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::pdl::OperationType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::FunctionalStyleTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::transform::TransformEachOpTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PromoteOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("alignment"), ::llvm::StringRef("operands_to_promote"), ::llvm::StringRef("use_alloca"), ::llvm::StringRef("use_full_tile_buffers"), ::llvm::StringRef("use_full_tiles_by_default")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getAlignmentAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAlignmentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandsToPromoteAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandsToPromoteAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getUseAllocaAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getUseAllocaAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getUseFullTileBuffersAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getUseFullTileBuffersAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getUseFullTilesByDefaultAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getUseFullTilesByDefaultAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.promote");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTransformed();
  ::mlir::ArrayAttr getOperandsToPromoteAttr();
  ::mlir::ArrayAttr getOperandsToPromote();
  ::mlir::ArrayAttr getUseFullTileBuffersAttr();
  ::mlir::ArrayAttr getUseFullTileBuffers();
  ::mlir::UnitAttr getUseFullTilesByDefaultAttr();
  bool getUseFullTilesByDefault();
  ::mlir::UnitAttr getUseAllocaAttr();
  bool getUseAlloca();
  ::mlir::IntegerAttr getAlignmentAttr();
  ::llvm::Optional<uint64_t> getAlignment();
  void setOperandsToPromoteAttr(::mlir::ArrayAttr attr);
  void setUseFullTileBuffersAttr(::mlir::ArrayAttr attr);
  void setUseFullTilesByDefaultAttr(::mlir::UnitAttr attr);
  void setUseAllocaAttr(::mlir::UnitAttr attr);
  void setAlignmentAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeUse_full_tiles_by_defaultAttr();
  ::mlir::Attribute removeUse_allocaAttr();
  ::mlir::Attribute removeAlignmentAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::ArrayAttr operands_to_promote, ::mlir::ArrayAttr use_full_tile_buffers, /*optional*/::mlir::UnitAttr use_full_tiles_by_default, /*optional*/::mlir::UnitAttr use_alloca, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr operands_to_promote, ::mlir::ArrayAttr use_full_tile_buffers, /*optional*/::mlir::UnitAttr use_full_tiles_by_default, /*optional*/::mlir::UnitAttr use_alloca, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::ArrayAttr operands_to_promote, ::mlir::ArrayAttr use_full_tile_buffers, /*optional*/bool use_full_tiles_by_default, /*optional*/bool use_alloca, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr operands_to_promote, ::mlir::ArrayAttr use_full_tile_buffers, /*optional*/bool use_full_tiles_by_default, /*optional*/bool use_alloca, /*optional*/::mlir::IntegerAttr alignment);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::linalg::LinalgOp target, 
      ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
      ::mlir::transform::TransformState &state);
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::PromoteOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::ScalarizeOp declarations
//===----------------------------------------------------------------------===//

class ScalarizeOpAdaptor {
public:
  ScalarizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ScalarizeOpAdaptor(ScalarizeOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ScalarizeOp : public ::mlir::Op<ScalarizeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::pdl::OperationType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::FunctionalStyleTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::transform::TransformEachOpTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ScalarizeOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.scalarize");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getResult();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value target);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
  ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::linalg::LinalgOp target, 
      ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
      ::mlir::transform::TransformState &state);
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::ScalarizeOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::SplitOp declarations
//===----------------------------------------------------------------------===//

class SplitOpAdaptor {
public:
  SplitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  SplitOpAdaptor(SplitOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::Value getDynamicSplitPoint();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getDimensionAttr();
  uint64_t getDimension();
  ::mlir::IntegerAttr getStaticSplitPointAttr();
  uint64_t getStaticSplitPoint();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class SplitOp : public ::mlir::Op<SplitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::NResults<2>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SplitOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("dimension"), ::llvm::StringRef("static_split_point")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getDimensionAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getDimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getStaticSplitPointAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getStaticSplitPointAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.split");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::Value getDynamicSplitPoint();
  ::mlir::MutableOperandRange getTargetMutable();
  ::mlir::MutableOperandRange getDynamicSplitPointMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getFirst();
  ::mlir::TypedValue<::mlir::pdl::OperationType> getSecond();
  ::mlir::IntegerAttr getDimensionAttr();
  uint64_t getDimension();
  ::mlir::IntegerAttr getStaticSplitPointAttr();
  uint64_t getStaticSplitPoint();
  void setDimensionAttr(::mlir::IntegerAttr attr);
  void setStaticSplitPointAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type first, ::mlir::Type second, ::mlir::Value target, ::mlir::IntegerAttr dimension, /*optional*/::mlir::Value dynamic_split_point, ::mlir::IntegerAttr static_split_point);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr dimension, /*optional*/::mlir::Value dynamic_split_point, ::mlir::IntegerAttr static_split_point);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type first, ::mlir::Type second, ::mlir::Value target, uint64_t dimension, /*optional*/::mlir::Value dynamic_split_point, uint64_t static_split_point);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t dimension, /*optional*/::mlir::Value dynamic_split_point, uint64_t static_split_point);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  ::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformResults &transformResults, ::mlir::transform::TransformState &state);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 2 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::SplitOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::SplitReductionOp declarations
//===----------------------------------------------------------------------===//

class SplitReductionOpAdaptor {
public:
  SplitReductionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  SplitReductionOpAdaptor(SplitReductionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getSplitFactorAttr();
  uint64_t getSplitFactor();
  ::mlir::IntegerAttr getInsertSplitDimensionAttr();
  uint64_t getInsertSplitDimension();
  ::mlir::UnitAttr getUseScalingAlgorithmAttr();
  bool getUseScalingAlgorithm();
  ::mlir::UnitAttr getUseAllocAttr();
  bool getUseAlloc();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class SplitReductionOp : public ::mlir::Op<SplitReductionOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::NResults<4>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::FunctionalStyleTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformEachOpTrait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SplitReductionOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("insert_split_dimension"), ::llvm::StringRef("split_factor"), ::llvm::StringRef("use_alloc"), ::llvm::StringRef("use_scaling_algorithm")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getInsertSplitDimensionAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInsertSplitDimensionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getSplitFactorAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getSplitFactorAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getUseAllocAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getUseAllocAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getUseScalingAlgorithmAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getUseScalingAlgorithmAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.split_reduction");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getInitOrAllocOp();
  ::mlir::TypedValue<::mlir::pdl::OperationType> getFillOp();
  ::mlir::TypedValue<::mlir::pdl::OperationType> getSplitLinalgOp();
  ::mlir::TypedValue<::mlir::pdl::OperationType> getCombiningLinalgOp();
  ::mlir::IntegerAttr getSplitFactorAttr();
  uint64_t getSplitFactor();
  ::mlir::IntegerAttr getInsertSplitDimensionAttr();
  uint64_t getInsertSplitDimension();
  ::mlir::UnitAttr getUseScalingAlgorithmAttr();
  bool getUseScalingAlgorithm();
  ::mlir::UnitAttr getUseAllocAttr();
  bool getUseAlloc();
  void setSplitFactorAttr(::mlir::IntegerAttr attr);
  void setInsertSplitDimensionAttr(::mlir::IntegerAttr attr);
  void setUseScalingAlgorithmAttr(::mlir::UnitAttr attr);
  void setUseAllocAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeUse_scaling_algorithmAttr();
  ::mlir::Attribute removeUse_allocAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type init_or_alloc_op, ::mlir::Type fill_op, ::mlir::Type split_linalg_op, ::mlir::Type combining_linalg_op, ::mlir::Value target, ::mlir::IntegerAttr split_factor, ::mlir::IntegerAttr insert_split_dimension, /*optional*/::mlir::UnitAttr use_scaling_algorithm, /*optional*/::mlir::UnitAttr use_alloc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr split_factor, ::mlir::IntegerAttr insert_split_dimension, /*optional*/::mlir::UnitAttr use_scaling_algorithm, /*optional*/::mlir::UnitAttr use_alloc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type init_or_alloc_op, ::mlir::Type fill_op, ::mlir::Type split_linalg_op, ::mlir::Type combining_linalg_op, ::mlir::Value target, uint64_t split_factor, uint64_t insert_split_dimension, /*optional*/bool use_scaling_algorithm, /*optional*/bool use_alloc);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t split_factor, uint64_t insert_split_dimension, /*optional*/bool use_scaling_algorithm, /*optional*/bool use_alloc);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::linalg::LinalgOp target, 
      ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
      ::mlir::transform::TransformState &state);
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::SplitReductionOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::TileOp declarations
//===----------------------------------------------------------------------===//

class TileOpAdaptor {
public:
  TileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TileOpAdaptor(TileOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::ValueRange getDynamicSizes();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getStaticSizesAttr();
  ::mlir::ArrayAttr getStaticSizes();
  ::mlir::ArrayAttr getInterchangeAttr();
  ::mlir::ArrayAttr getInterchange();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class TileOp : public ::mlir::Op<TileOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::AtLeastNResults<1>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::transform::TransformOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TileOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("interchange"), ::llvm::StringRef("static_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getInterchangeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getInterchangeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getStaticSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getStaticSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.tile");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::Operation::operand_range getDynamicSizes();
  ::mlir::MutableOperandRange getTargetMutable();
  ::mlir::MutableOperandRange getDynamicSizesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTiledLinalgOp();
  ::mlir::Operation::result_range getLoops();
  ::mlir::ArrayAttr getStaticSizesAttr();
  ::mlir::ArrayAttr getStaticSizes();
  ::mlir::ArrayAttr getInterchangeAttr();
  ::mlir::ArrayAttr getInterchange();
  void setStaticSizesAttr(::mlir::ArrayAttr attr);
  void setInterchangeAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type tiled_linalg_op, ::mlir::TypeRange loops, ::mlir::Value target, ::mlir::ValueRange dynamic_sizes, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr interchange);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ValueRange dynamic_sizes, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr interchange);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformResults &transformResults, ::mlir::transform::TransformState &state);
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
  /// Returns the list of tile sizes, which may be static (Attribute) or
  /// dynamic (Value).
  SmallVector<OpFoldResult> getMixedSizes();
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::TileOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::TileToForeachThreadOp declarations
//===----------------------------------------------------------------------===//

class TileToForeachThreadOpAdaptor {
public:
  TileToForeachThreadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TileToForeachThreadOpAdaptor(TileToForeachThreadOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::ValueRange getNumThreads();
  ::mlir::ValueRange getTileSizes();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getStaticNumThreadsAttr();
  ::mlir::ArrayAttr getStaticNumThreads();
  ::mlir::ArrayAttr getStaticTileSizesAttr();
  ::mlir::ArrayAttr getStaticTileSizes();
  ::mlir::ArrayAttr getThreadDimMappingAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getThreadDimMapping();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class TileToForeachThreadOp : public ::mlir::Op<TileToForeachThreadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::NResults<2>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::OpAsmOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TileToForeachThreadOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("static_num_threads"), ::llvm::StringRef("static_tile_sizes"), ::llvm::StringRef("thread_dim_mapping")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getStaticNumThreadsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getStaticNumThreadsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getStaticTileSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getStaticTileSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getThreadDimMappingAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getThreadDimMappingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  void getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn);
  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.tile_to_foreach_thread_op");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::Operation::operand_range getNumThreads();
  ::mlir::Operation::operand_range getTileSizes();
  ::mlir::MutableOperandRange getTargetMutable();
  ::mlir::MutableOperandRange getNumThreadsMutable();
  ::mlir::MutableOperandRange getTileSizesMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getForeachThreadOp();
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTiledOp();
  ::mlir::ArrayAttr getStaticNumThreadsAttr();
  ::mlir::ArrayAttr getStaticNumThreads();
  ::mlir::ArrayAttr getStaticTileSizesAttr();
  ::mlir::ArrayAttr getStaticTileSizes();
  ::mlir::ArrayAttr getThreadDimMappingAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > getThreadDimMapping();
  void setStaticNumThreadsAttr(::mlir::ArrayAttr attr);
  void setStaticTileSizesAttr(::mlir::ArrayAttr attr);
  void setThreadDimMappingAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeThread_dim_mappingAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type foreach_thread_op, ::mlir::Type tiled_op, ::mlir::Value target, ::mlir::ValueRange num_threads, ::mlir::ValueRange tile_sizes, ::mlir::ArrayAttr static_num_threads, ::mlir::ArrayAttr static_tile_sizes, /*optional*/::mlir::ArrayAttr thread_dim_mapping);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ValueRange num_threads, ::mlir::ValueRange tile_sizes, ::mlir::ArrayAttr static_num_threads, ::mlir::ArrayAttr static_tile_sizes, /*optional*/::mlir::ArrayAttr thread_dim_mapping);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  ::mlir::DiagnosedSilenceableFailure apply(
      ::mlir::transform::TransformResults &transformResults,
      ::mlir::transform::TransformState &state);

  ::llvm::SmallVector<::mlir::OpFoldResult> getMixedNumThreads();
  ::llvm::SmallVector<::mlir::OpFoldResult> getMixedTileSizes();
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::TileToForeachThreadOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::VectorizeOp declarations
//===----------------------------------------------------------------------===//

class VectorizeOpAdaptor {
public:
  VectorizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  VectorizeOpAdaptor(VectorizeOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getVectorizePaddingAttr();
  bool getVectorizePadding();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class VectorizeOp : public ::mlir::Op<VectorizeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::pdl::OperationType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::FunctionalStyleTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformEachOpTrait, ::mlir::transform::TransformOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = VectorizeOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("vectorize_padding")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getVectorizePaddingAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getVectorizePaddingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.structured.vectorize");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTransformed();
  ::mlir::BoolAttr getVectorizePaddingAttr();
  bool getVectorizePadding();
  void setVectorizePaddingAttr(::mlir::BoolAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::BoolAttr vectorize_padding);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::BoolAttr vectorize_padding);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, bool vectorize_padding = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, bool vectorize_padding = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::Operation *target, 
      ::llvm::SmallVectorImpl<::mlir::Operation *> &results, 
      ::mlir::transform::TransformState &state);
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::VectorizeOp)


#endif  // GET_OP_CLASSES

