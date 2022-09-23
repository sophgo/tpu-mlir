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
class OneShotBufferizeOp;
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
// ::mlir::transform::OneShotBufferizeOp declarations
//===----------------------------------------------------------------------===//

class OneShotBufferizeOpAdaptor {
public:
  OneShotBufferizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  OneShotBufferizeOpAdaptor(OneShotBufferizeOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getAllowReturnAllocsAttr();
  bool getAllowReturnAllocs();
  ::mlir::BoolAttr getAllowUnknownOpsAttr();
  bool getAllowUnknownOps();
  ::mlir::BoolAttr getBufferizeFunctionBoundariesAttr();
  bool getBufferizeFunctionBoundaries();
  ::mlir::BoolAttr getCreateDeallocsAttr();
  bool getCreateDeallocs();
  ::mlir::BoolAttr getTargetIsModuleAttr();
  bool getTargetIsModule();
  ::mlir::BoolAttr getTestAnalysisOnlyAttr();
  bool getTestAnalysisOnly();
  ::mlir::BoolAttr getPrintConflictsAttr();
  bool getPrintConflicts();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class OneShotBufferizeOp : public ::mlir::Op<OneShotBufferizeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::TransformOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = OneShotBufferizeOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("allow_return_allocs"), ::llvm::StringRef("allow_unknown_ops"), ::llvm::StringRef("bufferize_function_boundaries"), ::llvm::StringRef("create_deallocs"), ::llvm::StringRef("print_conflicts"), ::llvm::StringRef("target_is_module"), ::llvm::StringRef("test_analysis_only")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getAllowReturnAllocsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAllowReturnAllocsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getAllowUnknownOpsAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getAllowUnknownOpsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getBufferizeFunctionBoundariesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getBufferizeFunctionBoundariesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getCreateDeallocsAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getCreateDeallocsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getPrintConflictsAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getPrintConflictsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getTargetIsModuleAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getTargetIsModuleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getTestAnalysisOnlyAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getTestAnalysisOnlyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.bufferization.one_shot_bufferize");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr getAllowReturnAllocsAttr();
  bool getAllowReturnAllocs();
  ::mlir::BoolAttr getAllowUnknownOpsAttr();
  bool getAllowUnknownOps();
  ::mlir::BoolAttr getBufferizeFunctionBoundariesAttr();
  bool getBufferizeFunctionBoundaries();
  ::mlir::BoolAttr getCreateDeallocsAttr();
  bool getCreateDeallocs();
  ::mlir::BoolAttr getTargetIsModuleAttr();
  bool getTargetIsModule();
  ::mlir::BoolAttr getTestAnalysisOnlyAttr();
  bool getTestAnalysisOnly();
  ::mlir::BoolAttr getPrintConflictsAttr();
  bool getPrintConflicts();
  void setAllowReturnAllocsAttr(::mlir::BoolAttr attr);
  void setAllowUnknownOpsAttr(::mlir::BoolAttr attr);
  void setBufferizeFunctionBoundariesAttr(::mlir::BoolAttr attr);
  void setCreateDeallocsAttr(::mlir::BoolAttr attr);
  void setTargetIsModuleAttr(::mlir::BoolAttr attr);
  void setTestAnalysisOnlyAttr(::mlir::BoolAttr attr);
  void setPrintConflictsAttr(::mlir::BoolAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value target, ::mlir::BoolAttr allow_return_allocs, ::mlir::BoolAttr allow_unknown_ops, ::mlir::BoolAttr bufferize_function_boundaries, ::mlir::BoolAttr create_deallocs, ::mlir::BoolAttr target_is_module, ::mlir::BoolAttr test_analysis_only, ::mlir::BoolAttr print_conflicts);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::BoolAttr allow_return_allocs, ::mlir::BoolAttr allow_unknown_ops, ::mlir::BoolAttr bufferize_function_boundaries, ::mlir::BoolAttr create_deallocs, ::mlir::BoolAttr target_is_module, ::mlir::BoolAttr test_analysis_only, ::mlir::BoolAttr print_conflicts);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value target, bool allow_return_allocs = false, bool allow_unknown_ops = false, bool bufferize_function_boundaries = false, bool create_deallocs = true, bool target_is_module = true, bool test_analysis_only = false, bool print_conflicts = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, bool allow_return_allocs = false, bool allow_unknown_ops = false, bool bufferize_function_boundaries = false, bool create_deallocs = true, bool target_is_module = true, bool test_analysis_only = false, bool print_conflicts = false);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::DiagnosedSilenceableFailure apply(::mlir::transform::TransformResults &transformResults, ::mlir::transform::TransformState &state);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 7 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::OneShotBufferizeOp)


#endif  // GET_OP_CLASSES

