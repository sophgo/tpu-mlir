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
namespace scf {
class ConditionOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class ExecuteRegionOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class ForOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class ForeachThreadOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class IfOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class ParallelInsertSliceOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class ParallelOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class PerformConcurrentlyOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class ReduceOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class ReduceReturnOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class WhileOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class YieldOp;
} // namespace scf
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::ConditionOp declarations
//===----------------------------------------------------------------------===//

class ConditionOpAdaptor {
public:
  ConditionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ConditionOpAdaptor(ConditionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getCondition();
  ::mlir::ValueRange getArgs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ConditionOp : public ::mlir::Op<ConditionOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::HasParent<WhileOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::RegionBranchTerminatorOpInterface::Trait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConditionOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.condition");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getCondition();
  ::mlir::Operation::operand_range getArgs();
  ::mlir::MutableOperandRange getConditionMutable();
  ::mlir::MutableOperandRange getArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::mlir::ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::ValueRange args);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::MutableOperandRange getMutableSuccessorOperands(::mlir::Optional<unsigned> index);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::ConditionOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::ExecuteRegionOp declarations
//===----------------------------------------------------------------------===//

class ExecuteRegionOpAdaptor {
public:
  ExecuteRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ExecuteRegionOpAdaptor(ExecuteRegionOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ExecuteRegionOp : public ::mlir::Op<ExecuteRegionOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::RegionBranchOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExecuteRegionOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.execute_region");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getSuccessorRegions(::mlir::Optional<unsigned> index, ::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
public:
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::ExecuteRegionOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::ForOp declarations
//===----------------------------------------------------------------------===//

class ForOpAdaptor {
public:
  ForOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ForOpAdaptor(ForOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getLowerBound();
  ::mlir::Value getUpperBound();
  ::mlir::Value getStep();
  ::mlir::ValueRange getInitArgs();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ForOp : public ::mlir::Op<ForOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<3>::Impl, ::mlir::OpTrait::SingleBlockImplicitTerminator<scf::YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::LoopLikeOpInterface::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::OpTrait::HasRecursiveSideEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ForOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.for");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getLowerBound();
  ::mlir::Value getUpperBound();
  ::mlir::Value getStep();
  ::mlir::Operation::operand_range getInitArgs();
  ::mlir::MutableOperandRange getLowerBoundMutable();
  ::mlir::MutableOperandRange getUpperBoundMutable();
  ::mlir::MutableOperandRange getStepMutable();
  ::mlir::MutableOperandRange getInitArgsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value lowerBound, Value upperBound, Value step, ValueRange iterArgs = llvm::None, function_ref<void(OpBuilder &, Location, Value, ValueRange)> odsArg4 = nullptr);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult verifyRegions();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::Region &getLoopBody();
  ::mlir::Optional<::mlir::Value> getSingleInductionVar();
  ::mlir::Optional<::mlir::OpFoldResult> getSingleLowerBound();
  ::mlir::Optional<::mlir::OpFoldResult> getSingleStep();
  ::mlir::Optional<::mlir::OpFoldResult> getSingleUpperBound();
  void getSuccessorRegions(::mlir::Optional<unsigned> index, ::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
public:
  using BodyBuilderFn =
      function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

  Value getInductionVar() { return getBody()->getArgument(0); }
  Block::BlockArgListType getRegionIterArgs() {
    return getBody()->getArguments().drop_front(getNumInductionVars());
  }
  Operation::operand_range getIterOperands() {
    return getOperands().drop_front(getNumControlOperands());
  }
  MutableArrayRef<OpOperand> getIterOpOperands() {
    return
      getOperation()->getOpOperands().drop_front(getNumControlOperands());
  }

  void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
  void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
  void setStep(Value step) { getOperation()->setOperand(2, step); }

  /// Number of induction variables, always 1 for scf::ForOp.
  unsigned getNumInductionVars() { return 1; }
  /// Number of region arguments for loop-carried values
  unsigned getNumRegionIterArgs() {
    return getBody()->getNumArguments() - getNumInductionVars();
  }
  /// Number of operands controlling the loop: lb, ub, step
  unsigned getNumControlOperands() { return 3; }
  /// Does the operation hold operands for loop-carried values
  bool hasIterOperands() {
    return getOperation()->getNumOperands() > getNumControlOperands();
  }
  /// Get Number of loop-carried values
  unsigned getNumIterOperands() {
    return getOperation()->getNumOperands() - getNumControlOperands();
  }
  /// Get the region iter arg that corresponds to an OpOperand.
  /// This helper prevents internal op implementation detail leakage to
  /// clients by hiding the operand / block argument mapping.
  BlockArgument getRegionIterArgForOpOperand(OpOperand &opOperand) {
    assert(opOperand.getOperandNumber() >= getNumControlOperands() &&
           "expected an iter args operand");
    assert(opOperand.getOwner() == getOperation() &&
           "opOperand does not belong to this scf::ForOp operation");
    return getRegionIterArgs()[
      opOperand.getOperandNumber() - getNumControlOperands()];
  }
  /// Get the OpOperand& that corresponds to a region iter arg.
  /// This helper prevents internal op implementation detail leakage to
  /// clients by hiding the operand / block argument mapping.
  OpOperand &getOpOperandForRegionIterArg(BlockArgument bbArg) {
    assert(bbArg.getArgNumber() >= getNumInductionVars() &&
           "expected a bbArg that is not an induction variable");
    assert(bbArg.getOwner()->getParentOp() == getOperation() &&
           "bbArg does not belong to the scf::ForOp body");
    return getOperation()->getOpOperand(
      getNumControlOperands() + bbArg.getArgNumber() - getNumInductionVars());
  }
  /// Get the OpResult that corresponds to an OpOperand.
  /// Assert that opOperand is an iterArg.
  /// This helper prevents internal op implementation detail leakage to
  /// clients by hiding the operand / block argument mapping.
  OpResult getResultForOpOperand(OpOperand &opOperand) {
    assert(opOperand.getOperandNumber() >= getNumControlOperands() &&
           "expected an iter args operand");
    assert(opOperand.getOwner() == getOperation() &&
           "opOperand does not belong to this scf::ForOp operation");
    return getOperation()->getResult(
      opOperand.getOperandNumber() - getNumControlOperands());
  }
  /// Get the OpOperand& that corresponds to an OpResultOpOperand.
  /// This helper prevents internal op implementation detail leakage to
  /// clients by hiding the operand / block argument mapping.
  OpOperand &getOpOperandForResult(OpResult opResult) {
    assert(opResult.getDefiningOp() == getOperation() &&
           "opResult does not belong to the scf::ForOp operation");
    return getOperation()->getOpOperand(
      getNumControlOperands() + opResult.getResultNumber());
  }

  /// Return operands used when entering the region at 'index'. These operands
  /// correspond to the loop iterator operands, i.e., those exclusing the
  /// induction variable. LoopOp only has one region, so 0 is the only valid
  /// value for `index`.
  OperandRange getSuccessorEntryOperands(unsigned index);
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::ForOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::ForeachThreadOp declarations
//===----------------------------------------------------------------------===//

class ForeachThreadOpAdaptor {
public:
  ForeachThreadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ForeachThreadOpAdaptor(ForeachThreadOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getNumThreads();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ForeachThreadOp : public ::mlir::Op<ForeachThreadOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<scf::PerformConcurrentlyOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasRecursiveSideEffects, ::mlir::OpTrait::AutomaticAllocationScope> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ForeachThreadOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.foreach_thread");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getNumThreads();
  ::mlir::MutableOperandRange getNumThreadsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, ValueRange num_threads);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange num_threads, function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilder);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
public:
  int64_t getRank() { return getNumThreads().size(); }
  ::mlir::ValueRange getThreadIndices() { return getBody()->getArguments(); }
  ::mlir::Value getThreadIndex(int64_t idx) { return getBody()->getArgument(idx); }

  // The ensureTerminator method generated by SingleBlockImplicitTerminator is
  // unaware of the fact that our terminator also needs a region to be
  // well-formed. We override it here to ensure that we do the right thing.
  static void ensureTerminator(Region &region, OpBuilder &builder, Location loc);

  PerformConcurrentlyOp getTerminator();
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::ForeachThreadOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::IfOp declarations
//===----------------------------------------------------------------------===//

class IfOpAdaptor {
public:
  IfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  IfOpAdaptor(IfOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getCondition();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getThenRegion();
  ::mlir::Region &getElseRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class IfOp : public ::mlir::Op<IfOp, ::mlir::OpTrait::NRegions<2>::Impl, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::SingleBlockImplicitTerminator<scf::YieldOp>::Impl, ::mlir::OpTrait::NoRegionArguments, ::mlir::OpTrait::OpInvariants, ::mlir::RegionBranchOpInterface::Trait, ::mlir::OpTrait::HasRecursiveSideEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = IfOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.if");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getCondition();
  ::mlir::MutableOperandRange getConditionMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getThenRegion();
  ::mlir::Region &getElseRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value cond, bool withElseRegion);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, Value cond, bool withElseRegion);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, Value cond, function_ref<void(OpBuilder &, Location)> thenBuilder = buildTerminatedBody, function_ref<void(OpBuilder &, Location)> elseBuilder = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value cond, function_ref<void(OpBuilder &, Location)> thenBuilder = buildTerminatedBody, function_ref<void(OpBuilder &, Location)> elseBuilder = nullptr);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getSuccessorRegions(::mlir::Optional<unsigned> index, ::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
  void getRegionInvocationBounds(::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds);
public:
  OpBuilder getThenBodyBuilder(OpBuilder::Listener *listener = nullptr) {
    Block* body = getBody(0);
    return getResults().empty() ? OpBuilder::atBlockTerminator(body, listener)
                                : OpBuilder::atBlockEnd(body, listener);
  }
  OpBuilder getElseBodyBuilder(OpBuilder::Listener *listener = nullptr) {
    Block* body = getBody(1);
    return getResults().empty() ? OpBuilder::atBlockTerminator(body, listener)
                                : OpBuilder::atBlockEnd(body, listener);
  }
  Block* thenBlock();
  YieldOp thenYield();
  Block* elseBlock();
  YieldOp elseYield();
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::IfOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::ParallelInsertSliceOp declarations
//===----------------------------------------------------------------------===//

class ParallelInsertSliceOpAdaptor {
public:
  ParallelInsertSliceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ParallelInsertSliceOpAdaptor(ParallelInsertSliceOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getSource();
  ::mlir::Value getDest();
  ::mlir::ValueRange getOffsets();
  ::mlir::ValueRange getSizes();
  ::mlir::ValueRange getStrides();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr getStaticOffsetsAttr();
  ::mlir::ArrayAttr getStaticOffsets();
  ::mlir::ArrayAttr getStaticSizesAttr();
  ::mlir::ArrayAttr getStaticSizes();
  ::mlir::ArrayAttr getStaticStridesAttr();
  ::mlir::ArrayAttr getStaticStrides();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ParallelInsertSliceOp : public ::mlir::Op<ParallelInsertSliceOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::HasParent<PerformConcurrentlyOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OffsetSizeAndStrideOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelInsertSliceOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("static_offsets"), ::llvm::StringRef("static_sizes"), ::llvm::StringRef("static_strides")};
    return ::llvm::makeArrayRef(attrNames);
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
    return ::llvm::StringLiteral("scf.foreach_thread.parallel_insert_slice");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getSource();
  ::mlir::Value getDest();
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
  ::mlir::ArrayAttr getStaticOffsetsAttr();
  ::mlir::ArrayAttr getStaticOffsets();
  ::mlir::ArrayAttr getStaticSizesAttr();
  ::mlir::ArrayAttr getStaticSizes();
  ::mlir::ArrayAttr getStaticStridesAttr();
  ::mlir::ArrayAttr getStaticStrides();
  void setStaticOffsetsAttr(::mlir::ArrayAttr attr);
  void setStaticSizesAttr(::mlir::ArrayAttr attr);
  void setStaticStridesAttr(::mlir::ArrayAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value source, Value dest, ValueRange offsets, ValueRange sizes, ValueRange strides, ArrayRef<NamedAttribute> attrs = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value source, ::mlir::Value dest, ::mlir::ValueRange offsets, ::mlir::ValueRange sizes, ::mlir::ValueRange strides, ::mlir::ArrayAttr static_offsets, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr static_strides);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
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
  ::mlir::Operation::operand_range offsets() { return getOffsets(); }
  ::mlir::Operation::operand_range sizes() { return getSizes(); }
  ::mlir::Operation::operand_range strides() { return getStrides(); }
  ::mlir::ArrayAttr static_offsets() { return getStaticOffsets(); }
  ::mlir::ArrayAttr static_sizes() { return getStaticSizes(); }
  ::mlir::ArrayAttr static_strides() { return getStaticStrides(); }

  Type yieldedType() { return getDest().getType(); }

  RankedTensorType getSourceType() {
    return getSource().getType().cast<RankedTensorType>();
  }

  /// Return the expected rank of each of the `static_offsets`, `static_sizes`
  /// and `static_strides` attributes.
  std::array<unsigned, 3> getArrayAttrMaxRanks() {
    unsigned rank = getSourceType().getRank();
    return {rank, rank, rank};
  }

  /// Return the number of leading operands before `offsets`, `sizes` and
  /// `strides` operands.
  static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::ParallelInsertSliceOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::ParallelOp declarations
//===----------------------------------------------------------------------===//

class ParallelOpAdaptor {
public:
  ParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ParallelOpAdaptor(ParallelOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getLowerBound();
  ::mlir::ValueRange getUpperBound();
  ::mlir::ValueRange getStep();
  ::mlir::ValueRange getInitVals();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ParallelOp : public ::mlir::Op<ParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<scf::YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::LoopLikeOpInterface::Trait, ::mlir::OpTrait::HasRecursiveSideEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.parallel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getLowerBound();
  ::mlir::Operation::operand_range getUpperBound();
  ::mlir::Operation::operand_range getStep();
  ::mlir::Operation::operand_range getInitVals();
  ::mlir::MutableOperandRange getLowerBoundMutable();
  ::mlir::MutableOperandRange getUpperBoundMutable();
  ::mlir::MutableOperandRange getStepMutable();
  ::mlir::MutableOperandRange getInitValsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps, ValueRange initVals, function_ref<void (OpBuilder &, Location, ValueRange, ValueRange)> bodyBuilderFn = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange lowerBounds, ValueRange upperBounds, ValueRange steps, function_ref<void (OpBuilder &, Location, ValueRange)> bodyBuilderFn = nullptr);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::Region &getLoopBody();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  ValueRange getInductionVars() {
    return getBody()->getArguments();
  }
  unsigned getNumLoops() { return getStep().size(); }
  unsigned getNumReductions() { return getInitVals().size(); }
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::ParallelOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::PerformConcurrentlyOp declarations
//===----------------------------------------------------------------------===//

class PerformConcurrentlyOpAdaptor {
public:
  PerformConcurrentlyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  PerformConcurrentlyOpAdaptor(PerformConcurrentlyOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class PerformConcurrentlyOp : public ::mlir::Op<PerformConcurrentlyOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::HasParent<ForeachThreadOp>::Impl, ::mlir::OpTrait::NoTerminator, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator, ::mlir::RegionKindInterface::Trait, ::mlir::OpTrait::HasOnlyGraphRegion> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PerformConcurrentlyOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.foreach_thread.perform_concurrently");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  SmallVector<Type> yieldedTypes();
  ::llvm::iterator_range<Block::iterator> yieldingOps();
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::PerformConcurrentlyOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::ReduceOp declarations
//===----------------------------------------------------------------------===//

class ReduceOpAdaptor {
public:
  ReduceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ReduceOpAdaptor(ReduceOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getOperand();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getReductionOperator();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ReduceOp : public ::mlir::Op<ReduceOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::HasParent<ParallelOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReduceOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.reduce");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getOperand();
  ::mlir::MutableOperandRange getOperandMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getReductionOperator();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value operand, function_ref<void (OpBuilder &, Location, Value, Value)> bodyBuilderFn = nullptr);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifyRegions();
public:
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::ReduceOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::ReduceReturnOp declarations
//===----------------------------------------------------------------------===//

class ReduceReturnOpAdaptor {
public:
  ReduceReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ReduceReturnOpAdaptor(ReduceReturnOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getResult();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ReduceReturnOp : public ::mlir::Op<ReduceReturnOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::HasParent<ReduceOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReduceReturnOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.reduce.return");
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
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::ReduceReturnOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::WhileOp declarations
//===----------------------------------------------------------------------===//

class WhileOpAdaptor {
public:
  WhileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  WhileOpAdaptor(WhileOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getInits();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &getBefore();
  ::mlir::Region &getAfter();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class WhileOp : public ::mlir::Op<WhileOp, ::mlir::OpTrait::NRegions<2>::Impl, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::RegionBranchOpInterface::Trait, ::mlir::OpTrait::HasRecursiveSideEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WhileOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.while");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInits();
  ::mlir::MutableOperandRange getInitsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getBefore();
  ::mlir::Region &getAfter();
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verifyRegions();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getSuccessorRegions(::mlir::Optional<unsigned> index, ::mlir::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
public:
  OperandRange getSuccessorEntryOperands(unsigned index);
  ConditionOp getConditionOp();
  YieldOp getYieldOp();
  Block::BlockArgListType getBeforeArguments();
  Block::BlockArgListType getAfterArguments();
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::WhileOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::YieldOp declarations
//===----------------------------------------------------------------------===//

class YieldOpAdaptor {
public:
  YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  YieldOpAdaptor(YieldOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange getResults();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<ExecuteRegionOp, ForOp, IfOp, ParallelOp, WhileOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YieldOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.yield");
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
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::YieldOp)


#endif  // GET_OP_CLASSES

