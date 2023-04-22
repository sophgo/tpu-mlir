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
class ForallOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class IfOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class InParallelOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class IndexSwitchOp;
} // namespace scf
} // namespace mlir
namespace mlir {
namespace scf {
class ParallelOp;
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

namespace detail {
class ConditionOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ConditionOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ConditionOpGenericAdaptor : public detail::ConditionOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConditionOpGenericAdaptorBase;
public:
  ConditionOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getCondition() {
    return (*getODSOperands(0).begin());
  }

  RangeT getArgs() {
    return getODSOperands(1);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConditionOpAdaptor : public ConditionOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConditionOpGenericAdaptor::ConditionOpGenericAdaptor;
  ConditionOpAdaptor(ConditionOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ConditionOp : public ::mlir::Op<ConditionOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::HasParent<WhileOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::RegionBranchTerminatorOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConditionOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConditionOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.condition");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getCondition();
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
  ::mlir::MutableOperandRange getMutableSuccessorOperands(::std::optional<unsigned> index);
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

namespace detail {
class ExecuteRegionOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ExecuteRegionOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class ExecuteRegionOpGenericAdaptor : public detail::ExecuteRegionOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ExecuteRegionOpGenericAdaptorBase;
public:
  ExecuteRegionOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class ExecuteRegionOpAdaptor : public ExecuteRegionOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ExecuteRegionOpGenericAdaptor::ExecuteRegionOpGenericAdaptor;
  ExecuteRegionOpAdaptor(ExecuteRegionOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ExecuteRegionOp : public ::mlir::Op<ExecuteRegionOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::RegionBranchOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExecuteRegionOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ExecuteRegionOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
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
  void getSuccessorRegions(::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
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

namespace detail {
class ForOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ForOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class ForOpGenericAdaptor : public detail::ForOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ForOpGenericAdaptorBase;
public:
  ForOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLowerBound() {
    return (*getODSOperands(0).begin());
  }

  ValueT getUpperBound() {
    return (*getODSOperands(1).begin());
  }

  ValueT getStep() {
    return (*getODSOperands(2).begin());
  }

  RangeT getInitArgs() {
    return getODSOperands(3);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ForOpAdaptor : public ForOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ForOpGenericAdaptor::ForOpGenericAdaptor;
  ForOpAdaptor(ForOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ForOp : public ::mlir::Op<ForOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<3>::Impl, ::mlir::OpTrait::SingleBlockImplicitTerminator<scf::YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::LoopLikeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::RegionBranchOpInterface::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ForOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ForOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value lowerBound, Value upperBound, Value step, ValueRange iterArgs = std::nullopt, function_ref<void(OpBuilder &, Location, Value, ValueRange)> odsArg4 = nullptr);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  ::mlir::LogicalResult verifyRegions();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::Region &getLoopBody();
  ::std::optional<::mlir::Value> getSingleInductionVar();
  ::std::optional<::mlir::OpFoldResult> getSingleLowerBound();
  ::std::optional<::mlir::OpFoldResult> getSingleStep();
  ::std::optional<::mlir::OpFoldResult> getSingleUpperBound();
  void getSuccessorRegions(::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
public:
  using BodyBuilderFn =
      function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

  Value getInductionVar() { return getBody()->getArgument(0); }
  Block::BlockArgListType getRegionIterArgs() {
    return getBody()->getArguments().drop_front(getNumInductionVars());
  }
  /// Return the `index`-th region iteration argument.
  BlockArgument getRegionIterArg(unsigned index) {
    assert(index < getNumRegionIterArgs() &&
      "expected an index less than the number of region iter args");
    return getBody()->getArguments().drop_front(getNumInductionVars())[index];
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
  void setIterArg(unsigned iterArgNum, Value iterArgValue) {
    getOperation()->setOperand(iterArgNum + getNumControlOperands(), iterArgValue);
  }

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
  /// Get the iter arg number for an operand. If it isnt an iter arg
  /// operand return std::nullopt.
  std::optional<unsigned> getIterArgNumberForOpOperand(OpOperand &opOperand) {
    if (opOperand.getOwner() != getOperation())
      return std::nullopt;
    unsigned operandNumber = opOperand.getOperandNumber();
    if (operandNumber < getNumControlOperands())
      return std::nullopt;
    return operandNumber - getNumControlOperands();
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
  OperandRange getSuccessorEntryOperands(std::optional<unsigned> index);

  /// Returns the step as an `APInt` if it is constant.
  std::optional<APInt> getConstantStep();

  /// Interface method for ConditionallySpeculatable.
  Speculation::Speculatability getSpeculatability();
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::ForOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::ForallOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ForallOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ForallOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI64ArrayAttr getStaticLowerBoundAttr();
  ::llvm::ArrayRef<int64_t> getStaticLowerBound();
  ::mlir::DenseI64ArrayAttr getStaticUpperBoundAttr();
  ::llvm::ArrayRef<int64_t> getStaticUpperBound();
  ::mlir::DenseI64ArrayAttr getStaticStepAttr();
  ::llvm::ArrayRef<int64_t> getStaticStep();
  ::mlir::ArrayAttr getMappingAttr();
  ::std::optional< ::mlir::ArrayAttr > getMapping();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class ForallOpGenericAdaptor : public detail::ForallOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ForallOpGenericAdaptorBase;
public:
  ForallOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getDynamicLowerBound() {
    return getODSOperands(0);
  }

  RangeT getDynamicUpperBound() {
    return getODSOperands(1);
  }

  RangeT getDynamicStep() {
    return getODSOperands(2);
  }

  RangeT getOutputs() {
    return getODSOperands(3);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ForallOpAdaptor : public ForallOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ForallOpGenericAdaptor::ForallOpGenericAdaptor;
  ForallOpAdaptor(ForallOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ForallOp : public ::mlir::Op<ForallOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<scf::InParallelOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ForallOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ForallOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("mapping"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("staticLowerBound"), ::llvm::StringRef("staticStep"), ::llvm::StringRef("staticUpperBound")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getMappingAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getMappingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getStaticLowerBoundAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getStaticLowerBoundAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getStaticStepAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getStaticStepAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getStaticUpperBoundAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getStaticUpperBoundAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.forall");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getDynamicLowerBound();
  ::mlir::Operation::operand_range getDynamicUpperBound();
  ::mlir::Operation::operand_range getDynamicStep();
  ::mlir::Operation::operand_range getOutputs();
  ::mlir::MutableOperandRange getDynamicLowerBoundMutable();
  ::mlir::MutableOperandRange getDynamicUpperBoundMutable();
  ::mlir::MutableOperandRange getDynamicStepMutable();
  ::mlir::MutableOperandRange getOutputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getRegion();
  ::mlir::DenseI64ArrayAttr getStaticLowerBoundAttr();
  ::llvm::ArrayRef<int64_t> getStaticLowerBound();
  ::mlir::DenseI64ArrayAttr getStaticUpperBoundAttr();
  ::llvm::ArrayRef<int64_t> getStaticUpperBound();
  ::mlir::DenseI64ArrayAttr getStaticStepAttr();
  ::llvm::ArrayRef<int64_t> getStaticStep();
  ::mlir::ArrayAttr getMappingAttr();
  ::std::optional< ::mlir::ArrayAttr > getMapping();
  void setStaticLowerBoundAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticLowerBound(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticUpperBoundAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticUpperBound(::llvm::ArrayRef<int64_t> attrValue);
  void setStaticStepAttr(::mlir::DenseI64ArrayAttr attr);
  void setStaticStep(::llvm::ArrayRef<int64_t> attrValue);
  void setMappingAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeMappingAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<OpFoldResult> lbs, ArrayRef<OpFoldResult> ubs, ArrayRef<OpFoldResult> steps, ValueRange outputs, std::optional<ArrayAttr> mapping, function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn = nullptr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<OpFoldResult> ubs, ValueRange outputs, std::optional<ArrayAttr> mapping, function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuilderFn = nullptr);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 5 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  // Get lower bounds as OpFoldResult.
  SmallVector<OpFoldResult> getMixedLowerBound() {
    Builder b(getOperation()->getContext());
    return getMixedValues(getStaticLowerBound(), getDynamicLowerBound(), b);
  }

  // Get upper bounds as OpFoldResult.
  SmallVector<OpFoldResult> getMixedUpperBound() {
    Builder b(getOperation()->getContext());
    return getMixedValues(getStaticUpperBound(), getDynamicUpperBound(), b);
  }

  // Get steps as OpFoldResult.
  SmallVector<OpFoldResult> getMixedStep() {
    Builder b(getOperation()->getContext());
    return getMixedValues(getStaticStep(), getDynamicStep(), b);
  }

  /// Get lower bounds as values.
  SmallVector<Value> getLowerBound(OpBuilder &b) {
    return getValueOrCreateConstantIndexOp(b, getLoc(), getMixedLowerBound());
  }

  /// Get upper bounds as values.
  SmallVector<Value> getUpperBound(OpBuilder &b) {
    return getValueOrCreateConstantIndexOp(b, getLoc(), getMixedUpperBound());
  }

  /// Get steps as values.
  SmallVector<Value> getStep(OpBuilder &b) {
    return getValueOrCreateConstantIndexOp(b, getLoc(), getMixedStep());
  }

  int64_t getRank() { return getStaticLowerBound().size(); }

  /// Number of operands controlling the loop: lbs, ubs, steps
  unsigned getNumControlOperands() { return 3 * getRank(); }

  /// Number of dynamic operands controlling the loop: lbs, ubs, steps
  unsigned getNumDynamicControlOperands() {
    return getODSOperandIndexAndLength(3).first;
  }

  OpResult getTiedOpResult(OpOperand *opOperand) {
    assert(opOperand->getOperandNumber() >= getNumDynamicControlOperands() &&
           "invalid operand");
    return getOperation()->getOpResult(
        opOperand->getOperandNumber() - getNumDynamicControlOperands());
  }

  /// Return the num_threads operand that is tied to the given thread id
  /// block argument.
  OpOperand *getTiedOpOperand(BlockArgument bbArg) {
    assert(bbArg.getArgNumber() >= getRank() && "invalid bbArg");

    return &getOperation()->getOpOperand(getNumDynamicControlOperands() +
                                         bbArg.getArgNumber() - getRank());
  }

  /// Return the shared_outs operand that is tied to the given OpResult.
  OpOperand *getTiedOpOperand(OpResult opResult) {
    assert(opResult.getDefiningOp() == getOperation() && "invalid OpResult");
    return &getOperation()->getOpOperand(getNumDynamicControlOperands() +
                                         opResult.getResultNumber());
  }

  BlockArgument getTiedBlockArgument(OpOperand *opOperand) {
    assert(opOperand->getOperandNumber() >= getNumDynamicControlOperands() &&
           "invalid operand");

    return getBody()->getArgument(opOperand->getOperandNumber() -
                                  getNumDynamicControlOperands() + getRank());
  }

  ArrayRef<BlockArgument> getOutputBlockArguments() {
    return getBody()->getArguments().drop_front(getRank());
  }

  ::mlir::ValueRange getInductionVars() {
    return getBody()->getArguments().take_front(getRank());
  }

  ::mlir::Value getInductionVar(int64_t idx) {
    return getInductionVars()[idx];
  }

  ::mlir::Block::BlockArgListType getRegionOutArgs() {
    return getBody()->getArguments().drop_front(getRank());
  }

  /// Checks if the lbs are zeros and steps are ones.
  bool isNormalized();

  // The ensureTerminator method generated by SingleBlockImplicitTerminator is
  // unaware of the fact that our terminator also needs a region to be
  // well-formed. We override it here to ensure that we do the right thing.
  static void ensureTerminator(Region & region, OpBuilder & builder,
                               Location loc);

  InParallelOp getTerminator();
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::ForallOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::IfOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class IfOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  IfOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getThenRegion();
  ::mlir::Region &getElseRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class IfOpGenericAdaptor : public detail::IfOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::IfOpGenericAdaptorBase;
public:
  IfOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getCondition() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class IfOpAdaptor : public IfOpGenericAdaptor<::mlir::ValueRange> {
public:
  using IfOpGenericAdaptor::IfOpGenericAdaptor;
  IfOpAdaptor(IfOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class IfOp : public ::mlir::Op<IfOp, ::mlir::OpTrait::NRegions<2>::Impl, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::SingleBlockImplicitTerminator<scf::YieldOp>::Impl, ::mlir::OpTrait::NoRegionArguments, ::mlir::OpTrait::OpInvariants, ::mlir::RegionBranchOpInterface::Trait, ::mlir::InferTypeOpInterface::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = IfOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = IfOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.if");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IntegerType> getCondition();
  ::mlir::MutableOperandRange getConditionMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getThenRegion();
  ::mlir::Region &getElseRegion();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, Value cond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, Value cond, bool addThenBlock, bool addElseBlock);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value cond, bool withElseRegion);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, Value cond, bool withElseRegion);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value cond, function_ref<void(OpBuilder &, Location)> thenBuilder = buildTerminatedBody, function_ref<void(OpBuilder &, Location)> elseBuilder = nullptr);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  ::mlir::LogicalResult fold(FoldAdaptor adaptor, ::llvm::SmallVectorImpl<::mlir::OpFoldResult> &results);
  void getSuccessorRegions(::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
  void getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds);
  static ::mlir::LogicalResult inferReturnTypes(::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes);
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
// ::mlir::scf::InParallelOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class InParallelOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  InParallelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class InParallelOpGenericAdaptor : public detail::InParallelOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::InParallelOpGenericAdaptorBase;
public:
  InParallelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class InParallelOpAdaptor : public InParallelOpGenericAdaptor<::mlir::ValueRange> {
public:
  using InParallelOpGenericAdaptor::InParallelOpGenericAdaptor;
  InParallelOpAdaptor(InParallelOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class InParallelOp : public ::mlir::Op<InParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::HasParent<ForallOp>::Impl, ::mlir::OpTrait::NoTerminator, ::mlir::OpTrait::SingleBlock, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator, ::mlir::ParallelCombiningOpInterface::Trait, ::mlir::RegionKindInterface::Trait, ::mlir::OpTrait::HasOnlyGraphRegion> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InParallelOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = InParallelOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.forall.in_parallel");
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
  ::llvm::SmallVector<::mlir::BlockArgument> getDests();
  ::llvm::iterator_range<::mlir::Block::iterator> getYieldingOps();
  ::mlir::OpResult getParentResult(int64_t idx);
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::InParallelOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::IndexSwitchOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class IndexSwitchOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  IndexSwitchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseI64ArrayAttr getCasesAttr();
  ::llvm::ArrayRef<int64_t> getCases();
  ::mlir::Region &getDefaultRegion();
  ::mlir::RegionRange getCaseRegions();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class IndexSwitchOpGenericAdaptor : public detail::IndexSwitchOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::IndexSwitchOpGenericAdaptorBase;
public:
  IndexSwitchOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getArg() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class IndexSwitchOpAdaptor : public IndexSwitchOpGenericAdaptor<::mlir::ValueRange> {
public:
  using IndexSwitchOpGenericAdaptor::IndexSwitchOpGenericAdaptor;
  IndexSwitchOpAdaptor(IndexSwitchOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class IndexSwitchOp : public ::mlir::Op<IndexSwitchOp, ::mlir::OpTrait::AtLeastNRegions<1>::Impl, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::SingleBlockImplicitTerminator<scf::YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasRecursiveMemoryEffects, ::mlir::RegionBranchOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = IndexSwitchOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = IndexSwitchOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("cases")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getCasesAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCasesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("scf.index_switch");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::IndexType> getArg();
  ::mlir::MutableOperandRange getArgMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range getResults();
  ::mlir::Region &getDefaultRegion();
  ::mlir::MutableArrayRef<::mlir::Region> getCaseRegions();
  ::mlir::DenseI64ArrayAttr getCasesAttr();
  ::llvm::ArrayRef<int64_t> getCases();
  void setCasesAttr(::mlir::DenseI64ArrayAttr attr);
  void setCases(::llvm::ArrayRef<int64_t> attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value arg, ::mlir::DenseI64ArrayAttr cases, unsigned caseRegionsCount);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::Value arg, ::llvm::ArrayRef<int64_t> cases, unsigned caseRegionsCount);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes, unsigned numRegions);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getSuccessorRegions(::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
  void getRegionInvocationBounds(::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::InvocationBounds> &invocationBounds);
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
  /// Get the number of cases.
  unsigned getNumCases();

  /// Get the default region body.
  Block &getDefaultBlock();

  /// Get the body of a case region.
  Block &getCaseBlock(unsigned idx);
};
} // namespace scf
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::scf::IndexSwitchOp)

namespace mlir {
namespace scf {

//===----------------------------------------------------------------------===//
// ::mlir::scf::ParallelOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ParallelOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ParallelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getRegion();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class ParallelOpGenericAdaptor : public detail::ParallelOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ParallelOpGenericAdaptorBase;
public:
  ParallelOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getLowerBound() {
    return getODSOperands(0);
  }

  RangeT getUpperBound() {
    return getODSOperands(1);
  }

  RangeT getStep() {
    return getODSOperands(2);
  }

  RangeT getInitVals() {
    return getODSOperands(3);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ParallelOpAdaptor : public ParallelOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ParallelOpGenericAdaptor::ParallelOpGenericAdaptor;
  ParallelOpAdaptor(ParallelOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ParallelOp : public ::mlir::Op<ParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<scf::YieldOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::LoopLikeOpInterface::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ParallelOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::ArrayRef(attrNames);
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
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
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
// ::mlir::scf::ReduceOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ReduceOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ReduceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getReductionOperator();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class ReduceOpGenericAdaptor : public detail::ReduceOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReduceOpGenericAdaptorBase;
public:
  ReduceOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getOperand() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReduceOpAdaptor : public ReduceOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReduceOpGenericAdaptor::ReduceOpGenericAdaptor;
  ReduceOpAdaptor(ReduceOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ReduceOp : public ::mlir::Op<ReduceOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::HasParent<ParallelOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReduceOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReduceOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
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

namespace detail {
class ReduceReturnOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  ReduceReturnOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ReduceReturnOpGenericAdaptor : public detail::ReduceReturnOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReduceReturnOpGenericAdaptorBase;
public:
  ReduceReturnOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

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
class ReduceReturnOpAdaptor : public ReduceReturnOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReduceReturnOpGenericAdaptor::ReduceReturnOpGenericAdaptor;
  ReduceReturnOpAdaptor(ReduceReturnOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ReduceReturnOp : public ::mlir::Op<ReduceReturnOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::HasParent<ReduceOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReduceReturnOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReduceReturnOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
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

namespace detail {
class WhileOpGenericAdaptorBase {
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::std::optional<::mlir::OperationName> odsOpName;
public:
  WhileOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::Region &getBefore();
  ::mlir::Region &getAfter();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class WhileOpGenericAdaptor : public detail::WhileOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::WhileOpGenericAdaptorBase;
public:
  WhileOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {}) : Base(attrs, regions), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getInits() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class WhileOpAdaptor : public WhileOpGenericAdaptor<::mlir::ValueRange> {
public:
  using WhileOpGenericAdaptor::WhileOpGenericAdaptor;
  WhileOpAdaptor(WhileOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class WhileOp : public ::mlir::Op<WhileOp, ::mlir::OpTrait::NRegions<2>::Impl, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::RegionBranchOpInterface::Trait, ::mlir::OpTrait::HasRecursiveMemoryEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WhileOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = WhileOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
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
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange resultTypes, ValueRange operands, function_ref<void(OpBuilder &, Location, ValueRange)> beforeBuilder, function_ref<void(OpBuilder &, Location, ValueRange)> afterBuilder);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  void getSuccessorRegions(::std::optional<unsigned> index, ::llvm::ArrayRef<::mlir::Attribute> operands, ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
public:
  using BodyBuilderFn =
      function_ref<void(OpBuilder &, Location, ValueRange)>;

  OperandRange getSuccessorEntryOperands(std::optional<unsigned> index);
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

  RangeT getResults() {
    return getODSOperands(0);
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
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<ExecuteRegionOp, ForOp, IfOp, IndexSwitchOp, ParallelOp, WhileOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
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

