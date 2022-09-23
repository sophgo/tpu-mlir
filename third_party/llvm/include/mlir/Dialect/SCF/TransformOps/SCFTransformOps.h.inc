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
class GetParentForOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class LoopOutlineOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class LoopPeelOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class LoopPipelineOp;
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {
class LoopUnrollOp;
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
// ::mlir::transform::GetParentForOp declarations
//===----------------------------------------------------------------------===//

class GetParentForOpAdaptor {
public:
  GetParentForOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  GetParentForOpAdaptor(GetParentForOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getNumLoopsAttr();
  uint64_t getNumLoops();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class GetParentForOp : public ::mlir::Op<GetParentForOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::pdl::OperationType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::NavigationTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GetParentForOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("num_loops")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getNumLoopsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getNumLoopsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.loop.get_parent_for");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getParent();
  ::mlir::IntegerAttr getNumLoopsAttr();
  uint64_t getNumLoops();
  void setNumLoopsAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type parent, ::mlir::Value target, ::mlir::IntegerAttr num_loops);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr num_loops);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type parent, ::mlir::Value target, uint64_t num_loops = 1);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t num_loops = 1);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
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
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::GetParentForOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::LoopOutlineOp declarations
//===----------------------------------------------------------------------===//

class LoopOutlineOpAdaptor {
public:
  LoopOutlineOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  LoopOutlineOpAdaptor(LoopOutlineOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getFuncNameAttr();
  ::llvm::StringRef getFuncName();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class LoopOutlineOp : public ::mlir::Op<LoopOutlineOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::pdl::OperationType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::FunctionalStyleTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopOutlineOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("func_name")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getFuncNameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getFuncNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.loop.outline");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTransformed();
  ::mlir::StringAttr getFuncNameAttr();
  ::llvm::StringRef getFuncName();
  void setFuncNameAttr(::mlir::StringAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::StringAttr func_name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::StringAttr func_name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::llvm::StringRef func_name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::llvm::StringRef func_name);
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
    assert(index < 1 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::LoopOutlineOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::LoopPeelOp declarations
//===----------------------------------------------------------------------===//

class LoopPeelOpAdaptor {
public:
  LoopPeelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  LoopPeelOpAdaptor(LoopPeelOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getFailIfAlreadyDivisibleAttr();
  bool getFailIfAlreadyDivisible();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class LoopPeelOp : public ::mlir::Op<LoopPeelOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::pdl::OperationType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::FunctionalStyleTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::transform::TransformEachOpTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopPeelOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("fail_if_already_divisible")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getFailIfAlreadyDivisibleAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getFailIfAlreadyDivisibleAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.loop.peel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTransformed();
  ::mlir::BoolAttr getFailIfAlreadyDivisibleAttr();
  bool getFailIfAlreadyDivisible();
  void setFailIfAlreadyDivisibleAttr(::mlir::BoolAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::BoolAttr fail_if_already_divisible);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::BoolAttr fail_if_already_divisible);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, bool fail_if_already_divisible = false);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, bool fail_if_already_divisible = false);
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
      ::mlir::scf::ForOp target,
      ::llvm::SmallVector<::mlir::Operation *> &results,
      ::mlir::transform::TransformState &state);
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::LoopPeelOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::LoopPipelineOp declarations
//===----------------------------------------------------------------------===//

class LoopPipelineOpAdaptor {
public:
  LoopPipelineOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  LoopPipelineOpAdaptor(LoopPipelineOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getIterationIntervalAttr();
  uint64_t getIterationInterval();
  ::mlir::IntegerAttr getReadLatencyAttr();
  uint64_t getReadLatency();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class LoopPipelineOp : public ::mlir::Op<LoopPipelineOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::pdl::OperationType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::FunctionalStyleTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::transform::TransformEachOpTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopPipelineOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("iteration_interval"), ::llvm::StringRef("read_latency")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getIterationIntervalAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getIterationIntervalAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getReadLatencyAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getReadLatencyAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.loop.pipeline");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTransformed();
  ::mlir::IntegerAttr getIterationIntervalAttr();
  uint64_t getIterationInterval();
  ::mlir::IntegerAttr getReadLatencyAttr();
  uint64_t getReadLatency();
  void setIterationIntervalAttr(::mlir::IntegerAttr attr);
  void setReadLatencyAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::IntegerAttr iteration_interval, ::mlir::IntegerAttr read_latency);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr iteration_interval, ::mlir::IntegerAttr read_latency);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, uint64_t iteration_interval = 1, uint64_t read_latency = 10);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t iteration_interval = 1, uint64_t read_latency = 10);
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
    assert(index < 2 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::scf::ForOp target,
      ::llvm::SmallVector<::mlir::Operation *> &results,
      ::mlir::transform::TransformState &state);
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::LoopPipelineOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::LoopUnrollOp declarations
//===----------------------------------------------------------------------===//

class LoopUnrollOpAdaptor {
public:
  LoopUnrollOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  LoopUnrollOpAdaptor(LoopUnrollOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getTarget();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getFactorAttr();
  uint64_t getFactor();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class LoopUnrollOp : public ::mlir::Op<LoopUnrollOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::transform::FunctionalStyleTransformOpTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::transform::TransformOpInterface::Trait, ::mlir::transform::TransformEachOpTrait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopUnrollOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("factor")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getFactorAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getFactorAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("transform.loop.unroll");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::pdl::OperationType> getTarget();
  ::mlir::MutableOperandRange getTargetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr getFactorAttr();
  uint64_t getFactor();
  void setFactorAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value target, ::mlir::IntegerAttr factor);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr factor);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value target, uint64_t factor);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t factor);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
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
      ::mlir::scf::ForOp target,
      ::llvm::SmallVector<::mlir::Operation *> &results,
      ::mlir::transform::TransformState &state);
};
} // namespace transform
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::transform::LoopUnrollOp)


#endif  // GET_OP_CLASSES

