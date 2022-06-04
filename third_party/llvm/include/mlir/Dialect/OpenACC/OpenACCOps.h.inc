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
namespace acc {
class DataOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class EnterDataOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class ExitDataOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class InitOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class LoopOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class ParallelOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class ShutdownOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class TerminatorOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class UpdateOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class WaitOp;
} // namespace acc
} // namespace mlir
namespace mlir {
namespace acc {
class YieldOp;
} // namespace acc
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::DataOp declarations
//===----------------------------------------------------------------------===//

class DataOpAdaptor {
public:
  DataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  DataOpAdaptor(DataOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::ValueRange copyOperands();
  ::mlir::ValueRange copyinOperands();
  ::mlir::ValueRange copyinReadonlyOperands();
  ::mlir::ValueRange copyoutOperands();
  ::mlir::ValueRange copyoutZeroOperands();
  ::mlir::ValueRange createOperands();
  ::mlir::ValueRange createZeroOperands();
  ::mlir::ValueRange noCreateOperands();
  ::mlir::ValueRange presentOperands();
  ::mlir::ValueRange deviceptrOperands();
  ::mlir::ValueRange attachOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::acc::ClauseDefaultValueAttr defaultAttrAttr();
  ::llvm::Optional<::mlir::acc::ClauseDefaultValue> defaultAttr();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class DataOp : public ::mlir::Op<DataOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = DataOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("defaultAttr"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr defaultAttrAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr defaultAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.data");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::Operation::operand_range copyOperands();
  ::mlir::Operation::operand_range copyinOperands();
  ::mlir::Operation::operand_range copyinReadonlyOperands();
  ::mlir::Operation::operand_range copyoutOperands();
  ::mlir::Operation::operand_range copyoutZeroOperands();
  ::mlir::Operation::operand_range createOperands();
  ::mlir::Operation::operand_range createZeroOperands();
  ::mlir::Operation::operand_range noCreateOperands();
  ::mlir::Operation::operand_range presentOperands();
  ::mlir::Operation::operand_range deviceptrOperands();
  ::mlir::Operation::operand_range attachOperands();
  ::mlir::MutableOperandRange ifCondMutable();
  ::mlir::MutableOperandRange copyOperandsMutable();
  ::mlir::MutableOperandRange copyinOperandsMutable();
  ::mlir::MutableOperandRange copyinReadonlyOperandsMutable();
  ::mlir::MutableOperandRange copyoutOperandsMutable();
  ::mlir::MutableOperandRange copyoutZeroOperandsMutable();
  ::mlir::MutableOperandRange createOperandsMutable();
  ::mlir::MutableOperandRange createZeroOperandsMutable();
  ::mlir::MutableOperandRange noCreateOperandsMutable();
  ::mlir::MutableOperandRange presentOperandsMutable();
  ::mlir::MutableOperandRange deviceptrOperandsMutable();
  ::mlir::MutableOperandRange attachOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::acc::ClauseDefaultValueAttr defaultAttrAttr();
  ::llvm::Optional<::mlir::acc::ClauseDefaultValue> defaultAttr();
  void defaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr);
  ::mlir::Attribute removeDefaultAttrAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange deviceptrOperands, ::mlir::ValueRange attachOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
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
    assert(index < 2 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::DataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::EnterDataOp declarations
//===----------------------------------------------------------------------===//

class EnterDataOpAdaptor {
public:
  EnterDataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  EnterDataOpAdaptor(EnterDataOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::ValueRange waitOperands();
  ::mlir::ValueRange copyinOperands();
  ::mlir::ValueRange createOperands();
  ::mlir::ValueRange createZeroOperands();
  ::mlir::ValueRange attachOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr asyncAttr();
  bool async();
  ::mlir::UnitAttr waitAttr();
  bool wait();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class EnterDataOp : public ::mlir::Op<EnterDataOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = EnterDataOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("wait")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr asyncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr asyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr waitAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr waitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.enter_data");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::Operation::operand_range waitOperands();
  ::mlir::Operation::operand_range copyinOperands();
  ::mlir::Operation::operand_range createOperands();
  ::mlir::Operation::operand_range createZeroOperands();
  ::mlir::Operation::operand_range attachOperands();
  ::mlir::MutableOperandRange ifCondMutable();
  ::mlir::MutableOperandRange asyncOperandMutable();
  ::mlir::MutableOperandRange waitDevnumMutable();
  ::mlir::MutableOperandRange waitOperandsMutable();
  ::mlir::MutableOperandRange copyinOperandsMutable();
  ::mlir::MutableOperandRange createOperandsMutable();
  ::mlir::MutableOperandRange createZeroOperandsMutable();
  ::mlir::MutableOperandRange attachOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr asyncAttr();
  bool async();
  ::mlir::UnitAttr waitAttr();
  bool wait();
  void asyncAttr(::mlir::UnitAttr attr);
  void waitAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeAsyncAttr();
  ::mlir::Attribute removeWaitAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange attachOperands);
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
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::EnterDataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ExitDataOp declarations
//===----------------------------------------------------------------------===//

class ExitDataOpAdaptor {
public:
  ExitDataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ExitDataOpAdaptor(ExitDataOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::ValueRange waitOperands();
  ::mlir::ValueRange copyoutOperands();
  ::mlir::ValueRange deleteOperands();
  ::mlir::ValueRange detachOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr asyncAttr();
  bool async();
  ::mlir::UnitAttr waitAttr();
  bool wait();
  ::mlir::UnitAttr finalizeAttr();
  bool finalize();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ExitDataOp : public ::mlir::Op<ExitDataOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ExitDataOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async"), ::llvm::StringRef("finalize"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("wait")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr asyncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr asyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr finalizeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr finalizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr waitAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr waitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.exit_data");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::Operation::operand_range waitOperands();
  ::mlir::Operation::operand_range copyoutOperands();
  ::mlir::Operation::operand_range deleteOperands();
  ::mlir::Operation::operand_range detachOperands();
  ::mlir::MutableOperandRange ifCondMutable();
  ::mlir::MutableOperandRange asyncOperandMutable();
  ::mlir::MutableOperandRange waitDevnumMutable();
  ::mlir::MutableOperandRange waitOperandsMutable();
  ::mlir::MutableOperandRange copyoutOperandsMutable();
  ::mlir::MutableOperandRange deleteOperandsMutable();
  ::mlir::MutableOperandRange detachOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr asyncAttr();
  bool async();
  ::mlir::UnitAttr waitAttr();
  bool wait();
  ::mlir::UnitAttr finalizeAttr();
  bool finalize();
  void asyncAttr(::mlir::UnitAttr attr);
  void waitAttr(::mlir::UnitAttr attr);
  void finalizeAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeAsyncAttr();
  ::mlir::Attribute removeWaitAttr();
  ::mlir::Attribute removeFinalizeAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/::mlir::UnitAttr finalize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/::mlir::UnitAttr finalize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/bool finalize);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/bool async, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool wait, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange deleteOperands, ::mlir::ValueRange detachOperands, /*optional*/bool finalize);
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
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ExitDataOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::InitOp declarations
//===----------------------------------------------------------------------===//

class InitOpAdaptor {
public:
  InitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  InitOpAdaptor(InitOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange deviceTypeOperands();
  ::mlir::Value deviceNumOperand();
  ::mlir::Value ifCond();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class InitOp : public ::mlir::Op<InitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = InitOpAdaptor;
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
    return ::llvm::StringLiteral("acc.init");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range deviceTypeOperands();
  ::mlir::Value deviceNumOperand();
  ::mlir::Value ifCond();
  ::mlir::MutableOperandRange deviceTypeOperandsMutable();
  ::mlir::MutableOperandRange deviceNumOperandMutable();
  ::mlir::MutableOperandRange ifCondMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
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
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::InitOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::LoopOp declarations
//===----------------------------------------------------------------------===//

class LoopOpAdaptor {
public:
  LoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  LoopOpAdaptor(LoopOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value gangNum();
  ::mlir::Value gangStatic();
  ::mlir::Value workerNum();
  ::mlir::Value vectorLength();
  ::mlir::ValueRange tileOperands();
  ::mlir::ValueRange privateOperands();
  ::mlir::ValueRange reductionOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr collapseAttr();
  ::llvm::Optional<uint64_t> collapse();
  ::mlir::UnitAttr seqAttr();
  bool seq();
  ::mlir::UnitAttr independentAttr();
  bool independent();
  ::mlir::UnitAttr auto_Attr();
  bool auto_();
  ::mlir::acc::ReductionOpAttr reductionOpAttr();
  ::llvm::Optional<::mlir::acc::ReductionOp> reductionOp();
  ::mlir::IntegerAttr exec_mappingAttr();
  uint64_t exec_mapping();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class LoopOp : public ::mlir::Op<LoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::VariadicResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::SingleBlockImplicitTerminator<acc::YieldOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LoopOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("auto_"), ::llvm::StringRef("collapse"), ::llvm::StringRef("exec_mapping"), ::llvm::StringRef("independent"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("reductionOp"), ::llvm::StringRef("seq")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr auto_AttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr auto_AttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr collapseAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr collapseAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr exec_mappingAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr exec_mappingAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr independentAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr independentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr reductionOpAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr reductionOpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr seqAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr seqAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.loop");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value gangNum();
  ::mlir::Value gangStatic();
  ::mlir::Value workerNum();
  ::mlir::Value vectorLength();
  ::mlir::Operation::operand_range tileOperands();
  ::mlir::Operation::operand_range privateOperands();
  ::mlir::Operation::operand_range reductionOperands();
  ::mlir::MutableOperandRange gangNumMutable();
  ::mlir::MutableOperandRange gangStaticMutable();
  ::mlir::MutableOperandRange workerNumMutable();
  ::mlir::MutableOperandRange vectorLengthMutable();
  ::mlir::MutableOperandRange tileOperandsMutable();
  ::mlir::MutableOperandRange privateOperandsMutable();
  ::mlir::MutableOperandRange reductionOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Operation::result_range results();
  ::mlir::Region &region();
  ::mlir::IntegerAttr collapseAttr();
  ::llvm::Optional<uint64_t> collapse();
  ::mlir::UnitAttr seqAttr();
  bool seq();
  ::mlir::UnitAttr independentAttr();
  bool independent();
  ::mlir::UnitAttr auto_Attr();
  bool auto_();
  ::mlir::acc::ReductionOpAttr reductionOpAttr();
  ::llvm::Optional<::mlir::acc::ReductionOp> reductionOp();
  ::mlir::IntegerAttr exec_mappingAttr();
  uint64_t exec_mapping();
  void collapseAttr(::mlir::IntegerAttr attr);
  void seqAttr(::mlir::UnitAttr attr);
  void independentAttr(::mlir::UnitAttr attr);
  void auto_Attr(::mlir::UnitAttr attr);
  void reductionOpAttr(::mlir::acc::ReductionOpAttr attr);
  void exec_mappingAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeCollapseAttr();
  ::mlir::Attribute removeSeqAttr();
  ::mlir::Attribute removeIndependentAttr();
  ::mlir::Attribute removeAuto_Attr();
  ::mlir::Attribute removeReductionOpAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::UnitAttr seq, /*optional*/::mlir::UnitAttr independent, /*optional*/::mlir::UnitAttr auto_, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::IntegerAttr exec_mapping);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, /*optional*/::mlir::IntegerAttr collapse, /*optional*/::mlir::Value gangNum, /*optional*/::mlir::Value gangStatic, /*optional*/::mlir::Value workerNum, /*optional*/::mlir::Value vectorLength, /*optional*/bool seq, /*optional*/bool independent, /*optional*/bool auto_, ::mlir::ValueRange tileOperands, ::mlir::ValueRange privateOperands, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, uint64_t exec_mapping = 0);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 7 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  static StringRef getCollapseAttrName() { return "collapse"; }
  static StringRef getSeqAttrName() { return "seq"; }
  static StringRef getIndependentAttrName() { return "independent"; }
  static StringRef getAutoAttrName() { return "auto"; }
  static StringRef getExecutionMappingAttrName() { return "exec_mapping"; }
  static StringRef getGangKeyword() { return "gang"; }
  static StringRef getGangNumKeyword() { return "num"; }
  static StringRef getGangStaticKeyword() { return "static"; }
  static StringRef getVectorKeyword() { return "vector"; }
  static StringRef getWorkerKeyword() { return "worker"; }
  static StringRef getTileKeyword() { return "tile"; }
  static StringRef getPrivateKeyword() { return "private"; }
  static StringRef getReductionKeyword() { return "reduction"; }
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::LoopOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ParallelOp declarations
//===----------------------------------------------------------------------===//

class ParallelOpAdaptor {
public:
  ParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ParallelOpAdaptor(ParallelOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value async();
  ::mlir::ValueRange waitOperands();
  ::mlir::Value numGangs();
  ::mlir::Value numWorkers();
  ::mlir::Value vectorLength();
  ::mlir::Value ifCond();
  ::mlir::Value selfCond();
  ::mlir::ValueRange reductionOperands();
  ::mlir::ValueRange copyOperands();
  ::mlir::ValueRange copyinOperands();
  ::mlir::ValueRange copyinReadonlyOperands();
  ::mlir::ValueRange copyoutOperands();
  ::mlir::ValueRange copyoutZeroOperands();
  ::mlir::ValueRange createOperands();
  ::mlir::ValueRange createZeroOperands();
  ::mlir::ValueRange noCreateOperands();
  ::mlir::ValueRange presentOperands();
  ::mlir::ValueRange devicePtrOperands();
  ::mlir::ValueRange attachOperands();
  ::mlir::ValueRange gangPrivateOperands();
  ::mlir::ValueRange gangFirstPrivateOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr asyncAttrAttr();
  bool asyncAttr();
  ::mlir::UnitAttr waitAttrAttr();
  bool waitAttr();
  ::mlir::UnitAttr selfAttrAttr();
  bool selfAttr();
  ::mlir::acc::ReductionOpAttr reductionOpAttr();
  ::llvm::Optional<::mlir::acc::ReductionOp> reductionOp();
  ::mlir::acc::ClauseDefaultValueAttr defaultAttrAttr();
  ::llvm::Optional<::mlir::acc::ClauseDefaultValue> defaultAttr();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ParallelOp : public ::mlir::Op<ParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("asyncAttr"), ::llvm::StringRef("defaultAttr"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("reductionOp"), ::llvm::StringRef("selfAttr"), ::llvm::StringRef("waitAttr")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr asyncAttrAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr asyncAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr defaultAttrAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr defaultAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr reductionOpAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr reductionOpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr selfAttrAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr selfAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr waitAttrAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr waitAttrAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.parallel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value async();
  ::mlir::Operation::operand_range waitOperands();
  ::mlir::Value numGangs();
  ::mlir::Value numWorkers();
  ::mlir::Value vectorLength();
  ::mlir::Value ifCond();
  ::mlir::Value selfCond();
  ::mlir::Operation::operand_range reductionOperands();
  ::mlir::Operation::operand_range copyOperands();
  ::mlir::Operation::operand_range copyinOperands();
  ::mlir::Operation::operand_range copyinReadonlyOperands();
  ::mlir::Operation::operand_range copyoutOperands();
  ::mlir::Operation::operand_range copyoutZeroOperands();
  ::mlir::Operation::operand_range createOperands();
  ::mlir::Operation::operand_range createZeroOperands();
  ::mlir::Operation::operand_range noCreateOperands();
  ::mlir::Operation::operand_range presentOperands();
  ::mlir::Operation::operand_range devicePtrOperands();
  ::mlir::Operation::operand_range attachOperands();
  ::mlir::Operation::operand_range gangPrivateOperands();
  ::mlir::Operation::operand_range gangFirstPrivateOperands();
  ::mlir::MutableOperandRange asyncMutable();
  ::mlir::MutableOperandRange waitOperandsMutable();
  ::mlir::MutableOperandRange numGangsMutable();
  ::mlir::MutableOperandRange numWorkersMutable();
  ::mlir::MutableOperandRange vectorLengthMutable();
  ::mlir::MutableOperandRange ifCondMutable();
  ::mlir::MutableOperandRange selfCondMutable();
  ::mlir::MutableOperandRange reductionOperandsMutable();
  ::mlir::MutableOperandRange copyOperandsMutable();
  ::mlir::MutableOperandRange copyinOperandsMutable();
  ::mlir::MutableOperandRange copyinReadonlyOperandsMutable();
  ::mlir::MutableOperandRange copyoutOperandsMutable();
  ::mlir::MutableOperandRange copyoutZeroOperandsMutable();
  ::mlir::MutableOperandRange createOperandsMutable();
  ::mlir::MutableOperandRange createZeroOperandsMutable();
  ::mlir::MutableOperandRange noCreateOperandsMutable();
  ::mlir::MutableOperandRange presentOperandsMutable();
  ::mlir::MutableOperandRange devicePtrOperandsMutable();
  ::mlir::MutableOperandRange attachOperandsMutable();
  ::mlir::MutableOperandRange gangPrivateOperandsMutable();
  ::mlir::MutableOperandRange gangFirstPrivateOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::UnitAttr asyncAttrAttr();
  bool asyncAttr();
  ::mlir::UnitAttr waitAttrAttr();
  bool waitAttr();
  ::mlir::UnitAttr selfAttrAttr();
  bool selfAttr();
  ::mlir::acc::ReductionOpAttr reductionOpAttr();
  ::llvm::Optional<::mlir::acc::ReductionOp> reductionOp();
  ::mlir::acc::ClauseDefaultValueAttr defaultAttrAttr();
  ::llvm::Optional<::mlir::acc::ClauseDefaultValue> defaultAttr();
  void asyncAttrAttr(::mlir::UnitAttr attr);
  void waitAttrAttr(::mlir::UnitAttr attr);
  void selfAttrAttr(::mlir::UnitAttr attr);
  void reductionOpAttr(::mlir::acc::ReductionOpAttr attr);
  void defaultAttrAttr(::mlir::acc::ClauseDefaultValueAttr attr);
  ::mlir::Attribute removeAsyncAttrAttr();
  ::mlir::Attribute removeWaitAttrAttr();
  ::mlir::Attribute removeSelfAttrAttr();
  ::mlir::Attribute removeReductionOpAttr();
  ::mlir::Attribute removeDefaultAttrAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/::mlir::UnitAttr asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/::mlir::UnitAttr selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value async, /*optional*/bool asyncAttr, ::mlir::ValueRange waitOperands, /*optional*/bool waitAttr, /*optional*/::mlir::Value numGangs, /*optional*/::mlir::Value numWorkers, /*optional*/::mlir::Value vectorLength, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value selfCond, /*optional*/bool selfAttr, /*optional*/::mlir::acc::ReductionOpAttr reductionOp, ::mlir::ValueRange reductionOperands, ::mlir::ValueRange copyOperands, ::mlir::ValueRange copyinOperands, ::mlir::ValueRange copyinReadonlyOperands, ::mlir::ValueRange copyoutOperands, ::mlir::ValueRange copyoutZeroOperands, ::mlir::ValueRange createOperands, ::mlir::ValueRange createZeroOperands, ::mlir::ValueRange noCreateOperands, ::mlir::ValueRange presentOperands, ::mlir::ValueRange devicePtrOperands, ::mlir::ValueRange attachOperands, ::mlir::ValueRange gangPrivateOperands, ::mlir::ValueRange gangFirstPrivateOperands, /*optional*/::mlir::acc::ClauseDefaultValueAttr defaultAttr);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  static StringRef getAsyncKeyword() { return "async"; }
  static StringRef getAsyncAttrName() { return "asyncAttr"; }
  static StringRef getWaitKeyword() { return "wait"; }
  static StringRef getWaitAttrName() { return "waitAttr"; }
  static StringRef getNumGangsKeyword() { return "num_gangs"; }
  static StringRef getNumWorkersKeyword() { return "num_workers"; }
  static StringRef getVectorLengthKeyword() { return "vector_length"; }
  static StringRef getIfKeyword() { return "if"; }
  static StringRef getSelfKeyword() { return "self"; }
  static StringRef getSelfAttrName() { return "selfAttr"; }
  static StringRef getReductionKeyword() { return "reduction"; }
  static StringRef getCopyKeyword() { return "copy"; }
  static StringRef getCopyinKeyword() { return "copyin"; }
  static StringRef getCopyinReadonlyKeyword() { return "copyin_readonly"; }
  static StringRef getCopyoutKeyword() { return "copyout"; }
  static StringRef getCopyoutZeroKeyword() { return "copyout_zero"; }
  static StringRef getCreateKeyword() { return "create"; }
  static StringRef getCreateZeroKeyword() { return "create_zero"; }
  static StringRef getNoCreateKeyword() { return "no_create"; }
  static StringRef getPresentKeyword() { return "present"; }
  static StringRef getDevicePtrKeyword() { return "deviceptr"; }
  static StringRef getAttachKeyword() { return "attach"; }
  static StringRef getPrivateKeyword() { return "private"; }
  static StringRef getFirstPrivateKeyword() { return "firstprivate"; }

  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ParallelOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::ShutdownOp declarations
//===----------------------------------------------------------------------===//

class ShutdownOpAdaptor {
public:
  ShutdownOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ShutdownOpAdaptor(ShutdownOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange deviceTypeOperands();
  ::mlir::Value deviceNumOperand();
  ::mlir::Value ifCond();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class ShutdownOp : public ::mlir::Op<ShutdownOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ShutdownOpAdaptor;
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
    return ::llvm::StringLiteral("acc.shutdown");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range deviceTypeOperands();
  ::mlir::Value deviceNumOperand();
  ::mlir::Value ifCond();
  ::mlir::MutableOperandRange deviceTypeOperandsMutable();
  ::mlir::MutableOperandRange deviceNumOperandMutable();
  ::mlir::MutableOperandRange ifCondMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange deviceTypeOperands, /*optional*/::mlir::Value deviceNumOperand, /*optional*/::mlir::Value ifCond);
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
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::ShutdownOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::TerminatorOp declarations
//===----------------------------------------------------------------------===//

class TerminatorOpAdaptor {
public:
  TerminatorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TerminatorOpAdaptor(TerminatorOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class TerminatorOp : public ::mlir::Op<TerminatorOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TerminatorOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.terminator");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::TerminatorOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::UpdateOp declarations
//===----------------------------------------------------------------------===//

class UpdateOpAdaptor {
public:
  UpdateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  UpdateOpAdaptor(UpdateOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::ValueRange waitOperands();
  ::mlir::ValueRange deviceTypeOperands();
  ::mlir::ValueRange hostOperands();
  ::mlir::ValueRange deviceOperands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr asyncAttr();
  bool async();
  ::mlir::UnitAttr waitAttr();
  bool wait();
  ::mlir::UnitAttr ifPresentAttr();
  bool ifPresent();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class UpdateOp : public ::mlir::Op<UpdateOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = UpdateOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async"), ::llvm::StringRef("ifPresent"), ::llvm::StringRef("operand_segment_sizes"), ::llvm::StringRef("wait")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr asyncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr asyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr ifPresentAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr ifPresentAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr waitAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr waitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.update");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value ifCond();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::Operation::operand_range waitOperands();
  ::mlir::Operation::operand_range deviceTypeOperands();
  ::mlir::Operation::operand_range hostOperands();
  ::mlir::Operation::operand_range deviceOperands();
  ::mlir::MutableOperandRange ifCondMutable();
  ::mlir::MutableOperandRange asyncOperandMutable();
  ::mlir::MutableOperandRange waitDevnumMutable();
  ::mlir::MutableOperandRange waitOperandsMutable();
  ::mlir::MutableOperandRange deviceTypeOperandsMutable();
  ::mlir::MutableOperandRange hostOperandsMutable();
  ::mlir::MutableOperandRange deviceOperandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr asyncAttr();
  bool async();
  ::mlir::UnitAttr waitAttr();
  bool wait();
  ::mlir::UnitAttr ifPresentAttr();
  bool ifPresent();
  void asyncAttr(::mlir::UnitAttr attr);
  void waitAttr(::mlir::UnitAttr attr);
  void ifPresentAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeAsyncAttr();
  ::mlir::Attribute removeWaitAttr();
  ::mlir::Attribute removeIfPresentAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::UnitAttr wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/::mlir::UnitAttr ifPresent);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value ifCond, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, ::mlir::ValueRange waitOperands, /*optional*/bool async, /*optional*/bool wait, ::mlir::ValueRange deviceTypeOperands, ::mlir::ValueRange hostOperands, ::mlir::ValueRange deviceOperands, /*optional*/bool ifPresent);
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
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  /// The number of data operands.
  unsigned getNumDataOperands();

  /// The i-th data operand passed.
  Value getDataOperand(unsigned i);
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::UpdateOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::WaitOp declarations
//===----------------------------------------------------------------------===//

class WaitOpAdaptor {
public:
  WaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  WaitOpAdaptor(WaitOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange waitOperands();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::Value ifCond();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr asyncAttr();
  bool async();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class WaitOp : public ::mlir::Op<WaitOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WaitOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("async"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr asyncAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr asyncAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.wait");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range waitOperands();
  ::mlir::Value asyncOperand();
  ::mlir::Value waitDevnum();
  ::mlir::Value ifCond();
  ::mlir::MutableOperandRange waitOperandsMutable();
  ::mlir::MutableOperandRange asyncOperandMutable();
  ::mlir::MutableOperandRange waitDevnumMutable();
  ::mlir::MutableOperandRange ifCondMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::UnitAttr asyncAttr();
  bool async();
  void asyncAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeAsyncAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/::mlir::UnitAttr async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange waitOperands, /*optional*/::mlir::Value asyncOperand, /*optional*/::mlir::Value waitDevnum, /*optional*/bool async, /*optional*/::mlir::Value ifCond);
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
    assert(index < 2 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::WaitOp)

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// ::mlir::acc::YieldOp declarations
//===----------------------------------------------------------------------===//

class YieldOpAdaptor {
public:
  YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  YieldOpAdaptor(YieldOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange operands();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<ParallelOp, LoopOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YieldOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("acc.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range operands();
  ::mlir::MutableOperandRange operandsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace acc
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::acc::YieldOp)


#endif  // GET_OP_CLASSES

