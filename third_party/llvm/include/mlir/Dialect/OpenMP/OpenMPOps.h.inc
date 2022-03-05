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
namespace omp {
class AtomicCaptureOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class AtomicReadOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class AtomicUpdateOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class AtomicWriteOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class BarrierOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class CriticalDeclareOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class CriticalOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class FlushOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class MasterOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class OrderedOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class OrderedRegionOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ParallelOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ReductionDeclareOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class ReductionOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class SectionOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class SectionsOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TargetOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskwaitOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TaskyieldOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class TerminatorOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class WsLoopOp;
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {
class YieldOp;
} // namespace omp
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicCaptureOp declarations
//===----------------------------------------------------------------------===//

class AtomicCaptureOpAdaptor {
public:
  AtomicCaptureOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AtomicCaptureOpAdaptor(AtomicCaptureOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr hint_valAttr();
  uint64_t hint_val();
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> memory_order_val();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AtomicCaptureOp : public ::mlir::Op<AtomicCaptureOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::SingleBlockImplicitTerminator<TerminatorOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicCaptureOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hint_val"), ::llvm::StringRef("memory_order_val")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr hint_valAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr hint_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr memory_order_valAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr memory_order_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.atomic.capture");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::IntegerAttr hint_valAttr();
  uint64_t hint_val();
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> memory_order_val();
  void hint_valAttr(::mlir::IntegerAttr attr);
  void memory_order_valAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr);
  ::mlir::Attribute removeMemory_order_valAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicCaptureOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicReadOp declarations
//===----------------------------------------------------------------------===//

class AtomicReadOpAdaptor {
public:
  AtomicReadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AtomicReadOpAdaptor(AtomicReadOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::Value v();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr hint_valAttr();
  uint64_t hint_val();
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> memory_order_val();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AtomicReadOp : public ::mlir::Op<AtomicReadOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicReadOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hint_val"), ::llvm::StringRef("memory_order_val")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr hint_valAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr hint_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr memory_order_valAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr memory_order_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.atomic.read");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::Value v();
  ::mlir::MutableOperandRange xMutable();
  ::mlir::MutableOperandRange vMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr hint_valAttr();
  uint64_t hint_val();
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> memory_order_val();
  void hint_valAttr(::mlir::IntegerAttr attr);
  void memory_order_valAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr);
  ::mlir::Attribute removeMemory_order_valAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::Value v, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value v, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::Value v, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value v, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicReadOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicUpdateOp declarations
//===----------------------------------------------------------------------===//

class AtomicUpdateOpAdaptor {
public:
  AtomicUpdateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AtomicUpdateOpAdaptor(AtomicUpdateOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr hint_valAttr();
  uint64_t hint_val();
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> memory_order_val();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AtomicUpdateOp : public ::mlir::Op<AtomicUpdateOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::SingleBlockImplicitTerminator<YieldOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicUpdateOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hint_val"), ::llvm::StringRef("memory_order_val")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr hint_valAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr hint_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr memory_order_valAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr memory_order_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.atomic.update");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value x();
  ::mlir::MutableOperandRange xMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::IntegerAttr hint_valAttr();
  uint64_t hint_val();
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> memory_order_val();
  void hint_valAttr(::mlir::IntegerAttr attr);
  void memory_order_valAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr);
  ::mlir::Attribute removeMemory_order_valAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicUpdateOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicWriteOp declarations
//===----------------------------------------------------------------------===//

class AtomicWriteOpAdaptor {
public:
  AtomicWriteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  AtomicWriteOpAdaptor(AtomicWriteOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value address();
  ::mlir::Value value();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr hint_valAttr();
  uint64_t hint_val();
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> memory_order_val();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class AtomicWriteOp : public ::mlir::Op<AtomicWriteOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AtomicWriteOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("hint_val"), ::llvm::StringRef("memory_order_val")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr hint_valAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr hint_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr memory_order_valAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr memory_order_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.atomic.write");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value address();
  ::mlir::Value value();
  ::mlir::MutableOperandRange addressMutable();
  ::mlir::MutableOperandRange valueMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr hint_valAttr();
  uint64_t hint_val();
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> memory_order_val();
  void hint_valAttr(::mlir::IntegerAttr attr);
  void memory_order_valAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr);
  ::mlir::Attribute removeMemory_order_valAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value address, ::mlir::Value value, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value value, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value address, ::mlir::Value value, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value value, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val);
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicWriteOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::BarrierOp declarations
//===----------------------------------------------------------------------===//

class BarrierOpAdaptor {
public:
  BarrierOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  BarrierOpAdaptor(BarrierOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class BarrierOp : public ::mlir::Op<BarrierOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BarrierOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.barrier");
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::BarrierOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalDeclareOp declarations
//===----------------------------------------------------------------------===//

class CriticalDeclareOpAdaptor {
public:
  CriticalDeclareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CriticalDeclareOpAdaptor(CriticalDeclareOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr sym_nameAttr();
  ::llvm::StringRef sym_name();
  ::mlir::IntegerAttr hint_valAttr();
  uint64_t hint_val();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CriticalDeclareOp : public ::mlir::Op<CriticalDeclareOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CriticalDeclareOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("sym_name"), ::llvm::StringRef("hint_val")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr sym_nameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr sym_nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr hint_valAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr hint_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.critical.declare");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::StringAttr sym_nameAttr();
  ::llvm::StringRef sym_name();
  ::mlir::IntegerAttr hint_valAttr();
  uint64_t hint_val();
  void sym_nameAttr(::mlir::StringAttr attr);
  void hint_valAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr hint_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr hint_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, uint64_t hint_val = 0);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, uint64_t hint_val = 0);
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalOp declarations
//===----------------------------------------------------------------------===//

class CriticalOpAdaptor {
public:
  CriticalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CriticalOpAdaptor(CriticalOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FlatSymbolRefAttr nameAttr();
  ::llvm::Optional< ::llvm::StringRef > name();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CriticalOp : public ::mlir::Op<CriticalOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CriticalOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("name")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr nameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.critical");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::FlatSymbolRefAttr nameAttr();
  ::llvm::Optional< ::llvm::StringRef > name();
  void nameAttr(::mlir::FlatSymbolRefAttr attr);
  ::mlir::Attribute removeNameAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::FlatSymbolRefAttr name);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::FlatSymbolRefAttr name);
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::FlushOp declarations
//===----------------------------------------------------------------------===//

class FlushOpAdaptor {
public:
  FlushOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  FlushOpAdaptor(FlushOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange varList();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class FlushOp : public ::mlir::Op<FlushOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FlushOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.flush");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range varList();
  ::mlir::MutableOperandRange varListMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange varList);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::FlushOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::MasterOp declarations
//===----------------------------------------------------------------------===//

class MasterOpAdaptor {
public:
  MasterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  MasterOpAdaptor(MasterOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class MasterOp : public ::mlir::Op<MasterOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MasterOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.master");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::MasterOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::OrderedOp declarations
//===----------------------------------------------------------------------===//

class OrderedOpAdaptor {
public:
  OrderedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  OrderedOpAdaptor(OrderedOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange depend_vec_vars();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::omp::ClauseDependAttr depend_type_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseDepend> depend_type_val();
  ::mlir::IntegerAttr num_loops_valAttr();
  ::llvm::Optional<uint64_t> num_loops_val();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class OrderedOp : public ::mlir::Op<OrderedOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = OrderedOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("depend_type_val"), ::llvm::StringRef("num_loops_val")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr depend_type_valAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr depend_type_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr num_loops_valAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr num_loops_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.ordered");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range depend_vec_vars();
  ::mlir::MutableOperandRange depend_vec_varsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::omp::ClauseDependAttr depend_type_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseDepend> depend_type_val();
  ::mlir::IntegerAttr num_loops_valAttr();
  ::llvm::Optional<uint64_t> num_loops_val();
  void depend_type_valAttr(::mlir::omp::ClauseDependAttr attr);
  void num_loops_valAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeDepend_type_valAttr();
  ::mlir::Attribute removeNum_loops_valAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::omp::ClauseDependAttr depend_type_val, /*optional*/::mlir::IntegerAttr num_loops_val, ::mlir::ValueRange depend_vec_vars);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::omp::ClauseDependAttr depend_type_val, /*optional*/::mlir::IntegerAttr num_loops_val, ::mlir::ValueRange depend_vec_vars);
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::OrderedOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::OrderedRegionOp declarations
//===----------------------------------------------------------------------===//

class OrderedRegionOpAdaptor {
public:
  OrderedRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  OrderedRegionOpAdaptor(OrderedRegionOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr simdAttr();
  bool simd();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class OrderedRegionOp : public ::mlir::Op<OrderedRegionOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = OrderedRegionOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("simd")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr simdAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr simdAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.ordered_region");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::UnitAttr simdAttr();
  bool simd();
  void simdAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeSimdAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::UnitAttr simd);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr simd);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/bool simd);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool simd);
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::OrderedRegionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ParallelOp declarations
//===----------------------------------------------------------------------===//

class ParallelOpAdaptor {
public:
  ParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ParallelOpAdaptor(ParallelOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value if_expr_var();
  ::mlir::Value num_threads_var();
  ::mlir::ValueRange allocate_vars();
  ::mlir::ValueRange allocators_vars();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::omp::ClauseProcBindKindAttr proc_bind_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseProcBindKind> proc_bind_val();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ParallelOp : public ::mlir::Op<ParallelOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::AutomaticAllocationScope, ::mlir::omp::OutlineableOpenMPOpInterface::Trait, ::mlir::OpTrait::HasRecursiveSideEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ParallelOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("proc_bind_val"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr proc_bind_valAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr proc_bind_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.parallel");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value if_expr_var();
  ::mlir::Value num_threads_var();
  ::mlir::Operation::operand_range allocate_vars();
  ::mlir::Operation::operand_range allocators_vars();
  ::mlir::MutableOperandRange if_expr_varMutable();
  ::mlir::MutableOperandRange num_threads_varMutable();
  ::mlir::MutableOperandRange allocate_varsMutable();
  ::mlir::MutableOperandRange allocators_varsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::omp::ClauseProcBindKindAttr proc_bind_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseProcBindKind> proc_bind_val();
  void proc_bind_valAttr(::mlir::omp::ClauseProcBindKindAttr attr);
  ::mlir::Attribute removeProc_bind_valAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::omp::ClauseProcBindKindAttr proc_bind_val);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::omp::ClauseProcBindKindAttr proc_bind_val);
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ParallelOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionDeclareOp declarations
//===----------------------------------------------------------------------===//

class ReductionDeclareOpAdaptor {
public:
  ReductionDeclareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ReductionDeclareOpAdaptor(ReductionDeclareOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr sym_nameAttr();
  ::llvm::StringRef sym_name();
  ::mlir::TypeAttr typeAttr();
  ::mlir::Type type();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &initializerRegion();
  ::mlir::Region &reductionRegion();
  ::mlir::Region &atomicReductionRegion();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ReductionDeclareOp : public ::mlir::Op<ReductionDeclareOp, ::mlir::OpTrait::NRegions<3>::Impl, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::SymbolOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReductionDeclareOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("sym_name"), ::llvm::StringRef("type")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr sym_nameAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr sym_nameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr typeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr typeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.reduction.declare");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &initializerRegion();
  ::mlir::Region &reductionRegion();
  ::mlir::Region &atomicReductionRegion();
  ::mlir::StringAttr sym_nameAttr();
  ::llvm::StringRef sym_name();
  ::mlir::TypeAttr typeAttr();
  ::mlir::Type type();
  void sym_nameAttr(::mlir::StringAttr attr);
  void typeAttr(::mlir::TypeAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::mlir::Type type);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::mlir::Type type);
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
  PointerLikeType getAccumulatorType() {
    if (atomicReductionRegion().empty())
      return {};

    return atomicReductionRegion().front().getArgument(0).getType();
  }
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionOp declarations
//===----------------------------------------------------------------------===//

class ReductionOpAdaptor {
public:
  ReductionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ReductionOpAdaptor(ReductionOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value accumulator();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ReductionOp : public ::mlir::Op<ReductionOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReductionOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.reduction");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value operand();
  ::mlir::Value accumulator();
  ::mlir::MutableOperandRange operandMutable();
  ::mlir::MutableOperandRange accumulatorMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value accumulator);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value accumulator);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SectionOp declarations
//===----------------------------------------------------------------------===//

class SectionOpAdaptor {
public:
  SectionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  SectionOpAdaptor(SectionOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SectionOp : public ::mlir::Op<SectionOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::HasParent<SectionsOp>::Impl, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SectionOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.section");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::SectionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SectionsOp declarations
//===----------------------------------------------------------------------===//

class SectionsOpAdaptor {
public:
  SectionsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  SectionsOpAdaptor(SectionsOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange reduction_vars();
  ::mlir::ValueRange allocate_vars();
  ::mlir::ValueRange allocators_vars();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr reductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > reductions();
  ::mlir::UnitAttr nowaitAttr();
  bool nowait();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class SectionsOp : public ::mlir::Op<SectionsOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = SectionsOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("reductions"), ::llvm::StringRef("nowait"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr reductionsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr reductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr nowaitAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr nowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.sections");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range reduction_vars();
  ::mlir::Operation::operand_range allocate_vars();
  ::mlir::Operation::operand_range allocators_vars();
  ::mlir::MutableOperandRange reduction_varsMutable();
  ::mlir::MutableOperandRange allocate_varsMutable();
  ::mlir::MutableOperandRange allocators_varsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::ArrayAttr reductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > reductions();
  ::mlir::UnitAttr nowaitAttr();
  bool nowait();
  void reductionsAttr(::mlir::ArrayAttr attr);
  void nowaitAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeReductionsAttr();
  ::mlir::Attribute removeNowaitAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait);
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
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::SectionsOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TargetOp declarations
//===----------------------------------------------------------------------===//

class TargetOpAdaptor {
public:
  TargetOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TargetOpAdaptor(TargetOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value if_expr();
  ::mlir::Value device();
  ::mlir::Value thread_limit();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::UnitAttr nowaitAttr();
  bool nowait();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TargetOp : public ::mlir::Op<TargetOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TargetOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("nowait"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr nowaitAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr nowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.target");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value if_expr();
  ::mlir::Value device();
  ::mlir::Value thread_limit();
  ::mlir::MutableOperandRange if_exprMutable();
  ::mlir::MutableOperandRange deviceMutable();
  ::mlir::MutableOperandRange thread_limitMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::UnitAttr nowaitAttr();
  bool nowait();
  void nowaitAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeNowaitAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait);
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
    assert(index < 2 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TargetOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskwaitOp declarations
//===----------------------------------------------------------------------===//

class TaskwaitOpAdaptor {
public:
  TaskwaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TaskwaitOpAdaptor(TaskwaitOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TaskwaitOp : public ::mlir::Op<TaskwaitOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskwaitOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.taskwait");
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskwaitOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskyieldOp declarations
//===----------------------------------------------------------------------===//

class TaskyieldOpAdaptor {
public:
  TaskyieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TaskyieldOpAdaptor(TaskyieldOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TaskyieldOp : public ::mlir::Op<TaskyieldOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TaskyieldOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.taskyield");
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TaskyieldOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TerminatorOp declarations
//===----------------------------------------------------------------------===//

class TerminatorOpAdaptor {
public:
  TerminatorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  TerminatorOpAdaptor(TerminatorOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class TerminatorOp : public ::mlir::Op<TerminatorOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TerminatorOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.terminator");
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
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::TerminatorOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::WsLoopOp declarations
//===----------------------------------------------------------------------===//

class WsLoopOpAdaptor {
public:
  WsLoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  WsLoopOpAdaptor(WsLoopOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange lowerBound();
  ::mlir::ValueRange upperBound();
  ::mlir::ValueRange step();
  ::mlir::ValueRange linear_vars();
  ::mlir::ValueRange linear_step_vars();
  ::mlir::ValueRange reduction_vars();
  ::mlir::Value schedule_chunk_var();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::ArrayAttr reductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > reductions();
  ::mlir::omp::ClauseScheduleKindAttr schedule_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseScheduleKind> schedule_val();
  ::mlir::omp::ScheduleModifierAttr schedule_modifierAttr();
  ::llvm::Optional<::mlir::omp::ScheduleModifier> schedule_modifier();
  ::mlir::UnitAttr simd_modifierAttr();
  bool simd_modifier();
  ::mlir::IntegerAttr collapse_valAttr();
  ::llvm::Optional<uint64_t> collapse_val();
  ::mlir::UnitAttr nowaitAttr();
  bool nowait();
  ::mlir::IntegerAttr ordered_valAttr();
  ::llvm::Optional<uint64_t> ordered_val();
  ::mlir::omp::ClauseOrderKindAttr order_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseOrderKind> order_val();
  ::mlir::UnitAttr inclusiveAttr();
  bool inclusive();
  ::mlir::RegionRange getRegions();
  ::mlir::Region &region();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class WsLoopOp : public ::mlir::Op<WsLoopOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::OpTrait::HasRecursiveSideEffects> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WsLoopOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("reductions"), ::llvm::StringRef("schedule_val"), ::llvm::StringRef("schedule_modifier"), ::llvm::StringRef("simd_modifier"), ::llvm::StringRef("collapse_val"), ::llvm::StringRef("nowait"), ::llvm::StringRef("ordered_val"), ::llvm::StringRef("order_val"), ::llvm::StringRef("inclusive"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr reductionsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr reductionsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr schedule_valAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr schedule_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr schedule_modifierAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr schedule_modifierAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr simd_modifierAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr simd_modifierAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr collapse_valAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr collapse_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr nowaitAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr nowaitAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr ordered_valAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr ordered_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr order_valAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr order_valAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr inclusiveAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr inclusiveAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr operand_segment_sizesAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr operand_segment_sizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.wsloop");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range lowerBound();
  ::mlir::Operation::operand_range upperBound();
  ::mlir::Operation::operand_range step();
  ::mlir::Operation::operand_range linear_vars();
  ::mlir::Operation::operand_range linear_step_vars();
  ::mlir::Operation::operand_range reduction_vars();
  ::mlir::Value schedule_chunk_var();
  ::mlir::MutableOperandRange lowerBoundMutable();
  ::mlir::MutableOperandRange upperBoundMutable();
  ::mlir::MutableOperandRange stepMutable();
  ::mlir::MutableOperandRange linear_varsMutable();
  ::mlir::MutableOperandRange linear_step_varsMutable();
  ::mlir::MutableOperandRange reduction_varsMutable();
  ::mlir::MutableOperandRange schedule_chunk_varMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &region();
  ::mlir::ArrayAttr reductionsAttr();
  ::llvm::Optional< ::mlir::ArrayAttr > reductions();
  ::mlir::omp::ClauseScheduleKindAttr schedule_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseScheduleKind> schedule_val();
  ::mlir::omp::ScheduleModifierAttr schedule_modifierAttr();
  ::llvm::Optional<::mlir::omp::ScheduleModifier> schedule_modifier();
  ::mlir::UnitAttr simd_modifierAttr();
  bool simd_modifier();
  ::mlir::IntegerAttr collapse_valAttr();
  ::llvm::Optional<uint64_t> collapse_val();
  ::mlir::UnitAttr nowaitAttr();
  bool nowait();
  ::mlir::IntegerAttr ordered_valAttr();
  ::llvm::Optional<uint64_t> ordered_val();
  ::mlir::omp::ClauseOrderKindAttr order_valAttr();
  ::llvm::Optional<::mlir::omp::ClauseOrderKind> order_val();
  ::mlir::UnitAttr inclusiveAttr();
  bool inclusive();
  void reductionsAttr(::mlir::ArrayAttr attr);
  void schedule_valAttr(::mlir::omp::ClauseScheduleKindAttr attr);
  void schedule_modifierAttr(::mlir::omp::ScheduleModifierAttr attr);
  void simd_modifierAttr(::mlir::UnitAttr attr);
  void collapse_valAttr(::mlir::IntegerAttr attr);
  void nowaitAttr(::mlir::UnitAttr attr);
  void ordered_valAttr(::mlir::IntegerAttr attr);
  void order_valAttr(::mlir::omp::ClauseOrderKindAttr attr);
  void inclusiveAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeReductionsAttr();
  ::mlir::Attribute removeSchedule_valAttr();
  ::mlir::Attribute removeSchedule_modifierAttr();
  ::mlir::Attribute removeSimd_modifierAttr();
  ::mlir::Attribute removeCollapse_valAttr();
  ::mlir::Attribute removeNowaitAttr();
  ::mlir::Attribute removeOrdered_valAttr();
  ::mlir::Attribute removeOrder_valAttr();
  ::mlir::Attribute removeInclusiveAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange lowerBound, ValueRange upperBound, ValueRange step, ArrayRef<NamedAttribute> attributes = {});
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/::mlir::UnitAttr simd_modifier, /*optional*/::mlir::IntegerAttr collapse_val, /*optional*/::mlir::UnitAttr nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::UnitAttr inclusive);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/::mlir::UnitAttr simd_modifier, /*optional*/::mlir::IntegerAttr collapse_val, /*optional*/::mlir::UnitAttr nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::UnitAttr inclusive);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/bool simd_modifier, /*optional*/::mlir::IntegerAttr collapse_val, /*optional*/bool nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/bool inclusive);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/bool simd_modifier, /*optional*/::mlir::IntegerAttr collapse_val, /*optional*/bool nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/bool inclusive);
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
    assert(index < 10 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
  /// Returns the number of loops in the workshape loop nest.
  unsigned getNumLoops() { return lowerBound().size(); }

  /// Returns the number of reduction variables.
  unsigned getNumReductionVars() { return reduction_vars().size(); }
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::WsLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::YieldOp declarations
//===----------------------------------------------------------------------===//

class YieldOpAdaptor {
public:
  YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  YieldOpAdaptor(YieldOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange results();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class YieldOp : public ::mlir::Op<YieldOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<WsLoopOp, ReductionDeclareOp, AtomicUpdateOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::ReturnLike, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = YieldOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("omp.yield");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range results();
  ::mlir::MutableOperandRange resultsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace omp
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::omp::YieldOp)


#endif  // GET_OP_CLASSES

