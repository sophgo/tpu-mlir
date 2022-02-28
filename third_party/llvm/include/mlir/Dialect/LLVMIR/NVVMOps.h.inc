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
namespace NVVM {
class Barrier0Op;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class BlockDimXOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class BlockDimYOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class BlockDimZOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class BlockIdXOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class BlockIdYOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class BlockIdZOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class CpAsyncCommitGroupOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class CpAsyncOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class CpAsyncWaitGroupOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class GridDimXOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class GridDimYOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class GridDimZOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class LaneIdOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class MmaOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class ShflOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class ThreadIdXOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class ThreadIdYOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class ThreadIdZOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class VoteBallotOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class WMMALoadOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class WMMAMmaOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class WMMAStoreOp;
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {
class WarpSizeOp;
} // namespace NVVM
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::Barrier0Op declarations
//===----------------------------------------------------------------------===//

class Barrier0OpAdaptor {
public:
  Barrier0OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  Barrier0OpAdaptor(Barrier0Op &op);

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
class Barrier0Op : public ::mlir::Op<Barrier0Op, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = Barrier0OpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.barrier0");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::Barrier0Op)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockDimXOp declarations
//===----------------------------------------------------------------------===//

class BlockDimXOpAdaptor {
public:
  BlockDimXOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  BlockDimXOpAdaptor(BlockDimXOp &op);

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
class BlockDimXOp : public ::mlir::Op<BlockDimXOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockDimXOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.ntid.x");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockDimXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockDimYOp declarations
//===----------------------------------------------------------------------===//

class BlockDimYOpAdaptor {
public:
  BlockDimYOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  BlockDimYOpAdaptor(BlockDimYOp &op);

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
class BlockDimYOp : public ::mlir::Op<BlockDimYOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockDimYOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.ntid.y");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockDimYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockDimZOp declarations
//===----------------------------------------------------------------------===//

class BlockDimZOpAdaptor {
public:
  BlockDimZOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  BlockDimZOpAdaptor(BlockDimZOp &op);

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
class BlockDimZOp : public ::mlir::Op<BlockDimZOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockDimZOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.ntid.z");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockDimZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockIdXOp declarations
//===----------------------------------------------------------------------===//

class BlockIdXOpAdaptor {
public:
  BlockIdXOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  BlockIdXOpAdaptor(BlockIdXOp &op);

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
class BlockIdXOp : public ::mlir::Op<BlockIdXOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockIdXOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.ctaid.x");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockIdXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockIdYOp declarations
//===----------------------------------------------------------------------===//

class BlockIdYOpAdaptor {
public:
  BlockIdYOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  BlockIdYOpAdaptor(BlockIdYOp &op);

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
class BlockIdYOp : public ::mlir::Op<BlockIdYOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockIdYOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.ctaid.y");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockIdYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockIdZOp declarations
//===----------------------------------------------------------------------===//

class BlockIdZOpAdaptor {
public:
  BlockIdZOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  BlockIdZOpAdaptor(BlockIdZOp &op);

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
class BlockIdZOp : public ::mlir::Op<BlockIdZOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = BlockIdZOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.ctaid.z");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockIdZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::CpAsyncCommitGroupOp declarations
//===----------------------------------------------------------------------===//

class CpAsyncCommitGroupOpAdaptor {
public:
  CpAsyncCommitGroupOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CpAsyncCommitGroupOpAdaptor(CpAsyncCommitGroupOp &op);

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
class CpAsyncCommitGroupOp : public ::mlir::Op<CpAsyncCommitGroupOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CpAsyncCommitGroupOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.cp.async.commit.group");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::CpAsyncCommitGroupOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::CpAsyncOp declarations
//===----------------------------------------------------------------------===//

class CpAsyncOpAdaptor {
public:
  CpAsyncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CpAsyncOpAdaptor(CpAsyncOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value dst();
  ::mlir::Value src();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr sizeAttr();
  uint32_t size();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CpAsyncOp : public ::mlir::Op<CpAsyncOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CpAsyncOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("size")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr sizeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr sizeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.cp.async.shared.global");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value dst();
  ::mlir::Value src();
  ::mlir::MutableOperandRange dstMutable();
  ::mlir::MutableOperandRange srcMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr sizeAttr();
  uint32_t size();
  void sizeAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value dst, ::mlir::Value src, ::mlir::IntegerAttr size);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value src, ::mlir::IntegerAttr size);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value dst, ::mlir::Value src, uint32_t size);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value src, uint32_t size);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
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

  ::mlir::LogicalResult verify();
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::CpAsyncOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::CpAsyncWaitGroupOp declarations
//===----------------------------------------------------------------------===//

class CpAsyncWaitGroupOpAdaptor {
public:
  CpAsyncWaitGroupOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  CpAsyncWaitGroupOpAdaptor(CpAsyncWaitGroupOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr nAttr();
  uint32_t n();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class CpAsyncWaitGroupOp : public ::mlir::Op<CpAsyncWaitGroupOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = CpAsyncWaitGroupOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("n")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr nAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr nAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.cp.async.wait.group");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr nAttr();
  uint32_t n();
  void nAttr(::mlir::IntegerAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr n);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr n);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint32_t n);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t n);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
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
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::CpAsyncWaitGroupOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::GridDimXOp declarations
//===----------------------------------------------------------------------===//

class GridDimXOpAdaptor {
public:
  GridDimXOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  GridDimXOpAdaptor(GridDimXOp &op);

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
class GridDimXOp : public ::mlir::Op<GridDimXOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GridDimXOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.nctaid.x");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::GridDimXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::GridDimYOp declarations
//===----------------------------------------------------------------------===//

class GridDimYOpAdaptor {
public:
  GridDimYOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  GridDimYOpAdaptor(GridDimYOp &op);

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
class GridDimYOp : public ::mlir::Op<GridDimYOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GridDimYOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.nctaid.y");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::GridDimYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::GridDimZOp declarations
//===----------------------------------------------------------------------===//

class GridDimZOpAdaptor {
public:
  GridDimZOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  GridDimZOpAdaptor(GridDimZOp &op);

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
class GridDimZOp : public ::mlir::Op<GridDimZOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GridDimZOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.nctaid.z");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::GridDimZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::LaneIdOp declarations
//===----------------------------------------------------------------------===//

class LaneIdOpAdaptor {
public:
  LaneIdOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  LaneIdOpAdaptor(LaneIdOp &op);

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
class LaneIdOp : public ::mlir::Op<LaneIdOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LaneIdOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.laneid");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::LaneIdOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::MmaOp declarations
//===----------------------------------------------------------------------===//

class MmaOpAdaptor {
public:
  MmaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  MmaOpAdaptor(MmaOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange args();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class MmaOp : public ::mlir::Op<MmaOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MmaOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.mma.sync");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range args();
  ::mlir::MutableOperandRange argsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::ValueRange args);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::LogicalResult verify();
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::MmaOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ShflOp declarations
//===----------------------------------------------------------------------===//

class ShflOpAdaptor {
public:
  ShflOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ShflOpAdaptor(ShflOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value dst();
  ::mlir::Value val();
  ::mlir::Value offset();
  ::mlir::Value mask_and_clamp();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::NVVM::ShflKindAttr kindAttr();
  ::mlir::NVVM::ShflKind kind();
  ::mlir::UnitAttr return_value_and_is_validAttr();
  ::llvm::Optional<bool> return_value_and_is_valid();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class ShflOp : public ::mlir::Op<ShflOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<4>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ShflOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("kind"), ::llvm::StringRef("return_value_and_is_valid")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr kindAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr kindAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr return_value_and_is_validAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr return_value_and_is_validAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.shfl.sync");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value dst();
  ::mlir::Value val();
  ::mlir::Value offset();
  ::mlir::Value mask_and_clamp();
  ::mlir::MutableOperandRange dstMutable();
  ::mlir::MutableOperandRange valMutable();
  ::mlir::MutableOperandRange offsetMutable();
  ::mlir::MutableOperandRange mask_and_clampMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  ::mlir::NVVM::ShflKindAttr kindAttr();
  ::mlir::NVVM::ShflKind kind();
  ::mlir::UnitAttr return_value_and_is_validAttr();
  ::llvm::Optional<bool> return_value_and_is_valid();
  void kindAttr(::mlir::NVVM::ShflKindAttr attr);
  void return_value_and_is_validAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeReturn_value_and_is_validAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, ::mlir::NVVM::ShflKindAttr kind, /*optional*/::mlir::UnitAttr return_value_and_is_valid);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, ::mlir::NVVM::ShflKindAttr kind, /*optional*/::mlir::UnitAttr return_value_and_is_valid);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, ::mlir::NVVM::ShflKind kind, /*optional*/::mlir::UnitAttr return_value_and_is_valid);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, ::mlir::NVVM::ShflKind kind, /*optional*/::mlir::UnitAttr return_value_and_is_valid);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
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

  ::mlir::LogicalResult verify();
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::ShflOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ThreadIdXOp declarations
//===----------------------------------------------------------------------===//

class ThreadIdXOpAdaptor {
public:
  ThreadIdXOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ThreadIdXOpAdaptor(ThreadIdXOp &op);

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
class ThreadIdXOp : public ::mlir::Op<ThreadIdXOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ThreadIdXOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.tid.x");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::ThreadIdXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ThreadIdYOp declarations
//===----------------------------------------------------------------------===//

class ThreadIdYOpAdaptor {
public:
  ThreadIdYOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ThreadIdYOpAdaptor(ThreadIdYOp &op);

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
class ThreadIdYOp : public ::mlir::Op<ThreadIdYOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ThreadIdYOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.tid.y");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::ThreadIdYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ThreadIdZOp declarations
//===----------------------------------------------------------------------===//

class ThreadIdZOpAdaptor {
public:
  ThreadIdZOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  ThreadIdZOpAdaptor(ThreadIdZOp &op);

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
class ThreadIdZOp : public ::mlir::Op<ThreadIdZOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ThreadIdZOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.tid.z");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::ThreadIdZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::VoteBallotOp declarations
//===----------------------------------------------------------------------===//

class VoteBallotOpAdaptor {
public:
  VoteBallotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  VoteBallotOpAdaptor(VoteBallotOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value pred();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class VoteBallotOp : public ::mlir::Op<VoteBallotOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = VoteBallotOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.vote.ballot.sync");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value mask();
  ::mlir::Value pred();
  ::mlir::MutableOperandRange maskMutable();
  ::mlir::MutableOperandRange predMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value pred);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value pred);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariants();
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::VoteBallotOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMALoadOp declarations
//===----------------------------------------------------------------------===//

class WMMALoadOpAdaptor {
public:
  WMMALoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  WMMALoadOpAdaptor(WMMALoadOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value ptr();
  ::mlir::Value stride();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr mAttr();
  uint32_t m();
  ::mlir::IntegerAttr nAttr();
  uint32_t n();
  ::mlir::IntegerAttr kAttr();
  uint32_t k();
  ::mlir::NVVM::MMALayoutAttr layoutAttr();
  ::mlir::NVVM::MMALayout layout();
  ::mlir::NVVM::MMATypesAttr eltypeAttr();
  ::mlir::NVVM::MMATypes eltype();
  ::mlir::NVVM::MMAFragAttr fragAttr();
  ::mlir::NVVM::MMAFrag frag();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class WMMALoadOp : public ::mlir::Op<WMMALoadOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::NOperands<2>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WMMALoadOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("m"), ::llvm::StringRef("n"), ::llvm::StringRef("k"), ::llvm::StringRef("layout"), ::llvm::StringRef("eltype"), ::llvm::StringRef("frag")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr mAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr mAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr nAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr nAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr kAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr kAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr layoutAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr layoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr eltypeAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr eltypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr fragAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr fragAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.wmma.load");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value ptr();
  ::mlir::Value stride();
  ::mlir::MutableOperandRange ptrMutable();
  ::mlir::MutableOperandRange strideMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  ::mlir::IntegerAttr mAttr();
  uint32_t m();
  ::mlir::IntegerAttr nAttr();
  uint32_t n();
  ::mlir::IntegerAttr kAttr();
  uint32_t k();
  ::mlir::NVVM::MMALayoutAttr layoutAttr();
  ::mlir::NVVM::MMALayout layout();
  ::mlir::NVVM::MMATypesAttr eltypeAttr();
  ::mlir::NVVM::MMATypes eltype();
  ::mlir::NVVM::MMAFragAttr fragAttr();
  ::mlir::NVVM::MMAFrag frag();
  void mAttr(::mlir::IntegerAttr attr);
  void nAttr(::mlir::IntegerAttr attr);
  void kAttr(::mlir::IntegerAttr attr);
  void layoutAttr(::mlir::NVVM::MMALayoutAttr attr);
  void eltypeAttr(::mlir::NVVM::MMATypesAttr attr);
  void fragAttr(::mlir::NVVM::MMAFragAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value stride, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layout, ::mlir::NVVM::MMATypesAttr eltype, ::mlir::NVVM::MMAFragAttr frag);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value stride, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layout, ::mlir::NVVM::MMATypesAttr eltype, ::mlir::NVVM::MMAFragAttr frag);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value stride, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layout, ::mlir::NVVM::MMATypes eltype, ::mlir::NVVM::MMAFrag frag);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value stride, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layout, ::mlir::NVVM::MMATypes eltype, ::mlir::NVVM::MMAFrag frag);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 6 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

  ::mlir::LogicalResult verify();
public:
  static llvm::Intrinsic::ID getIntrinsicID(int m, int n, int k, mlir::NVVM::MMALayout layoutEnum,mlir::NVVM::MMATypes eltypeEnum,mlir::NVVM::MMAFrag fragEnum) {llvm::StringRef layout = stringifyEnum(layoutEnum);llvm::StringRef eltype = stringifyEnum(eltypeEnum);llvm::StringRef frag = stringifyEnum(fragEnum);

  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_a_f16_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_b_f16_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_a_f16_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_a_f16_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_b_f16_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_b_f16_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_a_f16_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_a_f16_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_b_f16_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_b_f16_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_f16_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k16_load_c_f32_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_c_f16_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_c_f16_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_c_f32_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m32n8k16_load_c_f32_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_c_f16_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_c_f16_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_c_f32_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m8n32k16_load_c_f32_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 8 && "tf32" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m16n16k8_load_a_tf32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 8 && "tf32" == eltype && frag == "a")  return llvm::Intrinsic::nvvm_wmma_m16n16k8_load_a_tf32_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 8 && "tf32" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m16n16k8_load_b_tf32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 8 && "tf32" == eltype && frag == "b")  return llvm::Intrinsic::nvvm_wmma_m16n16k8_load_b_tf32_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 8 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k8_load_c_f32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 8 && "f32" == eltype && frag == "c")  return llvm::Intrinsic::nvvm_wmma_m16n16k8_load_c_f32_col_stride;
  return 0;}
  /// Helpers to find valid n dimension based on mxk load shape.
  static int inferNDimension(int m, int k, mlir::NVVM::MMATypes eltypeEnum) {  llvm::StringRef eltype = stringifyEnum(eltypeEnum);
  if (m == 16 && k == 16 && "f16" == eltype)  return 16;
  if (m == 32 && k == 16 && "f16" == eltype)  return 8;
  if (m == 8 && k == 16 && "f16" == eltype)  return 32;
  if (m == 16 && k == 8 && "tf32" == eltype)  return 16;
  return 0;}
  /// Helpers to find valid m dimension based on kxn load shape.
  static int inferMDimension(int k, int n, mlir::NVVM::MMATypes eltypeEnum) {  llvm::StringRef eltype = stringifyEnum(eltypeEnum);
  if (n == 16 && k == 16 && "f16" == eltype)  return 16;
  if (n == 8 && k == 16 && "f16" == eltype)  return 32;
  if (n == 32 && k == 16 && "f16" == eltype)  return 8;
  if (n == 16 && k == 8 && "tf32" == eltype)  return 16;
  return 0;}
  /// Helpers to find valid k dimension based on mxn load shape.
  static int inferKDimension(int m, int n, mlir::NVVM::MMATypes eltypeEnum) {  llvm::StringRef eltype = stringifyEnum(eltypeEnum);
  if (m == 16 && n == 16 && "f16" == eltype)  return 16;
  if (m == 16 && n == 16 && "f32" == eltype)  return 16;
  if (m == 32 && n == 8 && "f16" == eltype)  return 16;
  if (m == 32 && n == 8 && "f32" == eltype)  return 16;
  if (m == 8 && n == 32 && "f16" == eltype)  return 16;
  if (m == 8 && n == 32 && "f32" == eltype)  return 16;
  if (m == 16 && n == 16 && "f32" == eltype)  return 8;
  return 0;}
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMALoadOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMAMmaOp declarations
//===----------------------------------------------------------------------===//

class WMMAMmaOpAdaptor {
public:
  WMMAMmaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  WMMAMmaOpAdaptor(WMMAMmaOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::ValueRange args();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr mAttr();
  uint32_t m();
  ::mlir::IntegerAttr nAttr();
  uint32_t n();
  ::mlir::IntegerAttr kAttr();
  uint32_t k();
  ::mlir::NVVM::MMALayoutAttr layoutAAttr();
  ::mlir::NVVM::MMALayout layoutA();
  ::mlir::NVVM::MMALayoutAttr layoutBAttr();
  ::mlir::NVVM::MMALayout layoutB();
  ::mlir::NVVM::MMATypesAttr eltypeAAttr();
  ::mlir::NVVM::MMATypes eltypeA();
  ::mlir::NVVM::MMATypesAttr eltypeBAttr();
  ::mlir::NVVM::MMATypes eltypeB();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class WMMAMmaOp : public ::mlir::Op<WMMAMmaOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::VariadicOperands> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WMMAMmaOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("m"), ::llvm::StringRef("n"), ::llvm::StringRef("k"), ::llvm::StringRef("layoutA"), ::llvm::StringRef("layoutB"), ::llvm::StringRef("eltypeA"), ::llvm::StringRef("eltypeB")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr mAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr mAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr nAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr nAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr kAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr kAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr layoutAAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr layoutAAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr layoutBAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr layoutBAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr eltypeAAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr eltypeAAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr eltypeBAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr eltypeBAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.wmma.mma");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range args();
  ::mlir::MutableOperandRange argsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  ::mlir::IntegerAttr mAttr();
  uint32_t m();
  ::mlir::IntegerAttr nAttr();
  uint32_t n();
  ::mlir::IntegerAttr kAttr();
  uint32_t k();
  ::mlir::NVVM::MMALayoutAttr layoutAAttr();
  ::mlir::NVVM::MMALayout layoutA();
  ::mlir::NVVM::MMALayoutAttr layoutBAttr();
  ::mlir::NVVM::MMALayout layoutB();
  ::mlir::NVVM::MMATypesAttr eltypeAAttr();
  ::mlir::NVVM::MMATypes eltypeA();
  ::mlir::NVVM::MMATypesAttr eltypeBAttr();
  ::mlir::NVVM::MMATypes eltypeB();
  void mAttr(::mlir::IntegerAttr attr);
  void nAttr(::mlir::IntegerAttr attr);
  void kAttr(::mlir::IntegerAttr attr);
  void layoutAAttr(::mlir::NVVM::MMALayoutAttr attr);
  void layoutBAttr(::mlir::NVVM::MMALayoutAttr attr);
  void eltypeAAttr(::mlir::NVVM::MMATypesAttr attr);
  void eltypeBAttr(::mlir::NVVM::MMATypesAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layoutA, ::mlir::NVVM::MMALayoutAttr layoutB, ::mlir::NVVM::MMATypesAttr eltypeA, ::mlir::NVVM::MMATypesAttr eltypeB, ::mlir::ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layoutA, ::mlir::NVVM::MMALayoutAttr layoutB, ::mlir::NVVM::MMATypesAttr eltypeA, ::mlir::NVVM::MMATypesAttr eltypeB, ::mlir::ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layoutA, ::mlir::NVVM::MMALayout layoutB, ::mlir::NVVM::MMATypes eltypeA, ::mlir::NVVM::MMATypes eltypeB, ::mlir::ValueRange args);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layoutA, ::mlir::NVVM::MMALayout layoutB, ::mlir::NVVM::MMATypes eltypeA, ::mlir::NVVM::MMATypes eltypeB, ::mlir::ValueRange args);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
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

  ::mlir::LogicalResult verify();
public:
  static llvm::Intrinsic::ID getIntrinsicID(int m, int n, int k, mlir::NVVM::MMALayout layoutAEnum,mlir::NVVM::MMALayout layoutBEnum, mlir::NVVM::MMATypes eltypeAEnum,mlir::NVVM::MMATypes eltypeBEnum) {llvm::StringRef layoutA = stringifyEnum(layoutAEnum);llvm::StringRef layoutB = stringifyEnum(layoutBEnum);llvm::StringRef eltypeA = stringifyEnum(eltypeAEnum);llvm::StringRef eltypeB = stringifyEnum(eltypeBEnum);


  if (layoutA == "row" && layoutB == "row" &&     m == 16 && n == 16 && k == 8    && "tf32" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k8_mma_row_row_tf32;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 16 && k == 8    && "tf32" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k8_mma_row_col_tf32;
  if (layoutA == "col" && layoutB == "row" &&     m == 16 && n == 16 && k == 8    && "tf32" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k8_mma_col_row_tf32;
  if (layoutA == "col" && layoutB == "col" &&     m == 16 && n == 16 && k == 8    && "tf32" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k8_mma_col_col_tf32;
  if (layoutA == "row" && layoutB == "row" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_f16_f16;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_col_f16_f16;
  if (layoutA == "col" && layoutB == "row" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_col_row_f16_f16;
  if (layoutA == "col" && layoutB == "col" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_col_col_f16_f16;
  if (layoutA == "row" && layoutB == "row" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_row_f32_f32;
  if (layoutA == "row" && layoutB == "col" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_row_col_f32_f32;
  if (layoutA == "col" && layoutB == "row" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_col_row_f32_f32;
  if (layoutA == "col" && layoutB == "col" &&     m == 16 && n == 16 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_mma_col_col_f32_f32;
  if (layoutA == "row" && layoutB == "row" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_row_row_f16_f16;
  if (layoutA == "row" && layoutB == "col" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_row_col_f16_f16;
  if (layoutA == "col" && layoutB == "row" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_col_row_f16_f16;
  if (layoutA == "col" && layoutB == "col" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_col_col_f16_f16;
  if (layoutA == "row" && layoutB == "row" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_row_row_f32_f32;
  if (layoutA == "row" && layoutB == "col" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_row_col_f32_f32;
  if (layoutA == "col" && layoutB == "row" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_col_row_f32_f32;
  if (layoutA == "col" && layoutB == "col" &&     m == 32 && n == 8 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_mma_col_col_f32_f32;
  if (layoutA == "row" && layoutB == "row" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_row_row_f16_f16;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_row_col_f16_f16;
  if (layoutA == "col" && layoutB == "row" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_col_row_f16_f16;
  if (layoutA == "col" && layoutB == "col" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f16" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_col_col_f16_f16;
  if (layoutA == "row" && layoutB == "row" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_row_row_f32_f32;
  if (layoutA == "row" && layoutB == "col" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_row_col_f32_f32;
  if (layoutA == "col" && layoutB == "row" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_col_row_f32_f32;
  if (layoutA == "col" && layoutB == "col" &&     m == 8 && n == 32 && k == 16    && "f16" == eltypeA && "f32" == eltypeB)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_mma_col_col_f32_f32;
  return 0;}};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMAMmaOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMAStoreOp declarations
//===----------------------------------------------------------------------===//

class WMMAStoreOpAdaptor {
public:
  WMMAStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  WMMAStoreOpAdaptor(WMMAStoreOp &op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value ptr();
  ::mlir::ValueRange args();
  ::mlir::Value stride();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr mAttr();
  uint32_t m();
  ::mlir::IntegerAttr nAttr();
  uint32_t n();
  ::mlir::IntegerAttr kAttr();
  uint32_t k();
  ::mlir::NVVM::MMALayoutAttr layoutAttr();
  ::mlir::NVVM::MMALayout layout();
  ::mlir::NVVM::MMATypesAttr eltypeAttr();
  ::mlir::NVVM::MMATypes eltype();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
};
class WMMAStoreOp : public ::mlir::Op<WMMAStoreOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::ZeroResult, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::AtLeastNOperands<2>::Impl> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WMMAStoreOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("m"), ::llvm::StringRef("n"), ::llvm::StringRef("k"), ::llvm::StringRef("layout"), ::llvm::StringRef("eltype")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr mAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr mAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr nAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr nAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr kAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr kAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr layoutAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr layoutAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr eltypeAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr eltypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.wmma.store");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value ptr();
  ::mlir::Operation::operand_range args();
  ::mlir::Value stride();
  ::mlir::MutableOperandRange ptrMutable();
  ::mlir::MutableOperandRange argsMutable();
  ::mlir::MutableOperandRange strideMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::IntegerAttr mAttr();
  uint32_t m();
  ::mlir::IntegerAttr nAttr();
  uint32_t n();
  ::mlir::IntegerAttr kAttr();
  uint32_t k();
  ::mlir::NVVM::MMALayoutAttr layoutAttr();
  ::mlir::NVVM::MMALayout layout();
  ::mlir::NVVM::MMATypesAttr eltypeAttr();
  ::mlir::NVVM::MMATypes eltype();
  void mAttr(::mlir::IntegerAttr attr);
  void nAttr(::mlir::IntegerAttr attr);
  void kAttr(::mlir::IntegerAttr attr);
  void layoutAttr(::mlir::NVVM::MMALayoutAttr attr);
  void eltypeAttr(::mlir::NVVM::MMATypesAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layout, ::mlir::NVVM::MMATypesAttr eltype, ::mlir::ValueRange args, ::mlir::Value stride);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layout, ::mlir::NVVM::MMATypesAttr eltype, ::mlir::ValueRange args, ::mlir::Value stride);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layout, ::mlir::NVVM::MMATypes eltype, ::mlir::ValueRange args, ::mlir::Value stride);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layout, ::mlir::NVVM::MMATypes eltype, ::mlir::ValueRange args, ::mlir::Value stride);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
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

  ::mlir::LogicalResult verify();
public:
  static llvm::Intrinsic::ID getIntrinsicID(int m, int n, int k, mlir::NVVM::MMALayout layoutEnum,mlir::NVVM::MMATypes eltypeEnum) {  llvm::StringRef layout = stringifyEnum(layoutEnum);  llvm::StringRef eltype = stringifyEnum(eltypeEnum);

  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "f16" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_f16_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 16 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 16 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k16_store_d_f32_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_store_d_f16_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "f16" == eltype)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_store_d_f16_col_stride;
  if (layout == "row" && m == 32 &&    n == 8 && k == 16 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_store_d_f32_row_stride;
  if (layout == "col" && m == 32 &&    n == 8 && k == 16 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m32n8k16_store_d_f32_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_store_d_f16_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "f16" == eltype)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_store_d_f16_col_stride;
  if (layout == "row" && m == 8 &&    n == 32 && k == 16 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_store_d_f32_row_stride;
  if (layout == "col" && m == 8 &&    n == 32 && k == 16 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m8n32k16_store_d_f32_col_stride;
  if (layout == "row" && m == 16 &&    n == 16 && k == 8 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k8_store_d_f32_row_stride;
  if (layout == "col" && m == 16 &&    n == 16 && k == 8 && "f32" == eltype)  return llvm::Intrinsic::nvvm_wmma_m16n16k8_store_d_f32_col_stride;
  return 0;}
  /// Helpers to find valid k dimension based on mxn store shape.
  static int inferKDimension(int m, int n, mlir::NVVM::MMATypes eltypeEnum) {  llvm::StringRef eltype = stringifyEnum(eltypeEnum);
  if (m == 16 && n == 16 && "f16" == eltype)  return 16;
  if (m == 16 && n == 16 && "f32" == eltype)  return 16;
  if (m == 32 && n == 8 && "f16" == eltype)  return 16;
  if (m == 32 && n == 8 && "f32" == eltype)  return 16;
  if (m == 8 && n == 32 && "f16" == eltype)  return 16;
  if (m == 8 && n == 32 && "f32" == eltype)  return 16;
  if (m == 16 && n == 16 && "f32" == eltype)  return 8;
  return 0;}};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMAStoreOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WarpSizeOp declarations
//===----------------------------------------------------------------------===//

class WarpSizeOpAdaptor {
public:
  WarpSizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  WarpSizeOpAdaptor(WarpSizeOp &op);

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
class WarpSizeOp : public ::mlir::Op<WarpSizeOp, ::mlir::OpTrait::ZeroRegion, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessor, ::mlir::OpTrait::ZeroOperands, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = WarpSizeOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("nvvm.read.ptx.sreg.warpsize");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value res();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace NVVM
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::NVVM::WarpSizeOp)


#endif  // GET_OP_CLASSES

