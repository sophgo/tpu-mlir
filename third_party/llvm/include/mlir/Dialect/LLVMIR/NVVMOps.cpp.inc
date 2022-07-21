/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::NVVM::Barrier0Op,
::mlir::NVVM::BlockDimXOp,
::mlir::NVVM::BlockDimYOp,
::mlir::NVVM::BlockDimZOp,
::mlir::NVVM::BlockIdXOp,
::mlir::NVVM::BlockIdYOp,
::mlir::NVVM::BlockIdZOp,
::mlir::NVVM::CpAsyncCommitGroupOp,
::mlir::NVVM::CpAsyncOp,
::mlir::NVVM::CpAsyncWaitGroupOp,
::mlir::NVVM::GridDimXOp,
::mlir::NVVM::GridDimYOp,
::mlir::NVVM::GridDimZOp,
::mlir::NVVM::LaneIdOp,
::mlir::NVVM::LdMatrixOp,
::mlir::NVVM::MmaOp,
::mlir::NVVM::ShflOp,
::mlir::NVVM::ThreadIdXOp,
::mlir::NVVM::ThreadIdYOp,
::mlir::NVVM::ThreadIdZOp,
::mlir::NVVM::VoteBallotOp,
::mlir::NVVM::WMMALoadOp,
::mlir::NVVM::WMMAMmaOp,
::mlir::NVVM::WMMAStoreOp,
::mlir::NVVM::WarpSizeOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace NVVM {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_NVVMOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::mlir::LLVM::isCompatibleOuterType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM dialect-compatible type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_NVVMOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) && (((type.cast<::mlir::LLVM::LLVMPointerType>().isOpaque())) || ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isSignlessInteger(8)))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer to 8-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_NVVMOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMPointerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_NVVMOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_NVVMOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMStructType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM structure type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_NVVMOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVVMOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVVMOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVVMOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::NVVM::MMALayoutAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: NVVM MMA layout";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVVMOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::NVVM::MMAShapeAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Attribute for MMA operation shape.";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVVMOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::NVVM::MMAB1OpAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: MMA binary operations";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVVMOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::NVVM::MMAIntOverflowAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: MMA overflow options";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVVMOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::NVVM::MMATypesAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: NVVM MMA types";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVVMOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::NVVM::ShflKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: NVVM shuffle kind";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_NVVMOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::NVVM::MMAFragAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: NVVM MMA frag type";
  }
  return ::mlir::success();
}
} // namespace NVVM
} // namespace mlir
namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::Barrier0Op definitions
//===----------------------------------------------------------------------===//

Barrier0OpAdaptor::Barrier0OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.barrier0", odsAttrs.getContext());
}

Barrier0OpAdaptor::Barrier0OpAdaptor(Barrier0Op op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Barrier0OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Barrier0OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Barrier0OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr Barrier0OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult Barrier0OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Barrier0Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Barrier0Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> Barrier0Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Barrier0Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void Barrier0Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void Barrier0Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Barrier0Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Barrier0Op::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult Barrier0Op::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult Barrier0Op::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void Barrier0Op::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::Barrier0Op)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockDimXOp definitions
//===----------------------------------------------------------------------===//

BlockDimXOpAdaptor::BlockDimXOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.ntid.x", odsAttrs.getContext());
}

BlockDimXOpAdaptor::BlockDimXOpAdaptor(BlockDimXOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BlockDimXOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockDimXOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockDimXOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockDimXOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BlockDimXOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BlockDimXOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockDimXOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockDimXOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockDimXOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BlockDimXOp::res() {
  return *getODSResults(0).begin();
}

void BlockDimXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void BlockDimXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockDimXOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockDimXOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BlockDimXOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BlockDimXOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void BlockDimXOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void BlockDimXOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockDimXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockDimYOp definitions
//===----------------------------------------------------------------------===//

BlockDimYOpAdaptor::BlockDimYOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.ntid.y", odsAttrs.getContext());
}

BlockDimYOpAdaptor::BlockDimYOpAdaptor(BlockDimYOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BlockDimYOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockDimYOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockDimYOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockDimYOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BlockDimYOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BlockDimYOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockDimYOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockDimYOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockDimYOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BlockDimYOp::res() {
  return *getODSResults(0).begin();
}

void BlockDimYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void BlockDimYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockDimYOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockDimYOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BlockDimYOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BlockDimYOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void BlockDimYOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void BlockDimYOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockDimYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockDimZOp definitions
//===----------------------------------------------------------------------===//

BlockDimZOpAdaptor::BlockDimZOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.ntid.z", odsAttrs.getContext());
}

BlockDimZOpAdaptor::BlockDimZOpAdaptor(BlockDimZOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BlockDimZOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockDimZOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockDimZOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockDimZOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BlockDimZOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BlockDimZOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockDimZOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockDimZOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockDimZOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BlockDimZOp::res() {
  return *getODSResults(0).begin();
}

void BlockDimZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void BlockDimZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockDimZOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockDimZOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BlockDimZOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BlockDimZOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void BlockDimZOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void BlockDimZOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockDimZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockIdXOp definitions
//===----------------------------------------------------------------------===//

BlockIdXOpAdaptor::BlockIdXOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.ctaid.x", odsAttrs.getContext());
}

BlockIdXOpAdaptor::BlockIdXOpAdaptor(BlockIdXOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BlockIdXOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockIdXOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockIdXOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockIdXOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BlockIdXOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BlockIdXOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockIdXOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockIdXOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockIdXOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BlockIdXOp::res() {
  return *getODSResults(0).begin();
}

void BlockIdXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void BlockIdXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockIdXOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockIdXOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BlockIdXOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BlockIdXOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void BlockIdXOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void BlockIdXOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockIdXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockIdYOp definitions
//===----------------------------------------------------------------------===//

BlockIdYOpAdaptor::BlockIdYOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.ctaid.y", odsAttrs.getContext());
}

BlockIdYOpAdaptor::BlockIdYOpAdaptor(BlockIdYOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BlockIdYOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockIdYOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockIdYOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockIdYOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BlockIdYOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BlockIdYOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockIdYOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockIdYOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockIdYOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BlockIdYOp::res() {
  return *getODSResults(0).begin();
}

void BlockIdYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void BlockIdYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockIdYOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockIdYOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BlockIdYOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BlockIdYOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void BlockIdYOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void BlockIdYOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockIdYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::BlockIdZOp definitions
//===----------------------------------------------------------------------===//

BlockIdZOpAdaptor::BlockIdZOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.ctaid.z", odsAttrs.getContext());
}

BlockIdZOpAdaptor::BlockIdZOpAdaptor(BlockIdZOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BlockIdZOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BlockIdZOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BlockIdZOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BlockIdZOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BlockIdZOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BlockIdZOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BlockIdZOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BlockIdZOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BlockIdZOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BlockIdZOp::res() {
  return *getODSResults(0).begin();
}

void BlockIdZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void BlockIdZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BlockIdZOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BlockIdZOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BlockIdZOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BlockIdZOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void BlockIdZOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void BlockIdZOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::BlockIdZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::CpAsyncCommitGroupOp definitions
//===----------------------------------------------------------------------===//

CpAsyncCommitGroupOpAdaptor::CpAsyncCommitGroupOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.cp.async.commit.group", odsAttrs.getContext());
}

CpAsyncCommitGroupOpAdaptor::CpAsyncCommitGroupOpAdaptor(CpAsyncCommitGroupOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CpAsyncCommitGroupOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CpAsyncCommitGroupOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CpAsyncCommitGroupOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CpAsyncCommitGroupOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CpAsyncCommitGroupOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CpAsyncCommitGroupOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CpAsyncCommitGroupOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CpAsyncCommitGroupOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CpAsyncCommitGroupOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void CpAsyncCommitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void CpAsyncCommitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CpAsyncCommitGroupOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CpAsyncCommitGroupOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult CpAsyncCommitGroupOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CpAsyncCommitGroupOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void CpAsyncCommitGroupOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::CpAsyncCommitGroupOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::CpAsyncOp definitions
//===----------------------------------------------------------------------===//

CpAsyncOpAdaptor::CpAsyncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.cp.async.shared.global", odsAttrs.getContext());
}

CpAsyncOpAdaptor::CpAsyncOpAdaptor(CpAsyncOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CpAsyncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CpAsyncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CpAsyncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CpAsyncOpAdaptor::dst() {
  return *getODSOperands(0).begin();
}

::mlir::Value CpAsyncOpAdaptor::src() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CpAsyncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr CpAsyncOpAdaptor::sizeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CpAsyncOp::sizeAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t CpAsyncOpAdaptor::size() {
  auto attr = sizeAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CpAsyncOpAdaptor::bypass_l1Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CpAsyncOp::bypass_l1AttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

::llvm::Optional<bool> CpAsyncOpAdaptor::bypass_l1() {
  auto attr = bypass_l1Attr();
  return attr ? ::llvm::Optional<bool>(attr != nullptr) : (::llvm::None);
}

::mlir::LogicalResult CpAsyncOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_size;
  ::mlir::Attribute tblgen_bypass_l1;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.cp.async.shared.global' op ""requires attribute 'size'");
    if (namedAttrIt->getName() == CpAsyncOp::sizeAttrName(*odsOpName)) {
      tblgen_size = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CpAsyncOp::bypass_l1AttrName(*odsOpName)) {
      tblgen_bypass_l1 = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_size && !(((tblgen_size.isa<::mlir::IntegerAttr>())) && ((tblgen_size.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvvm.cp.async.shared.global' op ""attribute 'size' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_bypass_l1 && !((tblgen_bypass_l1.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'nvvm.cp.async.shared.global' op ""attribute 'bypass_l1' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CpAsyncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CpAsyncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CpAsyncOp::dst() {
  return *getODSOperands(0).begin();
}

::mlir::Value CpAsyncOp::src() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CpAsyncOp::dstMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CpAsyncOp::srcMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CpAsyncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CpAsyncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr CpAsyncOp::sizeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, sizeAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t CpAsyncOp::size() {
  auto attr = sizeAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CpAsyncOp::bypass_l1Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, bypass_l1AttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

::llvm::Optional<bool> CpAsyncOp::bypass_l1() {
  auto attr = bypass_l1Attr();
  return attr ? ::llvm::Optional<bool>(attr != nullptr) : (::llvm::None);
}

void CpAsyncOp::sizeAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(sizeAttrName(), attr);
}

void CpAsyncOp::bypass_l1Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr(bypass_l1AttrName(), attr);
}

::mlir::Attribute CpAsyncOp::removeBypass_l1Attr() {
  return (*this)->removeAttr(bypass_l1AttrName());
}

void CpAsyncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value dst, ::mlir::Value src, ::mlir::IntegerAttr size, /*optional*/::mlir::UnitAttr bypass_l1) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addAttribute(sizeAttrName(odsState.name), size);
  if (bypass_l1) {
  odsState.addAttribute(bypass_l1AttrName(odsState.name), bypass_l1);
  }
}

void CpAsyncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value src, ::mlir::IntegerAttr size, /*optional*/::mlir::UnitAttr bypass_l1) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addAttribute(sizeAttrName(odsState.name), size);
  if (bypass_l1) {
  odsState.addAttribute(bypass_l1AttrName(odsState.name), bypass_l1);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CpAsyncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value dst, ::mlir::Value src, uint32_t size, /*optional*/::mlir::UnitAttr bypass_l1) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addAttribute(sizeAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), size));
  if (bypass_l1) {
  odsState.addAttribute(bypass_l1AttrName(odsState.name), bypass_l1);
  }
}

void CpAsyncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value src, uint32_t size, /*optional*/::mlir::UnitAttr bypass_l1) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addAttribute(sizeAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), size));
  if (bypass_l1) {
  odsState.addAttribute(bypass_l1AttrName(odsState.name), bypass_l1);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CpAsyncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CpAsyncOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_size;
  ::mlir::Attribute tblgen_bypass_l1;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'size'");
    if (namedAttrIt->getName() == sizeAttrName()) {
      tblgen_size = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == bypass_l1AttrName()) {
      tblgen_bypass_l1 = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps0(*this, tblgen_size, "size")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps1(*this, tblgen_bypass_l1, "bypass_l1")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CpAsyncOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CpAsyncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand dstRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dstOperands(dstRawOperands);  ::llvm::SMLoc dstOperandsLoc;
  (void)dstOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand srcRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> srcOperands(srcRawOperands);  ::llvm::SMLoc srcOperandsLoc;
  (void)srcOperandsLoc;
  ::mlir::IntegerAttr sizeAttr;

  dstOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dstRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  srcOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(srcRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(sizeAttr, parser.getBuilder().getIntegerType(32), "size",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8), 3);
  ::mlir::Type odsBuildableType1 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8), 1);
  if (parser.resolveOperands(dstOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(srcOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CpAsyncOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << dst();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << src();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(sizeAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"size"});
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::CpAsyncOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::CpAsyncWaitGroupOp definitions
//===----------------------------------------------------------------------===//

CpAsyncWaitGroupOpAdaptor::CpAsyncWaitGroupOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.cp.async.wait.group", odsAttrs.getContext());
}

CpAsyncWaitGroupOpAdaptor::CpAsyncWaitGroupOpAdaptor(CpAsyncWaitGroupOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CpAsyncWaitGroupOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CpAsyncWaitGroupOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CpAsyncWaitGroupOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CpAsyncWaitGroupOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr CpAsyncWaitGroupOpAdaptor::nAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CpAsyncWaitGroupOp::nAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t CpAsyncWaitGroupOpAdaptor::n() {
  auto attr = nAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult CpAsyncWaitGroupOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_n;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.cp.async.wait.group' op ""requires attribute 'n'");
    if (namedAttrIt->getName() == CpAsyncWaitGroupOp::nAttrName(*odsOpName)) {
      tblgen_n = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_n && !(((tblgen_n.isa<::mlir::IntegerAttr>())) && ((tblgen_n.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvvm.cp.async.wait.group' op ""attribute 'n' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CpAsyncWaitGroupOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CpAsyncWaitGroupOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CpAsyncWaitGroupOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CpAsyncWaitGroupOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr CpAsyncWaitGroupOp::nAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, nAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t CpAsyncWaitGroupOp::n() {
  auto attr = nAttr();
  return attr.getValue().getZExtValue();
}

void CpAsyncWaitGroupOp::nAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(nAttrName(), attr);
}

void CpAsyncWaitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr n) {
  odsState.addAttribute(nAttrName(odsState.name), n);
}

void CpAsyncWaitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr n) {
  odsState.addAttribute(nAttrName(odsState.name), n);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CpAsyncWaitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint32_t n) {
  odsState.addAttribute(nAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), n));
}

void CpAsyncWaitGroupOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t n) {
  odsState.addAttribute(nAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), n));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CpAsyncWaitGroupOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CpAsyncWaitGroupOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_n;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'n'");
    if (namedAttrIt->getName() == nAttrName()) {
      tblgen_n = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps0(*this, tblgen_n, "n")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult CpAsyncWaitGroupOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CpAsyncWaitGroupOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr nAttr;

  if (parser.parseCustomAttributeWithFallback(nAttr, parser.getBuilder().getIntegerType(32), "n",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void CpAsyncWaitGroupOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(nAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"n"});
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::CpAsyncWaitGroupOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::GridDimXOp definitions
//===----------------------------------------------------------------------===//

GridDimXOpAdaptor::GridDimXOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.nctaid.x", odsAttrs.getContext());
}

GridDimXOpAdaptor::GridDimXOpAdaptor(GridDimXOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GridDimXOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GridDimXOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GridDimXOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GridDimXOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GridDimXOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GridDimXOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GridDimXOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GridDimXOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GridDimXOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GridDimXOp::res() {
  return *getODSResults(0).begin();
}

void GridDimXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void GridDimXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GridDimXOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GridDimXOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GridDimXOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GridDimXOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void GridDimXOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void GridDimXOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::GridDimXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::GridDimYOp definitions
//===----------------------------------------------------------------------===//

GridDimYOpAdaptor::GridDimYOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.nctaid.y", odsAttrs.getContext());
}

GridDimYOpAdaptor::GridDimYOpAdaptor(GridDimYOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GridDimYOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GridDimYOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GridDimYOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GridDimYOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GridDimYOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GridDimYOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GridDimYOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GridDimYOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GridDimYOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GridDimYOp::res() {
  return *getODSResults(0).begin();
}

void GridDimYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void GridDimYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GridDimYOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GridDimYOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GridDimYOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GridDimYOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void GridDimYOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void GridDimYOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::GridDimYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::GridDimZOp definitions
//===----------------------------------------------------------------------===//

GridDimZOpAdaptor::GridDimZOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.nctaid.z", odsAttrs.getContext());
}

GridDimZOpAdaptor::GridDimZOpAdaptor(GridDimZOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GridDimZOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GridDimZOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GridDimZOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GridDimZOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GridDimZOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GridDimZOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GridDimZOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GridDimZOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GridDimZOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GridDimZOp::res() {
  return *getODSResults(0).begin();
}

void GridDimZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void GridDimZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GridDimZOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GridDimZOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GridDimZOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GridDimZOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void GridDimZOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void GridDimZOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::GridDimZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::LaneIdOp definitions
//===----------------------------------------------------------------------===//

LaneIdOpAdaptor::LaneIdOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.laneid", odsAttrs.getContext());
}

LaneIdOpAdaptor::LaneIdOpAdaptor(LaneIdOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LaneIdOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LaneIdOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LaneIdOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr LaneIdOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LaneIdOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LaneIdOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LaneIdOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> LaneIdOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LaneIdOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LaneIdOp::res() {
  return *getODSResults(0).begin();
}

void LaneIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void LaneIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LaneIdOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LaneIdOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LaneIdOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LaneIdOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void LaneIdOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void LaneIdOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::LaneIdOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::LdMatrixOp definitions
//===----------------------------------------------------------------------===//

LdMatrixOpAdaptor::LdMatrixOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.ldmatrix", odsAttrs.getContext());
}

LdMatrixOpAdaptor::LdMatrixOpAdaptor(LdMatrixOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LdMatrixOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LdMatrixOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LdMatrixOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LdMatrixOpAdaptor::ptr() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr LdMatrixOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr LdMatrixOpAdaptor::numAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, LdMatrixOp::numAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t LdMatrixOpAdaptor::num() {
  auto attr = numAttr();
  return attr.getValue().getZExtValue();
}

::mlir::NVVM::MMALayoutAttr LdMatrixOpAdaptor::layoutAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LdMatrixOp::layoutAttrName(*odsOpName)).cast<::mlir::NVVM::MMALayoutAttr>();
    return attr;
}

::mlir::NVVM::MMALayout LdMatrixOpAdaptor::layout() {
  auto attr = layoutAttr();
  return attr.getValue();
}

::mlir::LogicalResult LdMatrixOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_layout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.ldmatrix' op ""requires attribute 'layout'");
    if (namedAttrIt->getName() == LdMatrixOp::layoutAttrName(*odsOpName)) {
      tblgen_layout = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_num;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.ldmatrix' op ""requires attribute 'num'");
    if (namedAttrIt->getName() == LdMatrixOp::numAttrName(*odsOpName)) {
      tblgen_num = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_num && !(((tblgen_num.isa<::mlir::IntegerAttr>())) && ((tblgen_num.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvvm.ldmatrix' op ""attribute 'num' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_layout && !((tblgen_layout.isa<::mlir::NVVM::MMALayoutAttr>())))
    return emitError(loc, "'nvvm.ldmatrix' op ""attribute 'layout' failed to satisfy constraint: NVVM MMA layout");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LdMatrixOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LdMatrixOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LdMatrixOp::ptr() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange LdMatrixOp::ptrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LdMatrixOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LdMatrixOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LdMatrixOp::res() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr LdMatrixOp::numAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, numAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t LdMatrixOp::num() {
  auto attr = numAttr();
  return attr.getValue().getZExtValue();
}

::mlir::NVVM::MMALayoutAttr LdMatrixOp::layoutAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, layoutAttrName()).cast<::mlir::NVVM::MMALayoutAttr>();
}

::mlir::NVVM::MMALayout LdMatrixOp::layout() {
  auto attr = layoutAttr();
  return attr.getValue();
}

void LdMatrixOp::numAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(numAttrName(), attr);
}

void LdMatrixOp::layoutAttr(::mlir::NVVM::MMALayoutAttr attr) {
  (*this)->setAttr(layoutAttrName(), attr);
}

void LdMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::IntegerAttr num, ::mlir::NVVM::MMALayoutAttr layout) {
  odsState.addOperands(ptr);
  odsState.addAttribute(numAttrName(odsState.name), num);
  odsState.addAttribute(layoutAttrName(odsState.name), layout);
  odsState.addTypes(res);
}

void LdMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::IntegerAttr num, ::mlir::NVVM::MMALayoutAttr layout) {
  odsState.addOperands(ptr);
  odsState.addAttribute(numAttrName(odsState.name), num);
  odsState.addAttribute(layoutAttrName(odsState.name), layout);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LdMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, uint32_t num, ::mlir::NVVM::MMALayout layout) {
  odsState.addOperands(ptr);
  odsState.addAttribute(numAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), num));
  odsState.addAttribute(layoutAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layout));
  odsState.addTypes(res);
}

void LdMatrixOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, uint32_t num, ::mlir::NVVM::MMALayout layout) {
  odsState.addOperands(ptr);
  odsState.addAttribute(numAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), num));
  odsState.addAttribute(layoutAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layout));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LdMatrixOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LdMatrixOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_layout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'layout'");
    if (namedAttrIt->getName() == layoutAttrName()) {
      tblgen_layout = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_num;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'num'");
    if (namedAttrIt->getName() == numAttrName()) {
      tblgen_num = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps0(*this, tblgen_num, "num")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps2(*this, tblgen_layout, "layout")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LdMatrixOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult LdMatrixOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand ptrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> ptrOperands(ptrRawOperands);  ::llvm::SMLoc ptrOperandsLoc;
  (void)ptrOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> ptrTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(ptrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType ptr__res_functionType;
  if (parser.parseType(ptr__res_functionType))
    return ::mlir::failure();
  ptrTypes = ptr__res_functionType.getInputs();
  resTypes = ptr__res_functionType.getResults();
  result.addTypes(resTypes);
  if (parser.resolveOperands(ptrOperands, ptrTypes, ptrOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LdMatrixOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << ptr();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(ptr().getType()), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::LdMatrixOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::MmaOp definitions
//===----------------------------------------------------------------------===//

MmaOpAdaptor::MmaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.mma.sync", odsAttrs.getContext());
}

MmaOpAdaptor::MmaOpAdaptor(MmaOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MmaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MmaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, MmaOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange MmaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange MmaOpAdaptor::operandA() {
  return getODSOperands(0);
}

::mlir::ValueRange MmaOpAdaptor::operandB() {
  return getODSOperands(1);
}

::mlir::ValueRange MmaOpAdaptor::operandC() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr MmaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::NVVM::MMAShapeAttr MmaOpAdaptor::shapeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, MmaOp::shapeAttrName(*odsOpName)).cast<::mlir::NVVM::MMAShapeAttr>();
    return attr;
}

::mlir::NVVM::MMAShapeAttr MmaOpAdaptor::shape() {
  auto attr = shapeAttr();
  return attr;
}

::mlir::NVVM::MMAB1OpAttr MmaOpAdaptor::b1OpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 4, MmaOp::b1OpAttrName(*odsOpName)).dyn_cast_or_null<::mlir::NVVM::MMAB1OpAttr>();
    return attr;
}

::llvm::Optional<::mlir::NVVM::MMAB1Op> MmaOpAdaptor::b1Op() {
  auto attr = b1OpAttr();
  return attr ? ::llvm::Optional<::mlir::NVVM::MMAB1Op>(attr.getValue()) : (::llvm::None);
}

::mlir::NVVM::MMAIntOverflowAttr MmaOpAdaptor::intOverflowBehaviorAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 4, MmaOp::intOverflowBehaviorAttrName(*odsOpName)).dyn_cast_or_null<::mlir::NVVM::MMAIntOverflowAttr>();
    return attr;
}

::llvm::Optional<::mlir::NVVM::MMAIntOverflow> MmaOpAdaptor::intOverflowBehavior() {
  auto attr = intOverflowBehaviorAttr();
  return attr ? ::llvm::Optional<::mlir::NVVM::MMAIntOverflow>(attr.getValue()) : (::llvm::None);
}

::mlir::NVVM::MMALayoutAttr MmaOpAdaptor::layoutAAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, MmaOp::layoutAAttrName(*odsOpName)).cast<::mlir::NVVM::MMALayoutAttr>();
    return attr;
}

::mlir::NVVM::MMALayout MmaOpAdaptor::layoutA() {
  auto attr = layoutAAttr();
  return attr.getValue();
}

::mlir::NVVM::MMALayoutAttr MmaOpAdaptor::layoutBAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 2, MmaOp::layoutBAttrName(*odsOpName)).cast<::mlir::NVVM::MMALayoutAttr>();
    return attr;
}

::mlir::NVVM::MMALayout MmaOpAdaptor::layoutB() {
  auto attr = layoutBAttr();
  return attr.getValue();
}

::mlir::NVVM::MMATypesAttr MmaOpAdaptor::multiplicandAPtxTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 2, MmaOp::multiplicandAPtxTypeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::NVVM::MMATypesAttr>();
    return attr;
}

::llvm::Optional<::mlir::NVVM::MMATypes> MmaOpAdaptor::multiplicandAPtxType() {
  auto attr = multiplicandAPtxTypeAttr();
  return attr ? ::llvm::Optional<::mlir::NVVM::MMATypes>(attr.getValue()) : (::llvm::None);
}

::mlir::NVVM::MMATypesAttr MmaOpAdaptor::multiplicandBPtxTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 2, MmaOp::multiplicandBPtxTypeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::NVVM::MMATypesAttr>();
    return attr;
}

::llvm::Optional<::mlir::NVVM::MMATypes> MmaOpAdaptor::multiplicandBPtxType() {
  auto attr = multiplicandBPtxTypeAttr();
  return attr ? ::llvm::Optional<::mlir::NVVM::MMATypes>(attr.getValue()) : (::llvm::None);
}

::mlir::LogicalResult MmaOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_layoutA;
  ::mlir::Attribute tblgen_b1Op;
  ::mlir::Attribute tblgen_intOverflowBehavior;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.mma.sync' op ""requires attribute 'layoutA'");
    if (namedAttrIt->getName() == MmaOp::layoutAAttrName(*odsOpName)) {
      tblgen_layoutA = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == MmaOp::b1OpAttrName(*odsOpName)) {
      tblgen_b1Op = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == MmaOp::intOverflowBehaviorAttrName(*odsOpName)) {
      tblgen_intOverflowBehavior = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_layoutB;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.mma.sync' op ""requires attribute 'layoutB'");
    if (namedAttrIt->getName() == MmaOp::layoutBAttrName(*odsOpName)) {
      tblgen_layoutB = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_multiplicandAPtxType;
  ::mlir::Attribute tblgen_multiplicandBPtxType;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.mma.sync' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == MmaOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == MmaOp::multiplicandAPtxTypeAttrName(*odsOpName)) {
      tblgen_multiplicandAPtxType = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == MmaOp::multiplicandBPtxTypeAttrName(*odsOpName)) {
      tblgen_multiplicandBPtxType = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_shape;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.mma.sync' op ""requires attribute 'shape'");
    if (namedAttrIt->getName() == MmaOp::shapeAttrName(*odsOpName)) {
      tblgen_shape = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'nvvm.mma.sync' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_shape && !((tblgen_shape.isa<::mlir::NVVM::MMAShapeAttr>())))
    return emitError(loc, "'nvvm.mma.sync' op ""attribute 'shape' failed to satisfy constraint: Attribute for MMA operation shape.");

  if (tblgen_b1Op && !((tblgen_b1Op.isa<::mlir::NVVM::MMAB1OpAttr>())))
    return emitError(loc, "'nvvm.mma.sync' op ""attribute 'b1Op' failed to satisfy constraint: MMA binary operations");

  if (tblgen_intOverflowBehavior && !((tblgen_intOverflowBehavior.isa<::mlir::NVVM::MMAIntOverflowAttr>())))
    return emitError(loc, "'nvvm.mma.sync' op ""attribute 'intOverflowBehavior' failed to satisfy constraint: MMA overflow options");

  if (tblgen_layoutA && !((tblgen_layoutA.isa<::mlir::NVVM::MMALayoutAttr>())))
    return emitError(loc, "'nvvm.mma.sync' op ""attribute 'layoutA' failed to satisfy constraint: NVVM MMA layout");

  if (tblgen_layoutB && !((tblgen_layoutB.isa<::mlir::NVVM::MMALayoutAttr>())))
    return emitError(loc, "'nvvm.mma.sync' op ""attribute 'layoutB' failed to satisfy constraint: NVVM MMA layout");

  if (tblgen_multiplicandAPtxType && !((tblgen_multiplicandAPtxType.isa<::mlir::NVVM::MMATypesAttr>())))
    return emitError(loc, "'nvvm.mma.sync' op ""attribute 'multiplicandAPtxType' failed to satisfy constraint: NVVM MMA types");

  if (tblgen_multiplicandBPtxType && !((tblgen_multiplicandBPtxType.isa<::mlir::NVVM::MMATypesAttr>())))
    return emitError(loc, "'nvvm.mma.sync' op ""attribute 'multiplicandBPtxType' failed to satisfy constraint: NVVM MMA types");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MmaOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range MmaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range MmaOp::operandA() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range MmaOp::operandB() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range MmaOp::operandC() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange MmaOp::operandAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MmaOp::operandBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange MmaOp::operandCMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> MmaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MmaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MmaOp::res() {
  return *getODSResults(0).begin();
}

::mlir::NVVM::MMAShapeAttr MmaOp::shapeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, shapeAttrName()).cast<::mlir::NVVM::MMAShapeAttr>();
}

::mlir::NVVM::MMAShapeAttr MmaOp::shape() {
  auto attr = shapeAttr();
  return attr;
}

::mlir::NVVM::MMAB1OpAttr MmaOp::b1OpAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 4, b1OpAttrName()).dyn_cast_or_null<::mlir::NVVM::MMAB1OpAttr>();
}

::llvm::Optional<::mlir::NVVM::MMAB1Op> MmaOp::b1Op() {
  auto attr = b1OpAttr();
  return attr ? ::llvm::Optional<::mlir::NVVM::MMAB1Op>(attr.getValue()) : (::llvm::None);
}

::mlir::NVVM::MMAIntOverflowAttr MmaOp::intOverflowBehaviorAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 4, intOverflowBehaviorAttrName()).dyn_cast_or_null<::mlir::NVVM::MMAIntOverflowAttr>();
}

::llvm::Optional<::mlir::NVVM::MMAIntOverflow> MmaOp::intOverflowBehavior() {
  auto attr = intOverflowBehaviorAttr();
  return attr ? ::llvm::Optional<::mlir::NVVM::MMAIntOverflow>(attr.getValue()) : (::llvm::None);
}

::mlir::NVVM::MMALayoutAttr MmaOp::layoutAAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, layoutAAttrName()).cast<::mlir::NVVM::MMALayoutAttr>();
}

::mlir::NVVM::MMALayout MmaOp::layoutA() {
  auto attr = layoutAAttr();
  return attr.getValue();
}

::mlir::NVVM::MMALayoutAttr MmaOp::layoutBAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 2, layoutBAttrName()).cast<::mlir::NVVM::MMALayoutAttr>();
}

::mlir::NVVM::MMALayout MmaOp::layoutB() {
  auto attr = layoutBAttr();
  return attr.getValue();
}

::mlir::NVVM::MMATypesAttr MmaOp::multiplicandAPtxTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 2, multiplicandAPtxTypeAttrName()).dyn_cast_or_null<::mlir::NVVM::MMATypesAttr>();
}

::llvm::Optional<::mlir::NVVM::MMATypes> MmaOp::multiplicandAPtxType() {
  auto attr = multiplicandAPtxTypeAttr();
  return attr ? ::llvm::Optional<::mlir::NVVM::MMATypes>(attr.getValue()) : (::llvm::None);
}

::mlir::NVVM::MMATypesAttr MmaOp::multiplicandBPtxTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 2, multiplicandBPtxTypeAttrName()).dyn_cast_or_null<::mlir::NVVM::MMATypesAttr>();
}

::llvm::Optional<::mlir::NVVM::MMATypes> MmaOp::multiplicandBPtxType() {
  auto attr = multiplicandBPtxTypeAttr();
  return attr ? ::llvm::Optional<::mlir::NVVM::MMATypes>(attr.getValue()) : (::llvm::None);
}

void MmaOp::shapeAttr(::mlir::NVVM::MMAShapeAttr attr) {
  (*this)->setAttr(shapeAttrName(), attr);
}

void MmaOp::b1OpAttr(::mlir::NVVM::MMAB1OpAttr attr) {
  (*this)->setAttr(b1OpAttrName(), attr);
}

void MmaOp::intOverflowBehaviorAttr(::mlir::NVVM::MMAIntOverflowAttr attr) {
  (*this)->setAttr(intOverflowBehaviorAttrName(), attr);
}

void MmaOp::layoutAAttr(::mlir::NVVM::MMALayoutAttr attr) {
  (*this)->setAttr(layoutAAttrName(), attr);
}

void MmaOp::layoutBAttr(::mlir::NVVM::MMALayoutAttr attr) {
  (*this)->setAttr(layoutBAttrName(), attr);
}

void MmaOp::multiplicandAPtxTypeAttr(::mlir::NVVM::MMATypesAttr attr) {
  (*this)->setAttr(multiplicandAPtxTypeAttrName(), attr);
}

void MmaOp::multiplicandBPtxTypeAttr(::mlir::NVVM::MMATypesAttr attr) {
  (*this)->setAttr(multiplicandBPtxTypeAttrName(), attr);
}

::mlir::Attribute MmaOp::removeB1OpAttr() {
  return (*this)->removeAttr(b1OpAttrName());
}

::mlir::Attribute MmaOp::removeIntOverflowBehaviorAttr() {
  return (*this)->removeAttr(intOverflowBehaviorAttrName());
}

::mlir::Attribute MmaOp::removeMultiplicandAPtxTypeAttr() {
  return (*this)->removeAttr(multiplicandAPtxTypeAttrName());
}

::mlir::Attribute MmaOp::removeMultiplicandBPtxTypeAttr() {
  return (*this)->removeAttr(multiplicandBPtxTypeAttrName());
}

void MmaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::NVVM::MMAShapeAttr shape, /*optional*/::mlir::NVVM::MMAB1OpAttr b1Op, /*optional*/::mlir::NVVM::MMAIntOverflowAttr intOverflowBehavior, ::mlir::NVVM::MMALayoutAttr layoutA, ::mlir::NVVM::MMALayoutAttr layoutB, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandAPtxType, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandBPtxType, ::mlir::ValueRange operandA, ::mlir::ValueRange operandB, ::mlir::ValueRange operandC) {
  odsState.addOperands(operandA);
  odsState.addOperands(operandB);
  odsState.addOperands(operandC);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(operandA.size()), static_cast<int32_t>(operandB.size()), static_cast<int32_t>(operandC.size())}));
  odsState.addAttribute(shapeAttrName(odsState.name), shape);
  if (b1Op) {
  odsState.addAttribute(b1OpAttrName(odsState.name), b1Op);
  }
  if (intOverflowBehavior) {
  odsState.addAttribute(intOverflowBehaviorAttrName(odsState.name), intOverflowBehavior);
  }
  odsState.addAttribute(layoutAAttrName(odsState.name), layoutA);
  odsState.addAttribute(layoutBAttrName(odsState.name), layoutB);
  if (multiplicandAPtxType) {
  odsState.addAttribute(multiplicandAPtxTypeAttrName(odsState.name), multiplicandAPtxType);
  }
  if (multiplicandBPtxType) {
  odsState.addAttribute(multiplicandBPtxTypeAttrName(odsState.name), multiplicandBPtxType);
  }
  odsState.addTypes(res);
}

void MmaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::NVVM::MMAShapeAttr shape, /*optional*/::mlir::NVVM::MMAB1OpAttr b1Op, /*optional*/::mlir::NVVM::MMAIntOverflowAttr intOverflowBehavior, ::mlir::NVVM::MMALayoutAttr layoutA, ::mlir::NVVM::MMALayoutAttr layoutB, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandAPtxType, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandBPtxType, ::mlir::ValueRange operandA, ::mlir::ValueRange operandB, ::mlir::ValueRange operandC) {
  odsState.addOperands(operandA);
  odsState.addOperands(operandB);
  odsState.addOperands(operandC);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(operandA.size()), static_cast<int32_t>(operandB.size()), static_cast<int32_t>(operandC.size())}));
  odsState.addAttribute(shapeAttrName(odsState.name), shape);
  if (b1Op) {
  odsState.addAttribute(b1OpAttrName(odsState.name), b1Op);
  }
  if (intOverflowBehavior) {
  odsState.addAttribute(intOverflowBehaviorAttrName(odsState.name), intOverflowBehavior);
  }
  odsState.addAttribute(layoutAAttrName(odsState.name), layoutA);
  odsState.addAttribute(layoutBAttrName(odsState.name), layoutB);
  if (multiplicandAPtxType) {
  odsState.addAttribute(multiplicandAPtxTypeAttrName(odsState.name), multiplicandAPtxType);
  }
  if (multiplicandBPtxType) {
  odsState.addAttribute(multiplicandBPtxTypeAttrName(odsState.name), multiplicandBPtxType);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MmaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::NVVM::MMAShapeAttr shape, /*optional*/::mlir::NVVM::MMAB1OpAttr b1Op, /*optional*/::mlir::NVVM::MMAIntOverflowAttr intOverflowBehavior, ::mlir::NVVM::MMALayout layoutA, ::mlir::NVVM::MMALayout layoutB, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandAPtxType, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandBPtxType, ::mlir::ValueRange operandA, ::mlir::ValueRange operandB, ::mlir::ValueRange operandC) {
  odsState.addOperands(operandA);
  odsState.addOperands(operandB);
  odsState.addOperands(operandC);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(operandA.size()), static_cast<int32_t>(operandB.size()), static_cast<int32_t>(operandC.size())}));
  odsState.addAttribute(shapeAttrName(odsState.name), shape);
  if (b1Op) {
  odsState.addAttribute(b1OpAttrName(odsState.name), b1Op);
  }
  if (intOverflowBehavior) {
  odsState.addAttribute(intOverflowBehaviorAttrName(odsState.name), intOverflowBehavior);
  }
  odsState.addAttribute(layoutAAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layoutA));
  odsState.addAttribute(layoutBAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layoutB));
  if (multiplicandAPtxType) {
  odsState.addAttribute(multiplicandAPtxTypeAttrName(odsState.name), multiplicandAPtxType);
  }
  if (multiplicandBPtxType) {
  odsState.addAttribute(multiplicandBPtxTypeAttrName(odsState.name), multiplicandBPtxType);
  }
  odsState.addTypes(res);
}

void MmaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::NVVM::MMAShapeAttr shape, /*optional*/::mlir::NVVM::MMAB1OpAttr b1Op, /*optional*/::mlir::NVVM::MMAIntOverflowAttr intOverflowBehavior, ::mlir::NVVM::MMALayout layoutA, ::mlir::NVVM::MMALayout layoutB, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandAPtxType, /*optional*/::mlir::NVVM::MMATypesAttr multiplicandBPtxType, ::mlir::ValueRange operandA, ::mlir::ValueRange operandB, ::mlir::ValueRange operandC) {
  odsState.addOperands(operandA);
  odsState.addOperands(operandB);
  odsState.addOperands(operandC);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(operandA.size()), static_cast<int32_t>(operandB.size()), static_cast<int32_t>(operandC.size())}));
  odsState.addAttribute(shapeAttrName(odsState.name), shape);
  if (b1Op) {
  odsState.addAttribute(b1OpAttrName(odsState.name), b1Op);
  }
  if (intOverflowBehavior) {
  odsState.addAttribute(intOverflowBehaviorAttrName(odsState.name), intOverflowBehavior);
  }
  odsState.addAttribute(layoutAAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layoutA));
  odsState.addAttribute(layoutBAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layoutB));
  if (multiplicandAPtxType) {
  odsState.addAttribute(multiplicandAPtxTypeAttrName(odsState.name), multiplicandAPtxType);
  }
  if (multiplicandBPtxType) {
  odsState.addAttribute(multiplicandBPtxTypeAttrName(odsState.name), multiplicandBPtxType);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MmaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MmaOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_layoutA;
  ::mlir::Attribute tblgen_b1Op;
  ::mlir::Attribute tblgen_intOverflowBehavior;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'layoutA'");
    if (namedAttrIt->getName() == layoutAAttrName()) {
      tblgen_layoutA = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == b1OpAttrName()) {
      tblgen_b1Op = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == intOverflowBehaviorAttrName()) {
      tblgen_intOverflowBehavior = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_layoutB;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'layoutB'");
    if (namedAttrIt->getName() == layoutBAttrName()) {
      tblgen_layoutB = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_multiplicandAPtxType;
  ::mlir::Attribute tblgen_multiplicandBPtxType;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == multiplicandAPtxTypeAttrName()) {
      tblgen_multiplicandAPtxType = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == multiplicandBPtxTypeAttrName()) {
      tblgen_multiplicandBPtxType = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_shape;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'shape'");
    if (namedAttrIt->getName() == shapeAttrName()) {
      tblgen_shape = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps3(*this, tblgen_shape, "shape")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps4(*this, tblgen_b1Op, "b1Op")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps5(*this, tblgen_intOverflowBehavior, "intOverflowBehavior")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps2(*this, tblgen_layoutA, "layoutA")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps2(*this, tblgen_layoutB, "layoutB")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps6(*this, tblgen_multiplicandAPtxType, "multiplicandAPtxType")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps6(*this, tblgen_multiplicandBPtxType, "multiplicandBPtxType")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MmaOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::MmaOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ShflOp definitions
//===----------------------------------------------------------------------===//

ShflOpAdaptor::ShflOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.shfl.sync", odsAttrs.getContext());
}

ShflOpAdaptor::ShflOpAdaptor(ShflOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ShflOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ShflOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ShflOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShflOpAdaptor::dst() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShflOpAdaptor::val() {
  return *getODSOperands(1).begin();
}

::mlir::Value ShflOpAdaptor::offset() {
  return *getODSOperands(2).begin();
}

::mlir::Value ShflOpAdaptor::mask_and_clamp() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr ShflOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::NVVM::ShflKindAttr ShflOpAdaptor::kindAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ShflOp::kindAttrName(*odsOpName)).cast<::mlir::NVVM::ShflKindAttr>();
    return attr;
}

::mlir::NVVM::ShflKind ShflOpAdaptor::kind() {
  auto attr = kindAttr();
  return attr.getValue();
}

::mlir::UnitAttr ShflOpAdaptor::return_value_and_is_validAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ShflOp::return_value_and_is_validAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

::llvm::Optional<bool> ShflOpAdaptor::return_value_and_is_valid() {
  auto attr = return_value_and_is_validAttr();
  return attr ? ::llvm::Optional<bool>(attr != nullptr) : (::llvm::None);
}

::mlir::LogicalResult ShflOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_kind;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.shfl.sync' op ""requires attribute 'kind'");
    if (namedAttrIt->getName() == ShflOp::kindAttrName(*odsOpName)) {
      tblgen_kind = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_return_value_and_is_valid;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == ShflOp::return_value_and_is_validAttrName(*odsOpName)) {
      tblgen_return_value_and_is_valid = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_kind && !((tblgen_kind.isa<::mlir::NVVM::ShflKindAttr>())))
    return emitError(loc, "'nvvm.shfl.sync' op ""attribute 'kind' failed to satisfy constraint: NVVM shuffle kind");

  if (tblgen_return_value_and_is_valid && !((tblgen_return_value_and_is_valid.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'nvvm.shfl.sync' op ""attribute 'return_value_and_is_valid' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ShflOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ShflOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShflOp::dst() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShflOp::val() {
  return *getODSOperands(1).begin();
}

::mlir::Value ShflOp::offset() {
  return *getODSOperands(2).begin();
}

::mlir::Value ShflOp::mask_and_clamp() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange ShflOp::dstMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShflOp::valMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShflOp::offsetMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShflOp::mask_and_clampMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ShflOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShflOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShflOp::res() {
  return *getODSResults(0).begin();
}

::mlir::NVVM::ShflKindAttr ShflOp::kindAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, kindAttrName()).cast<::mlir::NVVM::ShflKindAttr>();
}

::mlir::NVVM::ShflKind ShflOp::kind() {
  auto attr = kindAttr();
  return attr.getValue();
}

::mlir::UnitAttr ShflOp::return_value_and_is_validAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, return_value_and_is_validAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

::llvm::Optional<bool> ShflOp::return_value_and_is_valid() {
  auto attr = return_value_and_is_validAttr();
  return attr ? ::llvm::Optional<bool>(attr != nullptr) : (::llvm::None);
}

void ShflOp::kindAttr(::mlir::NVVM::ShflKindAttr attr) {
  (*this)->setAttr(kindAttrName(), attr);
}

void ShflOp::return_value_and_is_validAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(return_value_and_is_validAttrName(), attr);
}

::mlir::Attribute ShflOp::removeReturn_value_and_is_validAttr() {
  return (*this)->removeAttr(return_value_and_is_validAttrName());
}

void ShflOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, ::mlir::NVVM::ShflKindAttr kind, /*optional*/::mlir::UnitAttr return_value_and_is_valid) {
  odsState.addOperands(dst);
  odsState.addOperands(val);
  odsState.addOperands(offset);
  odsState.addOperands(mask_and_clamp);
  odsState.addAttribute(kindAttrName(odsState.name), kind);
  if (return_value_and_is_valid) {
  odsState.addAttribute(return_value_and_is_validAttrName(odsState.name), return_value_and_is_valid);
  }
  odsState.addTypes(res);
}

void ShflOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, ::mlir::NVVM::ShflKindAttr kind, /*optional*/::mlir::UnitAttr return_value_and_is_valid) {
  odsState.addOperands(dst);
  odsState.addOperands(val);
  odsState.addOperands(offset);
  odsState.addOperands(mask_and_clamp);
  odsState.addAttribute(kindAttrName(odsState.name), kind);
  if (return_value_and_is_valid) {
  odsState.addAttribute(return_value_and_is_validAttrName(odsState.name), return_value_and_is_valid);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShflOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, ::mlir::NVVM::ShflKind kind, /*optional*/::mlir::UnitAttr return_value_and_is_valid) {
  odsState.addOperands(dst);
  odsState.addOperands(val);
  odsState.addOperands(offset);
  odsState.addOperands(mask_and_clamp);
  odsState.addAttribute(kindAttrName(odsState.name), ::mlir::NVVM::ShflKindAttr::get(odsBuilder.getContext(), kind));
  if (return_value_and_is_valid) {
  odsState.addAttribute(return_value_and_is_validAttrName(odsState.name), return_value_and_is_valid);
  }
  odsState.addTypes(res);
}

void ShflOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value offset, ::mlir::Value mask_and_clamp, ::mlir::NVVM::ShflKind kind, /*optional*/::mlir::UnitAttr return_value_and_is_valid) {
  odsState.addOperands(dst);
  odsState.addOperands(val);
  odsState.addOperands(offset);
  odsState.addOperands(mask_and_clamp);
  odsState.addAttribute(kindAttrName(odsState.name), ::mlir::NVVM::ShflKindAttr::get(odsBuilder.getContext(), kind));
  if (return_value_and_is_valid) {
  odsState.addAttribute(return_value_and_is_validAttrName(odsState.name), return_value_and_is_valid);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShflOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ShflOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_kind;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'kind'");
    if (namedAttrIt->getName() == kindAttrName()) {
      tblgen_kind = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_return_value_and_is_valid;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == return_value_and_is_validAttrName()) {
      tblgen_return_value_and_is_valid = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps7(*this, tblgen_kind, "kind")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps1(*this, tblgen_return_value_and_is_valid, "return_value_and_is_valid")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ShflOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ShflOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::NVVM::ShflKindAttr kindAttr;
  ::mlir::OpAsmParser::UnresolvedOperand dstRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dstOperands(dstRawOperands);  ::llvm::SMLoc dstOperandsLoc;
  (void)dstOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand valRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valOperands(valRawOperands);  ::llvm::SMLoc valOperandsLoc;
  (void)valOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand offsetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> offsetOperands(offsetRawOperands);  ::llvm::SMLoc offsetOperandsLoc;
  (void)offsetOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand mask_and_clampRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> mask_and_clampOperands(mask_and_clampRawOperands);  ::llvm::SMLoc mask_and_clampOperandsLoc;
  (void)mask_and_clampOperandsLoc;
  ::mlir::Type valRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valTypes(valRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  if (parser.parseCustomAttributeWithFallback(kindAttr, ::mlir::Type{}, "kind",
          result.attributes)) {
    return ::mlir::failure();
  }

  dstOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dstRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  valOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  offsetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(offsetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  mask_and_clampOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(mask_and_clampRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  result.addTypes(resTypes);
  if (parser.resolveOperands(dstOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valOperands, valTypes, valOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(offsetOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(mask_and_clampOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ShflOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(kindAttr());
  _odsPrinter << ' ';
  _odsPrinter << dst();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << val();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << offset();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << mask_and_clamp();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"kind"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = val().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::ShflOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ThreadIdXOp definitions
//===----------------------------------------------------------------------===//

ThreadIdXOpAdaptor::ThreadIdXOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.tid.x", odsAttrs.getContext());
}

ThreadIdXOpAdaptor::ThreadIdXOpAdaptor(ThreadIdXOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ThreadIdXOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ThreadIdXOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ThreadIdXOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ThreadIdXOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ThreadIdXOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ThreadIdXOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ThreadIdXOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ThreadIdXOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ThreadIdXOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ThreadIdXOp::res() {
  return *getODSResults(0).begin();
}

void ThreadIdXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void ThreadIdXOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ThreadIdXOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ThreadIdXOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ThreadIdXOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ThreadIdXOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void ThreadIdXOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ThreadIdXOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::ThreadIdXOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ThreadIdYOp definitions
//===----------------------------------------------------------------------===//

ThreadIdYOpAdaptor::ThreadIdYOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.tid.y", odsAttrs.getContext());
}

ThreadIdYOpAdaptor::ThreadIdYOpAdaptor(ThreadIdYOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ThreadIdYOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ThreadIdYOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ThreadIdYOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ThreadIdYOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ThreadIdYOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ThreadIdYOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ThreadIdYOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ThreadIdYOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ThreadIdYOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ThreadIdYOp::res() {
  return *getODSResults(0).begin();
}

void ThreadIdYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void ThreadIdYOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ThreadIdYOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ThreadIdYOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ThreadIdYOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ThreadIdYOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void ThreadIdYOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ThreadIdYOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::ThreadIdYOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::ThreadIdZOp definitions
//===----------------------------------------------------------------------===//

ThreadIdZOpAdaptor::ThreadIdZOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.tid.z", odsAttrs.getContext());
}

ThreadIdZOpAdaptor::ThreadIdZOpAdaptor(ThreadIdZOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ThreadIdZOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ThreadIdZOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ThreadIdZOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ThreadIdZOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ThreadIdZOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ThreadIdZOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ThreadIdZOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ThreadIdZOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ThreadIdZOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ThreadIdZOp::res() {
  return *getODSResults(0).begin();
}

void ThreadIdZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void ThreadIdZOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ThreadIdZOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ThreadIdZOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ThreadIdZOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ThreadIdZOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void ThreadIdZOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ThreadIdZOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::ThreadIdZOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::VoteBallotOp definitions
//===----------------------------------------------------------------------===//

VoteBallotOpAdaptor::VoteBallotOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.vote.ballot.sync", odsAttrs.getContext());
}

VoteBallotOpAdaptor::VoteBallotOpAdaptor(VoteBallotOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VoteBallotOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VoteBallotOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VoteBallotOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VoteBallotOpAdaptor::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value VoteBallotOpAdaptor::pred() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr VoteBallotOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VoteBallotOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VoteBallotOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VoteBallotOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VoteBallotOp::mask() {
  return *getODSOperands(0).begin();
}

::mlir::Value VoteBallotOp::pred() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange VoteBallotOp::maskMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VoteBallotOp::predMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VoteBallotOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VoteBallotOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VoteBallotOp::res() {
  return *getODSResults(0).begin();
}

void VoteBallotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value mask, ::mlir::Value pred) {
  odsState.addOperands(mask);
  odsState.addOperands(pred);
  odsState.addTypes(res);
}

void VoteBallotOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value mask, ::mlir::Value pred) {
  odsState.addOperands(mask);
  odsState.addOperands(pred);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VoteBallotOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VoteBallotOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VoteBallotOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::VoteBallotOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMALoadOp definitions
//===----------------------------------------------------------------------===//

WMMALoadOpAdaptor::WMMALoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.wmma.load", odsAttrs.getContext());
}

WMMALoadOpAdaptor::WMMALoadOpAdaptor(WMMALoadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange WMMALoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WMMALoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange WMMALoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WMMALoadOpAdaptor::ptr() {
  return *getODSOperands(0).begin();
}

::mlir::Value WMMALoadOpAdaptor::stride() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr WMMALoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr WMMALoadOpAdaptor::mAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 4, odsAttrs.end() - 1, WMMALoadOp::mAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t WMMALoadOpAdaptor::m() {
  auto attr = mAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr WMMALoadOpAdaptor::nAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 5, odsAttrs.end() - 0, WMMALoadOp::nAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t WMMALoadOpAdaptor::n() {
  auto attr = nAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr WMMALoadOpAdaptor::kAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 3, WMMALoadOp::kAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t WMMALoadOpAdaptor::k() {
  auto attr = kAttr();
  return attr.getValue().getZExtValue();
}

::mlir::NVVM::MMALayoutAttr WMMALoadOpAdaptor::layoutAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 2, WMMALoadOp::layoutAttrName(*odsOpName)).cast<::mlir::NVVM::MMALayoutAttr>();
    return attr;
}

::mlir::NVVM::MMALayout WMMALoadOpAdaptor::layout() {
  auto attr = layoutAttr();
  return attr.getValue();
}

::mlir::NVVM::MMATypesAttr WMMALoadOpAdaptor::eltypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 5, WMMALoadOp::eltypeAttrName(*odsOpName)).cast<::mlir::NVVM::MMATypesAttr>();
    return attr;
}

::mlir::NVVM::MMATypes WMMALoadOpAdaptor::eltype() {
  auto attr = eltypeAttr();
  return attr.getValue();
}

::mlir::NVVM::MMAFragAttr WMMALoadOpAdaptor::fragAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 4, WMMALoadOp::fragAttrName(*odsOpName)).cast<::mlir::NVVM::MMAFragAttr>();
    return attr;
}

::mlir::NVVM::MMAFrag WMMALoadOpAdaptor::frag() {
  auto attr = fragAttr();
  return attr.getValue();
}

::mlir::LogicalResult WMMALoadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_eltype;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.load' op ""requires attribute 'eltype'");
    if (namedAttrIt->getName() == WMMALoadOp::eltypeAttrName(*odsOpName)) {
      tblgen_eltype = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_frag;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.load' op ""requires attribute 'frag'");
    if (namedAttrIt->getName() == WMMALoadOp::fragAttrName(*odsOpName)) {
      tblgen_frag = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_k;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.load' op ""requires attribute 'k'");
    if (namedAttrIt->getName() == WMMALoadOp::kAttrName(*odsOpName)) {
      tblgen_k = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_layout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.load' op ""requires attribute 'layout'");
    if (namedAttrIt->getName() == WMMALoadOp::layoutAttrName(*odsOpName)) {
      tblgen_layout = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_m;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.load' op ""requires attribute 'm'");
    if (namedAttrIt->getName() == WMMALoadOp::mAttrName(*odsOpName)) {
      tblgen_m = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_n;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.load' op ""requires attribute 'n'");
    if (namedAttrIt->getName() == WMMALoadOp::nAttrName(*odsOpName)) {
      tblgen_n = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_m && !(((tblgen_m.isa<::mlir::IntegerAttr>())) && ((tblgen_m.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvvm.wmma.load' op ""attribute 'm' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_n && !(((tblgen_n.isa<::mlir::IntegerAttr>())) && ((tblgen_n.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvvm.wmma.load' op ""attribute 'n' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_k && !(((tblgen_k.isa<::mlir::IntegerAttr>())) && ((tblgen_k.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvvm.wmma.load' op ""attribute 'k' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_layout && !((tblgen_layout.isa<::mlir::NVVM::MMALayoutAttr>())))
    return emitError(loc, "'nvvm.wmma.load' op ""attribute 'layout' failed to satisfy constraint: NVVM MMA layout");

  if (tblgen_eltype && !((tblgen_eltype.isa<::mlir::NVVM::MMATypesAttr>())))
    return emitError(loc, "'nvvm.wmma.load' op ""attribute 'eltype' failed to satisfy constraint: NVVM MMA types");

  if (tblgen_frag && !((tblgen_frag.isa<::mlir::NVVM::MMAFragAttr>())))
    return emitError(loc, "'nvvm.wmma.load' op ""attribute 'frag' failed to satisfy constraint: NVVM MMA frag type");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WMMALoadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range WMMALoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WMMALoadOp::ptr() {
  return *getODSOperands(0).begin();
}

::mlir::Value WMMALoadOp::stride() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange WMMALoadOp::ptrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange WMMALoadOp::strideMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WMMALoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WMMALoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WMMALoadOp::res() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr WMMALoadOp::mAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 4, (*this)->getAttrs().end() - 1, mAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t WMMALoadOp::m() {
  auto attr = mAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr WMMALoadOp::nAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 5, (*this)->getAttrs().end() - 0, nAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t WMMALoadOp::n() {
  auto attr = nAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr WMMALoadOp::kAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 3, kAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t WMMALoadOp::k() {
  auto attr = kAttr();
  return attr.getValue().getZExtValue();
}

::mlir::NVVM::MMALayoutAttr WMMALoadOp::layoutAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 2, layoutAttrName()).cast<::mlir::NVVM::MMALayoutAttr>();
}

::mlir::NVVM::MMALayout WMMALoadOp::layout() {
  auto attr = layoutAttr();
  return attr.getValue();
}

::mlir::NVVM::MMATypesAttr WMMALoadOp::eltypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 5, eltypeAttrName()).cast<::mlir::NVVM::MMATypesAttr>();
}

::mlir::NVVM::MMATypes WMMALoadOp::eltype() {
  auto attr = eltypeAttr();
  return attr.getValue();
}

::mlir::NVVM::MMAFragAttr WMMALoadOp::fragAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 4, fragAttrName()).cast<::mlir::NVVM::MMAFragAttr>();
}

::mlir::NVVM::MMAFrag WMMALoadOp::frag() {
  auto attr = fragAttr();
  return attr.getValue();
}

void WMMALoadOp::mAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(mAttrName(), attr);
}

void WMMALoadOp::nAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(nAttrName(), attr);
}

void WMMALoadOp::kAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(kAttrName(), attr);
}

void WMMALoadOp::layoutAttr(::mlir::NVVM::MMALayoutAttr attr) {
  (*this)->setAttr(layoutAttrName(), attr);
}

void WMMALoadOp::eltypeAttr(::mlir::NVVM::MMATypesAttr attr) {
  (*this)->setAttr(eltypeAttrName(), attr);
}

void WMMALoadOp::fragAttr(::mlir::NVVM::MMAFragAttr attr) {
  (*this)->setAttr(fragAttrName(), attr);
}

void WMMALoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value stride, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layout, ::mlir::NVVM::MMATypesAttr eltype, ::mlir::NVVM::MMAFragAttr frag) {
  odsState.addOperands(ptr);
  odsState.addOperands(stride);
  odsState.addAttribute(mAttrName(odsState.name), m);
  odsState.addAttribute(nAttrName(odsState.name), n);
  odsState.addAttribute(kAttrName(odsState.name), k);
  odsState.addAttribute(layoutAttrName(odsState.name), layout);
  odsState.addAttribute(eltypeAttrName(odsState.name), eltype);
  odsState.addAttribute(fragAttrName(odsState.name), frag);
  odsState.addTypes(res);
}

void WMMALoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value stride, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layout, ::mlir::NVVM::MMATypesAttr eltype, ::mlir::NVVM::MMAFragAttr frag) {
  odsState.addOperands(ptr);
  odsState.addOperands(stride);
  odsState.addAttribute(mAttrName(odsState.name), m);
  odsState.addAttribute(nAttrName(odsState.name), n);
  odsState.addAttribute(kAttrName(odsState.name), k);
  odsState.addAttribute(layoutAttrName(odsState.name), layout);
  odsState.addAttribute(eltypeAttrName(odsState.name), eltype);
  odsState.addAttribute(fragAttrName(odsState.name), frag);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WMMALoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value stride, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layout, ::mlir::NVVM::MMATypes eltype, ::mlir::NVVM::MMAFrag frag) {
  odsState.addOperands(ptr);
  odsState.addOperands(stride);
  odsState.addAttribute(mAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), m));
  odsState.addAttribute(nAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), n));
  odsState.addAttribute(kAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), k));
  odsState.addAttribute(layoutAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layout));
  odsState.addAttribute(eltypeAttrName(odsState.name), ::mlir::NVVM::MMATypesAttr::get(odsBuilder.getContext(), eltype));
  odsState.addAttribute(fragAttrName(odsState.name), ::mlir::NVVM::MMAFragAttr::get(odsBuilder.getContext(), frag));
  odsState.addTypes(res);
}

void WMMALoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value stride, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layout, ::mlir::NVVM::MMATypes eltype, ::mlir::NVVM::MMAFrag frag) {
  odsState.addOperands(ptr);
  odsState.addOperands(stride);
  odsState.addAttribute(mAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), m));
  odsState.addAttribute(nAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), n));
  odsState.addAttribute(kAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), k));
  odsState.addAttribute(layoutAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layout));
  odsState.addAttribute(eltypeAttrName(odsState.name), ::mlir::NVVM::MMATypesAttr::get(odsBuilder.getContext(), eltype));
  odsState.addAttribute(fragAttrName(odsState.name), ::mlir::NVVM::MMAFragAttr::get(odsBuilder.getContext(), frag));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WMMALoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WMMALoadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_eltype;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'eltype'");
    if (namedAttrIt->getName() == eltypeAttrName()) {
      tblgen_eltype = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_frag;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'frag'");
    if (namedAttrIt->getName() == fragAttrName()) {
      tblgen_frag = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_k;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'k'");
    if (namedAttrIt->getName() == kAttrName()) {
      tblgen_k = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_layout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'layout'");
    if (namedAttrIt->getName() == layoutAttrName()) {
      tblgen_layout = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_m;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'm'");
    if (namedAttrIt->getName() == mAttrName()) {
      tblgen_m = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_n;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'n'");
    if (namedAttrIt->getName() == nAttrName()) {
      tblgen_n = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps0(*this, tblgen_m, "m")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps0(*this, tblgen_n, "n")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps0(*this, tblgen_k, "k")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps2(*this, tblgen_layout, "layout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps6(*this, tblgen_eltype, "eltype")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps8(*this, tblgen_frag, "frag")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult WMMALoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult WMMALoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand ptrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> ptrOperands(ptrRawOperands);  ::llvm::SMLoc ptrOperandsLoc;
  (void)ptrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand strideRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> strideOperands(strideRawOperands);  ::llvm::SMLoc strideOperandsLoc;
  (void)strideOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> ptrTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(ptrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  strideOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(strideRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType ptr__res_functionType;
  if (parser.parseType(ptr__res_functionType))
    return ::mlir::failure();
  ptrTypes = ptr__res_functionType.getInputs();
  resTypes = ptr__res_functionType.getResults();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  result.addTypes(resTypes);
  if (parser.resolveOperands(ptrOperands, ptrTypes, ptrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(strideOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WMMALoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << ptr();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << stride();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(ptr().getType()), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMALoadOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMAMmaOp definitions
//===----------------------------------------------------------------------===//

WMMAMmaOpAdaptor::WMMAMmaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.wmma.mma", odsAttrs.getContext());
}

WMMAMmaOpAdaptor::WMMAMmaOpAdaptor(WMMAMmaOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange WMMAMmaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WMMAMmaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange WMMAMmaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WMMAMmaOpAdaptor::args() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr WMMAMmaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr WMMAMmaOpAdaptor::mAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 5, odsAttrs.end() - 1, WMMAMmaOp::mAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t WMMAMmaOpAdaptor::m() {
  auto attr = mAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr WMMAMmaOpAdaptor::nAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 6, odsAttrs.end() - 0, WMMAMmaOp::nAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t WMMAMmaOpAdaptor::n() {
  auto attr = nAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr WMMAMmaOpAdaptor::kAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 4, WMMAMmaOp::kAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t WMMAMmaOpAdaptor::k() {
  auto attr = kAttr();
  return attr.getValue().getZExtValue();
}

::mlir::NVVM::MMALayoutAttr WMMAMmaOpAdaptor::layoutAAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 3, WMMAMmaOp::layoutAAttrName(*odsOpName)).cast<::mlir::NVVM::MMALayoutAttr>();
    return attr;
}

::mlir::NVVM::MMALayout WMMAMmaOpAdaptor::layoutA() {
  auto attr = layoutAAttr();
  return attr.getValue();
}

::mlir::NVVM::MMALayoutAttr WMMAMmaOpAdaptor::layoutBAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 4, odsAttrs.end() - 2, WMMAMmaOp::layoutBAttrName(*odsOpName)).cast<::mlir::NVVM::MMALayoutAttr>();
    return attr;
}

::mlir::NVVM::MMALayout WMMAMmaOpAdaptor::layoutB() {
  auto attr = layoutBAttr();
  return attr.getValue();
}

::mlir::NVVM::MMATypesAttr WMMAMmaOpAdaptor::eltypeAAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 6, WMMAMmaOp::eltypeAAttrName(*odsOpName)).cast<::mlir::NVVM::MMATypesAttr>();
    return attr;
}

::mlir::NVVM::MMATypes WMMAMmaOpAdaptor::eltypeA() {
  auto attr = eltypeAAttr();
  return attr.getValue();
}

::mlir::NVVM::MMATypesAttr WMMAMmaOpAdaptor::eltypeBAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 5, WMMAMmaOp::eltypeBAttrName(*odsOpName)).cast<::mlir::NVVM::MMATypesAttr>();
    return attr;
}

::mlir::NVVM::MMATypes WMMAMmaOpAdaptor::eltypeB() {
  auto attr = eltypeBAttr();
  return attr.getValue();
}

::mlir::LogicalResult WMMAMmaOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_eltypeA;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.mma' op ""requires attribute 'eltypeA'");
    if (namedAttrIt->getName() == WMMAMmaOp::eltypeAAttrName(*odsOpName)) {
      tblgen_eltypeA = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_eltypeB;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.mma' op ""requires attribute 'eltypeB'");
    if (namedAttrIt->getName() == WMMAMmaOp::eltypeBAttrName(*odsOpName)) {
      tblgen_eltypeB = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_k;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.mma' op ""requires attribute 'k'");
    if (namedAttrIt->getName() == WMMAMmaOp::kAttrName(*odsOpName)) {
      tblgen_k = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_layoutA;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.mma' op ""requires attribute 'layoutA'");
    if (namedAttrIt->getName() == WMMAMmaOp::layoutAAttrName(*odsOpName)) {
      tblgen_layoutA = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_layoutB;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.mma' op ""requires attribute 'layoutB'");
    if (namedAttrIt->getName() == WMMAMmaOp::layoutBAttrName(*odsOpName)) {
      tblgen_layoutB = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_m;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.mma' op ""requires attribute 'm'");
    if (namedAttrIt->getName() == WMMAMmaOp::mAttrName(*odsOpName)) {
      tblgen_m = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_n;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.mma' op ""requires attribute 'n'");
    if (namedAttrIt->getName() == WMMAMmaOp::nAttrName(*odsOpName)) {
      tblgen_n = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_m && !(((tblgen_m.isa<::mlir::IntegerAttr>())) && ((tblgen_m.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvvm.wmma.mma' op ""attribute 'm' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_n && !(((tblgen_n.isa<::mlir::IntegerAttr>())) && ((tblgen_n.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvvm.wmma.mma' op ""attribute 'n' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_k && !(((tblgen_k.isa<::mlir::IntegerAttr>())) && ((tblgen_k.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvvm.wmma.mma' op ""attribute 'k' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_layoutA && !((tblgen_layoutA.isa<::mlir::NVVM::MMALayoutAttr>())))
    return emitError(loc, "'nvvm.wmma.mma' op ""attribute 'layoutA' failed to satisfy constraint: NVVM MMA layout");

  if (tblgen_layoutB && !((tblgen_layoutB.isa<::mlir::NVVM::MMALayoutAttr>())))
    return emitError(loc, "'nvvm.wmma.mma' op ""attribute 'layoutB' failed to satisfy constraint: NVVM MMA layout");

  if (tblgen_eltypeA && !((tblgen_eltypeA.isa<::mlir::NVVM::MMATypesAttr>())))
    return emitError(loc, "'nvvm.wmma.mma' op ""attribute 'eltypeA' failed to satisfy constraint: NVVM MMA types");

  if (tblgen_eltypeB && !((tblgen_eltypeB.isa<::mlir::NVVM::MMATypesAttr>())))
    return emitError(loc, "'nvvm.wmma.mma' op ""attribute 'eltypeB' failed to satisfy constraint: NVVM MMA types");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WMMAMmaOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range WMMAMmaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WMMAMmaOp::args() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange WMMAMmaOp::argsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WMMAMmaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WMMAMmaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WMMAMmaOp::res() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr WMMAMmaOp::mAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 5, (*this)->getAttrs().end() - 1, mAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t WMMAMmaOp::m() {
  auto attr = mAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr WMMAMmaOp::nAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 6, (*this)->getAttrs().end() - 0, nAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t WMMAMmaOp::n() {
  auto attr = nAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr WMMAMmaOp::kAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 4, kAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t WMMAMmaOp::k() {
  auto attr = kAttr();
  return attr.getValue().getZExtValue();
}

::mlir::NVVM::MMALayoutAttr WMMAMmaOp::layoutAAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 3, layoutAAttrName()).cast<::mlir::NVVM::MMALayoutAttr>();
}

::mlir::NVVM::MMALayout WMMAMmaOp::layoutA() {
  auto attr = layoutAAttr();
  return attr.getValue();
}

::mlir::NVVM::MMALayoutAttr WMMAMmaOp::layoutBAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 4, (*this)->getAttrs().end() - 2, layoutBAttrName()).cast<::mlir::NVVM::MMALayoutAttr>();
}

::mlir::NVVM::MMALayout WMMAMmaOp::layoutB() {
  auto attr = layoutBAttr();
  return attr.getValue();
}

::mlir::NVVM::MMATypesAttr WMMAMmaOp::eltypeAAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 6, eltypeAAttrName()).cast<::mlir::NVVM::MMATypesAttr>();
}

::mlir::NVVM::MMATypes WMMAMmaOp::eltypeA() {
  auto attr = eltypeAAttr();
  return attr.getValue();
}

::mlir::NVVM::MMATypesAttr WMMAMmaOp::eltypeBAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 5, eltypeBAttrName()).cast<::mlir::NVVM::MMATypesAttr>();
}

::mlir::NVVM::MMATypes WMMAMmaOp::eltypeB() {
  auto attr = eltypeBAttr();
  return attr.getValue();
}

void WMMAMmaOp::mAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(mAttrName(), attr);
}

void WMMAMmaOp::nAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(nAttrName(), attr);
}

void WMMAMmaOp::kAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(kAttrName(), attr);
}

void WMMAMmaOp::layoutAAttr(::mlir::NVVM::MMALayoutAttr attr) {
  (*this)->setAttr(layoutAAttrName(), attr);
}

void WMMAMmaOp::layoutBAttr(::mlir::NVVM::MMALayoutAttr attr) {
  (*this)->setAttr(layoutBAttrName(), attr);
}

void WMMAMmaOp::eltypeAAttr(::mlir::NVVM::MMATypesAttr attr) {
  (*this)->setAttr(eltypeAAttrName(), attr);
}

void WMMAMmaOp::eltypeBAttr(::mlir::NVVM::MMATypesAttr attr) {
  (*this)->setAttr(eltypeBAttrName(), attr);
}

void WMMAMmaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layoutA, ::mlir::NVVM::MMALayoutAttr layoutB, ::mlir::NVVM::MMATypesAttr eltypeA, ::mlir::NVVM::MMATypesAttr eltypeB, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(mAttrName(odsState.name), m);
  odsState.addAttribute(nAttrName(odsState.name), n);
  odsState.addAttribute(kAttrName(odsState.name), k);
  odsState.addAttribute(layoutAAttrName(odsState.name), layoutA);
  odsState.addAttribute(layoutBAttrName(odsState.name), layoutB);
  odsState.addAttribute(eltypeAAttrName(odsState.name), eltypeA);
  odsState.addAttribute(eltypeBAttrName(odsState.name), eltypeB);
  odsState.addTypes(res);
}

void WMMAMmaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layoutA, ::mlir::NVVM::MMALayoutAttr layoutB, ::mlir::NVVM::MMATypesAttr eltypeA, ::mlir::NVVM::MMATypesAttr eltypeB, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(mAttrName(odsState.name), m);
  odsState.addAttribute(nAttrName(odsState.name), n);
  odsState.addAttribute(kAttrName(odsState.name), k);
  odsState.addAttribute(layoutAAttrName(odsState.name), layoutA);
  odsState.addAttribute(layoutBAttrName(odsState.name), layoutB);
  odsState.addAttribute(eltypeAAttrName(odsState.name), eltypeA);
  odsState.addAttribute(eltypeBAttrName(odsState.name), eltypeB);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WMMAMmaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layoutA, ::mlir::NVVM::MMALayout layoutB, ::mlir::NVVM::MMATypes eltypeA, ::mlir::NVVM::MMATypes eltypeB, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(mAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), m));
  odsState.addAttribute(nAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), n));
  odsState.addAttribute(kAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), k));
  odsState.addAttribute(layoutAAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layoutA));
  odsState.addAttribute(layoutBAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layoutB));
  odsState.addAttribute(eltypeAAttrName(odsState.name), ::mlir::NVVM::MMATypesAttr::get(odsBuilder.getContext(), eltypeA));
  odsState.addAttribute(eltypeBAttrName(odsState.name), ::mlir::NVVM::MMATypesAttr::get(odsBuilder.getContext(), eltypeB));
  odsState.addTypes(res);
}

void WMMAMmaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layoutA, ::mlir::NVVM::MMALayout layoutB, ::mlir::NVVM::MMATypes eltypeA, ::mlir::NVVM::MMATypes eltypeB, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.addAttribute(mAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), m));
  odsState.addAttribute(nAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), n));
  odsState.addAttribute(kAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), k));
  odsState.addAttribute(layoutAAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layoutA));
  odsState.addAttribute(layoutBAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layoutB));
  odsState.addAttribute(eltypeAAttrName(odsState.name), ::mlir::NVVM::MMATypesAttr::get(odsBuilder.getContext(), eltypeA));
  odsState.addAttribute(eltypeBAttrName(odsState.name), ::mlir::NVVM::MMATypesAttr::get(odsBuilder.getContext(), eltypeB));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WMMAMmaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WMMAMmaOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_eltypeA;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'eltypeA'");
    if (namedAttrIt->getName() == eltypeAAttrName()) {
      tblgen_eltypeA = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_eltypeB;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'eltypeB'");
    if (namedAttrIt->getName() == eltypeBAttrName()) {
      tblgen_eltypeB = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_k;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'k'");
    if (namedAttrIt->getName() == kAttrName()) {
      tblgen_k = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_layoutA;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'layoutA'");
    if (namedAttrIt->getName() == layoutAAttrName()) {
      tblgen_layoutA = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_layoutB;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'layoutB'");
    if (namedAttrIt->getName() == layoutBAttrName()) {
      tblgen_layoutB = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_m;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'm'");
    if (namedAttrIt->getName() == mAttrName()) {
      tblgen_m = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_n;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'n'");
    if (namedAttrIt->getName() == nAttrName()) {
      tblgen_n = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps0(*this, tblgen_m, "m")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps0(*this, tblgen_n, "n")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps0(*this, tblgen_k, "k")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps2(*this, tblgen_layoutA, "layoutA")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps2(*this, tblgen_layoutB, "layoutB")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps6(*this, tblgen_eltypeA, "eltypeA")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps6(*this, tblgen_eltypeB, "eltypeB")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult WMMAMmaOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult WMMAMmaOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> argsTypes;
  ::llvm::ArrayRef<::mlir::Type> resTypes;

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType args__res_functionType;
  if (parser.parseType(args__res_functionType))
    return ::mlir::failure();
  argsTypes = args__res_functionType.getInputs();
  resTypes = args__res_functionType.getResults();
  result.addTypes(resTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WMMAMmaOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << args();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(args().getTypes(), ::llvm::ArrayRef<::mlir::Type>(res().getType()));
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMAMmaOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WMMAStoreOp definitions
//===----------------------------------------------------------------------===//

WMMAStoreOpAdaptor::WMMAStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.wmma.store", odsAttrs.getContext());
}

WMMAStoreOpAdaptor::WMMAStoreOpAdaptor(WMMAStoreOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange WMMAStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WMMAStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange WMMAStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WMMAStoreOpAdaptor::ptr() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange WMMAStoreOpAdaptor::args() {
  return getODSOperands(1);
}

::mlir::Value WMMAStoreOpAdaptor::stride() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr WMMAStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr WMMAStoreOpAdaptor::mAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 1, WMMAStoreOp::mAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t WMMAStoreOpAdaptor::m() {
  auto attr = mAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr WMMAStoreOpAdaptor::nAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 4, odsAttrs.end() - 0, WMMAStoreOp::nAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t WMMAStoreOpAdaptor::n() {
  auto attr = nAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr WMMAStoreOpAdaptor::kAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 3, WMMAStoreOp::kAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t WMMAStoreOpAdaptor::k() {
  auto attr = kAttr();
  return attr.getValue().getZExtValue();
}

::mlir::NVVM::MMALayoutAttr WMMAStoreOpAdaptor::layoutAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 2, WMMAStoreOp::layoutAttrName(*odsOpName)).cast<::mlir::NVVM::MMALayoutAttr>();
    return attr;
}

::mlir::NVVM::MMALayout WMMAStoreOpAdaptor::layout() {
  auto attr = layoutAttr();
  return attr.getValue();
}

::mlir::NVVM::MMATypesAttr WMMAStoreOpAdaptor::eltypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 4, WMMAStoreOp::eltypeAttrName(*odsOpName)).cast<::mlir::NVVM::MMATypesAttr>();
    return attr;
}

::mlir::NVVM::MMATypes WMMAStoreOpAdaptor::eltype() {
  auto attr = eltypeAttr();
  return attr.getValue();
}

::mlir::LogicalResult WMMAStoreOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_eltype;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.store' op ""requires attribute 'eltype'");
    if (namedAttrIt->getName() == WMMAStoreOp::eltypeAttrName(*odsOpName)) {
      tblgen_eltype = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_k;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.store' op ""requires attribute 'k'");
    if (namedAttrIt->getName() == WMMAStoreOp::kAttrName(*odsOpName)) {
      tblgen_k = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_layout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.store' op ""requires attribute 'layout'");
    if (namedAttrIt->getName() == WMMAStoreOp::layoutAttrName(*odsOpName)) {
      tblgen_layout = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_m;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.store' op ""requires attribute 'm'");
    if (namedAttrIt->getName() == WMMAStoreOp::mAttrName(*odsOpName)) {
      tblgen_m = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_n;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'nvvm.wmma.store' op ""requires attribute 'n'");
    if (namedAttrIt->getName() == WMMAStoreOp::nAttrName(*odsOpName)) {
      tblgen_n = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_m && !(((tblgen_m.isa<::mlir::IntegerAttr>())) && ((tblgen_m.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvvm.wmma.store' op ""attribute 'm' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_n && !(((tblgen_n.isa<::mlir::IntegerAttr>())) && ((tblgen_n.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvvm.wmma.store' op ""attribute 'n' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_k && !(((tblgen_k.isa<::mlir::IntegerAttr>())) && ((tblgen_k.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'nvvm.wmma.store' op ""attribute 'k' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_layout && !((tblgen_layout.isa<::mlir::NVVM::MMALayoutAttr>())))
    return emitError(loc, "'nvvm.wmma.store' op ""attribute 'layout' failed to satisfy constraint: NVVM MMA layout");

  if (tblgen_eltype && !((tblgen_eltype.isa<::mlir::NVVM::MMATypesAttr>())))
    return emitError(loc, "'nvvm.wmma.store' op ""attribute 'eltype' failed to satisfy constraint: NVVM MMA types");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WMMAStoreOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true, false};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range WMMAStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WMMAStoreOp::ptr() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range WMMAStoreOp::args() {
  return getODSOperands(1);
}

::mlir::Value WMMAStoreOp::stride() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange WMMAStoreOp::ptrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange WMMAStoreOp::argsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange WMMAStoreOp::strideMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> WMMAStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WMMAStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr WMMAStoreOp::mAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 1, mAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t WMMAStoreOp::m() {
  auto attr = mAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr WMMAStoreOp::nAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 4, (*this)->getAttrs().end() - 0, nAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t WMMAStoreOp::n() {
  auto attr = nAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr WMMAStoreOp::kAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 3, kAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t WMMAStoreOp::k() {
  auto attr = kAttr();
  return attr.getValue().getZExtValue();
}

::mlir::NVVM::MMALayoutAttr WMMAStoreOp::layoutAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 2, layoutAttrName()).cast<::mlir::NVVM::MMALayoutAttr>();
}

::mlir::NVVM::MMALayout WMMAStoreOp::layout() {
  auto attr = layoutAttr();
  return attr.getValue();
}

::mlir::NVVM::MMATypesAttr WMMAStoreOp::eltypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 4, eltypeAttrName()).cast<::mlir::NVVM::MMATypesAttr>();
}

::mlir::NVVM::MMATypes WMMAStoreOp::eltype() {
  auto attr = eltypeAttr();
  return attr.getValue();
}

void WMMAStoreOp::mAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(mAttrName(), attr);
}

void WMMAStoreOp::nAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(nAttrName(), attr);
}

void WMMAStoreOp::kAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(kAttrName(), attr);
}

void WMMAStoreOp::layoutAttr(::mlir::NVVM::MMALayoutAttr attr) {
  (*this)->setAttr(layoutAttrName(), attr);
}

void WMMAStoreOp::eltypeAttr(::mlir::NVVM::MMATypesAttr attr) {
  (*this)->setAttr(eltypeAttrName(), attr);
}

void WMMAStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layout, ::mlir::NVVM::MMATypesAttr eltype, ::mlir::ValueRange args, ::mlir::Value stride) {
  odsState.addOperands(ptr);
  odsState.addOperands(args);
  odsState.addOperands(stride);
  odsState.addAttribute(mAttrName(odsState.name), m);
  odsState.addAttribute(nAttrName(odsState.name), n);
  odsState.addAttribute(kAttrName(odsState.name), k);
  odsState.addAttribute(layoutAttrName(odsState.name), layout);
  odsState.addAttribute(eltypeAttrName(odsState.name), eltype);
}

void WMMAStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::NVVM::MMALayoutAttr layout, ::mlir::NVVM::MMATypesAttr eltype, ::mlir::ValueRange args, ::mlir::Value stride) {
  odsState.addOperands(ptr);
  odsState.addOperands(args);
  odsState.addOperands(stride);
  odsState.addAttribute(mAttrName(odsState.name), m);
  odsState.addAttribute(nAttrName(odsState.name), n);
  odsState.addAttribute(kAttrName(odsState.name), k);
  odsState.addAttribute(layoutAttrName(odsState.name), layout);
  odsState.addAttribute(eltypeAttrName(odsState.name), eltype);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WMMAStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layout, ::mlir::NVVM::MMATypes eltype, ::mlir::ValueRange args, ::mlir::Value stride) {
  odsState.addOperands(ptr);
  odsState.addOperands(args);
  odsState.addOperands(stride);
  odsState.addAttribute(mAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), m));
  odsState.addAttribute(nAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), n));
  odsState.addAttribute(kAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), k));
  odsState.addAttribute(layoutAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layout));
  odsState.addAttribute(eltypeAttrName(odsState.name), ::mlir::NVVM::MMATypesAttr::get(odsBuilder.getContext(), eltype));
}

void WMMAStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, uint32_t m, uint32_t n, uint32_t k, ::mlir::NVVM::MMALayout layout, ::mlir::NVVM::MMATypes eltype, ::mlir::ValueRange args, ::mlir::Value stride) {
  odsState.addOperands(ptr);
  odsState.addOperands(args);
  odsState.addOperands(stride);
  odsState.addAttribute(mAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), m));
  odsState.addAttribute(nAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), n));
  odsState.addAttribute(kAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), k));
  odsState.addAttribute(layoutAttrName(odsState.name), ::mlir::NVVM::MMALayoutAttr::get(odsBuilder.getContext(), layout));
  odsState.addAttribute(eltypeAttrName(odsState.name), ::mlir::NVVM::MMATypesAttr::get(odsBuilder.getContext(), eltype));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WMMAStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WMMAStoreOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_eltype;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'eltype'");
    if (namedAttrIt->getName() == eltypeAttrName()) {
      tblgen_eltype = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_k;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'k'");
    if (namedAttrIt->getName() == kAttrName()) {
      tblgen_k = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_layout;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'layout'");
    if (namedAttrIt->getName() == layoutAttrName()) {
      tblgen_layout = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_m;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'm'");
    if (namedAttrIt->getName() == mAttrName()) {
      tblgen_m = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_n;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'n'");
    if (namedAttrIt->getName() == nAttrName()) {
      tblgen_n = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps0(*this, tblgen_m, "m")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps0(*this, tblgen_n, "n")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps0(*this, tblgen_k, "k")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps2(*this, tblgen_layout, "layout")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_NVVMOps6(*this, tblgen_eltype, "eltype")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult WMMAStoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult WMMAStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand ptrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> ptrOperands(ptrRawOperands);  ::llvm::SMLoc ptrOperandsLoc;
  (void)ptrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand strideRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> strideOperands(strideRawOperands);  ::llvm::SMLoc strideOperandsLoc;
  (void)strideOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::mlir::Type ptrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> ptrTypes(ptrRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;

  ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(ptrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  strideOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(strideRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::LLVM::LLVMPointerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    ptrRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(ptrOperands, ptrTypes, ptrOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(strideOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WMMAStoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << ptr();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << stride();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << args();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = ptr().getType();
    if (auto validType = type.dyn_cast<::mlir::LLVM::LLVMPointerType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << args().getTypes();
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::WMMAStoreOp)

namespace mlir {
namespace NVVM {

//===----------------------------------------------------------------------===//
// ::mlir::NVVM::WarpSizeOp definitions
//===----------------------------------------------------------------------===//

WarpSizeOpAdaptor::WarpSizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("nvvm.read.ptx.sreg.warpsize", odsAttrs.getContext());
}

WarpSizeOpAdaptor::WarpSizeOpAdaptor(WarpSizeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange WarpSizeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WarpSizeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange WarpSizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr WarpSizeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult WarpSizeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WarpSizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range WarpSizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> WarpSizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WarpSizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value WarpSizeOp::res() {
  return *getODSResults(0).begin();
}

void WarpSizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void WarpSizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WarpSizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WarpSizeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_NVVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult WarpSizeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult WarpSizeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void WarpSizeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = res().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void WarpSizeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace NVVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::NVVM::WarpSizeOp)


#endif  // GET_OP_CLASSES

