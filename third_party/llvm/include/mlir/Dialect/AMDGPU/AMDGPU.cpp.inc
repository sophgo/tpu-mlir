/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::amdgpu::LDSBarrierOp,
::mlir::amdgpu::MFMAOp,
::mlir::amdgpu::RawBufferAtomicFaddOp,
::mlir::amdgpu::RawBufferLoadOp,
::mlir::amdgpu::RawBufferStoreOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace amdgpu {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AMDGPU0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isF32())) || ((type.isF64())) || ((type.isSignlessInteger(32))) || ((type.isSignlessInteger(64))) || (((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isF32()); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ((type.cast<::mlir::VectorType>().getNumElements()
                           == 2)))) || (((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isF16()); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ((type.cast<::mlir::VectorType>().getNumElements()
                           == 4)))) || (((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isBF16()); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 2)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 4))))) || (((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 4)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8))))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit float or 64-bit float or 32-bit signless integer or 64-bit signless integer or vector of 32-bit float values of length 2 or vector of 16-bit float values of length 4 or vector of bfloat16 type values of length 2/4 or vector of 8-bit signless integer values of length 4/8, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AMDGPU1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isF64())) || (((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isF32()); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 4)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 32))))) || (((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(32)); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 4)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 16)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 32))))) || (((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isF64()); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ((type.cast<::mlir::VectorType>().getNumElements()
                           == 4)))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 64-bit float or vector of 32-bit float values of length 4/16/32 or vector of 32-bit signless integer values of length 4/16/32 or vector of 64-bit float values of length 4, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AMDGPU2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isF32()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit float, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AMDGPU3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::MemRefType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be memref of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AMDGPU4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_AMDGPU5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isBF16())) || ((type.isF16())) || ((type.isF32())) || ((type.isSignlessInteger(32))) || ((type.isSignlessInteger(8))) || (((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isF32())) || ((elementType.isSignlessInteger(32))); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 2)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 4))))) || (((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isF16())) || ((elementType.isBF16())); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 2)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 4)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8))))) || (((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return (elementType.isSignlessInteger(8)); }(type.cast<::mlir::ShapedType>().getElementType()))) && ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && (((type.cast<::mlir::VectorType>().getNumElements()
                           == 2)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 4)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 8)) || ((type.cast<::mlir::VectorType>().getNumElements()
                           == 16))))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be bfloat16 type or 16-bit float or 32-bit float or 32-bit signless integer or 8-bit signless integer or vector of 32-bit float or 32-bit signless integer values of length 2/4 or vector of 16-bit float or bfloat16 type values of length 2/4/8 or vector of 8-bit signless integer values of length 2/4/8/16, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_AMDGPU0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_AMDGPU1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::amdgpu::MFMAPermBAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: The possible permutations of the lanes storing B available in an MFMA";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_AMDGPU2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_AMDGPU3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
  }
  return ::mlir::success();
}
} // namespace amdgpu
} // namespace mlir
namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::LDSBarrierOp definitions
//===----------------------------------------------------------------------===//

LDSBarrierOpAdaptor::LDSBarrierOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("amdgpu.lds_barrier", odsAttrs.getContext());
}

LDSBarrierOpAdaptor::LDSBarrierOpAdaptor(LDSBarrierOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LDSBarrierOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LDSBarrierOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LDSBarrierOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr LDSBarrierOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LDSBarrierOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LDSBarrierOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LDSBarrierOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> LDSBarrierOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LDSBarrierOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void LDSBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void LDSBarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LDSBarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LDSBarrierOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult LDSBarrierOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LDSBarrierOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void LDSBarrierOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace amdgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::amdgpu::LDSBarrierOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::MFMAOp definitions
//===----------------------------------------------------------------------===//

MFMAOpAdaptor::MFMAOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("amdgpu.mfma", odsAttrs.getContext());
}

MFMAOpAdaptor::MFMAOpAdaptor(MFMAOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MFMAOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MFMAOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MFMAOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MFMAOpAdaptor::getSourceA() {
  return *getODSOperands(0).begin();
}

::mlir::Value MFMAOpAdaptor::getSourceB() {
  return *getODSOperands(1).begin();
}

::mlir::Value MFMAOpAdaptor::getDestC() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr MFMAOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MFMAOpAdaptor::getMAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, MFMAOp::getMAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MFMAOpAdaptor::getM() {
  auto attr = getMAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MFMAOpAdaptor::getNAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, MFMAOp::getNAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MFMAOpAdaptor::getN() {
  auto attr = getNAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MFMAOpAdaptor::getKAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 2, MFMAOp::getKAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MFMAOpAdaptor::getK() {
  auto attr = getKAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MFMAOpAdaptor::getBlocksAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, MFMAOp::getBlocksAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MFMAOpAdaptor::getBlocks() {
  auto attr = getBlocksAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MFMAOpAdaptor::getCbszAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 3, MFMAOp::getCbszAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MFMAOpAdaptor::getCbsz() {
  auto attr = getCbszAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MFMAOpAdaptor::getAbidAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 4, MFMAOp::getAbidAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint32_t MFMAOpAdaptor::getAbid() {
  auto attr = getAbidAttr();
  return attr.getValue().getZExtValue();
}

::mlir::amdgpu::MFMAPermBAttr MFMAOpAdaptor::getBlgpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 4, MFMAOp::getBlgpAttrName(*odsOpName)).dyn_cast_or_null<::mlir::amdgpu::MFMAPermBAttr>();
  return attr;
}

::mlir::amdgpu::MFMAPermB MFMAOpAdaptor::getBlgp() {
  auto attr = getBlgpAttr();
  return attr.getValue();
}

::mlir::UnitAttr MFMAOpAdaptor::getReducePrecisionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 4, odsAttrs.end() - 0, MFMAOp::getReducePrecisionAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool MFMAOpAdaptor::getReducePrecision() {
  auto attr = getReducePrecisionAttr();
  return attr != nullptr;
}

::mlir::UnitAttr MFMAOpAdaptor::getNegateAAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 4, odsAttrs.end() - 0, MFMAOp::getNegateAAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool MFMAOpAdaptor::getNegateA() {
  auto attr = getNegateAAttr();
  return attr != nullptr;
}

::mlir::UnitAttr MFMAOpAdaptor::getNegateBAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 4, odsAttrs.end() - 0, MFMAOp::getNegateBAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool MFMAOpAdaptor::getNegateB() {
  auto attr = getNegateBAttr();
  return attr != nullptr;
}

::mlir::UnitAttr MFMAOpAdaptor::getNegateCAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 4, odsAttrs.end() - 0, MFMAOp::getNegateCAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool MFMAOpAdaptor::getNegateC() {
  auto attr = getNegateCAttr();
  return attr != nullptr;
}

::mlir::LogicalResult MFMAOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_blocks;
  ::mlir::Attribute tblgen_abid;
  ::mlir::Attribute tblgen_blgp;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'amdgpu.mfma' op ""requires attribute 'blocks'");
    if (namedAttrIt->getName() == MFMAOp::getBlocksAttrName(*odsOpName)) {
      tblgen_blocks = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == MFMAOp::getAbidAttrName(*odsOpName)) {
      tblgen_abid = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == MFMAOp::getBlgpAttrName(*odsOpName)) {
      tblgen_blgp = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_k;
  ::mlir::Attribute tblgen_cbsz;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'amdgpu.mfma' op ""requires attribute 'k'");
    if (namedAttrIt->getName() == MFMAOp::getKAttrName(*odsOpName)) {
      tblgen_k = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == MFMAOp::getCbszAttrName(*odsOpName)) {
      tblgen_cbsz = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_m;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'amdgpu.mfma' op ""requires attribute 'm'");
    if (namedAttrIt->getName() == MFMAOp::getMAttrName(*odsOpName)) {
      tblgen_m = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_n;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'amdgpu.mfma' op ""requires attribute 'n'");
    if (namedAttrIt->getName() == MFMAOp::getNAttrName(*odsOpName)) {
      tblgen_n = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_negateA;
  ::mlir::Attribute tblgen_negateB;
  ::mlir::Attribute tblgen_negateC;
  ::mlir::Attribute tblgen_reducePrecision;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == MFMAOp::getNegateAAttrName(*odsOpName)) {
      tblgen_negateA = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == MFMAOp::getNegateBAttrName(*odsOpName)) {
      tblgen_negateB = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == MFMAOp::getNegateCAttrName(*odsOpName)) {
      tblgen_negateC = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == MFMAOp::getReducePrecisionAttrName(*odsOpName)) {
      tblgen_reducePrecision = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_m && !(((tblgen_m.isa<::mlir::IntegerAttr>())) && ((tblgen_m.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'amdgpu.mfma' op ""attribute 'm' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_n && !(((tblgen_n.isa<::mlir::IntegerAttr>())) && ((tblgen_n.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'amdgpu.mfma' op ""attribute 'n' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_k && !(((tblgen_k.isa<::mlir::IntegerAttr>())) && ((tblgen_k.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'amdgpu.mfma' op ""attribute 'k' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_blocks && !(((tblgen_blocks.isa<::mlir::IntegerAttr>())) && ((tblgen_blocks.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'amdgpu.mfma' op ""attribute 'blocks' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_cbsz && !(((tblgen_cbsz.isa<::mlir::IntegerAttr>())) && ((tblgen_cbsz.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'amdgpu.mfma' op ""attribute 'cbsz' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_abid && !(((tblgen_abid.isa<::mlir::IntegerAttr>())) && ((tblgen_abid.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'amdgpu.mfma' op ""attribute 'abid' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_blgp && !((tblgen_blgp.isa<::mlir::amdgpu::MFMAPermBAttr>())))
    return emitError(loc, "'amdgpu.mfma' op ""attribute 'blgp' failed to satisfy constraint: The possible permutations of the lanes storing B available in an MFMA");

  if (tblgen_reducePrecision && !((tblgen_reducePrecision.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'amdgpu.mfma' op ""attribute 'reducePrecision' failed to satisfy constraint: unit attribute");

  if (tblgen_negateA && !((tblgen_negateA.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'amdgpu.mfma' op ""attribute 'negateA' failed to satisfy constraint: unit attribute");

  if (tblgen_negateB && !((tblgen_negateB.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'amdgpu.mfma' op ""attribute 'negateB' failed to satisfy constraint: unit attribute");

  if (tblgen_negateC && !((tblgen_negateC.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'amdgpu.mfma' op ""attribute 'negateC' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MFMAOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MFMAOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MFMAOp::getSourceA() {
  return *getODSOperands(0).begin();
}

::mlir::Value MFMAOp::getSourceB() {
  return *getODSOperands(1).begin();
}

::mlir::Value MFMAOp::getDestC() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange MFMAOp::getSourceAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MFMAOp::getSourceBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MFMAOp::getDestCMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MFMAOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MFMAOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MFMAOp::getDestD() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr MFMAOp::getMAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getMAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MFMAOp::getM() {
  auto attr = getMAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MFMAOp::getNAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getNAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MFMAOp::getN() {
  auto attr = getNAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MFMAOp::getKAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 2, getKAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MFMAOp::getK() {
  auto attr = getKAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MFMAOp::getBlocksAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getBlocksAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MFMAOp::getBlocks() {
  auto attr = getBlocksAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MFMAOp::getCbszAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 3, getCbszAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint32_t MFMAOp::getCbsz() {
  auto attr = getCbszAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MFMAOp::getAbidAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 4, getAbidAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint32_t MFMAOp::getAbid() {
  auto attr = getAbidAttr();
  return attr.getValue().getZExtValue();
}

::mlir::amdgpu::MFMAPermBAttr MFMAOp::getBlgpAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 4, getBlgpAttrName()).dyn_cast_or_null<::mlir::amdgpu::MFMAPermBAttr>();
}

::mlir::amdgpu::MFMAPermB MFMAOp::getBlgp() {
  auto attr = getBlgpAttr();
  return attr.getValue();
}

::mlir::UnitAttr MFMAOp::getReducePrecisionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 4, (*this)->getAttrs().end() - 0, getReducePrecisionAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool MFMAOp::getReducePrecision() {
  auto attr = getReducePrecisionAttr();
  return attr != nullptr;
}

::mlir::UnitAttr MFMAOp::getNegateAAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 4, (*this)->getAttrs().end() - 0, getNegateAAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool MFMAOp::getNegateA() {
  auto attr = getNegateAAttr();
  return attr != nullptr;
}

::mlir::UnitAttr MFMAOp::getNegateBAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 4, (*this)->getAttrs().end() - 0, getNegateBAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool MFMAOp::getNegateB() {
  auto attr = getNegateBAttr();
  return attr != nullptr;
}

::mlir::UnitAttr MFMAOp::getNegateCAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 4, (*this)->getAttrs().end() - 0, getNegateCAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool MFMAOp::getNegateC() {
  auto attr = getNegateCAttr();
  return attr != nullptr;
}

void MFMAOp::setMAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getMAttrName(), attr);
}

void MFMAOp::setNAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getNAttrName(), attr);
}

void MFMAOp::setKAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getKAttrName(), attr);
}

void MFMAOp::setBlocksAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getBlocksAttrName(), attr);
}

void MFMAOp::setCbszAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getCbszAttrName(), attr);
}

void MFMAOp::setAbidAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAbidAttrName(), attr);
}

void MFMAOp::setBlgpAttr(::mlir::amdgpu::MFMAPermBAttr attr) {
  (*this)->setAttr(getBlgpAttrName(), attr);
}

void MFMAOp::setReducePrecisionAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getReducePrecisionAttrName(), attr);
}

void MFMAOp::setNegateAAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNegateAAttrName(), attr);
}

void MFMAOp::setNegateBAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNegateBAttrName(), attr);
}

void MFMAOp::setNegateCAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNegateCAttrName(), attr);
}

::mlir::Attribute MFMAOp::removeReducePrecisionAttr() {
  return (*this)->removeAttr(getReducePrecisionAttrName());
}

::mlir::Attribute MFMAOp::removeNegateAAttr() {
  return (*this)->removeAttr(getNegateAAttrName());
}

::mlir::Attribute MFMAOp::removeNegateBAttr() {
  return (*this)->removeAttr(getNegateBAttrName());
}

::mlir::Attribute MFMAOp::removeNegateCAttr() {
  return (*this)->removeAttr(getNegateCAttrName());
}

void MFMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type destD, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::IntegerAttr blocks, ::mlir::Value sourceA, ::mlir::Value sourceB, ::mlir::Value destC, ::mlir::IntegerAttr cbsz, ::mlir::IntegerAttr abid, ::mlir::amdgpu::MFMAPermBAttr blgp, /*optional*/::mlir::UnitAttr reducePrecision, /*optional*/::mlir::UnitAttr negateA, /*optional*/::mlir::UnitAttr negateB, /*optional*/::mlir::UnitAttr negateC) {
  odsState.addOperands(sourceA);
  odsState.addOperands(sourceB);
  odsState.addOperands(destC);
  odsState.addAttribute(getMAttrName(odsState.name), m);
  odsState.addAttribute(getNAttrName(odsState.name), n);
  odsState.addAttribute(getKAttrName(odsState.name), k);
  odsState.addAttribute(getBlocksAttrName(odsState.name), blocks);
  if (cbsz) {
  odsState.addAttribute(getCbszAttrName(odsState.name), cbsz);
  }
  if (abid) {
  odsState.addAttribute(getAbidAttrName(odsState.name), abid);
  }
  if (blgp) {
  odsState.addAttribute(getBlgpAttrName(odsState.name), blgp);
  }
  if (reducePrecision) {
  odsState.addAttribute(getReducePrecisionAttrName(odsState.name), reducePrecision);
  }
  if (negateA) {
  odsState.addAttribute(getNegateAAttrName(odsState.name), negateA);
  }
  if (negateB) {
  odsState.addAttribute(getNegateBAttrName(odsState.name), negateB);
  }
  if (negateC) {
  odsState.addAttribute(getNegateCAttrName(odsState.name), negateC);
  }
  odsState.addTypes(destD);
}

void MFMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::IntegerAttr blocks, ::mlir::Value sourceA, ::mlir::Value sourceB, ::mlir::Value destC, ::mlir::IntegerAttr cbsz, ::mlir::IntegerAttr abid, ::mlir::amdgpu::MFMAPermBAttr blgp, /*optional*/::mlir::UnitAttr reducePrecision, /*optional*/::mlir::UnitAttr negateA, /*optional*/::mlir::UnitAttr negateB, /*optional*/::mlir::UnitAttr negateC) {
  odsState.addOperands(sourceA);
  odsState.addOperands(sourceB);
  odsState.addOperands(destC);
  odsState.addAttribute(getMAttrName(odsState.name), m);
  odsState.addAttribute(getNAttrName(odsState.name), n);
  odsState.addAttribute(getKAttrName(odsState.name), k);
  odsState.addAttribute(getBlocksAttrName(odsState.name), blocks);
  if (cbsz) {
  odsState.addAttribute(getCbszAttrName(odsState.name), cbsz);
  }
  if (abid) {
  odsState.addAttribute(getAbidAttrName(odsState.name), abid);
  }
  if (blgp) {
  odsState.addAttribute(getBlgpAttrName(odsState.name), blgp);
  }
  if (reducePrecision) {
  odsState.addAttribute(getReducePrecisionAttrName(odsState.name), reducePrecision);
  }
  if (negateA) {
  odsState.addAttribute(getNegateAAttrName(odsState.name), negateA);
  }
  if (negateB) {
  odsState.addAttribute(getNegateBAttrName(odsState.name), negateB);
  }
  if (negateC) {
  odsState.addAttribute(getNegateCAttrName(odsState.name), negateC);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MFMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type destD, uint32_t m, uint32_t n, uint32_t k, uint32_t blocks, ::mlir::Value sourceA, ::mlir::Value sourceB, ::mlir::Value destC, uint32_t cbsz, uint32_t abid, ::mlir::amdgpu::MFMAPermB blgp, /*optional*/bool reducePrecision, /*optional*/bool negateA, /*optional*/bool negateB, /*optional*/bool negateC) {
  odsState.addOperands(sourceA);
  odsState.addOperands(sourceB);
  odsState.addOperands(destC);
  odsState.addAttribute(getMAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), m));
  odsState.addAttribute(getNAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), n));
  odsState.addAttribute(getKAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), k));
  odsState.addAttribute(getBlocksAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), blocks));
  odsState.addAttribute(getCbszAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), cbsz));
  odsState.addAttribute(getAbidAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), abid));
  odsState.addAttribute(getBlgpAttrName(odsState.name), ::mlir::amdgpu::MFMAPermBAttr::get(odsBuilder.getContext(), blgp));
  if (reducePrecision) {
  odsState.addAttribute(getReducePrecisionAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (negateA) {
  odsState.addAttribute(getNegateAAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (negateB) {
  odsState.addAttribute(getNegateBAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (negateC) {
  odsState.addAttribute(getNegateCAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  odsState.addTypes(destD);
}

void MFMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t m, uint32_t n, uint32_t k, uint32_t blocks, ::mlir::Value sourceA, ::mlir::Value sourceB, ::mlir::Value destC, uint32_t cbsz, uint32_t abid, ::mlir::amdgpu::MFMAPermB blgp, /*optional*/bool reducePrecision, /*optional*/bool negateA, /*optional*/bool negateB, /*optional*/bool negateC) {
  odsState.addOperands(sourceA);
  odsState.addOperands(sourceB);
  odsState.addOperands(destC);
  odsState.addAttribute(getMAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), m));
  odsState.addAttribute(getNAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), n));
  odsState.addAttribute(getKAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), k));
  odsState.addAttribute(getBlocksAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), blocks));
  odsState.addAttribute(getCbszAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), cbsz));
  odsState.addAttribute(getAbidAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), abid));
  odsState.addAttribute(getBlgpAttrName(odsState.name), ::mlir::amdgpu::MFMAPermBAttr::get(odsBuilder.getContext(), blgp));
  if (reducePrecision) {
  odsState.addAttribute(getReducePrecisionAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (negateA) {
  odsState.addAttribute(getNegateAAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (negateB) {
  odsState.addAttribute(getNegateBAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (negateC) {
  odsState.addAttribute(getNegateCAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MFMAOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MFMAOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[3])) {
     attributes.append(attrNames[3], odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), 0));
  }
  if (!attributes.get(attrNames[0])) {
     attributes.append(attrNames[0], odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), 0));
  }
  if (!attributes.get(attrNames[1])) {
     attributes.append(attrNames[1], ::mlir::amdgpu::MFMAPermBAttr::get(odsBuilder.getContext(), ::mlir::amdgpu::MFMAPermB::none));
  }
}

::mlir::LogicalResult MFMAOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_blocks;
  ::mlir::Attribute tblgen_abid;
  ::mlir::Attribute tblgen_blgp;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'blocks'");
    if (namedAttrIt->getName() == getBlocksAttrName()) {
      tblgen_blocks = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAbidAttrName()) {
      tblgen_abid = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getBlgpAttrName()) {
      tblgen_blgp = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_k;
  ::mlir::Attribute tblgen_cbsz;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'k'");
    if (namedAttrIt->getName() == getKAttrName()) {
      tblgen_k = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCbszAttrName()) {
      tblgen_cbsz = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_m;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'm'");
    if (namedAttrIt->getName() == getMAttrName()) {
      tblgen_m = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_n;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'n'");
    if (namedAttrIt->getName() == getNAttrName()) {
      tblgen_n = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_negateA;
  ::mlir::Attribute tblgen_negateB;
  ::mlir::Attribute tblgen_negateC;
  ::mlir::Attribute tblgen_reducePrecision;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getNegateAAttrName()) {
      tblgen_negateA = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNegateBAttrName()) {
      tblgen_negateB = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNegateCAttrName()) {
      tblgen_negateC = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getReducePrecisionAttrName()) {
      tblgen_reducePrecision = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_m, "m")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_n, "n")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_k, "k")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_blocks, "blocks")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_cbsz, "cbsz")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_abid, "abid")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU1(*this, tblgen_blgp, "blgp")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU2(*this, tblgen_reducePrecision, "reducePrecision")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU2(*this, tblgen_negateA, "negateA")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU2(*this, tblgen_negateB, "negateB")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU2(*this, tblgen_negateC, "negateC")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((((*this->getODSOperands(0).begin()).getType()) == ((*this->getODSOperands(1).begin()).getType()) && ((*this->getODSOperands(1).begin()).getType()) == ((*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that all of {sourceA, sourceB} have same type");
  if (!((((*this->getODSOperands(2).begin()).getType()) == ((*this->getODSResults(0).begin()).getType()) && ((*this->getODSResults(0).begin()).getType()) == ((*this->getODSOperands(2).begin()).getType()))))
    return emitOpError("failed to verify that all of {destC, destD} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult MFMAOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult MFMAOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceARawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceAOperands(sourceARawOperands);  ::llvm::SMLoc sourceAOperandsLoc;
  (void)sourceAOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand sourceBRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceBOperands(sourceBRawOperands);  ::llvm::SMLoc sourceBOperandsLoc;
  (void)sourceBOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand destCRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> destCOperands(destCRawOperands);  ::llvm::SMLoc destCOperandsLoc;
  (void)destCOperandsLoc;
  ::mlir::amdgpu::MFMAPermBAttr blgpAttr;
  ::mlir::Type sourceARawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sourceATypes(sourceARawTypes);
  ::mlir::Type destCRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> destCTypes(destCRawTypes);

  sourceAOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceARawOperands[0]))
    return ::mlir::failure();
  if (parser.parseStar())
    return ::mlir::failure();

  sourceBOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceBRawOperands[0]))
    return ::mlir::failure();
  if (parser.parsePlus())
    return ::mlir::failure();

  destCOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(destCRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseKeyword("blgp"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(blgpAttr, ::mlir::Type{}, "blgp",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceARawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    destCRawTypes[0] = type;
  }
  result.addTypes(destCTypes[0]);
  if (parser.resolveOperands(sourceAOperands, sourceATypes, sourceAOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sourceBOperands, sourceATypes[0], sourceBOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(destCOperands, destCTypes, destCOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MFMAOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSourceA();
  _odsPrinter << ' ' << "*";
  _odsPrinter << ' ';
  _odsPrinter << getSourceB();
  _odsPrinter << ' ' << "+";
  _odsPrinter << ' ';
  _odsPrinter << getDestC();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"blgp"});
  _odsPrinter << ' ' << "blgp";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getBlgpAttr());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSourceA().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getDestC().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void MFMAOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace amdgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::amdgpu::MFMAOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferAtomicFaddOp definitions
//===----------------------------------------------------------------------===//

RawBufferAtomicFaddOpAdaptor::RawBufferAtomicFaddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("amdgpu.raw_buffer_atomic_fadd", odsAttrs.getContext());
}

RawBufferAtomicFaddOpAdaptor::RawBufferAtomicFaddOpAdaptor(RawBufferAtomicFaddOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange RawBufferAtomicFaddOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RawBufferAtomicFaddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, RawBufferAtomicFaddOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange RawBufferAtomicFaddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RawBufferAtomicFaddOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value RawBufferAtomicFaddOpAdaptor::getMemref() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange RawBufferAtomicFaddOpAdaptor::getIndices() {
  return getODSOperands(2);
}

::mlir::Value RawBufferAtomicFaddOpAdaptor::getSgprOffset() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr RawBufferAtomicFaddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr RawBufferAtomicFaddOpAdaptor::getBoundsCheckAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, RawBufferAtomicFaddOp::getBoundsCheckAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool RawBufferAtomicFaddOpAdaptor::getBoundsCheck() {
  auto attr = getBoundsCheckAttr();
  return attr.getValue();
}

::mlir::IntegerAttr RawBufferAtomicFaddOpAdaptor::getIndexOffsetAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, RawBufferAtomicFaddOp::getIndexOffsetAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::llvm::Optional<uint32_t> RawBufferAtomicFaddOpAdaptor::getIndexOffset() {
  auto attr = getIndexOffsetAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult RawBufferAtomicFaddOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_boundsCheck;
  ::mlir::Attribute tblgen_indexOffset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'amdgpu.raw_buffer_atomic_fadd' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == RawBufferAtomicFaddOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == RawBufferAtomicFaddOp::getBoundsCheckAttrName(*odsOpName)) {
      tblgen_boundsCheck = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == RawBufferAtomicFaddOp::getIndexOffsetAttrName(*odsOpName)) {
      tblgen_indexOffset = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitError(loc, "'amdgpu.raw_buffer_atomic_fadd' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (tblgen_boundsCheck && !((tblgen_boundsCheck.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'amdgpu.raw_buffer_atomic_fadd' op ""attribute 'boundsCheck' failed to satisfy constraint: bool attribute");

  if (tblgen_indexOffset && !(((tblgen_indexOffset.isa<::mlir::IntegerAttr>())) && ((tblgen_indexOffset.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'amdgpu.raw_buffer_atomic_fadd' op ""attribute 'indexOffset' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RawBufferAtomicFaddOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range RawBufferAtomicFaddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::FloatType> RawBufferAtomicFaddOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::MemRefType> RawBufferAtomicFaddOp::getMemref() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range RawBufferAtomicFaddOp::getIndices() {
  return getODSOperands(2);
}

::mlir::TypedValue<::mlir::IntegerType> RawBufferAtomicFaddOp::getSgprOffset() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange RawBufferAtomicFaddOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferAtomicFaddOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferAtomicFaddOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferAtomicFaddOp::getSgprOffsetMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> RawBufferAtomicFaddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RawBufferAtomicFaddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::BoolAttr RawBufferAtomicFaddOp::getBoundsCheckAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getBoundsCheckAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool RawBufferAtomicFaddOp::getBoundsCheck() {
  auto attr = getBoundsCheckAttr();
  return attr.getValue();
}

::mlir::IntegerAttr RawBufferAtomicFaddOp::getIndexOffsetAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getIndexOffsetAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint32_t> RawBufferAtomicFaddOp::getIndexOffset() {
  auto attr = getIndexOffsetAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void RawBufferAtomicFaddOp::setBoundsCheckAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getBoundsCheckAttrName(), attr);
}

void RawBufferAtomicFaddOp::setIndexOffsetAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexOffsetAttrName(), attr);
}

::mlir::Attribute RawBufferAtomicFaddOp::removeIndexOffsetAttr() {
  return (*this)->removeAttr(getIndexOffsetAttrName());
}

void RawBufferAtomicFaddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  if (boundsCheck) {
  odsState.addAttribute(getBoundsCheckAttrName(odsState.name), boundsCheck);
  }
  if (indexOffset) {
  odsState.addAttribute(getIndexOffsetAttrName(odsState.name), indexOffset);
  }
}

void RawBufferAtomicFaddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  if (boundsCheck) {
  odsState.addAttribute(getBoundsCheckAttrName(odsState.name), boundsCheck);
  }
  if (indexOffset) {
  odsState.addAttribute(getIndexOffsetAttrName(odsState.name), indexOffset);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RawBufferAtomicFaddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(getBoundsCheckAttrName(odsState.name), odsBuilder.getBoolAttr(boundsCheck));
  if (indexOffset) {
  odsState.addAttribute(getIndexOffsetAttrName(odsState.name), indexOffset);
  }
}

void RawBufferAtomicFaddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(getBoundsCheckAttrName(odsState.name), odsBuilder.getBoolAttr(boundsCheck));
  if (indexOffset) {
  odsState.addAttribute(getIndexOffsetAttrName(odsState.name), indexOffset);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RawBufferAtomicFaddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void RawBufferAtomicFaddOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
     attributes.append(attrNames[0], odsBuilder.getBoolAttr(true));
  }
}

::mlir::LogicalResult RawBufferAtomicFaddOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_boundsCheck;
  ::mlir::Attribute tblgen_indexOffset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBoundsCheckAttrName()) {
      tblgen_boundsCheck = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getIndexOffsetAttrName()) {
      tblgen_indexOffset = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU3(*this, tblgen_boundsCheck, "boundsCheck")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_indexOffset, "indexOffset")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!(((getElementTypeOrSelf((*this->getODSOperands(0).begin()))) == (getElementTypeOrSelf((*this->getODSOperands(1).begin()))) && (getElementTypeOrSelf((*this->getODSOperands(1).begin()))) == (getElementTypeOrSelf((*this->getODSOperands(0).begin()))))))
    return emitOpError("failed to verify that all of {value, memref} have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult RawBufferAtomicFaddOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult RawBufferAtomicFaddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> sgprOffsetOperands;
  ::llvm::SMLoc sgprOffsetOperandsLoc;
  (void)sgprOffsetOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> indicesTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("sgprOffset"))) {

  {
    sgprOffsetOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      sgprOffsetOperands.push_back(operand);
    }
  }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::FloatType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseTypeList(indicesTypes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(indicesOperands.size()), static_cast<int32_t>(sgprOffsetOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, indicesTypes, indicesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sgprOffsetOperands, odsBuildableType0, sgprOffsetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RawBufferAtomicFaddOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getMemref();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  if (getSgprOffset()) {
  _odsPrinter << ' ' << "sgprOffset";
  _odsPrinter << ' ';
  if (::mlir::Value value = getSgprOffset())
    _odsPrinter << value;
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::FloatType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getMemref().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getIndices().getTypes();
}

void RawBufferAtomicFaddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace amdgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferAtomicFaddOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferLoadOp definitions
//===----------------------------------------------------------------------===//

RawBufferLoadOpAdaptor::RawBufferLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("amdgpu.raw_buffer_load", odsAttrs.getContext());
}

RawBufferLoadOpAdaptor::RawBufferLoadOpAdaptor(RawBufferLoadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange RawBufferLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RawBufferLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, RawBufferLoadOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange RawBufferLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RawBufferLoadOpAdaptor::getMemref() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange RawBufferLoadOpAdaptor::getIndices() {
  return getODSOperands(1);
}

::mlir::Value RawBufferLoadOpAdaptor::getSgprOffset() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr RawBufferLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr RawBufferLoadOpAdaptor::getBoundsCheckAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, RawBufferLoadOp::getBoundsCheckAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool RawBufferLoadOpAdaptor::getBoundsCheck() {
  auto attr = getBoundsCheckAttr();
  return attr.getValue();
}

::mlir::IntegerAttr RawBufferLoadOpAdaptor::getIndexOffsetAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, RawBufferLoadOp::getIndexOffsetAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::llvm::Optional<uint32_t> RawBufferLoadOpAdaptor::getIndexOffset() {
  auto attr = getIndexOffsetAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult RawBufferLoadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_boundsCheck;
  ::mlir::Attribute tblgen_indexOffset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'amdgpu.raw_buffer_load' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == RawBufferLoadOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == RawBufferLoadOp::getBoundsCheckAttrName(*odsOpName)) {
      tblgen_boundsCheck = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == RawBufferLoadOp::getIndexOffsetAttrName(*odsOpName)) {
      tblgen_indexOffset = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'amdgpu.raw_buffer_load' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_boundsCheck && !((tblgen_boundsCheck.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'amdgpu.raw_buffer_load' op ""attribute 'boundsCheck' failed to satisfy constraint: bool attribute");

  if (tblgen_indexOffset && !(((tblgen_indexOffset.isa<::mlir::IntegerAttr>())) && ((tblgen_indexOffset.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'amdgpu.raw_buffer_load' op ""attribute 'indexOffset' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RawBufferLoadOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range RawBufferLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::MemRefType> RawBufferLoadOp::getMemref() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range RawBufferLoadOp::getIndices() {
  return getODSOperands(1);
}

::mlir::TypedValue<::mlir::IntegerType> RawBufferLoadOp::getSgprOffset() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange RawBufferLoadOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferLoadOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferLoadOp::getSgprOffsetMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> RawBufferLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RawBufferLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RawBufferLoadOp::getValue() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr RawBufferLoadOp::getBoundsCheckAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getBoundsCheckAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool RawBufferLoadOp::getBoundsCheck() {
  auto attr = getBoundsCheckAttr();
  return attr.getValue();
}

::mlir::IntegerAttr RawBufferLoadOp::getIndexOffsetAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getIndexOffsetAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint32_t> RawBufferLoadOp::getIndexOffset() {
  auto attr = getIndexOffsetAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void RawBufferLoadOp::setBoundsCheckAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getBoundsCheckAttrName(), attr);
}

void RawBufferLoadOp::setIndexOffsetAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexOffsetAttrName(), attr);
}

::mlir::Attribute RawBufferLoadOp::removeIndexOffsetAttr() {
  return (*this)->removeAttr(getIndexOffsetAttrName());
}

void RawBufferLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  if (boundsCheck) {
  odsState.addAttribute(getBoundsCheckAttrName(odsState.name), boundsCheck);
  }
  if (indexOffset) {
  odsState.addAttribute(getIndexOffsetAttrName(odsState.name), indexOffset);
  }
  odsState.addTypes(value);
}

void RawBufferLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  if (boundsCheck) {
  odsState.addAttribute(getBoundsCheckAttrName(odsState.name), boundsCheck);
  }
  if (indexOffset) {
  odsState.addAttribute(getIndexOffsetAttrName(odsState.name), indexOffset);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RawBufferLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(getBoundsCheckAttrName(odsState.name), odsBuilder.getBoolAttr(boundsCheck));
  if (indexOffset) {
  odsState.addAttribute(getIndexOffsetAttrName(odsState.name), indexOffset);
  }
  odsState.addTypes(value);
}

void RawBufferLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(getBoundsCheckAttrName(odsState.name), odsBuilder.getBoolAttr(boundsCheck));
  if (indexOffset) {
  odsState.addAttribute(getIndexOffsetAttrName(odsState.name), indexOffset);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RawBufferLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void RawBufferLoadOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
     attributes.append(attrNames[0], odsBuilder.getBoolAttr(true));
  }
}

::mlir::LogicalResult RawBufferLoadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_boundsCheck;
  ::mlir::Attribute tblgen_indexOffset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBoundsCheckAttrName()) {
      tblgen_boundsCheck = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getIndexOffsetAttrName()) {
      tblgen_indexOffset = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU3(*this, tblgen_boundsCheck, "boundsCheck")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_indexOffset, "indexOffset")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!(((getElementTypeOrSelf((*this->getODSResults(0).begin()))) == (getElementTypeOrSelf((*this->getODSOperands(0).begin()))) && (getElementTypeOrSelf((*this->getODSOperands(0).begin()))) == (getElementTypeOrSelf((*this->getODSResults(0).begin()))))))
    return emitOpError("failed to verify that all of {value, memref} have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult RawBufferLoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult RawBufferLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> sgprOffsetOperands;
  ::llvm::SMLoc sgprOffsetOperandsLoc;
  (void)sgprOffsetOperandsLoc;
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> indicesTypes;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("sgprOffset"))) {

  {
    sgprOffsetOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      sgprOffsetOperands.push_back(operand);
    }
  }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseTypeList(indicesTypes))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(indicesOperands.size()), static_cast<int32_t>(sgprOffsetOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  result.addTypes(valueTypes);
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, indicesTypes, indicesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sgprOffsetOperands, odsBuildableType0, sgprOffsetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RawBufferLoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  _odsPrinter << ' ';
  _odsPrinter << getMemref();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  if (getSgprOffset()) {
  _odsPrinter << ' ' << "sgprOffset";
  _odsPrinter << ' ';
  if (::mlir::Value value = getSgprOffset())
    _odsPrinter << value;
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMemref().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getIndices().getTypes();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void RawBufferLoadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace amdgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferLoadOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferStoreOp definitions
//===----------------------------------------------------------------------===//

RawBufferStoreOpAdaptor::RawBufferStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("amdgpu.raw_buffer_store", odsAttrs.getContext());
}

RawBufferStoreOpAdaptor::RawBufferStoreOpAdaptor(RawBufferStoreOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange RawBufferStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> RawBufferStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, RawBufferStoreOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange RawBufferStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RawBufferStoreOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value RawBufferStoreOpAdaptor::getMemref() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange RawBufferStoreOpAdaptor::getIndices() {
  return getODSOperands(2);
}

::mlir::Value RawBufferStoreOpAdaptor::getSgprOffset() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr RawBufferStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr RawBufferStoreOpAdaptor::getBoundsCheckAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, RawBufferStoreOp::getBoundsCheckAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool RawBufferStoreOpAdaptor::getBoundsCheck() {
  auto attr = getBoundsCheckAttr();
  return attr.getValue();
}

::mlir::IntegerAttr RawBufferStoreOpAdaptor::getIndexOffsetAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, RawBufferStoreOp::getIndexOffsetAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::llvm::Optional<uint32_t> RawBufferStoreOpAdaptor::getIndexOffset() {
  auto attr = getIndexOffsetAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult RawBufferStoreOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_boundsCheck;
  ::mlir::Attribute tblgen_indexOffset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'amdgpu.raw_buffer_store' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == RawBufferStoreOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == RawBufferStoreOp::getBoundsCheckAttrName(*odsOpName)) {
      tblgen_boundsCheck = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == RawBufferStoreOp::getIndexOffsetAttrName(*odsOpName)) {
      tblgen_indexOffset = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitError(loc, "'amdgpu.raw_buffer_store' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (tblgen_boundsCheck && !((tblgen_boundsCheck.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'amdgpu.raw_buffer_store' op ""attribute 'boundsCheck' failed to satisfy constraint: bool attribute");

  if (tblgen_indexOffset && !(((tblgen_indexOffset.isa<::mlir::IntegerAttr>())) && ((tblgen_indexOffset.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'amdgpu.raw_buffer_store' op ""attribute 'indexOffset' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RawBufferStoreOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range RawBufferStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value RawBufferStoreOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::MemRefType> RawBufferStoreOp::getMemref() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range RawBufferStoreOp::getIndices() {
  return getODSOperands(2);
}

::mlir::TypedValue<::mlir::IntegerType> RawBufferStoreOp::getSgprOffset() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange RawBufferStoreOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferStoreOp::getMemrefMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferStoreOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange RawBufferStoreOp::getSgprOffsetMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> RawBufferStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RawBufferStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::BoolAttr RawBufferStoreOp::getBoundsCheckAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getBoundsCheckAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool RawBufferStoreOp::getBoundsCheck() {
  auto attr = getBoundsCheckAttr();
  return attr.getValue();
}

::mlir::IntegerAttr RawBufferStoreOp::getIndexOffsetAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getIndexOffsetAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint32_t> RawBufferStoreOp::getIndexOffset() {
  auto attr = getIndexOffsetAttr();
  return attr ? ::llvm::Optional<uint32_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void RawBufferStoreOp::setBoundsCheckAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getBoundsCheckAttrName(), attr);
}

void RawBufferStoreOp::setIndexOffsetAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexOffsetAttrName(), attr);
}

::mlir::Attribute RawBufferStoreOp::removeIndexOffsetAttr() {
  return (*this)->removeAttr(getIndexOffsetAttrName());
}

void RawBufferStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  if (boundsCheck) {
  odsState.addAttribute(getBoundsCheckAttrName(odsState.name), boundsCheck);
  }
  if (indexOffset) {
  odsState.addAttribute(getIndexOffsetAttrName(odsState.name), indexOffset);
  }
}

void RawBufferStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  if (boundsCheck) {
  odsState.addAttribute(getBoundsCheckAttrName(odsState.name), boundsCheck);
  }
  if (indexOffset) {
  odsState.addAttribute(getIndexOffsetAttrName(odsState.name), indexOffset);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RawBufferStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(getBoundsCheckAttrName(odsState.name), odsBuilder.getBoolAttr(boundsCheck));
  if (indexOffset) {
  odsState.addAttribute(getIndexOffsetAttrName(odsState.name), indexOffset);
  }
}

void RawBufferStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset) {
  odsState.addOperands(value);
  odsState.addOperands(memref);
  odsState.addOperands(indices);
  if (sgprOffset)
    odsState.addOperands(sgprOffset);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(indices.size()), (sgprOffset ? 1 : 0)}));
  odsState.addAttribute(getBoundsCheckAttrName(odsState.name), odsBuilder.getBoolAttr(boundsCheck));
  if (indexOffset) {
  odsState.addAttribute(getIndexOffsetAttrName(odsState.name), indexOffset);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RawBufferStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void RawBufferStoreOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
     attributes.append(attrNames[0], odsBuilder.getBoolAttr(true));
  }
}

::mlir::LogicalResult RawBufferStoreOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_boundsCheck;
  ::mlir::Attribute tblgen_indexOffset;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBoundsCheckAttrName()) {
      tblgen_boundsCheck = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getIndexOffsetAttrName()) {
      tblgen_indexOffset = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 4)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU3(*this, tblgen_boundsCheck, "boundsCheck")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_AMDGPU0(*this, tblgen_indexOffset, "indexOffset")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_AMDGPU4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!(((getElementTypeOrSelf((*this->getODSOperands(0).begin()))) == (getElementTypeOrSelf((*this->getODSOperands(1).begin()))) && (getElementTypeOrSelf((*this->getODSOperands(1).begin()))) == (getElementTypeOrSelf((*this->getODSOperands(0).begin()))))))
    return emitOpError("failed to verify that all of {value, memref} have same element type");
  return ::mlir::success();
}

::mlir::LogicalResult RawBufferStoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult RawBufferStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand memrefRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memrefOperands(memrefRawOperands);  ::llvm::SMLoc memrefOperandsLoc;
  (void)memrefOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> sgprOffsetOperands;
  ::llvm::SMLoc sgprOffsetOperandsLoc;
  (void)sgprOffsetOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type memrefRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> memrefTypes(memrefRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> indicesTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  memrefOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memrefRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  indicesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(indicesOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("sgprOffset"))) {

  {
    sgprOffsetOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.has_value()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      sgprOffsetOperands.push_back(operand);
    }
  }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::MemRefType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    memrefRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseTypeList(indicesTypes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(indicesOperands.size()), static_cast<int32_t>(sgprOffsetOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(memrefOperands, memrefTypes, memrefOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(indicesOperands, indicesTypes, indicesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(sgprOffsetOperands, odsBuildableType0, sgprOffsetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RawBufferStoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", });
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getMemref();
  _odsPrinter << "[";
  _odsPrinter << getIndices();
  _odsPrinter << "]";
  if (getSgprOffset()) {
  _odsPrinter << ' ' << "sgprOffset";
  _odsPrinter << ' ';
  if (::mlir::Value value = getSgprOffset())
    _odsPrinter << value;
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getMemref().getType();
    if (auto validType = type.dyn_cast<::mlir::MemRefType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getIndices().getTypes();
}

void RawBufferStoreOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace amdgpu
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferStoreOp)


#endif  // GET_OP_CLASSES

