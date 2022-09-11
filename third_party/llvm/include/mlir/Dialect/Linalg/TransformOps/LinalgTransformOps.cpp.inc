/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::transform::DecomposeOp,
::mlir::transform::FuseIntoContainingOp,
::mlir::transform::FuseOp,
::mlir::transform::GeneralizeOp,
::mlir::transform::InterchangeOp,
::mlir::transform::MatchOp,
::mlir::transform::MultiTileSizesOp,
::mlir::transform::PadOp,
::mlir::transform::PromoteOp,
::mlir::transform::ScalarizeOp,
::mlir::transform::SplitOp,
::mlir::transform::SplitReductionOp,
::mlir::transform::TileOp,
::mlir::transform::TileToForeachThreadOp,
::mlir::transform::VectorizeOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace transform {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LinalgTransformOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::OperationType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Operation *`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgTransformOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit integer array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgTransformOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgTransformOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<mlir::transform::MatchInterfaceEnumAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: An interface to match";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgTransformOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::DictionaryAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: dictionary of named attribute values";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgTransformOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgTransformOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgTransformOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: array of arrays of i64";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgTransformOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::BoolAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 1-bit boolean array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgTransformOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgTransformOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
  }
  return ::mlir::success();
}
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::DecomposeOp definitions
//===----------------------------------------------------------------------===//

DecomposeOpAdaptor::DecomposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.decompose", odsAttrs.getContext());
}

DecomposeOpAdaptor::DecomposeOpAdaptor(DecomposeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DecomposeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DecomposeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DecomposeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DecomposeOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr DecomposeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult DecomposeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DecomposeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DecomposeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> DecomposeOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange DecomposeOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DecomposeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DecomposeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> DecomposeOp::getTransformed() {
  return *getODSResults(0).begin();
}

void DecomposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target) {
  odsState.addOperands(target);
  odsState.addTypes(transformed);
}

void DecomposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target) {
  odsState.addOperands(target);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DecomposeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DecomposeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DecomposeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult DecomposeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(targetOperands, odsBuildableType0, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void DecomposeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::DecomposeOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::FuseIntoContainingOp definitions
//===----------------------------------------------------------------------===//

FuseIntoContainingOpAdaptor::FuseIntoContainingOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.fuse_into_containing_op", odsAttrs.getContext());
}

FuseIntoContainingOpAdaptor::FuseIntoContainingOpAdaptor(FuseIntoContainingOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FuseIntoContainingOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FuseIntoContainingOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FuseIntoContainingOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FuseIntoContainingOpAdaptor::getProducerOp() {
  return *getODSOperands(0).begin();
}

::mlir::Value FuseIntoContainingOpAdaptor::getContainingOp() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FuseIntoContainingOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FuseIntoContainingOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FuseIntoContainingOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FuseIntoContainingOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> FuseIntoContainingOp::getProducerOp() {
  return *getODSOperands(0).begin();
}

::mlir::TypedValue<::mlir::pdl::OperationType> FuseIntoContainingOp::getContainingOp() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FuseIntoContainingOp::getProducerOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FuseIntoContainingOp::getContainingOpMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FuseIntoContainingOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FuseIntoContainingOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> FuseIntoContainingOp::getFusedOp() {
  return *getODSResults(0).begin();
}

void FuseIntoContainingOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type fused_op, ::mlir::Value producer_op, ::mlir::Value containing_op) {
  odsState.addOperands(producer_op);
  odsState.addOperands(containing_op);
  odsState.addTypes(fused_op);
}

void FuseIntoContainingOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value producer_op, ::mlir::Value containing_op) {
  odsState.addOperands(producer_op);
  odsState.addOperands(containing_op);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FuseIntoContainingOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FuseIntoContainingOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FuseIntoContainingOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FuseIntoContainingOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand producer_opRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> producer_opOperands(producer_opRawOperands);  ::llvm::SMLoc producer_opOperandsLoc;
  (void)producer_opOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand containing_opRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> containing_opOperands(containing_opRawOperands);  ::llvm::SMLoc containing_opOperandsLoc;
  (void)containing_opOperandsLoc;

  producer_opOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(producer_opRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  containing_opOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(containing_opRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(producer_opOperands, odsBuildableType0, producer_opOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(containing_opOperands, odsBuildableType0, containing_opOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FuseIntoContainingOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getProducerOp();
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  _odsPrinter << getContainingOp();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void FuseIntoContainingOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::transform::TransformMappingResource::get());
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Free::get(), value, ::mlir::transform::TransformMappingResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::transform::TransformMappingResource::get());
  for (::mlir::Value value : getODSResults(0))
    effects.emplace_back(::mlir::MemoryEffects::Allocate::get(), value, ::mlir::transform::TransformMappingResource::get());
  for (::mlir::Value value : getODSResults(0))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::transform::TransformMappingResource::get());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::FuseIntoContainingOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::FuseOp definitions
//===----------------------------------------------------------------------===//

FuseOpAdaptor::FuseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.fuse", odsAttrs.getContext());
}

FuseOpAdaptor::FuseOpAdaptor(FuseOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FuseOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FuseOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FuseOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FuseOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FuseOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr FuseOpAdaptor::getTileSizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FuseOp::getTileSizesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr FuseOpAdaptor::getTileSizes() {
  auto attr = getTileSizesAttr();
  return attr;
}

::mlir::ArrayAttr FuseOpAdaptor::getTileInterchangeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FuseOp::getTileInterchangeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr FuseOpAdaptor::getTileInterchange() {
  auto attr = getTileInterchangeAttr();
  return attr;
}

::mlir::LogicalResult FuseOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_tile_interchange;
  ::mlir::Attribute tblgen_tile_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FuseOp::getTileInterchangeAttrName(*odsOpName)) {
      tblgen_tile_interchange = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == FuseOp::getTileSizesAttrName(*odsOpName)) {
      tblgen_tile_sizes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_tile_sizes && !(((tblgen_tile_sizes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_tile_sizes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.structured.fuse' op ""attribute 'tile_sizes' failed to satisfy constraint: 64-bit integer array attribute");

  if (tblgen_tile_interchange && !(((tblgen_tile_interchange.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_tile_interchange.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.structured.fuse' op ""attribute 'tile_interchange' failed to satisfy constraint: 64-bit integer array attribute");
  return ::mlir::success();
}

void FuseOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "transformed");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "loops");
}

std::pair<unsigned, unsigned> FuseOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FuseOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> FuseOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FuseOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FuseOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range FuseOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> FuseOp::getTransformed() {
  return *getODSResults(0).begin();
}

::mlir::Operation::result_range FuseOp::getLoops() {
  return getODSResults(1);
}

::mlir::ArrayAttr FuseOp::getTileSizesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTileSizesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr FuseOp::getTileSizes() {
  auto attr = getTileSizesAttr();
  return attr;
}

::mlir::ArrayAttr FuseOp::getTileInterchangeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTileInterchangeAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr FuseOp::getTileInterchange() {
  auto attr = getTileInterchangeAttr();
  return attr;
}

void FuseOp::setTileSizesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTileSizesAttrName(), attr);
}

void FuseOp::setTileInterchangeAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTileInterchangeAttrName(), attr);
}

void FuseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::TypeRange loops, ::mlir::Value target, ::mlir::ArrayAttr tile_sizes, ::mlir::ArrayAttr tile_interchange) {
  odsState.addOperands(target);
  if (tile_sizes) {
  odsState.addAttribute(getTileSizesAttrName(odsState.name), tile_sizes);
  }
  if (tile_interchange) {
  odsState.addAttribute(getTileInterchangeAttrName(odsState.name), tile_interchange);
  }
  odsState.addTypes(transformed);
  odsState.addTypes(loops);
}

void FuseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr tile_sizes, ::mlir::ArrayAttr tile_interchange) {
  odsState.addOperands(target);
  if (tile_sizes) {
  odsState.addAttribute(getTileSizesAttrName(odsState.name), tile_sizes);
  }
  if (tile_interchange) {
  odsState.addAttribute(getTileInterchangeAttrName(odsState.name), tile_interchange);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FuseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FuseOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[1])) {
     attributes.append(attrNames[1], odsBuilder.getI64ArrayAttr({}));
  }
  if (!attributes.get(attrNames[0])) {
     attributes.append(attrNames[0], odsBuilder.getI64ArrayAttr({}));
  }
}

::mlir::LogicalResult FuseOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_tile_interchange;
  ::mlir::Attribute tblgen_tile_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getTileInterchangeAttrName()) {
      tblgen_tile_interchange = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getTileSizesAttrName()) {
      tblgen_tile_sizes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_tile_sizes, "tile_sizes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_tile_interchange, "tile_interchange")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FuseOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::FuseOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::GeneralizeOp definitions
//===----------------------------------------------------------------------===//

GeneralizeOpAdaptor::GeneralizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.generalize", odsAttrs.getContext());
}

GeneralizeOpAdaptor::GeneralizeOpAdaptor(GeneralizeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GeneralizeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GeneralizeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GeneralizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GeneralizeOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GeneralizeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GeneralizeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GeneralizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GeneralizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> GeneralizeOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange GeneralizeOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GeneralizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GeneralizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> GeneralizeOp::getTransformed() {
  return *getODSResults(0).begin();
}

void GeneralizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target) {
  odsState.addOperands(target);
  odsState.addTypes(transformed);
}

void GeneralizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target) {
  odsState.addOperands(target);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GeneralizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GeneralizeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GeneralizeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GeneralizeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(targetOperands, odsBuildableType0, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GeneralizeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::GeneralizeOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::InterchangeOp definitions
//===----------------------------------------------------------------------===//

InterchangeOpAdaptor::InterchangeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.interchange", odsAttrs.getContext());
}

InterchangeOpAdaptor::InterchangeOpAdaptor(InterchangeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange InterchangeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InterchangeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange InterchangeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InterchangeOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr InterchangeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr InterchangeOpAdaptor::getIteratorInterchangeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, InterchangeOp::getIteratorInterchangeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr InterchangeOpAdaptor::getIteratorInterchange() {
  auto attr = getIteratorInterchangeAttr();
  return attr;
}

::mlir::LogicalResult InterchangeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_iterator_interchange;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == InterchangeOp::getIteratorInterchangeAttrName(*odsOpName)) {
      tblgen_iterator_interchange = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_iterator_interchange && !(((tblgen_iterator_interchange.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_iterator_interchange.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.structured.interchange' op ""attribute 'iterator_interchange' failed to satisfy constraint: 64-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InterchangeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range InterchangeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> InterchangeOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange InterchangeOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> InterchangeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InterchangeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> InterchangeOp::getTransformed() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr InterchangeOp::getIteratorInterchangeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIteratorInterchangeAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr InterchangeOp::getIteratorInterchange() {
  auto attr = getIteratorInterchangeAttr();
  return attr;
}

void InterchangeOp::setIteratorInterchangeAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getIteratorInterchangeAttrName(), attr);
}

void InterchangeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::ArrayAttr iterator_interchange) {
  odsState.addOperands(target);
  if (iterator_interchange) {
  odsState.addAttribute(getIteratorInterchangeAttrName(odsState.name), iterator_interchange);
  }
  odsState.addTypes(transformed);
}

void InterchangeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr iterator_interchange) {
  odsState.addOperands(target);
  if (iterator_interchange) {
  odsState.addAttribute(getIteratorInterchangeAttrName(odsState.name), iterator_interchange);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InterchangeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void InterchangeOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
     attributes.append(attrNames[0], odsBuilder.getI64ArrayAttr({}));
  }
}

::mlir::LogicalResult InterchangeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_iterator_interchange;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getIteratorInterchangeAttrName()) {
      tblgen_iterator_interchange = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_iterator_interchange, "iterator_interchange")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult InterchangeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult InterchangeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(targetOperands, odsBuildableType0, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InterchangeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::InterchangeOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::MatchOp definitions
//===----------------------------------------------------------------------===//

MatchOpAdaptor::MatchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.match", odsAttrs.getContext());
}

MatchOpAdaptor::MatchOpAdaptor(MatchOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MatchOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MatchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatchOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr MatchOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr MatchOpAdaptor::getOpsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MatchOp::getOpsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > MatchOpAdaptor::getOps() {
  auto attr = getOpsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

mlir::transform::MatchInterfaceEnumAttr MatchOpAdaptor::getInterfaceAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MatchOp::getInterfaceAttrName(*odsOpName)).dyn_cast_or_null<mlir::transform::MatchInterfaceEnumAttr>();
  return attr;
}

::llvm::Optional<mlir::transform::MatchInterfaceEnum> MatchOpAdaptor::getInterface() {
  auto attr = getInterfaceAttr();
  return attr ? ::llvm::Optional<mlir::transform::MatchInterfaceEnum>(attr.getValue()) : (::llvm::None);
}

::mlir::DictionaryAttr MatchOpAdaptor::getOpAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MatchOp::getOpAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::DictionaryAttr>();
  return attr;
}

::llvm::Optional< ::mlir::DictionaryAttr > MatchOpAdaptor::getOpAttrs() {
  auto attr = getOpAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult MatchOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_interface;
  ::mlir::Attribute tblgen_op_attrs;
  ::mlir::Attribute tblgen_ops;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == MatchOp::getInterfaceAttrName(*odsOpName)) {
      tblgen_interface = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == MatchOp::getOpAttrsAttrName(*odsOpName)) {
      tblgen_op_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == MatchOp::getOpsAttrName(*odsOpName)) {
      tblgen_ops = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_ops && !(((tblgen_ops.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_ops.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::StringAttr>())); }))))
    return emitError(loc, "'transform.structured.match' op ""attribute 'ops' failed to satisfy constraint: string array attribute");

  if (tblgen_interface && !((tblgen_interface.isa<mlir::transform::MatchInterfaceEnumAttr>())))
    return emitError(loc, "'transform.structured.match' op ""attribute 'interface' failed to satisfy constraint: An interface to match");

  if (tblgen_op_attrs && !((tblgen_op_attrs.isa<::mlir::DictionaryAttr>())))
    return emitError(loc, "'transform.structured.match' op ""attribute 'op_attrs' failed to satisfy constraint: dictionary of named attribute values");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MatchOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MatchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> MatchOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange MatchOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MatchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MatchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> MatchOp::getResults() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr MatchOp::getOpsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOpsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > MatchOp::getOps() {
  auto attr = getOpsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

mlir::transform::MatchInterfaceEnumAttr MatchOp::getInterfaceAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getInterfaceAttrName()).dyn_cast_or_null<mlir::transform::MatchInterfaceEnumAttr>();
}

::llvm::Optional<mlir::transform::MatchInterfaceEnum> MatchOp::getInterface() {
  auto attr = getInterfaceAttr();
  return attr ? ::llvm::Optional<mlir::transform::MatchInterfaceEnum>(attr.getValue()) : (::llvm::None);
}

::mlir::DictionaryAttr MatchOp::getOpAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOpAttrsAttrName()).dyn_cast_or_null<::mlir::DictionaryAttr>();
}

::llvm::Optional< ::mlir::DictionaryAttr > MatchOp::getOpAttrs() {
  auto attr = getOpAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::DictionaryAttr >(attr) : (::llvm::None);
}

void MatchOp::setOpsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOpsAttrName(), attr);
}

void MatchOp::setInterfaceAttr(mlir::transform::MatchInterfaceEnumAttr attr) {
  (*this)->setAttr(getInterfaceAttrName(), attr);
}

void MatchOp::setOpAttrsAttr(::mlir::DictionaryAttr attr) {
  (*this)->setAttr(getOpAttrsAttrName(), attr);
}

::mlir::Attribute MatchOp::removeOpsAttr() {
  return (*this)->removeAttr(getOpsAttrName());
}

::mlir::Attribute MatchOp::removeInterfaceAttr() {
  return (*this)->removeAttr(getInterfaceAttrName());
}

::mlir::Attribute MatchOp::removeOp_attrsAttr() {
  return (*this)->removeAttr(getOpAttrsAttrName());
}

void MatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type results, ::mlir::Value target, /*optional*/::mlir::ArrayAttr ops, /*optional*/mlir::transform::MatchInterfaceEnumAttr interface, /*optional*/::mlir::DictionaryAttr op_attrs) {
  odsState.addOperands(target);
  if (ops) {
  odsState.addAttribute(getOpsAttrName(odsState.name), ops);
  }
  if (interface) {
  odsState.addAttribute(getInterfaceAttrName(odsState.name), interface);
  }
  if (op_attrs) {
  odsState.addAttribute(getOpAttrsAttrName(odsState.name), op_attrs);
  }
  odsState.addTypes(results);
}

void MatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, /*optional*/::mlir::ArrayAttr ops, /*optional*/mlir::transform::MatchInterfaceEnumAttr interface, /*optional*/::mlir::DictionaryAttr op_attrs) {
  odsState.addOperands(target);
  if (ops) {
  odsState.addAttribute(getOpsAttrName(odsState.name), ops);
  }
  if (interface) {
  odsState.addAttribute(getInterfaceAttrName(odsState.name), interface);
  }
  if (op_attrs) {
  odsState.addAttribute(getOpAttrsAttrName(odsState.name), op_attrs);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MatchOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_interface;
  ::mlir::Attribute tblgen_op_attrs;
  ::mlir::Attribute tblgen_ops;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getInterfaceAttrName()) {
      tblgen_interface = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getOpAttrsAttrName()) {
      tblgen_op_attrs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getOpsAttrName()) {
      tblgen_ops = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps1(*this, tblgen_ops, "ops")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps2(*this, tblgen_interface, "interface")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps3(*this, tblgen_op_attrs, "op_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MatchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MatchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::ArrayAttr opsAttr;
  mlir::transform::MatchInterfaceEnumAttr interfaceAttr;
  ::mlir::DictionaryAttr op_attrsAttr;
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  if (succeeded(parser.parseOptionalKeyword("ops"))) {
  if (parser.parseLBrace())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(opsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "ops",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRBrace())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("interface"))) {
  if (parser.parseLBrace())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"LinalgOp","TilingInterface"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "interface", attrStorage);
      if (parseResult.has_value()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {

      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = mlir::transform::symbolizeMatchInterfaceEnum(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "interface attribute specification: \"" << attrStr << '"';;

      interfaceAttr = mlir::transform::MatchInterfaceEnumAttr::get(parser.getBuilder().getContext(), attrOptional.value());
      result.addAttribute("interface", interfaceAttr);
    }
  }
  if (parser.parseRBrace())
    return ::mlir::failure();
  }
  if (succeeded(parser.parseOptionalKeyword("attributes"))) {

  if (parser.parseCustomAttributeWithFallback(op_attrsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "op_attrs",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  if (parser.parseKeyword("in"))
    return ::mlir::failure();

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(targetOperands, odsBuildableType0, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("ops")) {
  _odsPrinter << ' ' << "ops";
  _odsPrinter << "{";
  _odsPrinter.printAttributeWithoutType(getOpsAttr());
  _odsPrinter << "}";
  }
  if ((*this)->getAttr("interface")) {
  _odsPrinter << ' ' << "interface";
  _odsPrinter << "{";

  {
    auto caseValue = *getInterface();
    auto caseValueStr = stringifyMatchInterfaceEnum(caseValue);
    _odsPrinter << caseValueStr;
  }
  _odsPrinter << "}";
  }
  if ((*this)->getAttr("op_attrs")) {
  _odsPrinter << ' ' << "attributes";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getOpAttrsAttr());
  }
  _odsPrinter << ' ' << "in";
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"ops", "interface", "op_attrs"});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::MatchOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::MultiTileSizesOp definitions
//===----------------------------------------------------------------------===//

MultiTileSizesOpAdaptor::MultiTileSizesOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.multitile_sizes", odsAttrs.getContext());
}

MultiTileSizesOpAdaptor::MultiTileSizesOpAdaptor(MultiTileSizesOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MultiTileSizesOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MultiTileSizesOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MultiTileSizesOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MultiTileSizesOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr MultiTileSizesOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MultiTileSizesOpAdaptor::getDimensionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, MultiTileSizesOp::getDimensionAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t MultiTileSizesOpAdaptor::getDimension() {
  auto attr = getDimensionAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MultiTileSizesOpAdaptor::getTargetSizeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, MultiTileSizesOp::getTargetSizeAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t MultiTileSizesOpAdaptor::getTargetSize() {
  auto attr = getTargetSizeAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MultiTileSizesOpAdaptor::getDivisorAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, MultiTileSizesOp::getDivisorAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint64_t MultiTileSizesOpAdaptor::getDivisor() {
  auto attr = getDivisorAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult MultiTileSizesOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dimension;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'transform.structured.multitile_sizes' op ""requires attribute 'dimension'");
    if (namedAttrIt->getName() == MultiTileSizesOp::getDimensionAttrName(*odsOpName)) {
      tblgen_dimension = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_target_size;
  ::mlir::Attribute tblgen_divisor;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'transform.structured.multitile_sizes' op ""requires attribute 'target_size'");
    if (namedAttrIt->getName() == MultiTileSizesOp::getTargetSizeAttrName(*odsOpName)) {
      tblgen_target_size = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == MultiTileSizesOp::getDivisorAttrName(*odsOpName)) {
      tblgen_divisor = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_dimension && !(((tblgen_dimension.isa<::mlir::IntegerAttr>())) && ((tblgen_dimension.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'transform.structured.multitile_sizes' op ""attribute 'dimension' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_target_size && !(((tblgen_target_size.isa<::mlir::IntegerAttr>())) && ((tblgen_target_size.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'transform.structured.multitile_sizes' op ""attribute 'target_size' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_divisor && !(((tblgen_divisor.isa<::mlir::IntegerAttr>())) && ((tblgen_divisor.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'transform.structured.multitile_sizes' op ""attribute 'divisor' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

void MultiTileSizesOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "low_size");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "high_size");
  auto resultGroup2 = getODSResults(2);
  if (!llvm::empty(resultGroup2))
    setNameFn(*resultGroup2.begin(), "split_point");
}

std::pair<unsigned, unsigned> MultiTileSizesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MultiTileSizesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> MultiTileSizesOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange MultiTileSizesOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MultiTileSizesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MultiTileSizesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> MultiTileSizesOp::getLowSize() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::pdl::OperationType> MultiTileSizesOp::getHighSize() {
  return *getODSResults(1).begin();
}

::mlir::TypedValue<::mlir::pdl::OperationType> MultiTileSizesOp::getSplitPoint() {
  return *getODSResults(2).begin();
}

::mlir::IntegerAttr MultiTileSizesOp::getDimensionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDimensionAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t MultiTileSizesOp::getDimension() {
  auto attr = getDimensionAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MultiTileSizesOp::getTargetSizeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTargetSizeAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t MultiTileSizesOp::getTargetSize() {
  auto attr = getTargetSizeAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MultiTileSizesOp::getDivisorAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getDivisorAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t MultiTileSizesOp::getDivisor() {
  auto attr = getDivisorAttr();
  return attr.getValue().getZExtValue();
}

void MultiTileSizesOp::setDimensionAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getDimensionAttrName(), attr);
}

void MultiTileSizesOp::setTargetSizeAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getTargetSizeAttrName(), attr);
}

void MultiTileSizesOp::setDivisorAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getDivisorAttrName(), attr);
}

void MultiTileSizesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type low_size, ::mlir::Type high_size, ::mlir::Type split_point, ::mlir::Value target, ::mlir::IntegerAttr dimension, ::mlir::IntegerAttr target_size, ::mlir::IntegerAttr divisor) {
  odsState.addOperands(target);
  odsState.addAttribute(getDimensionAttrName(odsState.name), dimension);
  odsState.addAttribute(getTargetSizeAttrName(odsState.name), target_size);
  if (divisor) {
  odsState.addAttribute(getDivisorAttrName(odsState.name), divisor);
  }
  odsState.addTypes(low_size);
  odsState.addTypes(high_size);
  odsState.addTypes(split_point);
}

void MultiTileSizesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr dimension, ::mlir::IntegerAttr target_size, ::mlir::IntegerAttr divisor) {
  odsState.addOperands(target);
  odsState.addAttribute(getDimensionAttrName(odsState.name), dimension);
  odsState.addAttribute(getTargetSizeAttrName(odsState.name), target_size);
  if (divisor) {
  odsState.addAttribute(getDivisorAttrName(odsState.name), divisor);
  }
  assert(resultTypes.size() == 3u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MultiTileSizesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type low_size, ::mlir::Type high_size, ::mlir::Type split_point, ::mlir::Value target, uint64_t dimension, uint64_t target_size, uint64_t divisor) {
  odsState.addOperands(target);
  odsState.addAttribute(getDimensionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), dimension));
  odsState.addAttribute(getTargetSizeAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), target_size));
  odsState.addAttribute(getDivisorAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), divisor));
  odsState.addTypes(low_size);
  odsState.addTypes(high_size);
  odsState.addTypes(split_point);
}

void MultiTileSizesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t dimension, uint64_t target_size, uint64_t divisor) {
  odsState.addOperands(target);
  odsState.addAttribute(getDimensionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), dimension));
  odsState.addAttribute(getTargetSizeAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), target_size));
  odsState.addAttribute(getDivisorAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), divisor));
  assert(resultTypes.size() == 3u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MultiTileSizesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 3u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MultiTileSizesOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[1])) {
     attributes.append(attrNames[1], odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1));
  }
}

::mlir::LogicalResult MultiTileSizesOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dimension;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'dimension'");
    if (namedAttrIt->getName() == getDimensionAttrName()) {
      tblgen_dimension = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_target_size;
  ::mlir::Attribute tblgen_divisor;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'target_size'");
    if (namedAttrIt->getName() == getTargetSizeAttrName()) {
      tblgen_target_size = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDivisorAttrName()) {
      tblgen_divisor = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps4(*this, tblgen_dimension, "dimension")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps4(*this, tblgen_target_size, "target_size")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps4(*this, tblgen_divisor, "divisor")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSResults(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MultiTileSizesOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MultiTileSizesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  result.addTypes(odsBuildableType0);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(targetOperands, odsBuildableType0, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MultiTileSizesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::MultiTileSizesOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::PadOp definitions
//===----------------------------------------------------------------------===//

PadOpAdaptor::PadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.pad", odsAttrs.getContext());
}

PadOpAdaptor::PadOpAdaptor(PadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PadOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr PadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr PadOpAdaptor::getPaddingValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PadOp::getPaddingValuesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getPaddingValues() {
  auto attr = getPaddingValuesAttr();
  return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getPaddingDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PadOp::getPaddingDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getPaddingDimensions() {
  auto attr = getPaddingDimensionsAttr();
  return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getPackPaddingsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PadOp::getPackPaddingsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getPackPaddings() {
  auto attr = getPackPaddingsAttr();
  return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getHoistPaddingsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PadOp::getHoistPaddingsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getHoistPaddings() {
  auto attr = getHoistPaddingsAttr();
  return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getTransposePaddingsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PadOp::getTransposePaddingsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getTransposePaddings() {
  auto attr = getTransposePaddingsAttr();
  return attr;
}

::mlir::LogicalResult PadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hoist_paddings;
  ::mlir::Attribute tblgen_pack_paddings;
  ::mlir::Attribute tblgen_padding_dimensions;
  ::mlir::Attribute tblgen_padding_values;
  ::mlir::Attribute tblgen_transpose_paddings;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PadOp::getHoistPaddingsAttrName(*odsOpName)) {
      tblgen_hoist_paddings = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == PadOp::getPackPaddingsAttrName(*odsOpName)) {
      tblgen_pack_paddings = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == PadOp::getPaddingDimensionsAttrName(*odsOpName)) {
      tblgen_padding_dimensions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == PadOp::getPaddingValuesAttrName(*odsOpName)) {
      tblgen_padding_values = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == PadOp::getTransposePaddingsAttrName(*odsOpName)) {
      tblgen_transpose_paddings = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_padding_values && !((tblgen_padding_values.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'transform.structured.pad' op ""attribute 'padding_values' failed to satisfy constraint: array attribute");

  if (tblgen_padding_dimensions && !(((tblgen_padding_dimensions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_padding_dimensions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.structured.pad' op ""attribute 'padding_dimensions' failed to satisfy constraint: 64-bit integer array attribute");

  if (tblgen_pack_paddings && !(((tblgen_pack_paddings.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_pack_paddings.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.structured.pad' op ""attribute 'pack_paddings' failed to satisfy constraint: 64-bit integer array attribute");

  if (tblgen_hoist_paddings && !(((tblgen_hoist_paddings.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_hoist_paddings.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.structured.pad' op ""attribute 'hoist_paddings' failed to satisfy constraint: 64-bit integer array attribute");

  if (tblgen_transpose_paddings && !(((tblgen_transpose_paddings.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_transpose_paddings.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))); }))))
    return emitError(loc, "'transform.structured.pad' op ""attribute 'transpose_paddings' failed to satisfy constraint: array of arrays of i64");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> PadOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange PadOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> PadOp::getTransformed() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr PadOp::getPaddingValuesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPaddingValuesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr PadOp::getPaddingValues() {
  auto attr = getPaddingValuesAttr();
  return attr;
}

::mlir::ArrayAttr PadOp::getPaddingDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPaddingDimensionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr PadOp::getPaddingDimensions() {
  auto attr = getPaddingDimensionsAttr();
  return attr;
}

::mlir::ArrayAttr PadOp::getPackPaddingsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPackPaddingsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr PadOp::getPackPaddings() {
  auto attr = getPackPaddingsAttr();
  return attr;
}

::mlir::ArrayAttr PadOp::getHoistPaddingsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getHoistPaddingsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr PadOp::getHoistPaddings() {
  auto attr = getHoistPaddingsAttr();
  return attr;
}

::mlir::ArrayAttr PadOp::getTransposePaddingsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTransposePaddingsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr PadOp::getTransposePaddings() {
  auto attr = getTransposePaddingsAttr();
  return attr;
}

void PadOp::setPaddingValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPaddingValuesAttrName(), attr);
}

void PadOp::setPaddingDimensionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPaddingDimensionsAttrName(), attr);
}

void PadOp::setPackPaddingsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPackPaddingsAttrName(), attr);
}

void PadOp::setHoistPaddingsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getHoistPaddingsAttrName(), attr);
}

void PadOp::setTransposePaddingsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTransposePaddingsAttrName(), attr);
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::ArrayAttr padding_values, ::mlir::ArrayAttr padding_dimensions, ::mlir::ArrayAttr pack_paddings, ::mlir::ArrayAttr hoist_paddings, ::mlir::ArrayAttr transpose_paddings) {
  odsState.addOperands(target);
  if (padding_values) {
  odsState.addAttribute(getPaddingValuesAttrName(odsState.name), padding_values);
  }
  if (padding_dimensions) {
  odsState.addAttribute(getPaddingDimensionsAttrName(odsState.name), padding_dimensions);
  }
  if (pack_paddings) {
  odsState.addAttribute(getPackPaddingsAttrName(odsState.name), pack_paddings);
  }
  if (hoist_paddings) {
  odsState.addAttribute(getHoistPaddingsAttrName(odsState.name), hoist_paddings);
  }
  if (transpose_paddings) {
  odsState.addAttribute(getTransposePaddingsAttrName(odsState.name), transpose_paddings);
  }
  odsState.addTypes(transformed);
}

void PadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr padding_values, ::mlir::ArrayAttr padding_dimensions, ::mlir::ArrayAttr pack_paddings, ::mlir::ArrayAttr hoist_paddings, ::mlir::ArrayAttr transpose_paddings) {
  odsState.addOperands(target);
  if (padding_values) {
  odsState.addAttribute(getPaddingValuesAttrName(odsState.name), padding_values);
  }
  if (padding_dimensions) {
  odsState.addAttribute(getPaddingDimensionsAttrName(odsState.name), padding_dimensions);
  }
  if (pack_paddings) {
  odsState.addAttribute(getPackPaddingsAttrName(odsState.name), pack_paddings);
  }
  if (hoist_paddings) {
  odsState.addAttribute(getHoistPaddingsAttrName(odsState.name), hoist_paddings);
  }
  if (transpose_paddings) {
  odsState.addAttribute(getTransposePaddingsAttrName(odsState.name), transpose_paddings);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void PadOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[3])) {
     attributes.append(attrNames[3], odsBuilder.getArrayAttr({}));
  }
  if (!attributes.get(attrNames[2])) {
     attributes.append(attrNames[2], odsBuilder.getI64ArrayAttr({}));
  }
  if (!attributes.get(attrNames[1])) {
     attributes.append(attrNames[1], odsBuilder.getI64ArrayAttr({}));
  }
  if (!attributes.get(attrNames[0])) {
     attributes.append(attrNames[0], odsBuilder.getI64ArrayAttr({}));
  }
  if (!attributes.get(attrNames[4])) {
     attributes.append(attrNames[4], odsBuilder.getArrayAttr({}));
  }
}

::mlir::LogicalResult PadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hoist_paddings;
  ::mlir::Attribute tblgen_pack_paddings;
  ::mlir::Attribute tblgen_padding_dimensions;
  ::mlir::Attribute tblgen_padding_values;
  ::mlir::Attribute tblgen_transpose_paddings;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getHoistPaddingsAttrName()) {
      tblgen_hoist_paddings = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getPackPaddingsAttrName()) {
      tblgen_pack_paddings = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getPaddingDimensionsAttrName()) {
      tblgen_padding_dimensions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getPaddingValuesAttrName()) {
      tblgen_padding_values = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getTransposePaddingsAttrName()) {
      tblgen_transpose_paddings = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps5(*this, tblgen_padding_values, "padding_values")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_padding_dimensions, "padding_dimensions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_pack_paddings, "pack_paddings")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_hoist_paddings, "hoist_paddings")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps6(*this, tblgen_transpose_paddings, "transpose_paddings")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult PadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(targetOperands, odsBuildableType0, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::PadOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::PromoteOp definitions
//===----------------------------------------------------------------------===//

PromoteOpAdaptor::PromoteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.promote", odsAttrs.getContext());
}

PromoteOpAdaptor::PromoteOpAdaptor(PromoteOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PromoteOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PromoteOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PromoteOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PromoteOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr PromoteOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr PromoteOpAdaptor::getOperandsToPromoteAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PromoteOp::getOperandsToPromoteAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PromoteOpAdaptor::getOperandsToPromote() {
  auto attr = getOperandsToPromoteAttr();
  return attr;
}

::mlir::ArrayAttr PromoteOpAdaptor::getUseFullTileBuffersAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PromoteOp::getUseFullTileBuffersAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr PromoteOpAdaptor::getUseFullTileBuffers() {
  auto attr = getUseFullTileBuffersAttr();
  return attr;
}

::mlir::UnitAttr PromoteOpAdaptor::getUseFullTilesByDefaultAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PromoteOp::getUseFullTilesByDefaultAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool PromoteOpAdaptor::getUseFullTilesByDefault() {
  auto attr = getUseFullTilesByDefaultAttr();
  return attr != nullptr;
}

::mlir::UnitAttr PromoteOpAdaptor::getUseAllocaAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PromoteOp::getUseAllocaAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool PromoteOpAdaptor::getUseAlloca() {
  auto attr = getUseAllocaAttr();
  return attr != nullptr;
}

::mlir::IntegerAttr PromoteOpAdaptor::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PromoteOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::llvm::Optional<uint64_t> PromoteOpAdaptor::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult PromoteOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_operands_to_promote;
  ::mlir::Attribute tblgen_use_alloca;
  ::mlir::Attribute tblgen_use_full_tile_buffers;
  ::mlir::Attribute tblgen_use_full_tiles_by_default;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == PromoteOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == PromoteOp::getOperandsToPromoteAttrName(*odsOpName)) {
      tblgen_operands_to_promote = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == PromoteOp::getUseAllocaAttrName(*odsOpName)) {
      tblgen_use_alloca = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == PromoteOp::getUseFullTileBuffersAttrName(*odsOpName)) {
      tblgen_use_full_tile_buffers = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == PromoteOp::getUseFullTilesByDefaultAttrName(*odsOpName)) {
      tblgen_use_full_tiles_by_default = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_operands_to_promote && !(((tblgen_operands_to_promote.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_operands_to_promote.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.structured.promote' op ""attribute 'operands_to_promote' failed to satisfy constraint: 64-bit integer array attribute");

  if (tblgen_use_full_tile_buffers && !(((tblgen_use_full_tile_buffers.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_use_full_tile_buffers.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::BoolAttr>())); }))))
    return emitError(loc, "'transform.structured.promote' op ""attribute 'use_full_tile_buffers' failed to satisfy constraint: 1-bit boolean array attribute");

  if (tblgen_use_full_tiles_by_default && !((tblgen_use_full_tiles_by_default.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'transform.structured.promote' op ""attribute 'use_full_tiles_by_default' failed to satisfy constraint: unit attribute");

  if (tblgen_use_alloca && !((tblgen_use_alloca.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'transform.structured.promote' op ""attribute 'use_alloca' failed to satisfy constraint: unit attribute");

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'transform.structured.promote' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PromoteOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PromoteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> PromoteOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange PromoteOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PromoteOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PromoteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> PromoteOp::getTransformed() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr PromoteOp::getOperandsToPromoteAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandsToPromoteAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr PromoteOp::getOperandsToPromote() {
  auto attr = getOperandsToPromoteAttr();
  return attr;
}

::mlir::ArrayAttr PromoteOp::getUseFullTileBuffersAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getUseFullTileBuffersAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr PromoteOp::getUseFullTileBuffers() {
  auto attr = getUseFullTileBuffersAttr();
  return attr;
}

::mlir::UnitAttr PromoteOp::getUseFullTilesByDefaultAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getUseFullTilesByDefaultAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool PromoteOp::getUseFullTilesByDefault() {
  auto attr = getUseFullTilesByDefaultAttr();
  return attr != nullptr;
}

::mlir::UnitAttr PromoteOp::getUseAllocaAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getUseAllocaAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool PromoteOp::getUseAlloca() {
  auto attr = getUseAllocaAttr();
  return attr != nullptr;
}

::mlir::IntegerAttr PromoteOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> PromoteOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void PromoteOp::setOperandsToPromoteAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOperandsToPromoteAttrName(), attr);
}

void PromoteOp::setUseFullTileBuffersAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getUseFullTileBuffersAttrName(), attr);
}

void PromoteOp::setUseFullTilesByDefaultAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getUseFullTilesByDefaultAttrName(), attr);
}

void PromoteOp::setUseAllocaAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getUseAllocaAttrName(), attr);
}

void PromoteOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

::mlir::Attribute PromoteOp::removeUse_full_tiles_by_defaultAttr() {
  return (*this)->removeAttr(getUseFullTilesByDefaultAttrName());
}

::mlir::Attribute PromoteOp::removeUse_allocaAttr() {
  return (*this)->removeAttr(getUseAllocaAttrName());
}

::mlir::Attribute PromoteOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

void PromoteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::ArrayAttr operands_to_promote, ::mlir::ArrayAttr use_full_tile_buffers, /*optional*/::mlir::UnitAttr use_full_tiles_by_default, /*optional*/::mlir::UnitAttr use_alloca, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(target);
  if (operands_to_promote) {
  odsState.addAttribute(getOperandsToPromoteAttrName(odsState.name), operands_to_promote);
  }
  if (use_full_tile_buffers) {
  odsState.addAttribute(getUseFullTileBuffersAttrName(odsState.name), use_full_tile_buffers);
  }
  if (use_full_tiles_by_default) {
  odsState.addAttribute(getUseFullTilesByDefaultAttrName(odsState.name), use_full_tiles_by_default);
  }
  if (use_alloca) {
  odsState.addAttribute(getUseAllocaAttrName(odsState.name), use_alloca);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  odsState.addTypes(transformed);
}

void PromoteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr operands_to_promote, ::mlir::ArrayAttr use_full_tile_buffers, /*optional*/::mlir::UnitAttr use_full_tiles_by_default, /*optional*/::mlir::UnitAttr use_alloca, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(target);
  if (operands_to_promote) {
  odsState.addAttribute(getOperandsToPromoteAttrName(odsState.name), operands_to_promote);
  }
  if (use_full_tile_buffers) {
  odsState.addAttribute(getUseFullTileBuffersAttrName(odsState.name), use_full_tile_buffers);
  }
  if (use_full_tiles_by_default) {
  odsState.addAttribute(getUseFullTilesByDefaultAttrName(odsState.name), use_full_tiles_by_default);
  }
  if (use_alloca) {
  odsState.addAttribute(getUseAllocaAttrName(odsState.name), use_alloca);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PromoteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::ArrayAttr operands_to_promote, ::mlir::ArrayAttr use_full_tile_buffers, /*optional*/bool use_full_tiles_by_default, /*optional*/bool use_alloca, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(target);
  odsState.addAttribute(getOperandsToPromoteAttrName(odsState.name), operands_to_promote);
  odsState.addAttribute(getUseFullTileBuffersAttrName(odsState.name), use_full_tile_buffers);
  if (use_full_tiles_by_default) {
  odsState.addAttribute(getUseFullTilesByDefaultAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (use_alloca) {
  odsState.addAttribute(getUseAllocaAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  odsState.addTypes(transformed);
}

void PromoteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr operands_to_promote, ::mlir::ArrayAttr use_full_tile_buffers, /*optional*/bool use_full_tiles_by_default, /*optional*/bool use_alloca, /*optional*/::mlir::IntegerAttr alignment) {
  odsState.addOperands(target);
  odsState.addAttribute(getOperandsToPromoteAttrName(odsState.name), operands_to_promote);
  odsState.addAttribute(getUseFullTileBuffersAttrName(odsState.name), use_full_tile_buffers);
  if (use_full_tiles_by_default) {
  odsState.addAttribute(getUseFullTilesByDefaultAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (use_alloca) {
  odsState.addAttribute(getUseAllocaAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PromoteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void PromoteOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[1])) {
     attributes.append(attrNames[1], odsBuilder.getI64ArrayAttr({}));
  }
  if (!attributes.get(attrNames[3])) {
     attributes.append(attrNames[3], odsBuilder.getBoolArrayAttr({}));
  }
}

::mlir::LogicalResult PromoteOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_operands_to_promote;
  ::mlir::Attribute tblgen_use_alloca;
  ::mlir::Attribute tblgen_use_full_tile_buffers;
  ::mlir::Attribute tblgen_use_full_tiles_by_default;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getOperandsToPromoteAttrName()) {
      tblgen_operands_to_promote = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getUseAllocaAttrName()) {
      tblgen_use_alloca = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getUseFullTileBuffersAttrName()) {
      tblgen_use_full_tile_buffers = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getUseFullTilesByDefaultAttrName()) {
      tblgen_use_full_tiles_by_default = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_operands_to_promote, "operands_to_promote")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps7(*this, tblgen_use_full_tile_buffers, "use_full_tile_buffers")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps8(*this, tblgen_use_full_tiles_by_default, "use_full_tiles_by_default")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps8(*this, tblgen_use_alloca, "use_alloca")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps4(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PromoteOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PromoteOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(targetOperands, odsBuildableType0, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PromoteOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::PromoteOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::ScalarizeOp definitions
//===----------------------------------------------------------------------===//

ScalarizeOpAdaptor::ScalarizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.scalarize", odsAttrs.getContext());
}

ScalarizeOpAdaptor::ScalarizeOpAdaptor(ScalarizeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ScalarizeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ScalarizeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ScalarizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ScalarizeOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ScalarizeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ScalarizeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ScalarizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ScalarizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> ScalarizeOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ScalarizeOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ScalarizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ScalarizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> ScalarizeOp::getResult() {
  return *getODSResults(0).begin();
}

void ScalarizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value target) {
  odsState.addOperands(target);
  odsState.addTypes(result);
}

void ScalarizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target) {
  odsState.addOperands(target);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ScalarizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ScalarizeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ScalarizeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ScalarizeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(targetOperands, odsBuildableType0, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ScalarizeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::ScalarizeOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::SplitOp definitions
//===----------------------------------------------------------------------===//

SplitOpAdaptor::SplitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.split", odsAttrs.getContext());
}

SplitOpAdaptor::SplitOpAdaptor(SplitOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SplitOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SplitOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange SplitOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SplitOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::Value SplitOpAdaptor::getDynamicSplitPoint() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr SplitOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr SplitOpAdaptor::getDimensionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SplitOp::getDimensionAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t SplitOpAdaptor::getDimension() {
  auto attr = getDimensionAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr SplitOpAdaptor::getStaticSplitPointAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SplitOp::getStaticSplitPointAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t SplitOpAdaptor::getStaticSplitPoint() {
  auto attr = getStaticSplitPointAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult SplitOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dimension;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'transform.structured.split' op ""requires attribute 'dimension'");
    if (namedAttrIt->getName() == SplitOp::getDimensionAttrName(*odsOpName)) {
      tblgen_dimension = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_split_point;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'transform.structured.split' op ""requires attribute 'static_split_point'");
    if (namedAttrIt->getName() == SplitOp::getStaticSplitPointAttrName(*odsOpName)) {
      tblgen_static_split_point = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_dimension && !(((tblgen_dimension.isa<::mlir::IntegerAttr>())) && ((tblgen_dimension.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'transform.structured.split' op ""attribute 'dimension' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_static_split_point && !(((tblgen_static_split_point.isa<::mlir::IntegerAttr>())) && ((tblgen_static_split_point.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'transform.structured.split' op ""attribute 'static_split_point' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

void SplitOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "first");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "second");
}

std::pair<unsigned, unsigned> SplitOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range SplitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> SplitOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::Value SplitOp::getDynamicSplitPoint() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange SplitOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SplitOp::getDynamicSplitPointMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SplitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SplitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> SplitOp::getFirst() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::pdl::OperationType> SplitOp::getSecond() {
  return *getODSResults(1).begin();
}

::mlir::IntegerAttr SplitOp::getDimensionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDimensionAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t SplitOp::getDimension() {
  auto attr = getDimensionAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr SplitOp::getStaticSplitPointAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStaticSplitPointAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t SplitOp::getStaticSplitPoint() {
  auto attr = getStaticSplitPointAttr();
  return attr.getValue().getZExtValue();
}

void SplitOp::setDimensionAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getDimensionAttrName(), attr);
}

void SplitOp::setStaticSplitPointAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getStaticSplitPointAttrName(), attr);
}

void SplitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type first, ::mlir::Type second, ::mlir::Value target, ::mlir::IntegerAttr dimension, /*optional*/::mlir::Value dynamic_split_point, ::mlir::IntegerAttr static_split_point) {
  odsState.addOperands(target);
  if (dynamic_split_point)
    odsState.addOperands(dynamic_split_point);
  odsState.addAttribute(getDimensionAttrName(odsState.name), dimension);
  odsState.addAttribute(getStaticSplitPointAttrName(odsState.name), static_split_point);
  odsState.addTypes(first);
  odsState.addTypes(second);
}

void SplitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr dimension, /*optional*/::mlir::Value dynamic_split_point, ::mlir::IntegerAttr static_split_point) {
  odsState.addOperands(target);
  if (dynamic_split_point)
    odsState.addOperands(dynamic_split_point);
  odsState.addAttribute(getDimensionAttrName(odsState.name), dimension);
  odsState.addAttribute(getStaticSplitPointAttrName(odsState.name), static_split_point);
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SplitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type first, ::mlir::Type second, ::mlir::Value target, uint64_t dimension, /*optional*/::mlir::Value dynamic_split_point, uint64_t static_split_point) {
  odsState.addOperands(target);
  if (dynamic_split_point)
    odsState.addOperands(dynamic_split_point);
  odsState.addAttribute(getDimensionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), dimension));
  odsState.addAttribute(getStaticSplitPointAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), static_split_point));
  odsState.addTypes(first);
  odsState.addTypes(second);
}

void SplitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t dimension, /*optional*/::mlir::Value dynamic_split_point, uint64_t static_split_point) {
  odsState.addOperands(target);
  if (dynamic_split_point)
    odsState.addOperands(dynamic_split_point);
  odsState.addAttribute(getDimensionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), dimension));
  odsState.addAttribute(getStaticSplitPointAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), static_split_point));
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SplitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SplitOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dimension;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'dimension'");
    if (namedAttrIt->getName() == getDimensionAttrName()) {
      tblgen_dimension = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_split_point;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'static_split_point'");
    if (namedAttrIt->getName() == getStaticSplitPointAttrName()) {
      tblgen_static_split_point = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps4(*this, tblgen_dimension, "dimension")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps4(*this, tblgen_static_split_point, "static_split_point")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SplitOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::SplitOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::SplitReductionOp definitions
//===----------------------------------------------------------------------===//

SplitReductionOpAdaptor::SplitReductionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.split_reduction", odsAttrs.getContext());
}

SplitReductionOpAdaptor::SplitReductionOpAdaptor(SplitReductionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SplitReductionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SplitReductionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SplitReductionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SplitReductionOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SplitReductionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr SplitReductionOpAdaptor::getSplitFactorAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SplitReductionOp::getSplitFactorAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint64_t SplitReductionOpAdaptor::getSplitFactor() {
  auto attr = getSplitFactorAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr SplitReductionOpAdaptor::getInsertSplitDimensionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SplitReductionOp::getInsertSplitDimensionAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint64_t SplitReductionOpAdaptor::getInsertSplitDimension() {
  auto attr = getInsertSplitDimensionAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr SplitReductionOpAdaptor::getUseScalingAlgorithmAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SplitReductionOp::getUseScalingAlgorithmAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool SplitReductionOpAdaptor::getUseScalingAlgorithm() {
  auto attr = getUseScalingAlgorithmAttr();
  return attr != nullptr;
}

::mlir::UnitAttr SplitReductionOpAdaptor::getUseAllocAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SplitReductionOp::getUseAllocAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool SplitReductionOpAdaptor::getUseAlloc() {
  auto attr = getUseAllocAttr();
  return attr != nullptr;
}

::mlir::LogicalResult SplitReductionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_insert_split_dimension;
  ::mlir::Attribute tblgen_split_factor;
  ::mlir::Attribute tblgen_use_alloc;
  ::mlir::Attribute tblgen_use_scaling_algorithm;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == SplitReductionOp::getInsertSplitDimensionAttrName(*odsOpName)) {
      tblgen_insert_split_dimension = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == SplitReductionOp::getSplitFactorAttrName(*odsOpName)) {
      tblgen_split_factor = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == SplitReductionOp::getUseAllocAttrName(*odsOpName)) {
      tblgen_use_alloc = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == SplitReductionOp::getUseScalingAlgorithmAttrName(*odsOpName)) {
      tblgen_use_scaling_algorithm = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_split_factor && !(((tblgen_split_factor.isa<::mlir::IntegerAttr>())) && ((tblgen_split_factor.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'transform.structured.split_reduction' op ""attribute 'split_factor' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_insert_split_dimension && !(((tblgen_insert_split_dimension.isa<::mlir::IntegerAttr>())) && ((tblgen_insert_split_dimension.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'transform.structured.split_reduction' op ""attribute 'insert_split_dimension' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_use_scaling_algorithm && !((tblgen_use_scaling_algorithm.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'transform.structured.split_reduction' op ""attribute 'use_scaling_algorithm' failed to satisfy constraint: unit attribute");

  if (tblgen_use_alloc && !((tblgen_use_alloc.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'transform.structured.split_reduction' op ""attribute 'use_alloc' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

void SplitReductionOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "init_or_alloc_op");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "fill_op");
  auto resultGroup2 = getODSResults(2);
  if (!llvm::empty(resultGroup2))
    setNameFn(*resultGroup2.begin(), "split_linalg_op");
  auto resultGroup3 = getODSResults(3);
  if (!llvm::empty(resultGroup3))
    setNameFn(*resultGroup3.begin(), "combining_linalg_op");
}

std::pair<unsigned, unsigned> SplitReductionOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SplitReductionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> SplitReductionOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SplitReductionOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SplitReductionOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SplitReductionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> SplitReductionOp::getInitOrAllocOp() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::pdl::OperationType> SplitReductionOp::getFillOp() {
  return *getODSResults(1).begin();
}

::mlir::TypedValue<::mlir::pdl::OperationType> SplitReductionOp::getSplitLinalgOp() {
  return *getODSResults(2).begin();
}

::mlir::TypedValue<::mlir::pdl::OperationType> SplitReductionOp::getCombiningLinalgOp() {
  return *getODSResults(3).begin();
}

::mlir::IntegerAttr SplitReductionOp::getSplitFactorAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSplitFactorAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t SplitReductionOp::getSplitFactor() {
  auto attr = getSplitFactorAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr SplitReductionOp::getInsertSplitDimensionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getInsertSplitDimensionAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t SplitReductionOp::getInsertSplitDimension() {
  auto attr = getInsertSplitDimensionAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr SplitReductionOp::getUseScalingAlgorithmAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getUseScalingAlgorithmAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool SplitReductionOp::getUseScalingAlgorithm() {
  auto attr = getUseScalingAlgorithmAttr();
  return attr != nullptr;
}

::mlir::UnitAttr SplitReductionOp::getUseAllocAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getUseAllocAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool SplitReductionOp::getUseAlloc() {
  auto attr = getUseAllocAttr();
  return attr != nullptr;
}

void SplitReductionOp::setSplitFactorAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getSplitFactorAttrName(), attr);
}

void SplitReductionOp::setInsertSplitDimensionAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getInsertSplitDimensionAttrName(), attr);
}

void SplitReductionOp::setUseScalingAlgorithmAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getUseScalingAlgorithmAttrName(), attr);
}

void SplitReductionOp::setUseAllocAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getUseAllocAttrName(), attr);
}

::mlir::Attribute SplitReductionOp::removeUse_scaling_algorithmAttr() {
  return (*this)->removeAttr(getUseScalingAlgorithmAttrName());
}

::mlir::Attribute SplitReductionOp::removeUse_allocAttr() {
  return (*this)->removeAttr(getUseAllocAttrName());
}

void SplitReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type init_or_alloc_op, ::mlir::Type fill_op, ::mlir::Type split_linalg_op, ::mlir::Type combining_linalg_op, ::mlir::Value target, ::mlir::IntegerAttr split_factor, ::mlir::IntegerAttr insert_split_dimension, /*optional*/::mlir::UnitAttr use_scaling_algorithm, /*optional*/::mlir::UnitAttr use_alloc) {
  odsState.addOperands(target);
  if (split_factor) {
  odsState.addAttribute(getSplitFactorAttrName(odsState.name), split_factor);
  }
  if (insert_split_dimension) {
  odsState.addAttribute(getInsertSplitDimensionAttrName(odsState.name), insert_split_dimension);
  }
  if (use_scaling_algorithm) {
  odsState.addAttribute(getUseScalingAlgorithmAttrName(odsState.name), use_scaling_algorithm);
  }
  if (use_alloc) {
  odsState.addAttribute(getUseAllocAttrName(odsState.name), use_alloc);
  }
  odsState.addTypes(init_or_alloc_op);
  odsState.addTypes(fill_op);
  odsState.addTypes(split_linalg_op);
  odsState.addTypes(combining_linalg_op);
}

void SplitReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr split_factor, ::mlir::IntegerAttr insert_split_dimension, /*optional*/::mlir::UnitAttr use_scaling_algorithm, /*optional*/::mlir::UnitAttr use_alloc) {
  odsState.addOperands(target);
  if (split_factor) {
  odsState.addAttribute(getSplitFactorAttrName(odsState.name), split_factor);
  }
  if (insert_split_dimension) {
  odsState.addAttribute(getInsertSplitDimensionAttrName(odsState.name), insert_split_dimension);
  }
  if (use_scaling_algorithm) {
  odsState.addAttribute(getUseScalingAlgorithmAttrName(odsState.name), use_scaling_algorithm);
  }
  if (use_alloc) {
  odsState.addAttribute(getUseAllocAttrName(odsState.name), use_alloc);
  }
  assert(resultTypes.size() == 4u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SplitReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type init_or_alloc_op, ::mlir::Type fill_op, ::mlir::Type split_linalg_op, ::mlir::Type combining_linalg_op, ::mlir::Value target, uint64_t split_factor, uint64_t insert_split_dimension, /*optional*/bool use_scaling_algorithm, /*optional*/bool use_alloc) {
  odsState.addOperands(target);
  odsState.addAttribute(getSplitFactorAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), split_factor));
  odsState.addAttribute(getInsertSplitDimensionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), insert_split_dimension));
  if (use_scaling_algorithm) {
  odsState.addAttribute(getUseScalingAlgorithmAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (use_alloc) {
  odsState.addAttribute(getUseAllocAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  odsState.addTypes(init_or_alloc_op);
  odsState.addTypes(fill_op);
  odsState.addTypes(split_linalg_op);
  odsState.addTypes(combining_linalg_op);
}

void SplitReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t split_factor, uint64_t insert_split_dimension, /*optional*/bool use_scaling_algorithm, /*optional*/bool use_alloc) {
  odsState.addOperands(target);
  odsState.addAttribute(getSplitFactorAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), split_factor));
  odsState.addAttribute(getInsertSplitDimensionAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), insert_split_dimension));
  if (use_scaling_algorithm) {
  odsState.addAttribute(getUseScalingAlgorithmAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (use_alloc) {
  odsState.addAttribute(getUseAllocAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 4u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SplitReductionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 4u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SplitReductionOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[1])) {
     attributes.append(attrNames[1], odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), {}));
  }
  if (!attributes.get(attrNames[0])) {
     attributes.append(attrNames[0], odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), {}));
  }
}

::mlir::LogicalResult SplitReductionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_insert_split_dimension;
  ::mlir::Attribute tblgen_split_factor;
  ::mlir::Attribute tblgen_use_alloc;
  ::mlir::Attribute tblgen_use_scaling_algorithm;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getInsertSplitDimensionAttrName()) {
      tblgen_insert_split_dimension = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getSplitFactorAttrName()) {
      tblgen_split_factor = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getUseAllocAttrName()) {
      tblgen_use_alloc = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getUseScalingAlgorithmAttrName()) {
      tblgen_use_scaling_algorithm = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps4(*this, tblgen_split_factor, "split_factor")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps4(*this, tblgen_insert_split_dimension, "insert_split_dimension")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps8(*this, tblgen_use_scaling_algorithm, "use_scaling_algorithm")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps8(*this, tblgen_use_alloc, "use_alloc")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSResults(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSResults(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SplitReductionOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SplitReductionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  result.addTypes(odsBuildableType0);
  result.addTypes(odsBuildableType0);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(targetOperands, odsBuildableType0, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SplitReductionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::SplitReductionOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::TileOp definitions
//===----------------------------------------------------------------------===//

TileOpAdaptor::TileOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.tile", odsAttrs.getContext());
}

TileOpAdaptor::TileOpAdaptor(TileOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange TileOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TileOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange TileOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TileOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange TileOpAdaptor::getDynamicSizes() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr TileOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr TileOpAdaptor::getStaticSizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TileOp::getStaticSizesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TileOpAdaptor::getStaticSizes() {
  auto attr = getStaticSizesAttr();
  return attr;
}

::mlir::ArrayAttr TileOpAdaptor::getInterchangeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TileOp::getInterchangeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TileOpAdaptor::getInterchange() {
  auto attr = getInterchangeAttr();
  return attr;
}

::mlir::LogicalResult TileOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_interchange;
  ::mlir::Attribute tblgen_static_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == TileOp::getInterchangeAttrName(*odsOpName)) {
      tblgen_interchange = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == TileOp::getStaticSizesAttrName(*odsOpName)) {
      tblgen_static_sizes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_static_sizes && !(((tblgen_static_sizes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_sizes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.structured.tile' op ""attribute 'static_sizes' failed to satisfy constraint: 64-bit integer array attribute");

  if (tblgen_interchange && !(((tblgen_interchange.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_interchange.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.structured.tile' op ""attribute 'interchange' failed to satisfy constraint: 64-bit integer array attribute");
  return ::mlir::success();
}

void TileOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "tiled_linalg_op");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "loops");
}

std::pair<unsigned, unsigned> TileOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range TileOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> TileOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range TileOp::getDynamicSizes() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange TileOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange TileOp::getDynamicSizesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TileOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range TileOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> TileOp::getTiledLinalgOp() {
  return *getODSResults(0).begin();
}

::mlir::Operation::result_range TileOp::getLoops() {
  return getODSResults(1);
}

::mlir::ArrayAttr TileOp::getStaticSizesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getStaticSizesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TileOp::getStaticSizes() {
  auto attr = getStaticSizesAttr();
  return attr;
}

::mlir::ArrayAttr TileOp::getInterchangeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getInterchangeAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TileOp::getInterchange() {
  auto attr = getInterchangeAttr();
  return attr;
}

void TileOp::setStaticSizesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStaticSizesAttrName(), attr);
}

void TileOp::setInterchangeAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getInterchangeAttrName(), attr);
}

void TileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type tiled_linalg_op, ::mlir::TypeRange loops, ::mlir::Value target, ::mlir::ValueRange dynamic_sizes, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr interchange) {
  odsState.addOperands(target);
  odsState.addOperands(dynamic_sizes);
  if (static_sizes) {
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), static_sizes);
  }
  if (interchange) {
  odsState.addAttribute(getInterchangeAttrName(odsState.name), interchange);
  }
  odsState.addTypes(tiled_linalg_op);
  odsState.addTypes(loops);
}

void TileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ValueRange dynamic_sizes, ::mlir::ArrayAttr static_sizes, ::mlir::ArrayAttr interchange) {
  odsState.addOperands(target);
  odsState.addOperands(dynamic_sizes);
  if (static_sizes) {
  odsState.addAttribute(getStaticSizesAttrName(odsState.name), static_sizes);
  }
  if (interchange) {
  odsState.addAttribute(getInterchangeAttrName(odsState.name), interchange);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TileOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void TileOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[1])) {
     attributes.append(attrNames[1], odsBuilder.getI64ArrayAttr({}));
  }
  if (!attributes.get(attrNames[0])) {
     attributes.append(attrNames[0], odsBuilder.getI64ArrayAttr({}));
  }
}

::mlir::LogicalResult TileOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_interchange;
  ::mlir::Attribute tblgen_static_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getInterchangeAttrName()) {
      tblgen_interchange = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getStaticSizesAttrName()) {
      tblgen_static_sizes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_static_sizes, "static_sizes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_interchange, "interchange")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TileOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::TileOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::TileToForeachThreadOp definitions
//===----------------------------------------------------------------------===//

TileToForeachThreadOpAdaptor::TileToForeachThreadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.tile_to_foreach_thread_op", odsAttrs.getContext());
}

TileToForeachThreadOpAdaptor::TileToForeachThreadOpAdaptor(TileToForeachThreadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange TileToForeachThreadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TileToForeachThreadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TileToForeachThreadOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::ValueRange TileToForeachThreadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TileToForeachThreadOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange TileToForeachThreadOpAdaptor::getNumThreads() {
  return getODSOperands(1);
}

::mlir::ValueRange TileToForeachThreadOpAdaptor::getTileSizes() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr TileToForeachThreadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr TileToForeachThreadOpAdaptor::getStaticNumThreadsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, TileToForeachThreadOp::getStaticNumThreadsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TileToForeachThreadOpAdaptor::getStaticNumThreads() {
  auto attr = getStaticNumThreadsAttr();
  return attr;
}

::mlir::ArrayAttr TileToForeachThreadOpAdaptor::getStaticTileSizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, TileToForeachThreadOp::getStaticTileSizesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr TileToForeachThreadOpAdaptor::getStaticTileSizes() {
  auto attr = getStaticTileSizesAttr();
  return attr;
}

::mlir::ArrayAttr TileToForeachThreadOpAdaptor::getThreadDimMappingAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, TileToForeachThreadOp::getThreadDimMappingAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > TileToForeachThreadOpAdaptor::getThreadDimMapping() {
  auto attr = getThreadDimMappingAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult TileToForeachThreadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'transform.structured.tile_to_foreach_thread_op' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == TileToForeachThreadOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_num_threads;
  ::mlir::Attribute tblgen_static_tile_sizes;
  ::mlir::Attribute tblgen_thread_dim_mapping;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == TileToForeachThreadOp::getStaticNumThreadsAttrName(*odsOpName)) {
      tblgen_static_num_threads = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == TileToForeachThreadOp::getStaticTileSizesAttrName(*odsOpName)) {
      tblgen_static_tile_sizes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == TileToForeachThreadOp::getThreadDimMappingAttrName(*odsOpName)) {
      tblgen_thread_dim_mapping = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'transform.structured.tile_to_foreach_thread_op' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_static_num_threads && !(((tblgen_static_num_threads.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_num_threads.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.structured.tile_to_foreach_thread_op' op ""attribute 'static_num_threads' failed to satisfy constraint: 64-bit integer array attribute");

  if (tblgen_static_tile_sizes && !(((tblgen_static_tile_sizes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_static_tile_sizes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.structured.tile_to_foreach_thread_op' op ""attribute 'static_tile_sizes' failed to satisfy constraint: 64-bit integer array attribute");

  if (tblgen_thread_dim_mapping && !(((tblgen_thread_dim_mapping.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_thread_dim_mapping.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.structured.tile_to_foreach_thread_op' op ""attribute 'thread_dim_mapping' failed to satisfy constraint: 64-bit integer array attribute");
  return ::mlir::success();
}

void TileToForeachThreadOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "foreach_thread_op");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "tiled_op");
}

std::pair<unsigned, unsigned> TileToForeachThreadOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range TileToForeachThreadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> TileToForeachThreadOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range TileToForeachThreadOp::getNumThreads() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range TileToForeachThreadOp::getTileSizes() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange TileToForeachThreadOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TileToForeachThreadOp::getNumThreadsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TileToForeachThreadOp::getTileSizesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> TileToForeachThreadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TileToForeachThreadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> TileToForeachThreadOp::getForeachThreadOp() {
  return *getODSResults(0).begin();
}

::mlir::TypedValue<::mlir::pdl::OperationType> TileToForeachThreadOp::getTiledOp() {
  return *getODSResults(1).begin();
}

::mlir::ArrayAttr TileToForeachThreadOp::getStaticNumThreadsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStaticNumThreadsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TileToForeachThreadOp::getStaticNumThreads() {
  auto attr = getStaticNumThreadsAttr();
  return attr;
}

::mlir::ArrayAttr TileToForeachThreadOp::getStaticTileSizesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getStaticTileSizesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TileToForeachThreadOp::getStaticTileSizes() {
  auto attr = getStaticTileSizesAttr();
  return attr;
}

::mlir::ArrayAttr TileToForeachThreadOp::getThreadDimMappingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getThreadDimMappingAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > TileToForeachThreadOp::getThreadDimMapping() {
  auto attr = getThreadDimMappingAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void TileToForeachThreadOp::setStaticNumThreadsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStaticNumThreadsAttrName(), attr);
}

void TileToForeachThreadOp::setStaticTileSizesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getStaticTileSizesAttrName(), attr);
}

void TileToForeachThreadOp::setThreadDimMappingAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getThreadDimMappingAttrName(), attr);
}

::mlir::Attribute TileToForeachThreadOp::removeThread_dim_mappingAttr() {
  return (*this)->removeAttr(getThreadDimMappingAttrName());
}

void TileToForeachThreadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type foreach_thread_op, ::mlir::Type tiled_op, ::mlir::Value target, ::mlir::ValueRange num_threads, ::mlir::ValueRange tile_sizes, ::mlir::ArrayAttr static_num_threads, ::mlir::ArrayAttr static_tile_sizes, /*optional*/::mlir::ArrayAttr thread_dim_mapping) {
  odsState.addOperands(target);
  odsState.addOperands(num_threads);
  odsState.addOperands(tile_sizes);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(num_threads.size()), static_cast<int32_t>(tile_sizes.size())}));
  if (static_num_threads) {
  odsState.addAttribute(getStaticNumThreadsAttrName(odsState.name), static_num_threads);
  }
  if (static_tile_sizes) {
  odsState.addAttribute(getStaticTileSizesAttrName(odsState.name), static_tile_sizes);
  }
  if (thread_dim_mapping) {
  odsState.addAttribute(getThreadDimMappingAttrName(odsState.name), thread_dim_mapping);
  }
  odsState.addTypes(foreach_thread_op);
  odsState.addTypes(tiled_op);
}

void TileToForeachThreadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ValueRange num_threads, ::mlir::ValueRange tile_sizes, ::mlir::ArrayAttr static_num_threads, ::mlir::ArrayAttr static_tile_sizes, /*optional*/::mlir::ArrayAttr thread_dim_mapping) {
  odsState.addOperands(target);
  odsState.addOperands(num_threads);
  odsState.addOperands(tile_sizes);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getDenseI32ArrayAttr({1, static_cast<int32_t>(num_threads.size()), static_cast<int32_t>(tile_sizes.size())}));
  if (static_num_threads) {
  odsState.addAttribute(getStaticNumThreadsAttrName(odsState.name), static_num_threads);
  }
  if (static_tile_sizes) {
  odsState.addAttribute(getStaticTileSizesAttrName(odsState.name), static_tile_sizes);
  }
  if (thread_dim_mapping) {
  odsState.addAttribute(getThreadDimMappingAttrName(odsState.name), thread_dim_mapping);
  }
  assert(resultTypes.size() == 2u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TileToForeachThreadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 2u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void TileToForeachThreadOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[1])) {
     attributes.append(attrNames[1], odsBuilder.getI64ArrayAttr({}));
  }
  if (!attributes.get(attrNames[2])) {
     attributes.append(attrNames[2], odsBuilder.getI64ArrayAttr({}));
  }
}

::mlir::LogicalResult TileToForeachThreadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_static_num_threads;
  ::mlir::Attribute tblgen_static_tile_sizes;
  ::mlir::Attribute tblgen_thread_dim_mapping;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getStaticNumThreadsAttrName()) {
      tblgen_static_num_threads = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getStaticTileSizesAttrName()) {
      tblgen_static_tile_sizes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getThreadDimMappingAttrName()) {
      tblgen_thread_dim_mapping = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_static_num_threads, "static_num_threads")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_static_tile_sizes, "static_tile_sizes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_thread_dim_mapping, "thread_dim_mapping")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TileToForeachThreadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TileToForeachThreadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> num_threadsOperands;
  ::llvm::SMLoc num_threadsOperandsLoc;
  (void)num_threadsOperandsLoc;
  ::mlir::ArrayAttr static_num_threadsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> tile_sizesOperands;
  ::llvm::SMLoc tile_sizesOperandsLoc;
  (void)tile_sizesOperandsLoc;
  ::mlir::ArrayAttr static_tile_sizesAttr;
  ::mlir::ArrayAttr thread_dim_mappingAttr;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  bool num_threadsClause = false;
  bool tile_sizesClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("num_threads"))) {

  if (num_threadsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`num_threads` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  num_threadsClause = true;
  {
    num_threadsOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, num_threadsOperands, static_num_threadsAttr, ShapedType::kDynamicSize))
      return ::mlir::failure();
    result.addAttribute("static_num_threads", static_num_threadsAttr);
  }
    } else if (succeeded(parser.parseOptionalKeyword("tile_sizes"))) {

  if (tile_sizesClause) {
    return parser.emitError(parser.getNameLoc())
          << "`tile_sizes` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  tile_sizesClause = true;
  {
    tile_sizesOperandsLoc = parser.getCurrentLocation();
    if (parseDynamicIndexList(parser, tile_sizesOperands, static_tile_sizesAttr, ShapedType::kDynamicSize))
      return ::mlir::failure();
    result.addAttribute("static_tile_sizes", static_tile_sizesAttr);
  }
    } else  {
    break;
  }
}
  if (succeeded(parser.parseOptionalLParen())) {
  if (parser.parseKeyword("mapped"))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();
  if (parser.parseKeyword("dims"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(thread_dim_mappingAttr, parser.getBuilder().getType<::mlir::NoneType>(), "thread_dim_mapping",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getDenseI32ArrayAttr({1, static_cast<int32_t>(num_threadsOperands.size()), static_cast<int32_t>(tile_sizesOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(targetOperands, odsBuildableType0, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(num_threadsOperands, odsBuildableType0, num_threadsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(tile_sizesOperands, odsBuildableType0, tile_sizesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TileToForeachThreadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter << ' ' << " ";
  if (false || getNumThreads().size() || getStaticNumThreadsAttr()) {
  _odsPrinter << ' ' << "num_threads";
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getNumThreads(), getStaticNumThreadsAttr(), ShapedType::kDynamicSize);
  }
  if (false || getTileSizes().size() || getStaticTileSizesAttr()) {
  _odsPrinter << ' ' << "tile_sizes";
  _odsPrinter << ' ';
  printDynamicIndexList(_odsPrinter, *this, getTileSizes(), getStaticTileSizesAttr(), ShapedType::kDynamicSize);
  }
  if ((*this)->getAttr("thread_dim_mapping")) {
  _odsPrinter << "(";
  _odsPrinter << "mapped";
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ' << "dims";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getThreadDimMappingAttr());
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "static_num_threads", "static_tile_sizes", "thread_dim_mapping"});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::TileToForeachThreadOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::VectorizeOp definitions
//===----------------------------------------------------------------------===//

VectorizeOpAdaptor::VectorizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.structured.vectorize", odsAttrs.getContext());
}

VectorizeOpAdaptor::VectorizeOpAdaptor(VectorizeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VectorizeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VectorizeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VectorizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VectorizeOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr VectorizeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr VectorizeOpAdaptor::getVectorizePaddingAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, VectorizeOp::getVectorizePaddingAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool VectorizeOpAdaptor::getVectorizePadding() {
  auto attr = getVectorizePaddingAttr();
  return attr.getValue();
}

::mlir::LogicalResult VectorizeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_vectorize_padding;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == VectorizeOp::getVectorizePaddingAttrName(*odsOpName)) {
      tblgen_vectorize_padding = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_vectorize_padding && !((tblgen_vectorize_padding.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'transform.structured.vectorize' op ""attribute 'vectorize_padding' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VectorizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VectorizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> VectorizeOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange VectorizeOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VectorizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VectorizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> VectorizeOp::getTransformed() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr VectorizeOp::getVectorizePaddingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getVectorizePaddingAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool VectorizeOp::getVectorizePadding() {
  auto attr = getVectorizePaddingAttr();
  return attr.getValue();
}

void VectorizeOp::setVectorizePaddingAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getVectorizePaddingAttrName(), attr);
}

void VectorizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::BoolAttr vectorize_padding) {
  odsState.addOperands(target);
  if (vectorize_padding) {
  odsState.addAttribute(getVectorizePaddingAttrName(odsState.name), vectorize_padding);
  }
  odsState.addTypes(transformed);
}

void VectorizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::BoolAttr vectorize_padding) {
  odsState.addOperands(target);
  if (vectorize_padding) {
  odsState.addAttribute(getVectorizePaddingAttrName(odsState.name), vectorize_padding);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VectorizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, bool vectorize_padding) {
  odsState.addOperands(target);
  odsState.addAttribute(getVectorizePaddingAttrName(odsState.name), odsBuilder.getBoolAttr(vectorize_padding));
  odsState.addTypes(transformed);
}

void VectorizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, bool vectorize_padding) {
  odsState.addOperands(target);
  odsState.addAttribute(getVectorizePaddingAttrName(odsState.name), odsBuilder.getBoolAttr(vectorize_padding));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VectorizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void VectorizeOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
     attributes.append(attrNames[0], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult VectorizeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_vectorize_padding;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getVectorizePaddingAttrName()) {
      tblgen_vectorize_padding = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps9(*this, tblgen_vectorize_padding, "vectorize_padding")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LinalgTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VectorizeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult VectorizeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(targetOperands, odsBuildableType0, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void VectorizeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::VectorizeOp)


#endif  // GET_OP_CLASSES

