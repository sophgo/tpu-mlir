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
::mlir::transform::GeneralizeOp,
::mlir::transform::InterchangeOp,
::mlir::transform::PadOp,
::mlir::transform::ScalarizeOp,
::mlir::transform::TileOp,
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
  if (attr && !((attr.isa<::mlir::ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgTransformOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: array of arrays of i64";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LinalgTransformOps3(
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

::mlir::Value DecomposeOp::getTarget() {
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

::mlir::Value DecomposeOp::getTransformed() {
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
  if (parser.resolveOperands(targetOperands, odsBuildableType0, result.operands))
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

::mlir::Value GeneralizeOp::getTarget() {
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

::mlir::Value GeneralizeOp::getTransformed() {
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
  if (parser.resolveOperands(targetOperands, odsBuildableType0, result.operands))
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
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getI64ArrayAttr({});
    return attr;
}

::mlir::ArrayAttr InterchangeOpAdaptor::getIteratorInterchange() {
  auto attr = getIteratorInterchangeAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getI64ArrayAttr({});
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

::mlir::Value InterchangeOp::getTarget() {
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

::mlir::Value InterchangeOp::getTransformed() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr InterchangeOp::getIteratorInterchangeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIteratorInterchangeAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr InterchangeOp::getIteratorInterchange() {
  auto attr = getIteratorInterchangeAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getI64ArrayAttr({});
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
  if (parser.resolveOperands(targetOperands, odsBuildableType0, result.operands))
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
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getArrayAttr({});
    return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getPaddingValues() {
  auto attr = getPaddingValuesAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getArrayAttr({});
  return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getPaddingDimensionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PadOp::getPaddingDimensionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getI64ArrayAttr({});
    return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getPaddingDimensions() {
  auto attr = getPaddingDimensionsAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getI64ArrayAttr({});
  return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getPackPaddingsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PadOp::getPackPaddingsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getI64ArrayAttr({});
    return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getPackPaddings() {
  auto attr = getPackPaddingsAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getI64ArrayAttr({});
  return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getHoistPaddingsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PadOp::getHoistPaddingsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getI64ArrayAttr({});
    return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getHoistPaddings() {
  auto attr = getHoistPaddingsAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getI64ArrayAttr({});
  return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getTransposePaddingsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, PadOp::getTransposePaddingsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getArrayAttr({});
    return attr;
}

::mlir::ArrayAttr PadOpAdaptor::getTransposePaddings() {
  auto attr = getTransposePaddingsAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getArrayAttr({});
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

::mlir::Value PadOp::getTarget() {
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

::mlir::Value PadOp::getTransformed() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr PadOp::getPaddingValuesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPaddingValuesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr PadOp::getPaddingValues() {
  auto attr = getPaddingValuesAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getArrayAttr({});
  return attr;
}

::mlir::ArrayAttr PadOp::getPaddingDimensionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPaddingDimensionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr PadOp::getPaddingDimensions() {
  auto attr = getPaddingDimensionsAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getI64ArrayAttr({});
  return attr;
}

::mlir::ArrayAttr PadOp::getPackPaddingsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPackPaddingsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr PadOp::getPackPaddings() {
  auto attr = getPackPaddingsAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getI64ArrayAttr({});
  return attr;
}

::mlir::ArrayAttr PadOp::getHoistPaddingsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getHoistPaddingsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr PadOp::getHoistPaddings() {
  auto attr = getHoistPaddingsAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getI64ArrayAttr({});
  return attr;
}

::mlir::ArrayAttr PadOp::getTransposePaddingsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTransposePaddingsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr PadOp::getTransposePaddings() {
  auto attr = getTransposePaddingsAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getArrayAttr({});
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps1(*this, tblgen_padding_values, "padding_values")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_padding_dimensions, "padding_dimensions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_pack_paddings, "pack_paddings")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_hoist_paddings, "hoist_paddings")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps2(*this, tblgen_transpose_paddings, "transpose_paddings")))
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
  if (parser.resolveOperands(targetOperands, odsBuildableType0, result.operands))
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

::mlir::Value ScalarizeOp::getTarget() {
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

::mlir::Value ScalarizeOp::getResult() {
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
  if (parser.resolveOperands(targetOperands, odsBuildableType0, result.operands))
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
  return {index, 1};
}

::mlir::ValueRange TileOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TileOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr TileOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr TileOpAdaptor::getSizesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TileOp::getSizesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getI64ArrayAttr({});
    return attr;
}

::mlir::ArrayAttr TileOpAdaptor::getSizes() {
  auto attr = getSizesAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getI64ArrayAttr({});
  return attr;
}

::mlir::ArrayAttr TileOpAdaptor::getInterchangeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TileOp::getInterchangeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getI64ArrayAttr({});
    return attr;
}

::mlir::ArrayAttr TileOpAdaptor::getInterchange() {
  auto attr = getInterchangeAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getI64ArrayAttr({});
  return attr;
}

::mlir::LogicalResult TileOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_interchange;
  ::mlir::Attribute tblgen_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == TileOp::getInterchangeAttrName(*odsOpName)) {
      tblgen_interchange = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == TileOp::getSizesAttrName(*odsOpName)) {
      tblgen_sizes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_sizes && !(((tblgen_sizes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_sizes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))); }))))
    return emitError(loc, "'transform.structured.tile' op ""attribute 'sizes' failed to satisfy constraint: 64-bit integer array attribute");

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
  return {index, 1};
}

::mlir::Operation::operand_range TileOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TileOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TileOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
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

::mlir::Value TileOp::getTiledLinalgOp() {
  return *getODSResults(0).begin();
}

::mlir::Operation::result_range TileOp::getLoops() {
  return getODSResults(1);
}

::mlir::ArrayAttr TileOp::getSizesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSizesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TileOp::getSizes() {
  auto attr = getSizesAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getI64ArrayAttr({});
  return attr;
}

::mlir::ArrayAttr TileOp::getInterchangeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getInterchangeAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr TileOp::getInterchange() {
  auto attr = getInterchangeAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getI64ArrayAttr({});
  return attr;
}

void TileOp::setSizesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getSizesAttrName(), attr);
}

void TileOp::setInterchangeAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getInterchangeAttrName(), attr);
}

void TileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type tiled_linalg_op, ::mlir::TypeRange loops, ::mlir::Value target, ::mlir::ArrayAttr sizes, ::mlir::ArrayAttr interchange) {
  odsState.addOperands(target);
  if (sizes) {
  odsState.addAttribute(getSizesAttrName(odsState.name), sizes);
  }
  if (interchange) {
  odsState.addAttribute(getInterchangeAttrName(odsState.name), interchange);
  }
  odsState.addTypes(tiled_linalg_op);
  odsState.addTypes(loops);
}

void TileOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::ArrayAttr sizes, ::mlir::ArrayAttr interchange) {
  odsState.addOperands(target);
  if (sizes) {
  odsState.addAttribute(getSizesAttrName(odsState.name), sizes);
  }
  if (interchange) {
  odsState.addAttribute(getInterchangeAttrName(odsState.name), interchange);
  }
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TileOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TileOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_interchange;
  ::mlir::Attribute tblgen_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getInterchangeAttrName()) {
      tblgen_interchange = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getSizesAttrName()) {
      tblgen_sizes = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps0(*this, tblgen_sizes, "sizes")))
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
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
    return attr;
}

bool VectorizeOpAdaptor::getVectorizePadding() {
  auto attr = getVectorizePaddingAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false).getValue();
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

::mlir::Value VectorizeOp::getTarget() {
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

::mlir::Value VectorizeOp::getTransformed() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr VectorizeOp::getVectorizePaddingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getVectorizePaddingAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool VectorizeOp::getVectorizePadding() {
  auto attr = getVectorizePaddingAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LinalgTransformOps3(*this, tblgen_vectorize_padding, "vectorize_padding")))
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
  if (parser.resolveOperands(targetOperands, odsBuildableType0, result.operands))
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

