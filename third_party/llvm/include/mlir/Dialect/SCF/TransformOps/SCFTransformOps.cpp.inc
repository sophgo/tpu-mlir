/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::transform::GetParentForOp,
::mlir::transform::LoopCoalesceOp,
::mlir::transform::LoopOutlineOp,
::mlir::transform::LoopPeelOp,
::mlir::transform::LoopPipelineOp,
::mlir::transform::LoopUnrollOp,
::mlir::transform::TakeAssumedBranchOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace transform {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SCFTransformOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::transform::TransformHandleTypeInterface>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be TransformHandleTypeInterface instance, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_SCFTransformOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::transform::OperationType>())) && ((type.cast<::mlir::transform::OperationType>().getOperationName() == "scf.for")))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be Transform IR handle to scf.for operations, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_SCFTransformOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((attr.cast<::mlir::IntegerAttr>().getValue().isStrictlyPositive())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_SCFTransformOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_SCFTransformOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_SCFTransformOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_SCFTransformOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::GetParentForOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetParentForOpGenericAdaptorBase::GetParentForOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.loop.get_parent_for", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetParentForOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetParentForOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetParentForOpGenericAdaptorBase::getNumLoopsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GetParentForOp::getNumLoopsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint64_t GetParentForOpGenericAdaptorBase::getNumLoops() {
  auto attr = getNumLoopsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr GetParentForOpGenericAdaptorBase::getAffineAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GetParentForOp::getAffineAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool GetParentForOpGenericAdaptorBase::getAffine() {
  auto attr = getAffineAttr();
  return attr.getValue();
}

} // namespace detail
GetParentForOpAdaptor::GetParentForOpAdaptor(GetParentForOp op) : GetParentForOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult GetParentForOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_affine;
  ::mlir::Attribute tblgen_num_loops;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == GetParentForOp::getAffineAttrName(*odsOpName)) {
      tblgen_affine = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GetParentForOp::getNumLoopsAttrName(*odsOpName)) {
      tblgen_num_loops = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_num_loops && !((((tblgen_num_loops.isa<::mlir::IntegerAttr>())) && ((tblgen_num_loops.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_num_loops.cast<::mlir::IntegerAttr>().getValue().isStrictlyPositive()))))
    return emitError(loc, "'transform.loop.get_parent_for' op ""attribute 'num_loops' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive");

  if (tblgen_affine && !((tblgen_affine.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'transform.loop.get_parent_for' op ""attribute 'affine' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetParentForOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetParentForOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface> GetParentForOp::getTarget() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange GetParentForOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetParentForOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetParentForOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface> GetParentForOp::getParent() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr GetParentForOp::getNumLoopsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNumLoopsAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t GetParentForOp::getNumLoops() {
  auto attr = getNumLoopsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr GetParentForOp::getAffineAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAffineAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool GetParentForOp::getAffine() {
  auto attr = getAffineAttr();
  return attr.getValue();
}

void GetParentForOp::setNumLoopsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getNumLoopsAttrName(), attr);
}

void GetParentForOp::setNumLoops(uint64_t attrValue) {
  (*this)->setAttr(getNumLoopsAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void GetParentForOp::setAffineAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getAffineAttrName(), attr);
}

void GetParentForOp::setAffine(bool attrValue) {
  (*this)->setAttr(getAffineAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void GetParentForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type parent, ::mlir::Value target, ::mlir::IntegerAttr num_loops, ::mlir::BoolAttr affine) {
  odsState.addOperands(target);
  if (num_loops) {
    odsState.addAttribute(getNumLoopsAttrName(odsState.name), num_loops);
  }
  if (affine) {
    odsState.addAttribute(getAffineAttrName(odsState.name), affine);
  }
  odsState.addTypes(parent);
}

void GetParentForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr num_loops, ::mlir::BoolAttr affine) {
  odsState.addOperands(target);
  if (num_loops) {
    odsState.addAttribute(getNumLoopsAttrName(odsState.name), num_loops);
  }
  if (affine) {
    odsState.addAttribute(getAffineAttrName(odsState.name), affine);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetParentForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type parent, ::mlir::Value target, uint64_t num_loops, bool affine) {
  odsState.addOperands(target);
  odsState.addAttribute(getNumLoopsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_loops));
  odsState.addAttribute(getAffineAttrName(odsState.name), odsBuilder.getBoolAttr(affine));
  odsState.addTypes(parent);
}

void GetParentForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t num_loops, bool affine) {
  odsState.addOperands(target);
  odsState.addAttribute(getNumLoopsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_loops));
  odsState.addAttribute(getAffineAttrName(odsState.name), odsBuilder.getBoolAttr(affine));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetParentForOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void GetParentForOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1));
  }
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult GetParentForOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_affine;
  ::mlir::Attribute tblgen_num_loops;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getAffineAttrName()) {
      tblgen_affine = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNumLoopsAttrName()) {
      tblgen_num_loops = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SCFTransformOps0(*this, tblgen_num_loops, "num_loops")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SCFTransformOps1(*this, tblgen_affine, "affine")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SCFTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SCFTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetParentForOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetParentForOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(targetOperands, allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetParentForOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getNumLoopsAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1)))
       elidedAttrs.push_back("num_loops");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getAffineAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("affine");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::GetParentForOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::LoopCoalesceOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LoopCoalesceOpGenericAdaptorBase::LoopCoalesceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.loop.coalesce", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LoopCoalesceOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LoopCoalesceOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
LoopCoalesceOpAdaptor::LoopCoalesceOpAdaptor(LoopCoalesceOp op) : LoopCoalesceOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult LoopCoalesceOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoopCoalesceOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LoopCoalesceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface> LoopCoalesceOp::getTarget() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange LoopCoalesceOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LoopCoalesceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LoopCoalesceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface> LoopCoalesceOp::getTransformed() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface>>(*getODSResults(0).begin());
}

void LoopCoalesceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target) {
  odsState.addOperands(target);
  odsState.addTypes(transformed);
}

void LoopCoalesceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target) {
  odsState.addOperands(target);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopCoalesceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LoopCoalesceOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SCFTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SCFTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoopCoalesceOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LoopCoalesceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> targetTypes;
  ::llvm::ArrayRef<::mlir::Type> transformedTypes;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType target__transformed_functionType;
  if (parser.parseType(target__transformed_functionType))
    return ::mlir::failure();
  targetTypes = target__transformed_functionType.getInputs();
  transformedTypes = target__transformed_functionType.getResults();
  result.addTypes(transformedTypes);
  if (parser.resolveOperands(targetOperands, targetTypes, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoopCoalesceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(::llvm::ArrayRef<::mlir::Type>(getTarget().getType()), ::llvm::ArrayRef<::mlir::Type>(getTransformed().getType()));
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::LoopCoalesceOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::LoopOutlineOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LoopOutlineOpGenericAdaptorBase::LoopOutlineOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.loop.outline", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LoopOutlineOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LoopOutlineOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr LoopOutlineOpGenericAdaptorBase::getFuncNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoopOutlineOp::getFuncNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef LoopOutlineOpGenericAdaptorBase::getFuncName() {
  auto attr = getFuncNameAttr();
  return attr.getValue();
}

} // namespace detail
LoopOutlineOpAdaptor::LoopOutlineOpAdaptor(LoopOutlineOp op) : LoopOutlineOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult LoopOutlineOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_func_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'transform.loop.outline' op ""requires attribute 'func_name'");
    if (namedAttrIt->getName() == LoopOutlineOp::getFuncNameAttrName(*odsOpName)) {
      tblgen_func_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_func_name && !((tblgen_func_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'transform.loop.outline' op ""attribute 'func_name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoopOutlineOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LoopOutlineOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface> LoopOutlineOp::getTarget() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange LoopOutlineOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LoopOutlineOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LoopOutlineOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface> LoopOutlineOp::getTransformed() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface>>(*getODSResults(0).begin());
}

::mlir::StringAttr LoopOutlineOp::getFuncNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFuncNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef LoopOutlineOp::getFuncName() {
  auto attr = getFuncNameAttr();
  return attr.getValue();
}

void LoopOutlineOp::setFuncNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getFuncNameAttrName(), attr);
}

void LoopOutlineOp::setFuncName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getFuncNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void LoopOutlineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::StringAttr func_name) {
  odsState.addOperands(target);
  odsState.addAttribute(getFuncNameAttrName(odsState.name), func_name);
  odsState.addTypes(transformed);
}

void LoopOutlineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::StringAttr func_name) {
  odsState.addOperands(target);
  odsState.addAttribute(getFuncNameAttrName(odsState.name), func_name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopOutlineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::llvm::StringRef func_name) {
  odsState.addOperands(target);
  odsState.addAttribute(getFuncNameAttrName(odsState.name), odsBuilder.getStringAttr(func_name));
  odsState.addTypes(transformed);
}

void LoopOutlineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::llvm::StringRef func_name) {
  odsState.addOperands(target);
  odsState.addAttribute(getFuncNameAttrName(odsState.name), odsBuilder.getStringAttr(func_name));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopOutlineOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LoopOutlineOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_func_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'func_name'");
    if (namedAttrIt->getName() == getFuncNameAttrName()) {
      tblgen_func_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SCFTransformOps2(*this, tblgen_func_name, "func_name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SCFTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SCFTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoopOutlineOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LoopOutlineOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(targetOperands, allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoopOutlineOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::LoopOutlineOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::LoopPeelOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LoopPeelOpGenericAdaptorBase::LoopPeelOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.loop.peel", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LoopPeelOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LoopPeelOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr LoopPeelOpGenericAdaptorBase::getFailIfAlreadyDivisibleAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoopPeelOp::getFailIfAlreadyDivisibleAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool LoopPeelOpGenericAdaptorBase::getFailIfAlreadyDivisible() {
  auto attr = getFailIfAlreadyDivisibleAttr();
  return attr.getValue();
}

} // namespace detail
LoopPeelOpAdaptor::LoopPeelOpAdaptor(LoopPeelOp op) : LoopPeelOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult LoopPeelOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fail_if_already_divisible;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == LoopPeelOp::getFailIfAlreadyDivisibleAttrName(*odsOpName)) {
      tblgen_fail_if_already_divisible = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fail_if_already_divisible && !((tblgen_fail_if_already_divisible.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'transform.loop.peel' op ""attribute 'fail_if_already_divisible' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoopPeelOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LoopPeelOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::OperationType> LoopPeelOp::getTarget() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::transform::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange LoopPeelOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LoopPeelOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LoopPeelOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface> LoopPeelOp::getTransformed() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface>>(*getODSResults(0).begin());
}

::mlir::BoolAttr LoopPeelOp::getFailIfAlreadyDivisibleAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFailIfAlreadyDivisibleAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool LoopPeelOp::getFailIfAlreadyDivisible() {
  auto attr = getFailIfAlreadyDivisibleAttr();
  return attr.getValue();
}

void LoopPeelOp::setFailIfAlreadyDivisibleAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getFailIfAlreadyDivisibleAttrName(), attr);
}

void LoopPeelOp::setFailIfAlreadyDivisible(bool attrValue) {
  (*this)->setAttr(getFailIfAlreadyDivisibleAttrName(), ::mlir::Builder((*this)->getContext()).getBoolAttr(attrValue));
}

void LoopPeelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::BoolAttr fail_if_already_divisible) {
  odsState.addOperands(target);
  if (fail_if_already_divisible) {
    odsState.addAttribute(getFailIfAlreadyDivisibleAttrName(odsState.name), fail_if_already_divisible);
  }
  odsState.addTypes(transformed);
}

void LoopPeelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::BoolAttr fail_if_already_divisible) {
  odsState.addOperands(target);
  if (fail_if_already_divisible) {
    odsState.addAttribute(getFailIfAlreadyDivisibleAttrName(odsState.name), fail_if_already_divisible);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopPeelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, bool fail_if_already_divisible) {
  odsState.addOperands(target);
  odsState.addAttribute(getFailIfAlreadyDivisibleAttrName(odsState.name), odsBuilder.getBoolAttr(fail_if_already_divisible));
  odsState.addTypes(transformed);
}

void LoopPeelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, bool fail_if_already_divisible) {
  odsState.addOperands(target);
  odsState.addAttribute(getFailIfAlreadyDivisibleAttrName(odsState.name), odsBuilder.getBoolAttr(fail_if_already_divisible));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopPeelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LoopPeelOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult LoopPeelOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fail_if_already_divisible;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFailIfAlreadyDivisibleAttrName()) {
      tblgen_fail_if_already_divisible = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SCFTransformOps1(*this, tblgen_fail_if_already_divisible, "fail_if_already_divisible")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SCFTransformOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SCFTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoopPeelOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LoopPeelOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(targetOperands, allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoopPeelOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getFailIfAlreadyDivisibleAttr();
     if(attr && (attr == odsBuilder.getBoolAttr(false)))
       elidedAttrs.push_back("fail_if_already_divisible");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::LoopPeelOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::LoopPipelineOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LoopPipelineOpGenericAdaptorBase::LoopPipelineOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.loop.pipeline", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LoopPipelineOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LoopPipelineOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr LoopPipelineOpGenericAdaptorBase::getIterationIntervalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoopPipelineOp::getIterationIntervalAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint64_t LoopPipelineOpGenericAdaptorBase::getIterationInterval() {
  auto attr = getIterationIntervalAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr LoopPipelineOpGenericAdaptorBase::getReadLatencyAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoopPipelineOp::getReadLatencyAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint64_t LoopPipelineOpGenericAdaptorBase::getReadLatency() {
  auto attr = getReadLatencyAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
LoopPipelineOpAdaptor::LoopPipelineOpAdaptor(LoopPipelineOp op) : LoopPipelineOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult LoopPipelineOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_iteration_interval;
  ::mlir::Attribute tblgen_read_latency;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == LoopPipelineOp::getIterationIntervalAttrName(*odsOpName)) {
      tblgen_iteration_interval = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoopPipelineOp::getReadLatencyAttrName(*odsOpName)) {
      tblgen_read_latency = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_iteration_interval && !(((tblgen_iteration_interval.isa<::mlir::IntegerAttr>())) && ((tblgen_iteration_interval.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'transform.loop.pipeline' op ""attribute 'iteration_interval' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_read_latency && !(((tblgen_read_latency.isa<::mlir::IntegerAttr>())) && ((tblgen_read_latency.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'transform.loop.pipeline' op ""attribute 'read_latency' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoopPipelineOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LoopPipelineOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::OperationType> LoopPipelineOp::getTarget() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::transform::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange LoopPipelineOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LoopPipelineOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LoopPipelineOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface> LoopPipelineOp::getTransformed() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface>>(*getODSResults(0).begin());
}

::mlir::IntegerAttr LoopPipelineOp::getIterationIntervalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getIterationIntervalAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t LoopPipelineOp::getIterationInterval() {
  auto attr = getIterationIntervalAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr LoopPipelineOp::getReadLatencyAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getReadLatencyAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t LoopPipelineOp::getReadLatency() {
  auto attr = getReadLatencyAttr();
  return attr.getValue().getZExtValue();
}

void LoopPipelineOp::setIterationIntervalAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIterationIntervalAttrName(), attr);
}

void LoopPipelineOp::setIterationInterval(uint64_t attrValue) {
  (*this)->setAttr(getIterationIntervalAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void LoopPipelineOp::setReadLatencyAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getReadLatencyAttrName(), attr);
}

void LoopPipelineOp::setReadLatency(uint64_t attrValue) {
  (*this)->setAttr(getReadLatencyAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void LoopPipelineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, ::mlir::IntegerAttr iteration_interval, ::mlir::IntegerAttr read_latency) {
  odsState.addOperands(target);
  if (iteration_interval) {
    odsState.addAttribute(getIterationIntervalAttrName(odsState.name), iteration_interval);
  }
  if (read_latency) {
    odsState.addAttribute(getReadLatencyAttrName(odsState.name), read_latency);
  }
  odsState.addTypes(transformed);
}

void LoopPipelineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr iteration_interval, ::mlir::IntegerAttr read_latency) {
  odsState.addOperands(target);
  if (iteration_interval) {
    odsState.addAttribute(getIterationIntervalAttrName(odsState.name), iteration_interval);
  }
  if (read_latency) {
    odsState.addAttribute(getReadLatencyAttrName(odsState.name), read_latency);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopPipelineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type transformed, ::mlir::Value target, uint64_t iteration_interval, uint64_t read_latency) {
  odsState.addOperands(target);
  odsState.addAttribute(getIterationIntervalAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), iteration_interval));
  odsState.addAttribute(getReadLatencyAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), read_latency));
  odsState.addTypes(transformed);
}

void LoopPipelineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t iteration_interval, uint64_t read_latency) {
  odsState.addOperands(target);
  odsState.addAttribute(getIterationIntervalAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), iteration_interval));
  odsState.addAttribute(getReadLatencyAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), read_latency));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopPipelineOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LoopPipelineOp::populateDefaultAttrs(const ::mlir::OperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
    attributes.append(attrNames[0], odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1));
  }
  if (!attributes.get(attrNames[1])) {
    attributes.append(attrNames[1], odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 10));
  }
}

::mlir::LogicalResult LoopPipelineOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_iteration_interval;
  ::mlir::Attribute tblgen_read_latency;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getIterationIntervalAttrName()) {
      tblgen_iteration_interval = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getReadLatencyAttrName()) {
      tblgen_read_latency = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SCFTransformOps3(*this, tblgen_iteration_interval, "iteration_interval")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SCFTransformOps3(*this, tblgen_read_latency, "read_latency")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SCFTransformOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SCFTransformOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoopPipelineOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LoopPipelineOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(targetOperands, allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoopPipelineOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getIterationIntervalAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1)))
       elidedAttrs.push_back("iteration_interval");
  }
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getReadLatencyAttr();
     if(attr && (attr == odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 10)))
       elidedAttrs.push_back("read_latency");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::LoopPipelineOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::LoopUnrollOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
LoopUnrollOpGenericAdaptorBase::LoopUnrollOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.loop.unroll", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> LoopUnrollOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr LoopUnrollOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr LoopUnrollOpGenericAdaptorBase::getFactorAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoopUnrollOp::getFactorAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t LoopUnrollOpGenericAdaptorBase::getFactor() {
  auto attr = getFactorAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
LoopUnrollOpAdaptor::LoopUnrollOpAdaptor(LoopUnrollOp op) : LoopUnrollOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult LoopUnrollOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_factor;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'transform.loop.unroll' op ""requires attribute 'factor'");
    if (namedAttrIt->getName() == LoopUnrollOp::getFactorAttrName(*odsOpName)) {
      tblgen_factor = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_factor && !((((tblgen_factor.isa<::mlir::IntegerAttr>())) && ((tblgen_factor.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_factor.cast<::mlir::IntegerAttr>().getValue().isStrictlyPositive()))))
    return emitError(loc, "'transform.loop.unroll' op ""attribute 'factor' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoopUnrollOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LoopUnrollOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface> LoopUnrollOp::getTarget() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange LoopUnrollOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LoopUnrollOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LoopUnrollOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr LoopUnrollOp::getFactorAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFactorAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t LoopUnrollOp::getFactor() {
  auto attr = getFactorAttr();
  return attr.getValue().getZExtValue();
}

void LoopUnrollOp::setFactorAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getFactorAttrName(), attr);
}

void LoopUnrollOp::setFactor(uint64_t attrValue) {
  (*this)->setAttr(getFactorAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), attrValue));
}

void LoopUnrollOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value target, ::mlir::IntegerAttr factor) {
  odsState.addOperands(target);
  odsState.addAttribute(getFactorAttrName(odsState.name), factor);
}

void LoopUnrollOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr factor) {
  odsState.addOperands(target);
  odsState.addAttribute(getFactorAttrName(odsState.name), factor);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopUnrollOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value target, uint64_t factor) {
  odsState.addOperands(target);
  odsState.addAttribute(getFactorAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), factor));
}

void LoopUnrollOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t factor) {
  odsState.addOperands(target);
  odsState.addAttribute(getFactorAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), factor));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoopUnrollOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LoopUnrollOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_factor;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'factor'");
    if (namedAttrIt->getName() == getFactorAttrName()) {
      tblgen_factor = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SCFTransformOps0(*this, tblgen_factor, "factor")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SCFTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoopUnrollOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LoopUnrollOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  ::mlir::Type targetRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> targetTypes(targetRawTypes);

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::transform::TransformHandleTypeInterface type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    targetRawTypes[0] = type;
  }
  if (parser.resolveOperands(targetOperands, targetTypes, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoopUnrollOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getTarget().getType();
    if (auto validType = type.dyn_cast<::mlir::transform::TransformHandleTypeInterface>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::LoopUnrollOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::TakeAssumedBranchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TakeAssumedBranchOpGenericAdaptorBase::TakeAssumedBranchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.scf.take_assumed_branch", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> TakeAssumedBranchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TakeAssumedBranchOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr TakeAssumedBranchOpGenericAdaptorBase::getTakeElseBranchAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TakeAssumedBranchOp::getTakeElseBranchAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

::std::optional<bool> TakeAssumedBranchOpGenericAdaptorBase::getTakeElseBranch() {
  auto attr = getTakeElseBranchAttr();
  return attr ? ::std::optional<bool>(attr != nullptr) : (::std::nullopt);
}

} // namespace detail
TakeAssumedBranchOpAdaptor::TakeAssumedBranchOpAdaptor(TakeAssumedBranchOp op) : TakeAssumedBranchOpAdaptor(op->getOperands(), op->getAttrDictionary(), op->getRegions()) {}

::mlir::LogicalResult TakeAssumedBranchOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_take_else_branch;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == TakeAssumedBranchOp::getTakeElseBranchAttrName(*odsOpName)) {
      tblgen_take_else_branch = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_take_else_branch && !((tblgen_take_else_branch.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'transform.scf.take_assumed_branch' op ""attribute 'take_else_branch' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TakeAssumedBranchOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TakeAssumedBranchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface> TakeAssumedBranchOp::getTarget() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::transform::TransformHandleTypeInterface>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange TakeAssumedBranchOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TakeAssumedBranchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TakeAssumedBranchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::UnitAttr TakeAssumedBranchOp::getTakeElseBranchAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTakeElseBranchAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

::std::optional<bool> TakeAssumedBranchOp::getTakeElseBranch() {
  auto attr = getTakeElseBranchAttr();
  return attr ? ::std::optional<bool>(attr != nullptr) : (::std::nullopt);
}

void TakeAssumedBranchOp::setTakeElseBranchAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getTakeElseBranchAttrName(), attr);
}

void TakeAssumedBranchOp::setTakeElseBranch(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getTakeElseBranchAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getTakeElseBranchAttrName());
}

::mlir::Attribute TakeAssumedBranchOp::removeTakeElseBranchAttr() {
  return (*this)->removeAttr(getTakeElseBranchAttrName());
}

void TakeAssumedBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value target, /*optional*/::mlir::UnitAttr take_else_branch) {
  odsState.addOperands(target);
  if (take_else_branch) {
    odsState.addAttribute(getTakeElseBranchAttrName(odsState.name), take_else_branch);
  }
}

void TakeAssumedBranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, /*optional*/::mlir::UnitAttr take_else_branch) {
  odsState.addOperands(target);
  if (take_else_branch) {
    odsState.addAttribute(getTakeElseBranchAttrName(odsState.name), take_else_branch);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TakeAssumedBranchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TakeAssumedBranchOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_take_else_branch;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getTakeElseBranchAttrName()) {
      tblgen_take_else_branch = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SCFTransformOps4(*this, tblgen_take_else_branch, "take_else_branch")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_SCFTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TakeAssumedBranchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TakeAssumedBranchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand targetRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> targetOperands(targetRawOperands);  ::llvm::SMLoc targetOperandsLoc;
  (void)targetOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("take_else_branch"))) {
    result.addAttribute("take_else_branch", parser.getBuilder().getUnitAttr());
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(targetOperands, allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TakeAssumedBranchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  if ((*this)->getAttr("take_else_branch")) {
    _odsPrinter << ' ' << "take_else_branch";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("take_else_branch");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::TakeAssumedBranchOp)


#endif  // GET_OP_CLASSES

