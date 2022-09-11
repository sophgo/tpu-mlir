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
::mlir::transform::LoopOutlineOp,
::mlir::transform::LoopPeelOp,
::mlir::transform::LoopPipelineOp,
::mlir::transform::LoopUnrollOp
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
  if (!((type.isa<::mlir::pdl::OperationType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Operation *`, but got " << type;
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
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_SCFTransformOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
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
} // namespace transform
} // namespace mlir
namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::GetParentForOp definitions
//===----------------------------------------------------------------------===//

GetParentForOpAdaptor::GetParentForOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.loop.get_parent_for", odsAttrs.getContext());
}

GetParentForOpAdaptor::GetParentForOpAdaptor(GetParentForOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GetParentForOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetParentForOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetParentForOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetParentForOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr GetParentForOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetParentForOpAdaptor::getNumLoopsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GetParentForOp::getNumLoopsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint64_t GetParentForOpAdaptor::getNumLoops() {
  auto attr = getNumLoopsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult GetParentForOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_num_loops;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == GetParentForOp::getNumLoopsAttrName(*odsOpName)) {
      tblgen_num_loops = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_num_loops && !((((tblgen_num_loops.isa<::mlir::IntegerAttr>())) && ((tblgen_num_loops.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_num_loops.cast<::mlir::IntegerAttr>().getValue().isStrictlyPositive()))))
    return emitError(loc, "'transform.loop.get_parent_for' op ""attribute 'num_loops' failed to satisfy constraint: 64-bit signless integer attribute whose value is positive");
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

::mlir::TypedValue<::mlir::pdl::OperationType> GetParentForOp::getTarget() {
  return *getODSOperands(0).begin();
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

::mlir::TypedValue<::mlir::pdl::OperationType> GetParentForOp::getParent() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr GetParentForOp::getNumLoopsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNumLoopsAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t GetParentForOp::getNumLoops() {
  auto attr = getNumLoopsAttr();
  return attr.getValue().getZExtValue();
}

void GetParentForOp::setNumLoopsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getNumLoopsAttrName(), attr);
}

void GetParentForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type parent, ::mlir::Value target, ::mlir::IntegerAttr num_loops) {
  odsState.addOperands(target);
  if (num_loops) {
  odsState.addAttribute(getNumLoopsAttrName(odsState.name), num_loops);
  }
  odsState.addTypes(parent);
}

void GetParentForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::IntegerAttr num_loops) {
  odsState.addOperands(target);
  if (num_loops) {
  odsState.addAttribute(getNumLoopsAttrName(odsState.name), num_loops);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetParentForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type parent, ::mlir::Value target, uint64_t num_loops) {
  odsState.addOperands(target);
  odsState.addAttribute(getNumLoopsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_loops));
  odsState.addTypes(parent);
}

void GetParentForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, uint64_t num_loops) {
  odsState.addOperands(target);
  odsState.addAttribute(getNumLoopsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_loops));
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

void GetParentForOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
     attributes.append(attrNames[0], odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), 1));
  }
}

::mlir::LogicalResult GetParentForOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_num_loops;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getNumLoopsAttrName()) {
      tblgen_num_loops = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SCFTransformOps0(*this, tblgen_num_loops, "num_loops")))
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

void GetParentForOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::GetParentForOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::LoopOutlineOp definitions
//===----------------------------------------------------------------------===//

LoopOutlineOpAdaptor::LoopOutlineOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.loop.outline", odsAttrs.getContext());
}

LoopOutlineOpAdaptor::LoopOutlineOpAdaptor(LoopOutlineOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LoopOutlineOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LoopOutlineOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LoopOutlineOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoopOutlineOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr LoopOutlineOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr LoopOutlineOpAdaptor::getFuncNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoopOutlineOp::getFuncNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef LoopOutlineOpAdaptor::getFuncName() {
  auto attr = getFuncNameAttr();
  return attr.getValue();
}

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

::mlir::TypedValue<::mlir::pdl::OperationType> LoopOutlineOp::getTarget() {
  return *getODSOperands(0).begin();
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

::mlir::TypedValue<::mlir::pdl::OperationType> LoopOutlineOp::getTransformed() {
  return *getODSResults(0).begin();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SCFTransformOps1(*this, tblgen_func_name, "func_name")))
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

void LoopOutlineOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::LoopOutlineOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::LoopPeelOp definitions
//===----------------------------------------------------------------------===//

LoopPeelOpAdaptor::LoopPeelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.loop.peel", odsAttrs.getContext());
}

LoopPeelOpAdaptor::LoopPeelOpAdaptor(LoopPeelOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LoopPeelOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LoopPeelOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LoopPeelOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoopPeelOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr LoopPeelOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr LoopPeelOpAdaptor::getFailIfAlreadyDivisibleAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoopPeelOp::getFailIfAlreadyDivisibleAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool LoopPeelOpAdaptor::getFailIfAlreadyDivisible() {
  auto attr = getFailIfAlreadyDivisibleAttr();
  return attr.getValue();
}

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

::mlir::TypedValue<::mlir::pdl::OperationType> LoopPeelOp::getTarget() {
  return *getODSOperands(0).begin();
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

::mlir::TypedValue<::mlir::pdl::OperationType> LoopPeelOp::getTransformed() {
  return *getODSResults(0).begin();
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

void LoopPeelOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_SCFTransformOps2(*this, tblgen_fail_if_already_divisible, "fail_if_already_divisible")))
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

::mlir::LogicalResult LoopPeelOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LoopPeelOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void LoopPeelOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::LoopPeelOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::LoopPipelineOp definitions
//===----------------------------------------------------------------------===//

LoopPipelineOpAdaptor::LoopPipelineOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.loop.pipeline", odsAttrs.getContext());
}

LoopPipelineOpAdaptor::LoopPipelineOpAdaptor(LoopPipelineOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LoopPipelineOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LoopPipelineOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LoopPipelineOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoopPipelineOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr LoopPipelineOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr LoopPipelineOpAdaptor::getIterationIntervalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoopPipelineOp::getIterationIntervalAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint64_t LoopPipelineOpAdaptor::getIterationInterval() {
  auto attr = getIterationIntervalAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr LoopPipelineOpAdaptor::getReadLatencyAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoopPipelineOp::getReadLatencyAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

uint64_t LoopPipelineOpAdaptor::getReadLatency() {
  auto attr = getReadLatencyAttr();
  return attr.getValue().getZExtValue();
}

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

::mlir::TypedValue<::mlir::pdl::OperationType> LoopPipelineOp::getTarget() {
  return *getODSOperands(0).begin();
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

::mlir::TypedValue<::mlir::pdl::OperationType> LoopPipelineOp::getTransformed() {
  return *getODSResults(0).begin();
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

void LoopPipelineOp::setReadLatencyAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getReadLatencyAttrName(), attr);
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

void LoopPipelineOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
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

::mlir::LogicalResult LoopPipelineOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult LoopPipelineOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void LoopPipelineOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::LoopPipelineOp)

namespace mlir {
namespace transform {

//===----------------------------------------------------------------------===//
// ::mlir::transform::LoopUnrollOp definitions
//===----------------------------------------------------------------------===//

LoopUnrollOpAdaptor::LoopUnrollOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.loop.unroll", odsAttrs.getContext());
}

LoopUnrollOpAdaptor::LoopUnrollOpAdaptor(LoopUnrollOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LoopUnrollOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LoopUnrollOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LoopUnrollOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoopUnrollOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr LoopUnrollOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr LoopUnrollOpAdaptor::getFactorAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoopUnrollOp::getFactorAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
  return attr;
}

uint64_t LoopUnrollOpAdaptor::getFactor() {
  auto attr = getFactorAttr();
  return attr.getValue().getZExtValue();
}

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

::mlir::TypedValue<::mlir::pdl::OperationType> LoopUnrollOp::getTarget() {
  return *getODSOperands(0).begin();
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

  targetOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(targetOperands, odsBuildableType0, targetOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LoopUnrollOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::LoopUnrollOp)


#endif  // GET_OP_CLASSES

