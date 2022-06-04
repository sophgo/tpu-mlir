/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::ml_program::FuncOp,
::mlir::ml_program::GlobalLoadConstOp,
::mlir::ml_program::GlobalLoadOp,
::mlir::ml_program::GlobalOp,
::mlir::ml_program::GlobalStoreOp,
::mlir::ml_program::OutputOp,
::mlir::ml_program::ReturnOp,
::mlir::ml_program::SubgraphOp,
::mlir::ml_program::TokenOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace ml_program {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MLProgramOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_MLProgramOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::ml_program::TokenType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be Token for establishing execution ordering in a graph, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MLProgramOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MLProgramOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::FunctionType>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: type attribute of function type";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MLProgramOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::SymbolRefAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: symbol reference attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MLProgramOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any type attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MLProgramOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_MLProgramOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((true))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_MLProgramOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}
} // namespace ml_program
} // namespace mlir
namespace mlir {
namespace ml_program {

//===----------------------------------------------------------------------===//
// ::mlir::ml_program::FuncOp definitions
//===----------------------------------------------------------------------===//

FuncOpAdaptor::FuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("ml_program.func", odsAttrs.getContext());
}

FuncOpAdaptor::FuncOpAdaptor(FuncOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FuncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FuncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FuncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr FuncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr FuncOpAdaptor::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, FuncOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef FuncOpAdaptor::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr FuncOpAdaptor::getFunctionTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, FuncOp::getFunctionTypeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
    return attr;
}

::mlir::FunctionType FuncOpAdaptor::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::FunctionType>();
}

::mlir::StringAttr FuncOpAdaptor::getSymVisibilityAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, FuncOp::getSymVisibilityAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > FuncOpAdaptor::getSymVisibility() {
  auto attr = getSymVisibilityAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::RegionRange FuncOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &FuncOpAdaptor::getBody() {
  return *odsRegions[0];
}

::mlir::LogicalResult FuncOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'ml_program.func' op ""requires attribute 'function_type'");
    if (namedAttrIt->getName() == FuncOp::getFunctionTypeAttrName(*odsOpName)) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'ml_program.func' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == FuncOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_visibility;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FuncOp::getSymVisibilityAttrName(*odsOpName)) {
      tblgen_sym_visibility = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'ml_program.func' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_function_type && !(((tblgen_function_type.isa<::mlir::TypeAttr>())) && ((tblgen_function_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::FunctionType>()))))
    return emitError(loc, "'ml_program.func' op ""attribute 'function_type' failed to satisfy constraint: type attribute of function type");

  if (tblgen_sym_visibility && !((tblgen_sym_visibility.isa<::mlir::StringAttr>())))
    return emitError(loc, "'ml_program.func' op ""attribute 'sym_visibility' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FuncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> FuncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &FuncOp::getBody() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr FuncOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef FuncOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr FuncOp::getFunctionTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFunctionTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::FunctionType FuncOp::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::FunctionType>();
}

::mlir::StringAttr FuncOp::getSymVisibilityAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getSymVisibilityAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > FuncOp::getSymVisibility() {
  auto attr = getSymVisibilityAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void FuncOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void FuncOp::setFunctionTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getFunctionTypeAttrName(), attr);
}

void FuncOp::setSymVisibilityAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymVisibilityAttrName(), attr);
}

::mlir::Attribute FuncOp::removeSym_visibilityAttr() {
  return (*this)->removeAttr(getSymVisibilityAttrName());
}

void FuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::TypeAttr function_type, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name), function_type);
  if (sym_visibility) {
  odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  (void)odsState.addRegion();
}

void FuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::TypeAttr function_type, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name), function_type);
  if (sym_visibility) {
  odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::mlir::FunctionType function_type, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name), ::mlir::TypeAttr::get(function_type));
  if (sym_visibility) {
  odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  (void)odsState.addRegion();
}

void FuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::mlir::FunctionType function_type, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name), ::mlir::TypeAttr::get(function_type));
  if (sym_visibility) {
  odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FuncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FuncOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'function_type'");
    if (namedAttrIt->getName() == getFunctionTypeAttrName()) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_visibility;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getSymVisibilityAttrName()) {
      tblgen_sym_visibility = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps1(*this, tblgen_function_type, "function_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps0(*this, tblgen_sym_visibility, "sym_visibility")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_MLProgramOps0(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult FuncOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace ml_program
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ml_program::FuncOp)

namespace mlir {
namespace ml_program {

//===----------------------------------------------------------------------===//
// ::mlir::ml_program::GlobalLoadConstOp definitions
//===----------------------------------------------------------------------===//

GlobalLoadConstOpAdaptor::GlobalLoadConstOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("ml_program.global_load_const", odsAttrs.getContext());
}

GlobalLoadConstOpAdaptor::GlobalLoadConstOpAdaptor(GlobalLoadConstOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GlobalLoadConstOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GlobalLoadConstOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GlobalLoadConstOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GlobalLoadConstOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::SymbolRefAttr GlobalLoadConstOpAdaptor::getGlobalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GlobalLoadConstOp::getGlobalAttrName(*odsOpName)).cast<::mlir::SymbolRefAttr>();
    return attr;
}

::mlir::SymbolRefAttr GlobalLoadConstOpAdaptor::getGlobal() {
  auto attr = getGlobalAttr();
  return attr;
}

::mlir::LogicalResult GlobalLoadConstOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'ml_program.global_load_const' op ""requires attribute 'global'");
    if (namedAttrIt->getName() == GlobalLoadConstOp::getGlobalAttrName(*odsOpName)) {
      tblgen_global = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_global && !((tblgen_global.isa<::mlir::SymbolRefAttr>())))
    return emitError(loc, "'ml_program.global_load_const' op ""attribute 'global' failed to satisfy constraint: symbol reference attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GlobalLoadConstOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GlobalLoadConstOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GlobalLoadConstOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GlobalLoadConstOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GlobalLoadConstOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::SymbolRefAttr GlobalLoadConstOp::getGlobalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getGlobalAttrName()).cast<::mlir::SymbolRefAttr>();
}

::mlir::SymbolRefAttr GlobalLoadConstOp::getGlobal() {
  auto attr = getGlobalAttr();
  return attr;
}

void GlobalLoadConstOp::setGlobalAttr(::mlir::SymbolRefAttr attr) {
  (*this)->setAttr(getGlobalAttrName(), attr);
}

void GlobalLoadConstOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::SymbolRefAttr global) {
  odsState.addAttribute(getGlobalAttrName(odsState.name), global);
  odsState.addTypes(result);
}

void GlobalLoadConstOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::SymbolRefAttr global) {
  odsState.addAttribute(getGlobalAttrName(odsState.name), global);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalLoadConstOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GlobalLoadConstOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'global'");
    if (namedAttrIt->getName() == getGlobalAttrName()) {
      tblgen_global = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps2(*this, tblgen_global, "global")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MLProgramOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GlobalLoadConstOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GlobalLoadConstOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SymbolRefAttr globalAttr;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  if (parser.parseCustomAttributeWithFallback(globalAttr, parser.getBuilder().getType<::mlir::NoneType>(), "global",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(resultTypes);
  return ::mlir::success();
}

void GlobalLoadConstOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getGlobalAttr());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"global"});
}

void GlobalLoadConstOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace ml_program
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ml_program::GlobalLoadConstOp)

namespace mlir {
namespace ml_program {

//===----------------------------------------------------------------------===//
// ::mlir::ml_program::GlobalLoadOp definitions
//===----------------------------------------------------------------------===//

GlobalLoadOpAdaptor::GlobalLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("ml_program.global_load", odsAttrs.getContext());
}

GlobalLoadOpAdaptor::GlobalLoadOpAdaptor(GlobalLoadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GlobalLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GlobalLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange GlobalLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange GlobalLoadOpAdaptor::getConsumeTokens() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr GlobalLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::SymbolRefAttr GlobalLoadOpAdaptor::getGlobalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GlobalLoadOp::getGlobalAttrName(*odsOpName)).cast<::mlir::SymbolRefAttr>();
    return attr;
}

::mlir::SymbolRefAttr GlobalLoadOpAdaptor::getGlobal() {
  auto attr = getGlobalAttr();
  return attr;
}

::mlir::LogicalResult GlobalLoadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'ml_program.global_load' op ""requires attribute 'global'");
    if (namedAttrIt->getName() == GlobalLoadOp::getGlobalAttrName(*odsOpName)) {
      tblgen_global = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_global && !((tblgen_global.isa<::mlir::SymbolRefAttr>())))
    return emitError(loc, "'ml_program.global_load' op ""attribute 'global' failed to satisfy constraint: symbol reference attribute");
  return ::mlir::success();
}

void GlobalLoadOp::getAsmResultNames(::mlir::OpAsmSetValueNameFn setNameFn) {
  auto resultGroup0 = getODSResults(0);
  if (!llvm::empty(resultGroup0))
    setNameFn(*resultGroup0.begin(), "result");
  auto resultGroup1 = getODSResults(1);
  if (!llvm::empty(resultGroup1))
    setNameFn(*resultGroup1.begin(), "produceToken");
}

std::pair<unsigned, unsigned> GlobalLoadOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range GlobalLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range GlobalLoadOp::getConsumeTokens() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange GlobalLoadOp::getConsumeTokensMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GlobalLoadOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range GlobalLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GlobalLoadOp::getResult() {
  return *getODSResults(0).begin();
}

::mlir::Value GlobalLoadOp::getProduceToken() {
  auto results = getODSResults(1);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

::mlir::SymbolRefAttr GlobalLoadOp::getGlobalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getGlobalAttrName()).cast<::mlir::SymbolRefAttr>();
}

::mlir::SymbolRefAttr GlobalLoadOp::getGlobal() {
  auto attr = getGlobalAttr();
  return attr;
}

void GlobalLoadOp::setGlobalAttr(::mlir::SymbolRefAttr attr) {
  (*this)->setAttr(getGlobalAttrName(), attr);
}

void GlobalLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, /*optional*/::mlir::Type produceToken, ::mlir::SymbolRefAttr global, ::mlir::ValueRange consumeTokens) {
  odsState.addOperands(consumeTokens);
  odsState.addAttribute(getGlobalAttrName(odsState.name), global);
  odsState.addTypes(result);
  if (produceToken)
    odsState.addTypes(produceToken);
}

void GlobalLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::SymbolRefAttr global, ::mlir::ValueRange consumeTokens) {
  odsState.addOperands(consumeTokens);
  odsState.addAttribute(getGlobalAttrName(odsState.name), global);
  assert(resultTypes.size() >= 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() >= 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GlobalLoadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'global'");
    if (namedAttrIt->getName() == getGlobalAttrName()) {
      tblgen_global = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps2(*this, tblgen_global, "global")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MLProgramOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MLProgramOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSResults(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("result group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MLProgramOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GlobalLoadOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GlobalLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SymbolRefAttr globalAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> consumeTokensOperands;
  ::llvm::SMLoc consumeTokensOperandsLoc;
  (void)consumeTokensOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> produceTokenTypes;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  if (parser.parseCustomAttributeWithFallback(globalAttr, parser.getBuilder().getType<::mlir::NoneType>(), "global",
          result.attributes)) {
    return ::mlir::failure();
  }
  {
    consumeTokensOperandsLoc = parser.getCurrentLocation();
    ::mlir::Type produceTokenType;
    if (parseTokenOrdering(parser, consumeTokensOperands, produceTokenType))
      return ::mlir::failure();
    if (produceTokenType)
      produceTokenTypes.push_back(produceTokenType);
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::ml_program::TokenType>();
  result.addTypes(resultTypes);
  result.addTypes(produceTokenTypes);
  if (parser.resolveOperands(consumeTokensOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GlobalLoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getGlobalAttr());
  printTokenOrdering(_odsPrinter, *this, getConsumeTokens(), (getProduceToken() ? getProduceToken().getType() : Type()));
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"global"});
}

void GlobalLoadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Read::get(), getGlobalAttr(), ::mlir::SideEffects::DefaultResource::get());
}

} // namespace ml_program
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ml_program::GlobalLoadOp)

namespace mlir {
namespace ml_program {

//===----------------------------------------------------------------------===//
// ::mlir::ml_program::GlobalOp definitions
//===----------------------------------------------------------------------===//

GlobalOpAdaptor::GlobalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("ml_program.global", odsAttrs.getContext());
}

GlobalOpAdaptor::GlobalOpAdaptor(GlobalOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GlobalOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GlobalOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GlobalOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GlobalOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr GlobalOpAdaptor::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GlobalOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef GlobalOpAdaptor::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr GlobalOpAdaptor::getTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, GlobalOp::getTypeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
    return attr;
}

::mlir::Type GlobalOpAdaptor::getType() {
  auto attr = getTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::UnitAttr GlobalOpAdaptor::getIsMutableAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, GlobalOp::getIsMutableAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool GlobalOpAdaptor::getIsMutable() {
  auto attr = getIsMutableAttr();
  return attr != nullptr;
}

::mlir::Attribute GlobalOpAdaptor::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, GlobalOp::getValueAttrName(*odsOpName)).dyn_cast_or_null<::mlir::Attribute>();
    return attr;
}

::llvm::Optional<::mlir::Attribute> GlobalOpAdaptor::getValue() {
  auto attr = getValueAttr();
  return attr ? ::llvm::Optional<::mlir::Attribute>(attr) : (::llvm::None);
}

::mlir::StringAttr GlobalOpAdaptor::getSymVisibilityAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, GlobalOp::getSymVisibilityAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > GlobalOpAdaptor::getSymVisibility() {
  auto attr = getSymVisibilityAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::LogicalResult GlobalOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_is_mutable;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'ml_program.global' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == GlobalOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GlobalOp::getIsMutableAttrName(*odsOpName)) {
      tblgen_is_mutable = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_type;
  ::mlir::Attribute tblgen_sym_visibility;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'ml_program.global' op ""requires attribute 'type'");
    if (namedAttrIt->getName() == GlobalOp::getTypeAttrName(*odsOpName)) {
      tblgen_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GlobalOp::getSymVisibilityAttrName(*odsOpName)) {
      tblgen_sym_visibility = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == GlobalOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'ml_program.global' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_type && !(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return emitError(loc, "'ml_program.global' op ""attribute 'type' failed to satisfy constraint: any type attribute");

  if (tblgen_is_mutable && !((tblgen_is_mutable.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'ml_program.global' op ""attribute 'is_mutable' failed to satisfy constraint: unit attribute");

  if (tblgen_value && !((true)))
    return emitError(loc, "'ml_program.global' op ""attribute 'value' failed to satisfy constraint: any attribute");

  if (tblgen_sym_visibility && !((tblgen_sym_visibility.isa<::mlir::StringAttr>())))
    return emitError(loc, "'ml_program.global' op ""attribute 'sym_visibility' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GlobalOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GlobalOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GlobalOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GlobalOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr GlobalOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef GlobalOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr GlobalOp::getTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::Type GlobalOp::getType() {
  auto attr = getTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::UnitAttr GlobalOp::getIsMutableAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getIsMutableAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool GlobalOp::getIsMutable() {
  auto attr = getIsMutableAttr();
  return attr != nullptr;
}

::mlir::Attribute GlobalOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getValueAttrName()).dyn_cast_or_null<::mlir::Attribute>();
}

::llvm::Optional<::mlir::Attribute> GlobalOp::getValue() {
  auto attr = getValueAttr();
  return attr ? ::llvm::Optional<::mlir::Attribute>(attr) : (::llvm::None);
}

::mlir::StringAttr GlobalOp::getSymVisibilityAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getSymVisibilityAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GlobalOp::getSymVisibility() {
  auto attr = getSymVisibilityAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void GlobalOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void GlobalOp::setTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getTypeAttrName(), attr);
}

void GlobalOp::setIsMutableAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getIsMutableAttrName(), attr);
}

void GlobalOp::setValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void GlobalOp::setSymVisibilityAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymVisibilityAttrName(), attr);
}

::mlir::Attribute GlobalOp::removeIs_mutableAttr() {
  return (*this)->removeAttr(getIsMutableAttrName());
}

::mlir::Attribute GlobalOp::removeValueAttr() {
  return (*this)->removeAttr(getValueAttrName());
}

::mlir::Attribute GlobalOp::removeSym_visibilityAttr() {
  return (*this)->removeAttr(getSymVisibilityAttrName());
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type, /*optional*/::mlir::UnitAttr is_mutable, /*optional*/::mlir::Attribute value, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getTypeAttrName(odsState.name), type);
  if (is_mutable) {
  odsState.addAttribute(getIsMutableAttrName(odsState.name), is_mutable);
  }
  if (value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  if (sym_visibility) {
  odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type, /*optional*/::mlir::UnitAttr is_mutable, /*optional*/::mlir::Attribute value, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getTypeAttrName(odsState.name), type);
  if (is_mutable) {
  odsState.addAttribute(getIsMutableAttrName(odsState.name), is_mutable);
  }
  if (value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  if (sym_visibility) {
  odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::mlir::Type type, /*optional*/bool is_mutable, /*optional*/::mlir::Attribute value, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getTypeAttrName(odsState.name), ::mlir::TypeAttr::get(type));
  if (is_mutable) {
  odsState.addAttribute(getIsMutableAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  if (sym_visibility) {
  odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::mlir::Type type, /*optional*/bool is_mutable, /*optional*/::mlir::Attribute value, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getTypeAttrName(odsState.name), ::mlir::TypeAttr::get(type));
  if (is_mutable) {
  odsState.addAttribute(getIsMutableAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  if (sym_visibility) {
  odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GlobalOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_is_mutable;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getIsMutableAttrName()) {
      tblgen_is_mutable = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_type;
  ::mlir::Attribute tblgen_sym_visibility;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'type'");
    if (namedAttrIt->getName() == getTypeAttrName()) {
      tblgen_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getSymVisibilityAttrName()) {
      tblgen_sym_visibility = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps3(*this, tblgen_type, "type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps4(*this, tblgen_is_mutable, "is_mutable")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps5(*this, tblgen_value, "value")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps0(*this, tblgen_sym_visibility, "sym_visibility")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult GlobalOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GlobalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_visibilityAttr;
  ::mlir::StringAttr sym_nameAttr;
  ::mlir::TypeAttr typeAttr;
  ::mlir::Attribute valueAttr;
  {
    if (parseSymbolVisibility(parser, sym_visibilityAttr))
      return ::mlir::failure();
    if (sym_visibilityAttr)
      result.addAttribute("sym_visibility", sym_visibilityAttr);
  }
  if (succeeded(parser.parseOptionalKeyword("mutable"))) {
    result.addAttribute("is_mutable", parser.getBuilder().getUnitAttr());
  }

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  {
    if (parseTypedInitialValue(parser, typeAttr, valueAttr))
      return ::mlir::failure();
    result.addAttribute("type", typeAttr);
    if (valueAttr)
      result.addAttribute("value", valueAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void GlobalOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  printSymbolVisibility(_odsPrinter, *this, getSymVisibilityAttr());
  if ((*this)->getAttr("is_mutable")) {
  _odsPrinter << ' ' << "mutable";
  }
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  printTypedInitialValue(_odsPrinter, *this, getTypeAttr(), getValueAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"sym_visibility", "is_mutable", "sym_name", "type", "value"});
}

} // namespace ml_program
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ml_program::GlobalOp)

namespace mlir {
namespace ml_program {

//===----------------------------------------------------------------------===//
// ::mlir::ml_program::GlobalStoreOp definitions
//===----------------------------------------------------------------------===//

GlobalStoreOpAdaptor::GlobalStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("ml_program.global_store", odsAttrs.getContext());
}

GlobalStoreOpAdaptor::GlobalStoreOpAdaptor(GlobalStoreOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GlobalStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GlobalStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange GlobalStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GlobalStoreOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange GlobalStoreOpAdaptor::getConsumeTokens() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr GlobalStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::SymbolRefAttr GlobalStoreOpAdaptor::getGlobalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GlobalStoreOp::getGlobalAttrName(*odsOpName)).cast<::mlir::SymbolRefAttr>();
    return attr;
}

::mlir::SymbolRefAttr GlobalStoreOpAdaptor::getGlobal() {
  auto attr = getGlobalAttr();
  return attr;
}

::mlir::LogicalResult GlobalStoreOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'ml_program.global_store' op ""requires attribute 'global'");
    if (namedAttrIt->getName() == GlobalStoreOp::getGlobalAttrName(*odsOpName)) {
      tblgen_global = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_global && !((tblgen_global.isa<::mlir::SymbolRefAttr>())))
    return emitError(loc, "'ml_program.global_store' op ""attribute 'global' failed to satisfy constraint: symbol reference attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GlobalStoreOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range GlobalStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GlobalStoreOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range GlobalStoreOp::getConsumeTokens() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange GlobalStoreOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GlobalStoreOp::getConsumeTokensMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GlobalStoreOp::getODSResultIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumResults() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::result_range GlobalStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GlobalStoreOp::getProduceToken() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

::mlir::SymbolRefAttr GlobalStoreOp::getGlobalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getGlobalAttrName()).cast<::mlir::SymbolRefAttr>();
}

::mlir::SymbolRefAttr GlobalStoreOp::getGlobal() {
  auto attr = getGlobalAttr();
  return attr;
}

void GlobalStoreOp::setGlobalAttr(::mlir::SymbolRefAttr attr) {
  (*this)->setAttr(getGlobalAttrName(), attr);
}

void GlobalStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type produceToken, ::mlir::SymbolRefAttr global, ::mlir::Value value, ::mlir::ValueRange consumeTokens) {
  odsState.addOperands(value);
  odsState.addOperands(consumeTokens);
  odsState.addAttribute(getGlobalAttrName(odsState.name), global);
  if (produceToken)
    odsState.addTypes(produceToken);
}

void GlobalStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::SymbolRefAttr global, ::mlir::Value value, ::mlir::ValueRange consumeTokens) {
  odsState.addOperands(value);
  odsState.addOperands(consumeTokens);
  odsState.addAttribute(getGlobalAttrName(odsState.name), global);
  odsState.addTypes(resultTypes);
}

void GlobalStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GlobalStoreOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'global'");
    if (namedAttrIt->getName() == getGlobalAttrName()) {
      tblgen_global = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps2(*this, tblgen_global, "global")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MLProgramOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MLProgramOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("result group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MLProgramOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GlobalStoreOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GlobalStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SymbolRefAttr globalAttr;
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> consumeTokensOperands;
  ::llvm::SMLoc consumeTokensOperandsLoc;
  (void)consumeTokensOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> produceTokenTypes;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  if (parser.parseCustomAttributeWithFallback(globalAttr, parser.getBuilder().getType<::mlir::NoneType>(), "global",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseEqual())
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  {
    consumeTokensOperandsLoc = parser.getCurrentLocation();
    ::mlir::Type produceTokenType;
    if (parseTokenOrdering(parser, consumeTokensOperands, produceTokenType))
      return ::mlir::failure();
    if (produceTokenType)
      produceTokenTypes.push_back(produceTokenType);
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::ml_program::TokenType>();
  result.addTypes(produceTokenTypes);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(consumeTokensOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GlobalStoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getGlobalAttr());
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  printTokenOrdering(_odsPrinter, *this, getConsumeTokens(), (getProduceToken() ? getProduceToken().getType() : Type()));
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"global"});
}

void GlobalStoreOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(::mlir::MemoryEffects::Write::get(), getGlobalAttr(), ::mlir::SideEffects::DefaultResource::get());
}

} // namespace ml_program
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ml_program::GlobalStoreOp)

namespace mlir {
namespace ml_program {

//===----------------------------------------------------------------------===//
// ::mlir::ml_program::OutputOp definitions
//===----------------------------------------------------------------------===//

OutputOpAdaptor::OutputOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("ml_program.output", odsAttrs.getContext());
}

OutputOpAdaptor::OutputOpAdaptor(OutputOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange OutputOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> OutputOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange OutputOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange OutputOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr OutputOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult OutputOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OutputOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range OutputOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range OutputOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange OutputOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> OutputOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OutputOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void OutputOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
    build(odsBuilder, odsState, llvm::None);
  
}

void OutputOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
}

void OutputOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult OutputOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MLProgramOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult OutputOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult OutputOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> operandsTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (!operandsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(operandsTypes))
    return ::mlir::failure();
  }
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OutputOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  if (!operands().empty()) {
  _odsPrinter << ' ';
  _odsPrinter << operands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << operands().getTypes();
  }
}

void OutputOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace ml_program
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ml_program::OutputOp)

namespace mlir {
namespace ml_program {

//===----------------------------------------------------------------------===//
// ::mlir::ml_program::ReturnOp definitions
//===----------------------------------------------------------------------===//

ReturnOpAdaptor::ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("ml_program.return", odsAttrs.getContext());
}

ReturnOpAdaptor::ReturnOpAdaptor(ReturnOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ReturnOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReturnOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange ReturnOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange ReturnOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr ReturnOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ReturnOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReturnOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ReturnOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReturnOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
    build(odsBuilder, odsState, llvm::None);
  
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands) {
  odsState.addOperands(operands);
}

void ReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReturnOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MLProgramOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReturnOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ReturnOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> operandsOperands;
  ::llvm::SMLoc operandsOperandsLoc;
  (void)operandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> operandsTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  operandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(operandsOperands))
    return ::mlir::failure();
  if (!operandsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(operandsTypes))
    return ::mlir::failure();
  }
  if (parser.resolveOperands(operandsOperands, operandsTypes, operandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReturnOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  if (!operands().empty()) {
  _odsPrinter << ' ';
  _odsPrinter << operands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << operands().getTypes();
  }
}

void ReturnOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace ml_program
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ml_program::ReturnOp)

namespace mlir {
namespace ml_program {

//===----------------------------------------------------------------------===//
// ::mlir::ml_program::SubgraphOp definitions
//===----------------------------------------------------------------------===//

SubgraphOpAdaptor::SubgraphOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("ml_program.subgraph", odsAttrs.getContext());
}

SubgraphOpAdaptor::SubgraphOpAdaptor(SubgraphOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SubgraphOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SubgraphOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SubgraphOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr SubgraphOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr SubgraphOpAdaptor::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SubgraphOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef SubgraphOpAdaptor::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr SubgraphOpAdaptor::getFunctionTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SubgraphOp::getFunctionTypeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
    return attr;
}

::mlir::FunctionType SubgraphOpAdaptor::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::FunctionType>();
}

::mlir::StringAttr SubgraphOpAdaptor::getSymVisibilityAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, SubgraphOp::getSymVisibilityAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > SubgraphOpAdaptor::getSymVisibility() {
  auto attr = getSymVisibilityAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::RegionRange SubgraphOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &SubgraphOpAdaptor::getBody() {
  return *odsRegions[0];
}

::mlir::LogicalResult SubgraphOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'ml_program.subgraph' op ""requires attribute 'function_type'");
    if (namedAttrIt->getName() == SubgraphOp::getFunctionTypeAttrName(*odsOpName)) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'ml_program.subgraph' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == SubgraphOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_visibility;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == SubgraphOp::getSymVisibilityAttrName(*odsOpName)) {
      tblgen_sym_visibility = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'ml_program.subgraph' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_function_type && !(((tblgen_function_type.isa<::mlir::TypeAttr>())) && ((tblgen_function_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::FunctionType>()))))
    return emitError(loc, "'ml_program.subgraph' op ""attribute 'function_type' failed to satisfy constraint: type attribute of function type");

  if (tblgen_sym_visibility && !((tblgen_sym_visibility.isa<::mlir::StringAttr>())))
    return emitError(loc, "'ml_program.subgraph' op ""attribute 'sym_visibility' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SubgraphOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SubgraphOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> SubgraphOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubgraphOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &SubgraphOp::getBody() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr SubgraphOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef SubgraphOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr SubgraphOp::getFunctionTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFunctionTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::FunctionType SubgraphOp::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::FunctionType>();
}

::mlir::StringAttr SubgraphOp::getSymVisibilityAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getSymVisibilityAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > SubgraphOp::getSymVisibility() {
  auto attr = getSymVisibilityAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void SubgraphOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void SubgraphOp::setFunctionTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getFunctionTypeAttrName(), attr);
}

void SubgraphOp::setSymVisibilityAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymVisibilityAttrName(), attr);
}

::mlir::Attribute SubgraphOp::removeSym_visibilityAttr() {
  return (*this)->removeAttr(getSymVisibilityAttrName());
}

void SubgraphOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::TypeAttr function_type, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name), function_type);
  if (sym_visibility) {
  odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  (void)odsState.addRegion();
}

void SubgraphOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::TypeAttr function_type, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name), function_type);
  if (sym_visibility) {
  odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubgraphOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::mlir::FunctionType function_type, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name), ::mlir::TypeAttr::get(function_type));
  if (sym_visibility) {
  odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  (void)odsState.addRegion();
}

void SubgraphOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::mlir::FunctionType function_type, /*optional*/::mlir::StringAttr sym_visibility) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getFunctionTypeAttrName(odsState.name), ::mlir::TypeAttr::get(function_type));
  if (sym_visibility) {
  odsState.addAttribute(getSymVisibilityAttrName(odsState.name), sym_visibility);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubgraphOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SubgraphOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'function_type'");
    if (namedAttrIt->getName() == getFunctionTypeAttrName()) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_visibility;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getSymVisibilityAttrName()) {
      tblgen_sym_visibility = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps1(*this, tblgen_function_type, "function_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_MLProgramOps0(*this, tblgen_sym_visibility, "sym_visibility")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_MLProgramOps0(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult SubgraphOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace ml_program
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ml_program::SubgraphOp)

namespace mlir {
namespace ml_program {

//===----------------------------------------------------------------------===//
// ::mlir::ml_program::TokenOp definitions
//===----------------------------------------------------------------------===//

TokenOpAdaptor::TokenOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("ml_program.token", odsAttrs.getContext());
}

TokenOpAdaptor::TokenOpAdaptor(TokenOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange TokenOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TokenOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TokenOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr TokenOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TokenOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TokenOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TokenOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TokenOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TokenOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TokenOp::getToken() {
  return *getODSResults(0).begin();
}

void TokenOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type token) {
  odsState.addTypes(token);
}

void TokenOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TokenOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TokenOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_MLProgramOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TokenOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TokenOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::ml_program::TokenType>();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void TokenOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void TokenOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace ml_program
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ml_program::TokenOp)


#endif  // GET_OP_CLASSES

