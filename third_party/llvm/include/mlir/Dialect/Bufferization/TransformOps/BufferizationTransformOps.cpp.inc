/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::transform::OneShotBufferizeOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace transform {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_BufferizationTransformOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::pdl::OperationType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Operation *`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_BufferizationTransformOps0(
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
// ::mlir::transform::OneShotBufferizeOp definitions
//===----------------------------------------------------------------------===//

OneShotBufferizeOpAdaptor::OneShotBufferizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("transform.bufferization.one_shot_bufferize", odsAttrs.getContext());
}

OneShotBufferizeOpAdaptor::OneShotBufferizeOpAdaptor(OneShotBufferizeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange OneShotBufferizeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> OneShotBufferizeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange OneShotBufferizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OneShotBufferizeOpAdaptor::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr OneShotBufferizeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr OneShotBufferizeOpAdaptor::getAllowReturnAllocsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, OneShotBufferizeOp::getAllowReturnAllocsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool OneShotBufferizeOpAdaptor::getAllowReturnAllocs() {
  auto attr = getAllowReturnAllocsAttr();
  return attr.getValue();
}

::mlir::BoolAttr OneShotBufferizeOpAdaptor::getAllowUnknownOpsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, OneShotBufferizeOp::getAllowUnknownOpsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool OneShotBufferizeOpAdaptor::getAllowUnknownOps() {
  auto attr = getAllowUnknownOpsAttr();
  return attr.getValue();
}

::mlir::BoolAttr OneShotBufferizeOpAdaptor::getBufferizeFunctionBoundariesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, OneShotBufferizeOp::getBufferizeFunctionBoundariesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool OneShotBufferizeOpAdaptor::getBufferizeFunctionBoundaries() {
  auto attr = getBufferizeFunctionBoundariesAttr();
  return attr.getValue();
}

::mlir::BoolAttr OneShotBufferizeOpAdaptor::getCreateDeallocsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, OneShotBufferizeOp::getCreateDeallocsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool OneShotBufferizeOpAdaptor::getCreateDeallocs() {
  auto attr = getCreateDeallocsAttr();
  return attr.getValue();
}

::mlir::BoolAttr OneShotBufferizeOpAdaptor::getTargetIsModuleAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, OneShotBufferizeOp::getTargetIsModuleAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool OneShotBufferizeOpAdaptor::getTargetIsModule() {
  auto attr = getTargetIsModuleAttr();
  return attr.getValue();
}

::mlir::BoolAttr OneShotBufferizeOpAdaptor::getTestAnalysisOnlyAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, OneShotBufferizeOp::getTestAnalysisOnlyAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool OneShotBufferizeOpAdaptor::getTestAnalysisOnly() {
  auto attr = getTestAnalysisOnlyAttr();
  return attr.getValue();
}

::mlir::BoolAttr OneShotBufferizeOpAdaptor::getPrintConflictsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, OneShotBufferizeOp::getPrintConflictsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
  return attr;
}

bool OneShotBufferizeOpAdaptor::getPrintConflicts() {
  auto attr = getPrintConflictsAttr();
  return attr.getValue();
}

::mlir::LogicalResult OneShotBufferizeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_allow_return_allocs;
  ::mlir::Attribute tblgen_allow_unknown_ops;
  ::mlir::Attribute tblgen_bufferize_function_boundaries;
  ::mlir::Attribute tblgen_create_deallocs;
  ::mlir::Attribute tblgen_print_conflicts;
  ::mlir::Attribute tblgen_target_is_module;
  ::mlir::Attribute tblgen_test_analysis_only;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == OneShotBufferizeOp::getAllowReturnAllocsAttrName(*odsOpName)) {
      tblgen_allow_return_allocs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == OneShotBufferizeOp::getAllowUnknownOpsAttrName(*odsOpName)) {
      tblgen_allow_unknown_ops = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == OneShotBufferizeOp::getBufferizeFunctionBoundariesAttrName(*odsOpName)) {
      tblgen_bufferize_function_boundaries = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == OneShotBufferizeOp::getCreateDeallocsAttrName(*odsOpName)) {
      tblgen_create_deallocs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == OneShotBufferizeOp::getPrintConflictsAttrName(*odsOpName)) {
      tblgen_print_conflicts = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == OneShotBufferizeOp::getTargetIsModuleAttrName(*odsOpName)) {
      tblgen_target_is_module = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == OneShotBufferizeOp::getTestAnalysisOnlyAttrName(*odsOpName)) {
      tblgen_test_analysis_only = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_allow_return_allocs && !((tblgen_allow_return_allocs.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'transform.bufferization.one_shot_bufferize' op ""attribute 'allow_return_allocs' failed to satisfy constraint: bool attribute");

  if (tblgen_allow_unknown_ops && !((tblgen_allow_unknown_ops.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'transform.bufferization.one_shot_bufferize' op ""attribute 'allow_unknown_ops' failed to satisfy constraint: bool attribute");

  if (tblgen_bufferize_function_boundaries && !((tblgen_bufferize_function_boundaries.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'transform.bufferization.one_shot_bufferize' op ""attribute 'bufferize_function_boundaries' failed to satisfy constraint: bool attribute");

  if (tblgen_create_deallocs && !((tblgen_create_deallocs.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'transform.bufferization.one_shot_bufferize' op ""attribute 'create_deallocs' failed to satisfy constraint: bool attribute");

  if (tblgen_target_is_module && !((tblgen_target_is_module.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'transform.bufferization.one_shot_bufferize' op ""attribute 'target_is_module' failed to satisfy constraint: bool attribute");

  if (tblgen_test_analysis_only && !((tblgen_test_analysis_only.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'transform.bufferization.one_shot_bufferize' op ""attribute 'test_analysis_only' failed to satisfy constraint: bool attribute");

  if (tblgen_print_conflicts && !((tblgen_print_conflicts.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'transform.bufferization.one_shot_bufferize' op ""attribute 'print_conflicts' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OneShotBufferizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range OneShotBufferizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> OneShotBufferizeOp::getTarget() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange OneShotBufferizeOp::getTargetMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> OneShotBufferizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OneShotBufferizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::BoolAttr OneShotBufferizeOp::getAllowReturnAllocsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAllowReturnAllocsAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool OneShotBufferizeOp::getAllowReturnAllocs() {
  auto attr = getAllowReturnAllocsAttr();
  return attr.getValue();
}

::mlir::BoolAttr OneShotBufferizeOp::getAllowUnknownOpsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAllowUnknownOpsAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool OneShotBufferizeOp::getAllowUnknownOps() {
  auto attr = getAllowUnknownOpsAttr();
  return attr.getValue();
}

::mlir::BoolAttr OneShotBufferizeOp::getBufferizeFunctionBoundariesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getBufferizeFunctionBoundariesAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool OneShotBufferizeOp::getBufferizeFunctionBoundaries() {
  auto attr = getBufferizeFunctionBoundariesAttr();
  return attr.getValue();
}

::mlir::BoolAttr OneShotBufferizeOp::getCreateDeallocsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCreateDeallocsAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool OneShotBufferizeOp::getCreateDeallocs() {
  auto attr = getCreateDeallocsAttr();
  return attr.getValue();
}

::mlir::BoolAttr OneShotBufferizeOp::getTargetIsModuleAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTargetIsModuleAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool OneShotBufferizeOp::getTargetIsModule() {
  auto attr = getTargetIsModuleAttr();
  return attr.getValue();
}

::mlir::BoolAttr OneShotBufferizeOp::getTestAnalysisOnlyAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getTestAnalysisOnlyAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool OneShotBufferizeOp::getTestAnalysisOnly() {
  auto attr = getTestAnalysisOnlyAttr();
  return attr.getValue();
}

::mlir::BoolAttr OneShotBufferizeOp::getPrintConflictsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPrintConflictsAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool OneShotBufferizeOp::getPrintConflicts() {
  auto attr = getPrintConflictsAttr();
  return attr.getValue();
}

void OneShotBufferizeOp::setAllowReturnAllocsAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getAllowReturnAllocsAttrName(), attr);
}

void OneShotBufferizeOp::setAllowUnknownOpsAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getAllowUnknownOpsAttrName(), attr);
}

void OneShotBufferizeOp::setBufferizeFunctionBoundariesAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getBufferizeFunctionBoundariesAttrName(), attr);
}

void OneShotBufferizeOp::setCreateDeallocsAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getCreateDeallocsAttrName(), attr);
}

void OneShotBufferizeOp::setTargetIsModuleAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getTargetIsModuleAttrName(), attr);
}

void OneShotBufferizeOp::setTestAnalysisOnlyAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getTestAnalysisOnlyAttrName(), attr);
}

void OneShotBufferizeOp::setPrintConflictsAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getPrintConflictsAttrName(), attr);
}

void OneShotBufferizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value target, ::mlir::BoolAttr allow_return_allocs, ::mlir::BoolAttr allow_unknown_ops, ::mlir::BoolAttr bufferize_function_boundaries, ::mlir::BoolAttr create_deallocs, ::mlir::BoolAttr target_is_module, ::mlir::BoolAttr test_analysis_only, ::mlir::BoolAttr print_conflicts) {
  odsState.addOperands(target);
  if (allow_return_allocs) {
  odsState.addAttribute(getAllowReturnAllocsAttrName(odsState.name), allow_return_allocs);
  }
  if (allow_unknown_ops) {
  odsState.addAttribute(getAllowUnknownOpsAttrName(odsState.name), allow_unknown_ops);
  }
  if (bufferize_function_boundaries) {
  odsState.addAttribute(getBufferizeFunctionBoundariesAttrName(odsState.name), bufferize_function_boundaries);
  }
  if (create_deallocs) {
  odsState.addAttribute(getCreateDeallocsAttrName(odsState.name), create_deallocs);
  }
  if (target_is_module) {
  odsState.addAttribute(getTargetIsModuleAttrName(odsState.name), target_is_module);
  }
  if (test_analysis_only) {
  odsState.addAttribute(getTestAnalysisOnlyAttrName(odsState.name), test_analysis_only);
  }
  if (print_conflicts) {
  odsState.addAttribute(getPrintConflictsAttrName(odsState.name), print_conflicts);
  }
}

void OneShotBufferizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, ::mlir::BoolAttr allow_return_allocs, ::mlir::BoolAttr allow_unknown_ops, ::mlir::BoolAttr bufferize_function_boundaries, ::mlir::BoolAttr create_deallocs, ::mlir::BoolAttr target_is_module, ::mlir::BoolAttr test_analysis_only, ::mlir::BoolAttr print_conflicts) {
  odsState.addOperands(target);
  if (allow_return_allocs) {
  odsState.addAttribute(getAllowReturnAllocsAttrName(odsState.name), allow_return_allocs);
  }
  if (allow_unknown_ops) {
  odsState.addAttribute(getAllowUnknownOpsAttrName(odsState.name), allow_unknown_ops);
  }
  if (bufferize_function_boundaries) {
  odsState.addAttribute(getBufferizeFunctionBoundariesAttrName(odsState.name), bufferize_function_boundaries);
  }
  if (create_deallocs) {
  odsState.addAttribute(getCreateDeallocsAttrName(odsState.name), create_deallocs);
  }
  if (target_is_module) {
  odsState.addAttribute(getTargetIsModuleAttrName(odsState.name), target_is_module);
  }
  if (test_analysis_only) {
  odsState.addAttribute(getTestAnalysisOnlyAttrName(odsState.name), test_analysis_only);
  }
  if (print_conflicts) {
  odsState.addAttribute(getPrintConflictsAttrName(odsState.name), print_conflicts);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OneShotBufferizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value target, bool allow_return_allocs, bool allow_unknown_ops, bool bufferize_function_boundaries, bool create_deallocs, bool target_is_module, bool test_analysis_only, bool print_conflicts) {
  odsState.addOperands(target);
  odsState.addAttribute(getAllowReturnAllocsAttrName(odsState.name), odsBuilder.getBoolAttr(allow_return_allocs));
  odsState.addAttribute(getAllowUnknownOpsAttrName(odsState.name), odsBuilder.getBoolAttr(allow_unknown_ops));
  odsState.addAttribute(getBufferizeFunctionBoundariesAttrName(odsState.name), odsBuilder.getBoolAttr(bufferize_function_boundaries));
  odsState.addAttribute(getCreateDeallocsAttrName(odsState.name), odsBuilder.getBoolAttr(create_deallocs));
  odsState.addAttribute(getTargetIsModuleAttrName(odsState.name), odsBuilder.getBoolAttr(target_is_module));
  odsState.addAttribute(getTestAnalysisOnlyAttrName(odsState.name), odsBuilder.getBoolAttr(test_analysis_only));
  odsState.addAttribute(getPrintConflictsAttrName(odsState.name), odsBuilder.getBoolAttr(print_conflicts));
}

void OneShotBufferizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value target, bool allow_return_allocs, bool allow_unknown_ops, bool bufferize_function_boundaries, bool create_deallocs, bool target_is_module, bool test_analysis_only, bool print_conflicts) {
  odsState.addOperands(target);
  odsState.addAttribute(getAllowReturnAllocsAttrName(odsState.name), odsBuilder.getBoolAttr(allow_return_allocs));
  odsState.addAttribute(getAllowUnknownOpsAttrName(odsState.name), odsBuilder.getBoolAttr(allow_unknown_ops));
  odsState.addAttribute(getBufferizeFunctionBoundariesAttrName(odsState.name), odsBuilder.getBoolAttr(bufferize_function_boundaries));
  odsState.addAttribute(getCreateDeallocsAttrName(odsState.name), odsBuilder.getBoolAttr(create_deallocs));
  odsState.addAttribute(getTargetIsModuleAttrName(odsState.name), odsBuilder.getBoolAttr(target_is_module));
  odsState.addAttribute(getTestAnalysisOnlyAttrName(odsState.name), odsBuilder.getBoolAttr(test_analysis_only));
  odsState.addAttribute(getPrintConflictsAttrName(odsState.name), odsBuilder.getBoolAttr(print_conflicts));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OneShotBufferizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void OneShotBufferizeOp::populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes) {
  auto attrNames = opName.getAttributeNames();
  ::mlir::Builder odsBuilder(attrNames.front().getContext());
  if (!attributes.get(attrNames[0])) {
     attributes.append(attrNames[0], odsBuilder.getBoolAttr(false));
  }
  if (!attributes.get(attrNames[1])) {
     attributes.append(attrNames[1], odsBuilder.getBoolAttr(false));
  }
  if (!attributes.get(attrNames[2])) {
     attributes.append(attrNames[2], odsBuilder.getBoolAttr(false));
  }
  if (!attributes.get(attrNames[3])) {
     attributes.append(attrNames[3], odsBuilder.getBoolAttr(true));
  }
  if (!attributes.get(attrNames[5])) {
     attributes.append(attrNames[5], odsBuilder.getBoolAttr(true));
  }
  if (!attributes.get(attrNames[6])) {
     attributes.append(attrNames[6], odsBuilder.getBoolAttr(false));
  }
  if (!attributes.get(attrNames[4])) {
     attributes.append(attrNames[4], odsBuilder.getBoolAttr(false));
  }
}

::mlir::LogicalResult OneShotBufferizeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_allow_return_allocs;
  ::mlir::Attribute tblgen_allow_unknown_ops;
  ::mlir::Attribute tblgen_bufferize_function_boundaries;
  ::mlir::Attribute tblgen_create_deallocs;
  ::mlir::Attribute tblgen_print_conflicts;
  ::mlir::Attribute tblgen_target_is_module;
  ::mlir::Attribute tblgen_test_analysis_only;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getAllowReturnAllocsAttrName()) {
      tblgen_allow_return_allocs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAllowUnknownOpsAttrName()) {
      tblgen_allow_unknown_ops = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getBufferizeFunctionBoundariesAttrName()) {
      tblgen_bufferize_function_boundaries = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getCreateDeallocsAttrName()) {
      tblgen_create_deallocs = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getPrintConflictsAttrName()) {
      tblgen_print_conflicts = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getTargetIsModuleAttrName()) {
      tblgen_target_is_module = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getTestAnalysisOnlyAttrName()) {
      tblgen_test_analysis_only = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BufferizationTransformOps0(*this, tblgen_allow_return_allocs, "allow_return_allocs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BufferizationTransformOps0(*this, tblgen_allow_unknown_ops, "allow_unknown_ops")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BufferizationTransformOps0(*this, tblgen_bufferize_function_boundaries, "bufferize_function_boundaries")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BufferizationTransformOps0(*this, tblgen_create_deallocs, "create_deallocs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BufferizationTransformOps0(*this, tblgen_target_is_module, "target_is_module")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BufferizationTransformOps0(*this, tblgen_test_analysis_only, "test_analysis_only")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_BufferizationTransformOps0(*this, tblgen_print_conflicts, "print_conflicts")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_BufferizationTransformOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult OneShotBufferizeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult OneShotBufferizeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void OneShotBufferizeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTarget();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace transform
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::transform::OneShotBufferizeOp)


#endif  // GET_OP_CLASSES

