/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::omp::AtomicCaptureOp,
::mlir::omp::AtomicReadOp,
::mlir::omp::AtomicUpdateOp,
::mlir::omp::AtomicWriteOp,
::mlir::omp::BarrierOp,
::mlir::omp::CancelOp,
::mlir::omp::CancellationPointOp,
::mlir::omp::CriticalDeclareOp,
::mlir::omp::CriticalOp,
::mlir::omp::FlushOp,
::mlir::omp::MasterOp,
::mlir::omp::OrderedOp,
::mlir::omp::OrderedRegionOp,
::mlir::omp::ParallelOp,
::mlir::omp::ReductionDeclareOp,
::mlir::omp::ReductionOp,
::mlir::omp::SectionOp,
::mlir::omp::SectionsOp,
::mlir::omp::SimdLoopOp,
::mlir::omp::SingleOp,
::mlir::omp::TargetOp,
::mlir::omp::TaskOp,
::mlir::omp::TaskwaitOp,
::mlir::omp::TaskyieldOp,
::mlir::omp::TerminatorOp,
::mlir::omp::ThreadprivateOp,
::mlir::omp::WsLoopOp,
::mlir::omp::YieldOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace omp {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::omp::PointerLikeType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be OpenMP-compatible variable type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || ((type.isa<::mlir::IndexType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer or index, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IntegerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseMemoryOrderKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: MemoryOrderKind Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseCancellationConstructTypeAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: CancellationConstructType Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::FlatSymbolRefAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: flat symbol reference attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseDependAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: depend clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((attr.cast<::mlir::IntegerAttr>().getInt() >= 0)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: symbol ref array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseProcBindKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: ProcBindKind Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any type attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseScheduleKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: ScheduleKind Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps12(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ScheduleModifierAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: OpenMP Schedule Modifier";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps13(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseOrderKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: OrderKind Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_OpenMPOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_OpenMPOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}
} // namespace omp
} // namespace mlir
namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicCaptureOp definitions
//===----------------------------------------------------------------------===//

AtomicCaptureOpAdaptor::AtomicCaptureOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.atomic.capture", odsAttrs.getContext());
}

AtomicCaptureOpAdaptor::AtomicCaptureOpAdaptor(AtomicCaptureOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AtomicCaptureOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AtomicCaptureOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AtomicCaptureOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr AtomicCaptureOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr AtomicCaptureOpAdaptor::hint_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicCaptureOp::hint_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0);
    return attr;
}

uint64_t AtomicCaptureOpAdaptor::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicCaptureOpAdaptor::memory_order_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicCaptureOp::memory_order_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
    return attr;
}

::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> AtomicCaptureOpAdaptor::memory_order_val() {
  auto attr = memory_order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::llvm::None);
}

::mlir::RegionRange AtomicCaptureOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &AtomicCaptureOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult AtomicCaptureOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AtomicCaptureOp::hint_valAttrName(*odsOpName)) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicCaptureOp::memory_order_valAttrName(*odsOpName)) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'omp.atomic.capture' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_memory_order_val && !((tblgen_memory_order_val.isa<::mlir::omp::ClauseMemoryOrderKindAttr>())))
    return emitError(loc, "'omp.atomic.capture' op ""attribute 'memory_order_val' failed to satisfy constraint: MemoryOrderKind Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicCaptureOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicCaptureOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AtomicCaptureOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicCaptureOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &AtomicCaptureOp::region() {
  return (*this)->getRegion(0);
}

::mlir::IntegerAttr AtomicCaptureOp::hint_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, hint_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t AtomicCaptureOp::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicCaptureOp::memory_order_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, memory_order_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
}

::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> AtomicCaptureOp::memory_order_val() {
  auto attr = memory_order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::llvm::None);
}

void AtomicCaptureOp::hint_valAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(hint_valAttrName(), attr);
}

void AtomicCaptureOp::memory_order_valAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr) {
  (*this)->setAttr(memory_order_valAttrName(), attr);
}

::mlir::Attribute AtomicCaptureOp::removeMemory_order_valAttr() {
  return (*this)->removeAttr(memory_order_valAttrName());
}

void AtomicCaptureOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  if (hint_val) {
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
}

void AtomicCaptureOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  if (hint_val) {
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicCaptureOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addAttribute(hint_valAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
}

void AtomicCaptureOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addAttribute(hint_valAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicCaptureOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AtomicCaptureOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == hint_valAttrName()) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == memory_order_valAttrName()) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps1(*this, tblgen_memory_order_val, "memory_order_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult AtomicCaptureOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AtomicCaptureOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr;
  ::mlir::IntegerAttr hint_valAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool memory_orderClause = false;
  bool hintClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("memory_order"))) {

  if (memory_orderClause) {
    return parser.emitError(parser.getNameLoc())
          << "`memory_order` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  memory_orderClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, memory_order_valAttr))
      return ::mlir::failure();
    if (memory_order_valAttr)
      result.addAttribute("memory_order_val", memory_order_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("hint"))) {

  if (hintClause) {
    return parser.emitError(parser.getNameLoc())
          << "`hint` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  hintClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseSynchronizationHint(parser, hint_valAttr))
      return ::mlir::failure();
    result.addAttribute("hint_val", hint_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();

  ensureTerminator(*regionRegion, parser.getBuilder(), result.location);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void AtomicCaptureOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || memory_order_valAttr()) {
  _odsPrinter << ' ' << "memory_order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, memory_order_valAttr());
  _odsPrinter << ")";
  }
  if (false || hint_valAttr()) {
  _odsPrinter << ' ' << "hint";
  _odsPrinter << "(";
  printSynchronizationHint(_odsPrinter, *this, hint_valAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';

  {
    bool printTerminator = true;
    if (auto *term = region().empty() ? nullptr : region().begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    _odsPrinter.printRegion(region(), /*printEntryBlockArgs=*/true,
      /*printBlockTerminators=*/printTerminator);
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"memory_order_val", "hint_val"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicCaptureOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicReadOp definitions
//===----------------------------------------------------------------------===//

AtomicReadOpAdaptor::AtomicReadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.atomic.read", odsAttrs.getContext());
}

AtomicReadOpAdaptor::AtomicReadOpAdaptor(AtomicReadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AtomicReadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AtomicReadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AtomicReadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicReadOpAdaptor::x() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicReadOpAdaptor::v() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr AtomicReadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr AtomicReadOpAdaptor::hint_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicReadOp::hint_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0);
    return attr;
}

uint64_t AtomicReadOpAdaptor::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicReadOpAdaptor::memory_order_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicReadOp::memory_order_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
    return attr;
}

::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> AtomicReadOpAdaptor::memory_order_val() {
  auto attr = memory_order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::llvm::None);
}

::mlir::LogicalResult AtomicReadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AtomicReadOp::hint_valAttrName(*odsOpName)) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicReadOp::memory_order_valAttrName(*odsOpName)) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'omp.atomic.read' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_memory_order_val && !((tblgen_memory_order_val.isa<::mlir::omp::ClauseMemoryOrderKindAttr>())))
    return emitError(loc, "'omp.atomic.read' op ""attribute 'memory_order_val' failed to satisfy constraint: MemoryOrderKind Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicReadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicReadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicReadOp::x() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicReadOp::v() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AtomicReadOp::xMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtomicReadOp::vMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtomicReadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicReadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr AtomicReadOp::hint_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, hint_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t AtomicReadOp::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicReadOp::memory_order_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, memory_order_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
}

::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> AtomicReadOp::memory_order_val() {
  auto attr = memory_order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::llvm::None);
}

void AtomicReadOp::hint_valAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(hint_valAttrName(), attr);
}

void AtomicReadOp::memory_order_valAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr) {
  (*this)->setAttr(memory_order_valAttrName(), attr);
}

::mlir::Attribute AtomicReadOp::removeMemory_order_valAttr() {
  return (*this)->removeAttr(memory_order_valAttrName());
}

void AtomicReadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::Value v, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addOperands(v);
  if (hint_val) {
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
}

void AtomicReadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value v, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addOperands(v);
  if (hint_val) {
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicReadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::Value v, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addOperands(v);
  odsState.addAttribute(hint_valAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
}

void AtomicReadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value v, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addOperands(v);
  odsState.addAttribute(hint_valAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicReadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AtomicReadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == hint_valAttrName()) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == memory_order_valAttrName()) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps1(*this, tblgen_memory_order_val, "memory_order_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(0).begin()).getType(), (*this->getODSOperands(1).begin()).getType()})))))
    return emitOpError("failed to verify that all of {x, v} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult AtomicReadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AtomicReadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand vRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> vOperands(vRawOperands);  ::llvm::SMLoc vOperandsLoc;
  (void)vOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand xRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> xOperands(xRawOperands);  ::llvm::SMLoc xOperandsLoc;
  (void)xOperandsLoc;
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr;
  ::mlir::IntegerAttr hint_valAttr;
  ::mlir::Type xRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> xTypes(xRawTypes);

  vOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(vRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  xOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(xRawOperands[0]))
    return ::mlir::failure();
  bool memory_orderClause = false;
  bool hintClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("memory_order"))) {

  if (memory_orderClause) {
    return parser.emitError(parser.getNameLoc())
          << "`memory_order` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  memory_orderClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, memory_order_valAttr))
      return ::mlir::failure();
    if (memory_order_valAttr)
      result.addAttribute("memory_order_val", memory_order_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("hint"))) {

  if (hintClause) {
    return parser.emitError(parser.getNameLoc())
          << "`hint` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  hintClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseSynchronizationHint(parser, hint_valAttr))
      return ::mlir::failure();
    result.addAttribute("hint_val", hint_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::omp::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    xRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(xOperands, xTypes, xOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(vOperands, xTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtomicReadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << v();
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter << x();
  _odsPrinter << ' ' << " ";
  if (false || memory_order_valAttr()) {
  _odsPrinter << ' ' << "memory_order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, memory_order_valAttr());
  _odsPrinter << ")";
  }
  if (false || hint_valAttr()) {
  _odsPrinter << ' ' << "hint";
  _odsPrinter << "(";
  printSynchronizationHint(_odsPrinter, *this, hint_valAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = x().getType();
    if (auto validType = type.dyn_cast<::mlir::omp::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"memory_order_val", "hint_val"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicReadOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicUpdateOp definitions
//===----------------------------------------------------------------------===//

AtomicUpdateOpAdaptor::AtomicUpdateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.atomic.update", odsAttrs.getContext());
}

AtomicUpdateOpAdaptor::AtomicUpdateOpAdaptor(AtomicUpdateOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AtomicUpdateOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AtomicUpdateOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AtomicUpdateOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicUpdateOpAdaptor::x() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AtomicUpdateOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr AtomicUpdateOpAdaptor::hint_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicUpdateOp::hint_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0);
    return attr;
}

uint64_t AtomicUpdateOpAdaptor::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicUpdateOpAdaptor::memory_order_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicUpdateOp::memory_order_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
    return attr;
}

::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> AtomicUpdateOpAdaptor::memory_order_val() {
  auto attr = memory_order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::llvm::None);
}

::mlir::RegionRange AtomicUpdateOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &AtomicUpdateOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult AtomicUpdateOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AtomicUpdateOp::hint_valAttrName(*odsOpName)) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicUpdateOp::memory_order_valAttrName(*odsOpName)) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'omp.atomic.update' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_memory_order_val && !((tblgen_memory_order_val.isa<::mlir::omp::ClauseMemoryOrderKindAttr>())))
    return emitError(loc, "'omp.atomic.update' op ""attribute 'memory_order_val' failed to satisfy constraint: MemoryOrderKind Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicUpdateOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicUpdateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicUpdateOp::x() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AtomicUpdateOp::xMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtomicUpdateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicUpdateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &AtomicUpdateOp::region() {
  return (*this)->getRegion(0);
}

::mlir::IntegerAttr AtomicUpdateOp::hint_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, hint_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t AtomicUpdateOp::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicUpdateOp::memory_order_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, memory_order_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
}

::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> AtomicUpdateOp::memory_order_val() {
  auto attr = memory_order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::llvm::None);
}

void AtomicUpdateOp::hint_valAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(hint_valAttrName(), attr);
}

void AtomicUpdateOp::memory_order_valAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr) {
  (*this)->setAttr(memory_order_valAttrName(), attr);
}

::mlir::Attribute AtomicUpdateOp::removeMemory_order_valAttr() {
  return (*this)->removeAttr(memory_order_valAttrName());
}

void AtomicUpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  if (hint_val) {
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
}

void AtomicUpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  if (hint_val) {
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicUpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value x, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addAttribute(hint_valAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
}

void AtomicUpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addAttribute(hint_valAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicUpdateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AtomicUpdateOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == hint_valAttrName()) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == memory_order_valAttrName()) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps1(*this, tblgen_memory_order_val, "memory_order_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult AtomicUpdateOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AtomicUpdateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr;
  ::mlir::IntegerAttr hint_valAttr;
  ::mlir::OpAsmParser::UnresolvedOperand xRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> xOperands(xRawOperands);  ::llvm::SMLoc xOperandsLoc;
  (void)xOperandsLoc;
  ::mlir::Type xRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> xTypes(xRawTypes);
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool memory_orderClause = false;
  bool hintClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("memory_order"))) {

  if (memory_orderClause) {
    return parser.emitError(parser.getNameLoc())
          << "`memory_order` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  memory_orderClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, memory_order_valAttr))
      return ::mlir::failure();
    if (memory_order_valAttr)
      result.addAttribute("memory_order_val", memory_order_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("hint"))) {

  if (hintClause) {
    return parser.emitError(parser.getNameLoc())
          << "`hint` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  hintClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseSynchronizationHint(parser, hint_valAttr))
      return ::mlir::failure();
    result.addAttribute("hint_val", hint_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  xOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(xRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::omp::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    xRawTypes[0] = type;
  }

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();

  ensureTerminator(*regionRegion, parser.getBuilder(), result.location);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  if (parser.resolveOperands(xOperands, xTypes, xOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtomicUpdateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || memory_order_valAttr()) {
  _odsPrinter << ' ' << "memory_order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, memory_order_valAttr());
  _odsPrinter << ")";
  }
  if (false || hint_valAttr()) {
  _odsPrinter << ' ' << "hint";
  _odsPrinter << "(";
  printSynchronizationHint(_odsPrinter, *this, hint_valAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter << x();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = x().getType();
    if (auto validType = type.dyn_cast<::mlir::omp::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ';

  {
    bool printTerminator = true;
    if (auto *term = region().empty() ? nullptr : region().begin()->getTerminator()) {
      printTerminator = !term->getAttrDictionary().empty() ||
                        term->getNumOperands() != 0 ||
                        term->getNumResults() != 0;
    }
    _odsPrinter.printRegion(region(), /*printEntryBlockArgs=*/true,
      /*printBlockTerminators=*/printTerminator);
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"memory_order_val", "hint_val"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicUpdateOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicWriteOp definitions
//===----------------------------------------------------------------------===//

AtomicWriteOpAdaptor::AtomicWriteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.atomic.write", odsAttrs.getContext());
}

AtomicWriteOpAdaptor::AtomicWriteOpAdaptor(AtomicWriteOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AtomicWriteOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AtomicWriteOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AtomicWriteOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicWriteOpAdaptor::address() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicWriteOpAdaptor::value() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr AtomicWriteOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr AtomicWriteOpAdaptor::hint_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicWriteOp::hint_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0);
    return attr;
}

uint64_t AtomicWriteOpAdaptor::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicWriteOpAdaptor::memory_order_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AtomicWriteOp::memory_order_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
    return attr;
}

::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> AtomicWriteOpAdaptor::memory_order_val() {
  auto attr = memory_order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::llvm::None);
}

::mlir::LogicalResult AtomicWriteOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AtomicWriteOp::hint_valAttrName(*odsOpName)) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AtomicWriteOp::memory_order_valAttrName(*odsOpName)) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'omp.atomic.write' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_memory_order_val && !((tblgen_memory_order_val.isa<::mlir::omp::ClauseMemoryOrderKindAttr>())))
    return emitError(loc, "'omp.atomic.write' op ""attribute 'memory_order_val' failed to satisfy constraint: MemoryOrderKind Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicWriteOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicWriteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicWriteOp::address() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicWriteOp::value() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AtomicWriteOp::addressMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtomicWriteOp::valueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtomicWriteOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicWriteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr AtomicWriteOp::hint_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, hint_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t AtomicWriteOp::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicWriteOp::memory_order_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, memory_order_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
}

::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> AtomicWriteOp::memory_order_val() {
  auto attr = memory_order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::llvm::None);
}

void AtomicWriteOp::hint_valAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(hint_valAttrName(), attr);
}

void AtomicWriteOp::memory_order_valAttr(::mlir::omp::ClauseMemoryOrderKindAttr attr) {
  (*this)->setAttr(memory_order_valAttrName(), attr);
}

::mlir::Attribute AtomicWriteOp::removeMemory_order_valAttr() {
  return (*this)->removeAttr(memory_order_valAttrName());
}

void AtomicWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value address, ::mlir::Value value, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(address);
  odsState.addOperands(value);
  if (hint_val) {
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
}

void AtomicWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value value, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(address);
  odsState.addOperands(value);
  if (hint_val) {
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  }
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value address, ::mlir::Value value, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(address);
  odsState.addOperands(value);
  odsState.addAttribute(hint_valAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
}

void AtomicWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value value, uint64_t hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(address);
  odsState.addOperands(value);
  odsState.addAttribute(hint_valAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicWriteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AtomicWriteOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_hint_val;
  ::mlir::Attribute tblgen_memory_order_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == hint_valAttrName()) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == memory_order_valAttrName()) {
      tblgen_memory_order_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps1(*this, tblgen_memory_order_val, "memory_order_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AtomicWriteOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AtomicWriteOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand addressRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> addressOperands(addressRawOperands);  ::llvm::SMLoc addressOperandsLoc;
  (void)addressOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::IntegerAttr hint_valAttr;
  ::mlir::omp::ClauseMemoryOrderKindAttr memory_order_valAttr;
  ::mlir::Type addressRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> addressTypes(addressRawTypes);
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  addressOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(addressRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  bool hintClause = false;
  bool memory_orderClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("hint"))) {

  if (hintClause) {
    return parser.emitError(parser.getNameLoc())
          << "`hint` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  hintClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseSynchronizationHint(parser, hint_valAttr))
      return ::mlir::failure();
    result.addAttribute("hint_val", hint_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("memory_order"))) {

  if (memory_orderClause) {
    return parser.emitError(parser.getNameLoc())
          << "`memory_order` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  memory_orderClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, memory_order_valAttr))
      return ::mlir::failure();
    if (memory_order_valAttr)
      result.addAttribute("memory_order_val", memory_order_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::omp::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    addressRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(addressOperands, addressTypes, addressOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AtomicWriteOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << address();
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter << value();
  _odsPrinter << ' ' << " ";
  if (false || hint_valAttr()) {
  _odsPrinter << ' ' << "hint";
  _odsPrinter << "(";
  printSynchronizationHint(_odsPrinter, *this, hint_valAttr());
  _odsPrinter << ")";
  }
  if (false || memory_order_valAttr()) {
  _odsPrinter << ' ' << "memory_order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, memory_order_valAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = address().getType();
    if (auto validType = type.dyn_cast<::mlir::omp::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = value().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"hint_val", "memory_order_val"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicWriteOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::BarrierOp definitions
//===----------------------------------------------------------------------===//

BarrierOpAdaptor::BarrierOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.barrier", odsAttrs.getContext());
}

BarrierOpAdaptor::BarrierOpAdaptor(BarrierOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BarrierOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BarrierOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BarrierOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr BarrierOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BarrierOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BarrierOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BarrierOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BarrierOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BarrierOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void BarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void BarrierOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BarrierOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BarrierOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult BarrierOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BarrierOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void BarrierOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::BarrierOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CancelOp definitions
//===----------------------------------------------------------------------===//

CancelOpAdaptor::CancelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.cancel", odsAttrs.getContext());
}

CancelOpAdaptor::CancelOpAdaptor(CancelOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CancelOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CancelOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange CancelOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CancelOpAdaptor::if_expr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr CancelOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::omp::ClauseCancellationConstructTypeAttr CancelOpAdaptor::cancellation_construct_type_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CancelOp::cancellation_construct_type_valAttrName(*odsOpName)).cast<::mlir::omp::ClauseCancellationConstructTypeAttr>();
    return attr;
}

::mlir::omp::ClauseCancellationConstructType CancelOpAdaptor::cancellation_construct_type_val() {
  auto attr = cancellation_construct_type_valAttr();
  return attr.getValue();
}

::mlir::LogicalResult CancelOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_cancellation_construct_type_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.cancel' op ""requires attribute 'cancellation_construct_type_val'");
    if (namedAttrIt->getName() == CancelOp::cancellation_construct_type_valAttrName(*odsOpName)) {
      tblgen_cancellation_construct_type_val = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_cancellation_construct_type_val && !((tblgen_cancellation_construct_type_val.isa<::mlir::omp::ClauseCancellationConstructTypeAttr>())))
    return emitError(loc, "'omp.cancel' op ""attribute 'cancellation_construct_type_val' failed to satisfy constraint: CancellationConstructType Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CancelOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range CancelOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CancelOp::if_expr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange CancelOp::if_exprMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CancelOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CancelOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::omp::ClauseCancellationConstructTypeAttr CancelOp::cancellation_construct_type_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, cancellation_construct_type_valAttrName()).cast<::mlir::omp::ClauseCancellationConstructTypeAttr>();
}

::mlir::omp::ClauseCancellationConstructType CancelOp::cancellation_construct_type_val() {
  auto attr = cancellation_construct_type_valAttr();
  return attr.getValue();
}

void CancelOp::cancellation_construct_type_valAttr(::mlir::omp::ClauseCancellationConstructTypeAttr attr) {
  (*this)->setAttr(cancellation_construct_type_valAttrName(), attr);
}

void CancelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val, /*optional*/::mlir::Value if_expr) {
  if (if_expr)
    odsState.addOperands(if_expr);
  odsState.addAttribute(cancellation_construct_type_valAttrName(odsState.name), cancellation_construct_type_val);
}

void CancelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val, /*optional*/::mlir::Value if_expr) {
  if (if_expr)
    odsState.addOperands(if_expr);
  odsState.addAttribute(cancellation_construct_type_valAttrName(odsState.name), cancellation_construct_type_val);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CancelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val, /*optional*/::mlir::Value if_expr) {
  if (if_expr)
    odsState.addOperands(if_expr);
  odsState.addAttribute(cancellation_construct_type_valAttrName(odsState.name), ::mlir::omp::ClauseCancellationConstructTypeAttr::get(odsBuilder.getContext(), cancellation_construct_type_val));
}

void CancelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val, /*optional*/::mlir::Value if_expr) {
  if (if_expr)
    odsState.addOperands(if_expr);
  odsState.addAttribute(cancellation_construct_type_valAttrName(odsState.name), ::mlir::omp::ClauseCancellationConstructTypeAttr::get(odsBuilder.getContext(), cancellation_construct_type_val));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CancelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CancelOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_cancellation_construct_type_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'cancellation_construct_type_val'");
    if (namedAttrIt->getName() == cancellation_construct_type_valAttrName()) {
      tblgen_cancellation_construct_type_val = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps2(*this, tblgen_cancellation_construct_type_val, "cancellation_construct_type_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CancelOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CancelOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_valAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> if_exprOperands;
  ::llvm::SMLoc if_exprOperandsLoc;
  (void)if_exprOperandsLoc;
  if (parser.parseKeyword("cancellation_construct_type"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, cancellation_construct_type_valAttr))
      return ::mlir::failure();
    result.addAttribute("cancellation_construct_type_val", cancellation_construct_type_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("if"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    if_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_exprOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(if_exprOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CancelOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "cancellation_construct_type";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, cancellation_construct_type_valAttr());
  _odsPrinter << ")";
  if (if_expr()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = if_expr())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"cancellation_construct_type_val"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::CancelOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CancellationPointOp definitions
//===----------------------------------------------------------------------===//

CancellationPointOpAdaptor::CancellationPointOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.cancellationpoint", odsAttrs.getContext());
}

CancellationPointOpAdaptor::CancellationPointOpAdaptor(CancellationPointOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CancellationPointOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CancellationPointOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CancellationPointOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CancellationPointOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::omp::ClauseCancellationConstructTypeAttr CancellationPointOpAdaptor::cancellation_construct_type_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CancellationPointOp::cancellation_construct_type_valAttrName(*odsOpName)).cast<::mlir::omp::ClauseCancellationConstructTypeAttr>();
    return attr;
}

::mlir::omp::ClauseCancellationConstructType CancellationPointOpAdaptor::cancellation_construct_type_val() {
  auto attr = cancellation_construct_type_valAttr();
  return attr.getValue();
}

::mlir::LogicalResult CancellationPointOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_cancellation_construct_type_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.cancellationpoint' op ""requires attribute 'cancellation_construct_type_val'");
    if (namedAttrIt->getName() == CancellationPointOp::cancellation_construct_type_valAttrName(*odsOpName)) {
      tblgen_cancellation_construct_type_val = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_cancellation_construct_type_val && !((tblgen_cancellation_construct_type_val.isa<::mlir::omp::ClauseCancellationConstructTypeAttr>())))
    return emitError(loc, "'omp.cancellationpoint' op ""attribute 'cancellation_construct_type_val' failed to satisfy constraint: CancellationConstructType Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CancellationPointOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CancellationPointOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CancellationPointOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CancellationPointOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::omp::ClauseCancellationConstructTypeAttr CancellationPointOp::cancellation_construct_type_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, cancellation_construct_type_valAttrName()).cast<::mlir::omp::ClauseCancellationConstructTypeAttr>();
}

::mlir::omp::ClauseCancellationConstructType CancellationPointOp::cancellation_construct_type_val() {
  auto attr = cancellation_construct_type_valAttr();
  return attr.getValue();
}

void CancellationPointOp::cancellation_construct_type_valAttr(::mlir::omp::ClauseCancellationConstructTypeAttr attr) {
  (*this)->setAttr(cancellation_construct_type_valAttrName(), attr);
}

void CancellationPointOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val) {
  odsState.addAttribute(cancellation_construct_type_valAttrName(odsState.name), cancellation_construct_type_val);
}

void CancellationPointOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_val) {
  odsState.addAttribute(cancellation_construct_type_valAttrName(odsState.name), cancellation_construct_type_val);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CancellationPointOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val) {
  odsState.addAttribute(cancellation_construct_type_valAttrName(odsState.name), ::mlir::omp::ClauseCancellationConstructTypeAttr::get(odsBuilder.getContext(), cancellation_construct_type_val));
}

void CancellationPointOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::omp::ClauseCancellationConstructType cancellation_construct_type_val) {
  odsState.addAttribute(cancellation_construct_type_valAttrName(odsState.name), ::mlir::omp::ClauseCancellationConstructTypeAttr::get(odsBuilder.getContext(), cancellation_construct_type_val));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CancellationPointOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CancellationPointOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_cancellation_construct_type_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'cancellation_construct_type_val'");
    if (namedAttrIt->getName() == cancellation_construct_type_valAttrName()) {
      tblgen_cancellation_construct_type_val = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps2(*this, tblgen_cancellation_construct_type_val, "cancellation_construct_type_val")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult CancellationPointOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CancellationPointOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::omp::ClauseCancellationConstructTypeAttr cancellation_construct_type_valAttr;
  if (parser.parseKeyword("cancellation_construct_type"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, cancellation_construct_type_valAttr))
      return ::mlir::failure();
    result.addAttribute("cancellation_construct_type_val", cancellation_construct_type_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void CancellationPointOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "cancellation_construct_type";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, cancellation_construct_type_valAttr());
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"cancellation_construct_type_val"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::CancellationPointOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalDeclareOp definitions
//===----------------------------------------------------------------------===//

CriticalDeclareOpAdaptor::CriticalDeclareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.critical.declare", odsAttrs.getContext());
}

CriticalDeclareOpAdaptor::CriticalDeclareOpAdaptor(CriticalDeclareOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CriticalDeclareOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CriticalDeclareOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CriticalDeclareOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CriticalDeclareOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CriticalDeclareOpAdaptor::sym_nameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CriticalDeclareOp::sym_nameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef CriticalDeclareOpAdaptor::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

::mlir::IntegerAttr CriticalDeclareOpAdaptor::hint_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CriticalDeclareOp::hint_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0);
    return attr;
}

uint64_t CriticalDeclareOpAdaptor::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult CriticalDeclareOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_hint_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.critical.declare' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == CriticalDeclareOp::sym_nameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CriticalDeclareOp::hint_valAttrName(*odsOpName)) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'omp.critical.declare' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'omp.critical.declare' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CriticalDeclareOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CriticalDeclareOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CriticalDeclareOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CriticalDeclareOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr CriticalDeclareOp::sym_nameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, sym_nameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef CriticalDeclareOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

::mlir::IntegerAttr CriticalDeclareOp::hint_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, hint_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t CriticalDeclareOp::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

void CriticalDeclareOp::sym_nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(sym_nameAttrName(), attr);
}

void CriticalDeclareOp::hint_valAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(hint_valAttrName(), attr);
}

void CriticalDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr hint_val) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), sym_name);
  if (hint_val) {
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  }
}

void CriticalDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr hint_val) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), sym_name);
  if (hint_val) {
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CriticalDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, uint64_t hint_val) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(hint_valAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
}

void CriticalDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, uint64_t hint_val) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(hint_valAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), hint_val));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CriticalDeclareOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CriticalDeclareOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_hint_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == sym_nameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == hint_valAttrName()) {
      tblgen_hint_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps3(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult CriticalDeclareOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CriticalDeclareOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;
  ::mlir::IntegerAttr hint_valAttr;

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  bool hintClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("hint"))) {

  if (hintClause) {
    return parser.emitError(parser.getNameLoc())
          << "`hint` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  hintClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseSynchronizationHint(parser, hint_valAttr))
      return ::mlir::failure();
    result.addAttribute("hint_val", hint_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void CriticalDeclareOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(sym_nameAttr().getValue());
  _odsPrinter << ' ' << " ";
  if (false || hint_valAttr()) {
  _odsPrinter << ' ' << "hint";
  _odsPrinter << "(";
  printSynchronizationHint(_odsPrinter, *this, hint_valAttr());
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"sym_name", "hint_val"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalOp definitions
//===----------------------------------------------------------------------===//

CriticalOpAdaptor::CriticalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.critical", odsAttrs.getContext());
}

CriticalOpAdaptor::CriticalOpAdaptor(CriticalOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CriticalOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CriticalOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CriticalOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CriticalOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr CriticalOpAdaptor::nameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CriticalOp::nameAttrName(*odsOpName)).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > CriticalOpAdaptor::name() {
  auto attr = nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::RegionRange CriticalOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &CriticalOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult CriticalOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == CriticalOp::nameAttrName(*odsOpName)) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_name && !((tblgen_name.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'omp.critical' op ""attribute 'name' failed to satisfy constraint: flat symbol reference attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CriticalOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CriticalOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CriticalOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CriticalOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &CriticalOp::region() {
  return (*this)->getRegion(0);
}

::mlir::FlatSymbolRefAttr CriticalOp::nameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, nameAttrName()).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
}

::llvm::Optional< ::llvm::StringRef > CriticalOp::name() {
  auto attr = nameAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void CriticalOp::nameAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(nameAttrName(), attr);
}

::mlir::Attribute CriticalOp::removeNameAttr() {
  return (*this)->removeAttr(nameAttrName());
}

void CriticalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::FlatSymbolRefAttr name) {
  if (name) {
  odsState.addAttribute(nameAttrName(odsState.name), name);
  }
  (void)odsState.addRegion();
}

void CriticalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::FlatSymbolRefAttr name) {
  if (name) {
  odsState.addAttribute(nameAttrName(odsState.name), name);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CriticalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CriticalOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == nameAttrName()) {
      tblgen_name = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps4(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult CriticalOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CriticalOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::FlatSymbolRefAttr nameAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  if (succeeded(parser.parseOptionalLParen())) {

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void CriticalOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("name")) {
  _odsPrinter << "(";
  _odsPrinter.printAttributeWithoutType(nameAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(region());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"name"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::FlushOp definitions
//===----------------------------------------------------------------------===//

FlushOpAdaptor::FlushOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.flush", odsAttrs.getContext());
}

FlushOpAdaptor::FlushOpAdaptor(FlushOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FlushOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FlushOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange FlushOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange FlushOpAdaptor::varList() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr FlushOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FlushOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FlushOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range FlushOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range FlushOp::varList() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange FlushOp::varListMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FlushOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FlushOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void FlushOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange varList) {
  odsState.addOperands(varList);
}

void FlushOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FlushOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FlushOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FlushOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> varListOperands;
  ::llvm::SMLoc varListOperandsLoc;
  (void)varListOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> varListTypes;
  if (succeeded(parser.parseOptionalLParen())) {

  varListOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(varListOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(varListTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(varListOperands, varListTypes, varListOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FlushOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!varList().empty()) {
  _odsPrinter << "(";
  _odsPrinter << varList();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << varList().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::FlushOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::MasterOp definitions
//===----------------------------------------------------------------------===//

MasterOpAdaptor::MasterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.master", odsAttrs.getContext());
}

MasterOpAdaptor::MasterOpAdaptor(MasterOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MasterOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MasterOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MasterOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr MasterOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange MasterOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MasterOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult MasterOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MasterOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MasterOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> MasterOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MasterOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &MasterOp::region() {
  return (*this)->getRegion(0);
}

void MasterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
  (void)odsState.addRegion();
}

void MasterOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MasterOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MasterOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult MasterOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MasterOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void MasterOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printRegion(region());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::MasterOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::OrderedOp definitions
//===----------------------------------------------------------------------===//

OrderedOpAdaptor::OrderedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.ordered", odsAttrs.getContext());
}

OrderedOpAdaptor::OrderedOpAdaptor(OrderedOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange OrderedOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> OrderedOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange OrderedOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange OrderedOpAdaptor::depend_vec_vars() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr OrderedOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::omp::ClauseDependAttr OrderedOpAdaptor::depend_type_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, OrderedOp::depend_type_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseDependAttr>();
    return attr;
}

::llvm::Optional<::mlir::omp::ClauseDepend> OrderedOpAdaptor::depend_type_val() {
  auto attr = depend_type_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseDepend>(attr.getValue()) : (::llvm::None);
}

::mlir::IntegerAttr OrderedOpAdaptor::num_loops_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, OrderedOp::num_loops_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    return attr;
}

::llvm::Optional<uint64_t> OrderedOpAdaptor::num_loops_val() {
  auto attr = num_loops_valAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult OrderedOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_depend_type_val;
  ::mlir::Attribute tblgen_num_loops_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == OrderedOp::depend_type_valAttrName(*odsOpName)) {
      tblgen_depend_type_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == OrderedOp::num_loops_valAttrName(*odsOpName)) {
      tblgen_num_loops_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_depend_type_val && !((tblgen_depend_type_val.isa<::mlir::omp::ClauseDependAttr>())))
    return emitError(loc, "'omp.ordered' op ""attribute 'depend_type_val' failed to satisfy constraint: depend clause");

  if (tblgen_num_loops_val && !((((tblgen_num_loops_val.isa<::mlir::IntegerAttr>())) && ((tblgen_num_loops_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_num_loops_val.cast<::mlir::IntegerAttr>().getInt() >= 0))))
    return emitError(loc, "'omp.ordered' op ""attribute 'num_loops_val' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OrderedOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range OrderedOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range OrderedOp::depend_vec_vars() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange OrderedOp::depend_vec_varsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> OrderedOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OrderedOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::omp::ClauseDependAttr OrderedOp::depend_type_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, depend_type_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseDependAttr>();
}

::llvm::Optional<::mlir::omp::ClauseDepend> OrderedOp::depend_type_val() {
  auto attr = depend_type_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseDepend>(attr.getValue()) : (::llvm::None);
}

::mlir::IntegerAttr OrderedOp::num_loops_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, num_loops_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> OrderedOp::num_loops_val() {
  auto attr = num_loops_valAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void OrderedOp::depend_type_valAttr(::mlir::omp::ClauseDependAttr attr) {
  (*this)->setAttr(depend_type_valAttrName(), attr);
}

void OrderedOp::num_loops_valAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(num_loops_valAttrName(), attr);
}

::mlir::Attribute OrderedOp::removeDepend_type_valAttr() {
  return (*this)->removeAttr(depend_type_valAttrName());
}

::mlir::Attribute OrderedOp::removeNum_loops_valAttr() {
  return (*this)->removeAttr(num_loops_valAttrName());
}

void OrderedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::omp::ClauseDependAttr depend_type_val, /*optional*/::mlir::IntegerAttr num_loops_val, ::mlir::ValueRange depend_vec_vars) {
  odsState.addOperands(depend_vec_vars);
  if (depend_type_val) {
  odsState.addAttribute(depend_type_valAttrName(odsState.name), depend_type_val);
  }
  if (num_loops_val) {
  odsState.addAttribute(num_loops_valAttrName(odsState.name), num_loops_val);
  }
}

void OrderedOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::omp::ClauseDependAttr depend_type_val, /*optional*/::mlir::IntegerAttr num_loops_val, ::mlir::ValueRange depend_vec_vars) {
  odsState.addOperands(depend_vec_vars);
  if (depend_type_val) {
  odsState.addAttribute(depend_type_valAttrName(odsState.name), depend_type_val);
  }
  if (num_loops_val) {
  odsState.addAttribute(num_loops_valAttrName(odsState.name), num_loops_val);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OrderedOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult OrderedOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_depend_type_val;
  ::mlir::Attribute tblgen_num_loops_val;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == depend_type_valAttrName()) {
      tblgen_depend_type_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == num_loops_valAttrName()) {
      tblgen_num_loops_val = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps5(*this, tblgen_depend_type_val, "depend_type_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps6(*this, tblgen_num_loops_val, "num_loops_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult OrderedOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult OrderedOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::omp::ClauseDependAttr depend_type_valAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> depend_vec_varsOperands;
  ::llvm::SMLoc depend_vec_varsOperandsLoc;
  (void)depend_vec_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> depend_vec_varsTypes;
  if (succeeded(parser.parseOptionalKeyword("depend_type"))) {

  if (parser.parseCustomAttributeWithFallback(depend_type_valAttr, ::mlir::Type{}, "depend_type_val",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  if (succeeded(parser.parseOptionalKeyword("depend_vec"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  depend_vec_varsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(depend_vec_varsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(depend_vec_varsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(depend_vec_varsOperands, depend_vec_varsTypes, depend_vec_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OrderedOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("depend_type_val")) {
  _odsPrinter << ' ' << "depend_type";
_odsPrinter.printStrippedAttrOrType(depend_type_valAttr());
  }
  if (!depend_vec_vars().empty()) {
  _odsPrinter << ' ' << "depend_vec";
  _odsPrinter << "(";
  _odsPrinter << depend_vec_vars();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << depend_vec_vars().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"depend_type_val"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::OrderedOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::OrderedRegionOp definitions
//===----------------------------------------------------------------------===//

OrderedRegionOpAdaptor::OrderedRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.ordered_region", odsAttrs.getContext());
}

OrderedRegionOpAdaptor::OrderedRegionOpAdaptor(OrderedRegionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange OrderedRegionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> OrderedRegionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange OrderedRegionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr OrderedRegionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr OrderedRegionOpAdaptor::simdAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, OrderedRegionOp::simdAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool OrderedRegionOpAdaptor::simd() {
  auto attr = simdAttr();
  return attr != nullptr;
}

::mlir::RegionRange OrderedRegionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &OrderedRegionOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult OrderedRegionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_simd;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == OrderedRegionOp::simdAttrName(*odsOpName)) {
      tblgen_simd = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_simd && !((tblgen_simd.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.ordered_region' op ""attribute 'simd' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OrderedRegionOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range OrderedRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> OrderedRegionOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OrderedRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &OrderedRegionOp::region() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr OrderedRegionOp::simdAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, simdAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool OrderedRegionOp::simd() {
  auto attr = simdAttr();
  return attr != nullptr;
}

void OrderedRegionOp::simdAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(simdAttrName(), attr);
}

::mlir::Attribute OrderedRegionOp::removeSimdAttr() {
  return (*this)->removeAttr(simdAttrName());
}

void OrderedRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::UnitAttr simd) {
  if (simd) {
  odsState.addAttribute(simdAttrName(odsState.name), simd);
  }
  (void)odsState.addRegion();
}

void OrderedRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr simd) {
  if (simd) {
  odsState.addAttribute(simdAttrName(odsState.name), simd);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OrderedRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/bool simd) {
  if (simd) {
  odsState.addAttribute(simdAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  (void)odsState.addRegion();
}

void OrderedRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool simd) {
  if (simd) {
  odsState.addAttribute(simdAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OrderedRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult OrderedRegionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_simd;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == simdAttrName()) {
      tblgen_simd = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps7(*this, tblgen_simd, "simd")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult OrderedRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult OrderedRegionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  if (succeeded(parser.parseOptionalKeyword("simd"))) {
    result.addAttribute("simd", parser.getBuilder().getUnitAttr());
  }

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void OrderedRegionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("simd")) {
  _odsPrinter << ' ' << "simd";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(region());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"simd"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::OrderedRegionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ParallelOp definitions
//===----------------------------------------------------------------------===//

ParallelOpAdaptor::ParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.parallel", odsAttrs.getContext());
}

ParallelOpAdaptor::ParallelOpAdaptor(ParallelOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ParallelOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ParallelOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ParallelOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange ParallelOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ParallelOpAdaptor::if_expr_var() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOpAdaptor::num_threads_var() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange ParallelOpAdaptor::allocate_vars() {
  return getODSOperands(2);
}

::mlir::ValueRange ParallelOpAdaptor::allocators_vars() {
  return getODSOperands(3);
}

::mlir::ValueRange ParallelOpAdaptor::reduction_vars() {
  return getODSOperands(4);
}

::mlir::DictionaryAttr ParallelOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ParallelOpAdaptor::reductionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ParallelOp::reductionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > ParallelOpAdaptor::reductions() {
  auto attr = reductionsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::omp::ClauseProcBindKindAttr ParallelOpAdaptor::proc_bind_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ParallelOp::proc_bind_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseProcBindKindAttr>();
    return attr;
}

::llvm::Optional<::mlir::omp::ClauseProcBindKind> ParallelOpAdaptor::proc_bind_val() {
  auto attr = proc_bind_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseProcBindKind>(attr.getValue()) : (::llvm::None);
}

::mlir::RegionRange ParallelOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ParallelOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult ParallelOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.parallel' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == ParallelOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_proc_bind_val;
  ::mlir::Attribute tblgen_reductions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == ParallelOp::proc_bind_valAttrName(*odsOpName)) {
      tblgen_proc_bind_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ParallelOp::reductionsAttrName(*odsOpName)) {
      tblgen_reductions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 5)
      return emitError(loc, "'omp.parallel' op ""'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (tblgen_reductions && !(((tblgen_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'omp.parallel' op ""attribute 'reductions' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_proc_bind_val && !((tblgen_proc_bind_val.isa<::mlir::omp::ClauseProcBindKindAttr>())))
    return emitError(loc, "'omp.parallel' op ""attribute 'proc_bind_val' failed to satisfy constraint: ProcBindKind Clause");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ParallelOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range ParallelOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ParallelOp::if_expr_var() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value ParallelOp::num_threads_var() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range ParallelOp::allocate_vars() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range ParallelOp::allocators_vars() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range ParallelOp::reduction_vars() {
  return getODSOperands(4);
}

::mlir::MutableOperandRange ParallelOp::if_expr_varMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::num_threads_varMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::allocate_varsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::allocators_varsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::reduction_varsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> ParallelOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ParallelOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &ParallelOp::region() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr ParallelOp::reductionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, reductionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > ParallelOp::reductions() {
  auto attr = reductionsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::omp::ClauseProcBindKindAttr ParallelOp::proc_bind_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, proc_bind_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseProcBindKindAttr>();
}

::llvm::Optional<::mlir::omp::ClauseProcBindKind> ParallelOp::proc_bind_val() {
  auto attr = proc_bind_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseProcBindKind>(attr.getValue()) : (::llvm::None);
}

void ParallelOp::reductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(reductionsAttrName(), attr);
}

void ParallelOp::proc_bind_valAttr(::mlir::omp::ClauseProcBindKindAttr attr) {
  (*this)->setAttr(proc_bind_valAttrName(), attr);
}

::mlir::Attribute ParallelOp::removeReductionsAttr() {
  return (*this)->removeAttr(reductionsAttrName());
}

::mlir::Attribute ParallelOp::removeProc_bind_valAttr() {
  return (*this)->removeAttr(proc_bind_valAttrName());
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseProcBindKindAttr proc_bind_val) {
  if (if_expr_var)
    odsState.addOperands(if_expr_var);
  if (num_threads_var)
    odsState.addOperands(num_threads_var);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addOperands(reduction_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr_var ? 1 : 0), (num_threads_var ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size()), static_cast<int32_t>(reduction_vars.size())}));
  if (reductions) {
  odsState.addAttribute(reductionsAttrName(odsState.name), reductions);
  }
  if (proc_bind_val) {
  odsState.addAttribute(proc_bind_valAttrName(odsState.name), proc_bind_val);
  }
  (void)odsState.addRegion();
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseProcBindKindAttr proc_bind_val) {
  if (if_expr_var)
    odsState.addOperands(if_expr_var);
  if (num_threads_var)
    odsState.addOperands(num_threads_var);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addOperands(reduction_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr_var ? 1 : 0), (num_threads_var ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size()), static_cast<int32_t>(reduction_vars.size())}));
  if (reductions) {
  odsState.addAttribute(reductionsAttrName(odsState.name), reductions);
  }
  if (proc_bind_val) {
  odsState.addAttribute(proc_bind_valAttrName(odsState.name), proc_bind_val);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ParallelOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ParallelOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_proc_bind_val;
  ::mlir::Attribute tblgen_reductions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == proc_bind_valAttrName()) {
      tblgen_proc_bind_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == reductionsAttrName()) {
      tblgen_reductions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 5)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 5 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_reductions, "reductions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps9(*this, tblgen_proc_bind_val, "proc_bind_val")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ParallelOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ParallelOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> reduction_varsOperands;
  ::llvm::SMLoc reduction_varsOperandsLoc;
  (void)reduction_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> reduction_varsTypes;
  ::mlir::ArrayAttr reductionsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> if_expr_varOperands;
  ::llvm::SMLoc if_expr_varOperandsLoc;
  (void)if_expr_varOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> if_expr_varTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> num_threads_varOperands;
  ::llvm::SMLoc num_threads_varOperandsLoc;
  (void)num_threads_varOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> num_threads_varTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocate_varsOperands;
  ::llvm::SMLoc allocate_varsOperandsLoc;
  (void)allocate_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocate_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocators_varsOperands;
  ::llvm::SMLoc allocators_varsOperandsLoc;
  (void)allocators_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocators_varsTypes;
  ::mlir::omp::ClauseProcBindKindAttr proc_bind_valAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool reductionClause = false;
  bool ifClause = false;
  bool num_threadsClause = false;
  bool allocateClause = false;
  bool proc_bindClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("reduction"))) {

  if (reductionClause) {
    return parser.emitError(parser.getNameLoc())
          << "`reduction` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  reductionClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    reduction_varsOperandsLoc = parser.getCurrentLocation();
    if (parseReductionVarList(parser, reduction_varsOperands, reduction_varsTypes, reductionsAttr))
      return ::mlir::failure();
    if (reductionsAttr)
      result.addAttribute("reductions", reductionsAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    if_expr_varOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_expr_varOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_expr_varTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("num_threads"))) {

  if (num_threadsClause) {
    return parser.emitError(parser.getNameLoc())
          << "`num_threads` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  num_threadsClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    num_threads_varOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      num_threads_varOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      num_threads_varTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("allocate"))) {

  if (allocateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`allocate` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  allocateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    allocate_varsOperandsLoc = parser.getCurrentLocation();
    allocators_varsOperandsLoc = parser.getCurrentLocation();
    if (parseAllocateAndAllocator(parser, allocate_varsOperands, allocate_varsTypes, allocators_varsOperands, allocators_varsTypes))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("proc_bind"))) {

  if (proc_bindClause) {
    return parser.emitError(parser.getNameLoc())
          << "`proc_bind` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  proc_bindClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, proc_bind_valAttr))
      return ::mlir::failure();
    if (proc_bind_valAttr)
      result.addAttribute("proc_bind_val", proc_bind_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(if_expr_varOperands.size()), static_cast<int32_t>(num_threads_varOperands.size()), static_cast<int32_t>(allocate_varsOperands.size()), static_cast<int32_t>(allocators_varsOperands.size()), static_cast<int32_t>(reduction_varsOperands.size())}));
  if (parser.resolveOperands(if_expr_varOperands, if_expr_varTypes, if_expr_varOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(num_threads_varOperands, num_threads_varTypes, num_threads_varOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocate_varsOperands, allocate_varsTypes, allocate_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocators_varsOperands, allocators_varsTypes, allocators_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(reduction_varsOperands, reduction_varsTypes, reduction_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ParallelOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || reduction_vars().size() || reductionsAttr()) {
  _odsPrinter << ' ' << "reduction";
  _odsPrinter << "(";
  printReductionVarList(_odsPrinter, *this, reduction_vars(), reduction_vars().getTypes(), reductionsAttr());
  _odsPrinter << ")";
  }
  if (false || if_expr_var()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = if_expr_var())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (if_expr_var() ? ::llvm::ArrayRef<::mlir::Type>(if_expr_var().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || num_threads_var()) {
  _odsPrinter << ' ' << "num_threads";
  _odsPrinter << "(";
  if (::mlir::Value value = num_threads_var())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (num_threads_var() ? ::llvm::ArrayRef<::mlir::Type>(num_threads_var().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || allocate_vars().size() || allocators_vars().size()) {
  _odsPrinter << ' ' << "allocate";
  _odsPrinter << "(";
  printAllocateAndAllocator(_odsPrinter, *this, allocate_vars(), allocate_vars().getTypes(), allocators_vars(), allocators_vars().getTypes());
  _odsPrinter << ")";
  }
  if (false || proc_bind_valAttr()) {
  _odsPrinter << ' ' << "proc_bind";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, proc_bind_valAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(region());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "reductions", "proc_bind_val"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ParallelOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionDeclareOp definitions
//===----------------------------------------------------------------------===//

ReductionDeclareOpAdaptor::ReductionDeclareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.reduction.declare", odsAttrs.getContext());
}

ReductionDeclareOpAdaptor::ReductionDeclareOpAdaptor(ReductionDeclareOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ReductionDeclareOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReductionDeclareOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReductionDeclareOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ReductionDeclareOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ReductionDeclareOpAdaptor::sym_nameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, ReductionDeclareOp::sym_nameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef ReductionDeclareOpAdaptor::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

::mlir::TypeAttr ReductionDeclareOpAdaptor::typeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, ReductionDeclareOp::typeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
    return attr;
}

::mlir::Type ReductionDeclareOpAdaptor::type() {
  auto attr = typeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::RegionRange ReductionDeclareOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &ReductionDeclareOpAdaptor::initializerRegion() {
  return *odsRegions[0];
}

::mlir::Region &ReductionDeclareOpAdaptor::reductionRegion() {
  return *odsRegions[1];
}

::mlir::Region &ReductionDeclareOpAdaptor::atomicReductionRegion() {
  return *odsRegions[2];
}

::mlir::LogicalResult ReductionDeclareOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.reduction.declare' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == ReductionDeclareOp::sym_nameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.reduction.declare' op ""requires attribute 'type'");
    if (namedAttrIt->getName() == ReductionDeclareOp::typeAttrName(*odsOpName)) {
      tblgen_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'omp.reduction.declare' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_type && !(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return emitError(loc, "'omp.reduction.declare' op ""attribute 'type' failed to satisfy constraint: any type attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReductionDeclareOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReductionDeclareOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ReductionDeclareOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReductionDeclareOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &ReductionDeclareOp::initializerRegion() {
  return (*this)->getRegion(0);
}

::mlir::Region &ReductionDeclareOp::reductionRegion() {
  return (*this)->getRegion(1);
}

::mlir::Region &ReductionDeclareOp::atomicReductionRegion() {
  return (*this)->getRegion(2);
}

::mlir::StringAttr ReductionDeclareOp::sym_nameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, sym_nameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ReductionDeclareOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

::mlir::TypeAttr ReductionDeclareOp::typeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, typeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::Type ReductionDeclareOp::type() {
  auto attr = typeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

void ReductionDeclareOp::sym_nameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(sym_nameAttrName(), attr);
}

void ReductionDeclareOp::typeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(typeAttrName(), attr);
}

void ReductionDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), sym_name);
  odsState.addAttribute(typeAttrName(odsState.name), type);
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
}

void ReductionDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::TypeAttr type) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), sym_name);
  odsState.addAttribute(typeAttrName(odsState.name), type);
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReductionDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::mlir::Type type) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(typeAttrName(odsState.name), ::mlir::TypeAttr::get(type));
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
}

void ReductionDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::mlir::Type type) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(typeAttrName(odsState.name), ::mlir::TypeAttr::get(type));
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReductionDeclareOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 3; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReductionDeclareOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == sym_nameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'type'");
    if (namedAttrIt->getName() == typeAttrName()) {
      tblgen_type = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps3(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps10(*this, tblgen_type, "type")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "initializerRegion", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(1)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "reductionRegion", index++)))
        return ::mlir::failure();

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(2)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "atomicReductionRegion", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReductionDeclareOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ReductionDeclareOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;
  ::mlir::TypeAttr typeAttr;
  std::unique_ptr<::mlir::Region> initializerRegionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> reductionRegionRegion = std::make_unique<::mlir::Region>();
  std::unique_ptr<::mlir::Region> atomicReductionRegionRegion = std::make_unique<::mlir::Region>();

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(typeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "type",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();
  if (parser.parseKeyword("init"))
    return ::mlir::failure();

  if (parser.parseRegion(*initializerRegionRegion))
    return ::mlir::failure();
  if (parser.parseKeyword("combiner"))
    return ::mlir::failure();

  if (parser.parseRegion(*reductionRegionRegion))
    return ::mlir::failure();
  {
    if (parseAtomicReductionRegion(parser, *atomicReductionRegionRegion))
      return ::mlir::failure();
  }
  result.addRegion(std::move(initializerRegionRegion));
  result.addRegion(std::move(reductionRegionRegion));
  result.addRegion(std::move(atomicReductionRegionRegion));
  return ::mlir::success();
}

void ReductionDeclareOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(sym_nameAttr().getValue());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(typeAttr());
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"sym_name", "type"});
  _odsPrinter << ' ' << "init";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(initializerRegion());
  _odsPrinter << ' ' << "combiner";
  _odsPrinter << ' ';
  _odsPrinter.printRegion(reductionRegion());
  _odsPrinter << ' ';
  printAtomicReductionRegion(_odsPrinter, *this, atomicReductionRegion());
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionOp definitions
//===----------------------------------------------------------------------===//

ReductionOpAdaptor::ReductionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.reduction", odsAttrs.getContext());
}

ReductionOpAdaptor::ReductionOpAdaptor(ReductionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ReductionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ReductionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ReductionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReductionOpAdaptor::operand() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReductionOpAdaptor::accumulator() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ReductionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ReductionOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReductionOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReductionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ReductionOp::operand() {
  return *getODSOperands(0).begin();
}

::mlir::Value ReductionOp::accumulator() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ReductionOp::operandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReductionOp::accumulatorMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReductionOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReductionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::Value accumulator) {
  odsState.addOperands(operand);
  odsState.addOperands(accumulator);
}

void ReductionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::Value accumulator) {
  odsState.addOperands(operand);
  odsState.addOperands(accumulator);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReductionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReductionOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()((*this->getODSOperands(1).begin()).getType().cast<::mlir::omp::PointerLikeType>().getElementType(), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that value types matches accumulator element type");
  return ::mlir::success();
}

::mlir::LogicalResult ReductionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ReductionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand accumulatorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> accumulatorOperands(accumulatorRawOperands);  ::llvm::SMLoc accumulatorOperandsLoc;
  (void)accumulatorOperandsLoc;
  ::mlir::Type accumulatorRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> accumulatorTypes(accumulatorRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  accumulatorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(accumulatorRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::omp::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    accumulatorRawTypes[0] = type;
  }
  for (::mlir::Type type : accumulatorTypes) {
    (void)type;
    if (!((type.isa<::mlir::omp::PointerLikeType>()))) {
      return parser.emitError(parser.getNameLoc()) << "'accumulator' must be OpenMP-compatible variable type, but got " << type;
    }
  }
  if (parser.resolveOperands(operandOperands, accumulatorTypes[0].cast<::mlir::omp::PointerLikeType>().getElementType(), operandOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(accumulatorOperands, accumulatorTypes, accumulatorOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReductionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << operand();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << accumulator();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = accumulator().getType();
    if (auto validType = type.dyn_cast<::mlir::omp::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SectionOp definitions
//===----------------------------------------------------------------------===//

SectionOpAdaptor::SectionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.section", odsAttrs.getContext());
}

SectionOpAdaptor::SectionOpAdaptor(SectionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SectionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SectionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SectionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr SectionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange SectionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &SectionOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult SectionOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SectionOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SectionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> SectionOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SectionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &SectionOp::region() {
  return (*this)->getRegion(0);
}

void SectionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
  (void)odsState.addRegion();
}

void SectionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SectionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SectionOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult SectionOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SectionOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  return ::mlir::success();
}

void SectionOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printRegion(region());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::SectionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SectionsOp definitions
//===----------------------------------------------------------------------===//

SectionsOpAdaptor::SectionsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.sections", odsAttrs.getContext());
}

SectionsOpAdaptor::SectionsOpAdaptor(SectionsOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SectionsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SectionsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SectionsOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange SectionsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange SectionsOpAdaptor::reduction_vars() {
  return getODSOperands(0);
}

::mlir::ValueRange SectionsOpAdaptor::allocate_vars() {
  return getODSOperands(1);
}

::mlir::ValueRange SectionsOpAdaptor::allocators_vars() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr SectionsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SectionsOpAdaptor::reductionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SectionsOp::reductionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > SectionsOpAdaptor::reductions() {
  auto attr = reductionsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::UnitAttr SectionsOpAdaptor::nowaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SectionsOp::nowaitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool SectionsOpAdaptor::nowait() {
  auto attr = nowaitAttr();
  return attr != nullptr;
}

::mlir::RegionRange SectionsOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &SectionsOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult SectionsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.sections' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == SectionsOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == SectionsOp::nowaitAttrName(*odsOpName)) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == SectionsOp::reductionsAttrName(*odsOpName)) {
      tblgen_reductions = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'omp.sections' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_reductions && !(((tblgen_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'omp.sections' op ""attribute 'reductions' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_nowait && !((tblgen_nowait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.sections' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SectionsOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range SectionsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range SectionsOp::reduction_vars() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range SectionsOp::allocate_vars() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range SectionsOp::allocators_vars() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange SectionsOp::reduction_varsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SectionsOp::allocate_varsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SectionsOp::allocators_varsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> SectionsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SectionsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &SectionsOp::region() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr SectionsOp::reductionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, reductionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > SectionsOp::reductions() {
  auto attr = reductionsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::UnitAttr SectionsOp::nowaitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, nowaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool SectionsOp::nowait() {
  auto attr = nowaitAttr();
  return attr != nullptr;
}

void SectionsOp::reductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(reductionsAttrName(), attr);
}

void SectionsOp::nowaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(nowaitAttrName(), attr);
}

::mlir::Attribute SectionsOp::removeReductionsAttr() {
  return (*this)->removeAttr(reductionsAttrName());
}

::mlir::Attribute SectionsOp::removeNowaitAttr() {
  return (*this)->removeAttr(nowaitAttrName());
}

void SectionsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait) {
  odsState.addOperands(reduction_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(reduction_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (reductions) {
  odsState.addAttribute(reductionsAttrName(odsState.name), reductions);
  }
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), nowait);
  }
  (void)odsState.addRegion();
}

void SectionsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait) {
  odsState.addOperands(reduction_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(reduction_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (reductions) {
  odsState.addAttribute(reductionsAttrName(odsState.name), reductions);
  }
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), nowait);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SectionsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait) {
  odsState.addOperands(reduction_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(reduction_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (reductions) {
  odsState.addAttribute(reductionsAttrName(odsState.name), reductions);
  }
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  (void)odsState.addRegion();
}

void SectionsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait) {
  odsState.addOperands(reduction_vars);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(reduction_vars.size()), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (reductions) {
  odsState.addAttribute(reductionsAttrName(odsState.name), reductions);
  }
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SectionsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SectionsOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == nowaitAttrName()) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_reductions;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == reductionsAttrName()) {
      tblgen_reductions = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_reductions, "reductions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps7(*this, tblgen_nowait, "nowait")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult SectionsOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SectionsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> reduction_varsOperands;
  ::llvm::SMLoc reduction_varsOperandsLoc;
  (void)reduction_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> reduction_varsTypes;
  ::mlir::ArrayAttr reductionsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocate_varsOperands;
  ::llvm::SMLoc allocate_varsOperandsLoc;
  (void)allocate_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocate_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocators_varsOperands;
  ::llvm::SMLoc allocators_varsOperandsLoc;
  (void)allocators_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocators_varsTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool reductionClause = false;
  bool allocateClause = false;
  bool nowaitClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("reduction"))) {

  if (reductionClause) {
    return parser.emitError(parser.getNameLoc())
          << "`reduction` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  reductionClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    reduction_varsOperandsLoc = parser.getCurrentLocation();
    if (parseReductionVarList(parser, reduction_varsOperands, reduction_varsTypes, reductionsAttr))
      return ::mlir::failure();
    if (reductionsAttr)
      result.addAttribute("reductions", reductionsAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("allocate"))) {

  if (allocateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`allocate` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  allocateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    allocate_varsOperandsLoc = parser.getCurrentLocation();
    allocators_varsOperandsLoc = parser.getCurrentLocation();
    if (parseAllocateAndAllocator(parser, allocate_varsOperands, allocate_varsTypes, allocators_varsOperands, allocators_varsTypes))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("nowait"))) {

  if (nowaitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`nowait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  nowaitClause = true;
  result.addAttribute("nowait", UnitAttr::get(parser.getContext()));
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(reduction_varsOperands.size()), static_cast<int32_t>(allocate_varsOperands.size()), static_cast<int32_t>(allocators_varsOperands.size())}));
  if (parser.resolveOperands(reduction_varsOperands, reduction_varsTypes, reduction_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocate_varsOperands, allocate_varsTypes, allocate_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocators_varsOperands, allocators_varsTypes, allocators_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SectionsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || reduction_vars().size() || reductionsAttr()) {
  _odsPrinter << ' ' << "reduction";
  _odsPrinter << "(";
  printReductionVarList(_odsPrinter, *this, reduction_vars(), reduction_vars().getTypes(), reductionsAttr());
  _odsPrinter << ")";
  }
  if (false || allocate_vars().size() || allocators_vars().size()) {
  _odsPrinter << ' ' << "allocate";
  _odsPrinter << "(";
  printAllocateAndAllocator(_odsPrinter, *this, allocate_vars(), allocate_vars().getTypes(), allocators_vars(), allocators_vars().getTypes());
  _odsPrinter << ")";
  }
  if (false || nowaitAttr()) {
  _odsPrinter << ' ' << "nowait";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(region());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "reductions", "nowait"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::SectionsOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SimdLoopOp definitions
//===----------------------------------------------------------------------===//

SimdLoopOpAdaptor::SimdLoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.simdloop", odsAttrs.getContext());
}

SimdLoopOpAdaptor::SimdLoopOpAdaptor(SimdLoopOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SimdLoopOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SimdLoopOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SimdLoopOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange SimdLoopOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange SimdLoopOpAdaptor::lowerBound() {
  return getODSOperands(0);
}

::mlir::ValueRange SimdLoopOpAdaptor::upperBound() {
  return getODSOperands(1);
}

::mlir::ValueRange SimdLoopOpAdaptor::step() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr SimdLoopOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::RegionRange SimdLoopOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &SimdLoopOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult SimdLoopOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.simdloop' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == SimdLoopOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'omp.simdloop' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    return ::mlir::success();
}

std::pair<unsigned, unsigned> SimdLoopOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range SimdLoopOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range SimdLoopOp::lowerBound() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range SimdLoopOp::upperBound() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range SimdLoopOp::step() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange SimdLoopOp::lowerBoundMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SimdLoopOp::upperBoundMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SimdLoopOp::stepMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> SimdLoopOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SimdLoopOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &SimdLoopOp::region() {
  return (*this)->getRegion(0);
}

void SimdLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size())}));
  (void)odsState.addRegion();
}

void SimdLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size())}));
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SimdLoopOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SimdLoopOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
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
    {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({this->getODSOperands(0).getType(), this->getODSOperands(1).getType(), this->getODSOperands(2).getType()})))))
    return emitOpError("failed to verify that all of {lowerBound, upperBound, step} have same type");
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult SimdLoopOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::SimdLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SingleOp definitions
//===----------------------------------------------------------------------===//

SingleOpAdaptor::SingleOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.single", odsAttrs.getContext());
}

SingleOpAdaptor::SingleOpAdaptor(SingleOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SingleOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SingleOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, SingleOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange SingleOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange SingleOpAdaptor::allocate_vars() {
  return getODSOperands(0);
}

::mlir::ValueRange SingleOpAdaptor::allocators_vars() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr SingleOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr SingleOpAdaptor::nowaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SingleOp::nowaitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool SingleOpAdaptor::nowait() {
  auto attr = nowaitAttr();
  return attr != nullptr;
}

::mlir::RegionRange SingleOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &SingleOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult SingleOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.single' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == SingleOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == SingleOp::nowaitAttrName(*odsOpName)) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitError(loc, "'omp.single' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_nowait && !((tblgen_nowait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.single' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SingleOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range SingleOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range SingleOp::allocate_vars() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range SingleOp::allocators_vars() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange SingleOp::allocate_varsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SingleOp::allocators_varsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> SingleOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SingleOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &SingleOp::region() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr SingleOp::nowaitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, nowaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool SingleOp::nowait() {
  auto attr = nowaitAttr();
  return attr != nullptr;
}

void SingleOp::nowaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(nowaitAttrName(), attr);
}

::mlir::Attribute SingleOp::removeNowaitAttr() {
  return (*this)->removeAttr(nowaitAttrName());
}

void SingleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait) {
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), nowait);
  }
  (void)odsState.addRegion();
}

void SingleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::UnitAttr nowait) {
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), nowait);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SingleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait) {
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  (void)odsState.addRegion();
}

void SingleOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/bool nowait) {
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SingleOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SingleOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == nowaitAttrName()) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps7(*this, tblgen_nowait, "nowait")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult SingleOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SingleOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocate_varsOperands;
  ::llvm::SMLoc allocate_varsOperandsLoc;
  (void)allocate_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocate_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocators_varsOperands;
  ::llvm::SMLoc allocators_varsOperandsLoc;
  (void)allocators_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocators_varsTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool allocateClause = false;
  bool nowaitClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("allocate"))) {

  if (allocateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`allocate` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  allocateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    allocate_varsOperandsLoc = parser.getCurrentLocation();
    allocators_varsOperandsLoc = parser.getCurrentLocation();
    if (parseAllocateAndAllocator(parser, allocate_varsOperands, allocate_varsTypes, allocators_varsOperands, allocators_varsTypes))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("nowait"))) {

  if (nowaitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`nowait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  nowaitClause = true;
  result.addAttribute("nowait", UnitAttr::get(parser.getContext()));
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(allocate_varsOperands.size()), static_cast<int32_t>(allocators_varsOperands.size())}));
  if (parser.resolveOperands(allocate_varsOperands, allocate_varsTypes, allocate_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocators_varsOperands, allocators_varsTypes, allocators_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SingleOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || allocate_vars().size() || allocators_vars().size()) {
  _odsPrinter << ' ' << "allocate";
  _odsPrinter << "(";
  printAllocateAndAllocator(_odsPrinter, *this, allocate_vars(), allocate_vars().getTypes(), allocators_vars(), allocators_vars().getTypes());
  _odsPrinter << ")";
  }
  if (false || nowaitAttr()) {
  _odsPrinter << ' ' << "nowait";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(region());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "nowait"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::SingleOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TargetOp definitions
//===----------------------------------------------------------------------===//

TargetOpAdaptor::TargetOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.target", odsAttrs.getContext());
}

TargetOpAdaptor::TargetOpAdaptor(TargetOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange TargetOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TargetOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TargetOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange TargetOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TargetOpAdaptor::if_expr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TargetOpAdaptor::device() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TargetOpAdaptor::thread_limit() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr TargetOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr TargetOpAdaptor::nowaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TargetOp::nowaitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool TargetOpAdaptor::nowait() {
  auto attr = nowaitAttr();
  return attr != nullptr;
}

::mlir::RegionRange TargetOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &TargetOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult TargetOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.target' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == TargetOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == TargetOp::nowaitAttrName(*odsOpName)) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'omp.target' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_nowait && !((tblgen_nowait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.target' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TargetOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range TargetOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TargetOp::if_expr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TargetOp::device() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TargetOp::thread_limit() {
  auto operands = getODSOperands(2);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange TargetOp::if_exprMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TargetOp::deviceMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TargetOp::thread_limitMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> TargetOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TargetOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &TargetOp::region() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr TargetOp::nowaitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, nowaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool TargetOp::nowait() {
  auto attr = nowaitAttr();
  return attr != nullptr;
}

void TargetOp::nowaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(nowaitAttrName(), attr);
}

::mlir::Attribute TargetOp::removeNowaitAttr() {
  return (*this)->removeAttr(nowaitAttrName());
}

void TargetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  if (thread_limit)
    odsState.addOperands(thread_limit);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr ? 1 : 0), (device ? 1 : 0), (thread_limit ? 1 : 0)}));
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), nowait);
  }
  (void)odsState.addRegion();
}

void TargetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/::mlir::UnitAttr nowait) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  if (thread_limit)
    odsState.addOperands(thread_limit);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr ? 1 : 0), (device ? 1 : 0), (thread_limit ? 1 : 0)}));
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), nowait);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TargetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  if (thread_limit)
    odsState.addOperands(thread_limit);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr ? 1 : 0), (device ? 1 : 0), (thread_limit ? 1 : 0)}));
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  (void)odsState.addRegion();
}

void TargetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value device, /*optional*/::mlir::Value thread_limit, /*optional*/bool nowait) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (device)
    odsState.addOperands(device);
  if (thread_limit)
    odsState.addOperands(thread_limit);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr ? 1 : 0), (device ? 1 : 0), (thread_limit ? 1 : 0)}));
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TargetOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TargetOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == nowaitAttrName()) {
      tblgen_nowait = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps7(*this, tblgen_nowait, "nowait")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult TargetOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TargetOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> if_exprOperands;
  ::llvm::SMLoc if_exprOperandsLoc;
  (void)if_exprOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> deviceOperands;
  ::llvm::SMLoc deviceOperandsLoc;
  (void)deviceOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> deviceTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> thread_limitOperands;
  ::llvm::SMLoc thread_limitOperandsLoc;
  (void)thread_limitOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> thread_limitTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool ifClause = false;
  bool deviceClause = false;
  bool thread_limitClause = false;
  bool nowaitClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    if_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_exprOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("device"))) {

  if (deviceClause) {
    return parser.emitError(parser.getNameLoc())
          << "`device` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  deviceClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    deviceOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      deviceTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("thread_limit"))) {

  if (thread_limitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`thread_limit` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  thread_limitClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    thread_limitOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      thread_limitOperands.push_back(operand);
    }
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type optionalType;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalType(optionalType);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      thread_limitTypes.push_back(optionalType);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("nowait"))) {

  if (nowaitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`nowait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  nowaitClause = true;
  result.addAttribute("nowait", UnitAttr::get(parser.getContext()));
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(if_exprOperands.size()), static_cast<int32_t>(deviceOperands.size()), static_cast<int32_t>(thread_limitOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(if_exprOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(deviceOperands, deviceTypes, deviceOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(thread_limitOperands, thread_limitTypes, thread_limitOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TargetOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || if_expr()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = if_expr())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || device()) {
  _odsPrinter << ' ' << "device";
  _odsPrinter << "(";
  if (::mlir::Value value = device())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (device() ? ::llvm::ArrayRef<::mlir::Type>(device().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || thread_limit()) {
  _odsPrinter << ' ' << "thread_limit";
  _odsPrinter << "(";
  if (::mlir::Value value = thread_limit())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (thread_limit() ? ::llvm::ArrayRef<::mlir::Type>(thread_limit().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if (false || nowaitAttr()) {
  _odsPrinter << ' ' << "nowait";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(region());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "nowait"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TargetOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskOp definitions
//===----------------------------------------------------------------------===//

TaskOpAdaptor::TaskOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.task", odsAttrs.getContext());
}

TaskOpAdaptor::TaskOpAdaptor(TaskOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange TaskOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TaskOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, TaskOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange TaskOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TaskOpAdaptor::if_expr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TaskOpAdaptor::final_expr() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange TaskOpAdaptor::in_reduction_vars() {
  return getODSOperands(2);
}

::mlir::Value TaskOpAdaptor::priority() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::ValueRange TaskOpAdaptor::allocate_vars() {
  return getODSOperands(4);
}

::mlir::ValueRange TaskOpAdaptor::allocators_vars() {
  return getODSOperands(5);
}

::mlir::DictionaryAttr TaskOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr TaskOpAdaptor::untiedAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, TaskOp::untiedAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool TaskOpAdaptor::untied() {
  auto attr = untiedAttr();
  return attr != nullptr;
}

::mlir::UnitAttr TaskOpAdaptor::mergeableAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TaskOp::mergeableAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool TaskOpAdaptor::mergeable() {
  auto attr = mergeableAttr();
  return attr != nullptr;
}

::mlir::ArrayAttr TaskOpAdaptor::in_reductionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, TaskOp::in_reductionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > TaskOpAdaptor::in_reductions() {
  auto attr = in_reductionsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::RegionRange TaskOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &TaskOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult TaskOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_in_reductions;
  ::mlir::Attribute tblgen_mergeable;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.task' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == TaskOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == TaskOp::in_reductionsAttrName(*odsOpName)) {
      tblgen_in_reductions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == TaskOp::mergeableAttrName(*odsOpName)) {
      tblgen_mergeable = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_untied;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == TaskOp::untiedAttrName(*odsOpName)) {
      tblgen_untied = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 6)
      return emitError(loc, "'omp.task' op ""'operand_segment_sizes' attribute for specifying operand segments must have 6 "
                "elements, but got ") << numElements;
  }

  if (tblgen_untied && !((tblgen_untied.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.task' op ""attribute 'untied' failed to satisfy constraint: unit attribute");

  if (tblgen_mergeable && !((tblgen_mergeable.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.task' op ""attribute 'mergeable' failed to satisfy constraint: unit attribute");

  if (tblgen_in_reductions && !(((tblgen_in_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_in_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'omp.task' op ""attribute 'in_reductions' failed to satisfy constraint: symbol ref array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TaskOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range TaskOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TaskOp::if_expr() {
  auto operands = getODSOperands(0);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Value TaskOp::final_expr() {
  auto operands = getODSOperands(1);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range TaskOp::in_reduction_vars() {
  return getODSOperands(2);
}

::mlir::Value TaskOp::priority() {
  auto operands = getODSOperands(3);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::Operation::operand_range TaskOp::allocate_vars() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range TaskOp::allocators_vars() {
  return getODSOperands(5);
}

::mlir::MutableOperandRange TaskOp::if_exprMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskOp::final_exprMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskOp::in_reduction_varsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskOp::priorityMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskOp::allocate_varsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TaskOp::allocators_varsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> TaskOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TaskOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &TaskOp::region() {
  return (*this)->getRegion(0);
}

::mlir::UnitAttr TaskOp::untiedAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, untiedAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool TaskOp::untied() {
  auto attr = untiedAttr();
  return attr != nullptr;
}

::mlir::UnitAttr TaskOp::mergeableAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, mergeableAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool TaskOp::mergeable() {
  auto attr = mergeableAttr();
  return attr != nullptr;
}

::mlir::ArrayAttr TaskOp::in_reductionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, in_reductionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > TaskOp::in_reductions() {
  auto attr = in_reductionsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void TaskOp::untiedAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(untiedAttrName(), attr);
}

void TaskOp::mergeableAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(mergeableAttrName(), attr);
}

void TaskOp::in_reductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(in_reductionsAttrName(), attr);
}

::mlir::Attribute TaskOp::removeUntiedAttr() {
  return (*this)->removeAttr(untiedAttrName());
}

::mlir::Attribute TaskOp::removeMergeableAttr() {
  return (*this)->removeAttr(mergeableAttrName());
}

::mlir::Attribute TaskOp::removeIn_reductionsAttr() {
  return (*this)->removeAttr(in_reductionsAttrName());
}

void TaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (final_expr)
    odsState.addOperands(final_expr);
  odsState.addOperands(in_reduction_vars);
  if (priority)
    odsState.addOperands(priority);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr ? 1 : 0), (final_expr ? 1 : 0), static_cast<int32_t>(in_reduction_vars.size()), (priority ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (untied) {
  odsState.addAttribute(untiedAttrName(odsState.name), untied);
  }
  if (mergeable) {
  odsState.addAttribute(mergeableAttrName(odsState.name), mergeable);
  }
  if (in_reductions) {
  odsState.addAttribute(in_reductionsAttrName(odsState.name), in_reductions);
  }
  (void)odsState.addRegion();
}

void TaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/::mlir::UnitAttr untied, /*optional*/::mlir::UnitAttr mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (final_expr)
    odsState.addOperands(final_expr);
  odsState.addOperands(in_reduction_vars);
  if (priority)
    odsState.addOperands(priority);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr ? 1 : 0), (final_expr ? 1 : 0), static_cast<int32_t>(in_reduction_vars.size()), (priority ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (untied) {
  odsState.addAttribute(untiedAttrName(odsState.name), untied);
  }
  if (mergeable) {
  odsState.addAttribute(mergeableAttrName(odsState.name), mergeable);
  }
  if (in_reductions) {
  odsState.addAttribute(in_reductionsAttrName(odsState.name), in_reductions);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (final_expr)
    odsState.addOperands(final_expr);
  odsState.addOperands(in_reduction_vars);
  if (priority)
    odsState.addOperands(priority);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr ? 1 : 0), (final_expr ? 1 : 0), static_cast<int32_t>(in_reduction_vars.size()), (priority ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (untied) {
  odsState.addAttribute(untiedAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (mergeable) {
  odsState.addAttribute(mergeableAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (in_reductions) {
  odsState.addAttribute(in_reductionsAttrName(odsState.name), in_reductions);
  }
  (void)odsState.addRegion();
}

void TaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr, /*optional*/::mlir::Value final_expr, /*optional*/bool untied, /*optional*/bool mergeable, ::mlir::ValueRange in_reduction_vars, /*optional*/::mlir::ArrayAttr in_reductions, /*optional*/::mlir::Value priority, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars) {
  if (if_expr)
    odsState.addOperands(if_expr);
  if (final_expr)
    odsState.addOperands(final_expr);
  odsState.addOperands(in_reduction_vars);
  if (priority)
    odsState.addOperands(priority);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr ? 1 : 0), (final_expr ? 1 : 0), static_cast<int32_t>(in_reduction_vars.size()), (priority ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (untied) {
  odsState.addAttribute(untiedAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (mergeable) {
  odsState.addAttribute(mergeableAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (in_reductions) {
  odsState.addAttribute(in_reductionsAttrName(odsState.name), in_reductions);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TaskOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TaskOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_in_reductions;
  ::mlir::Attribute tblgen_mergeable;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == in_reductionsAttrName()) {
      tblgen_in_reductions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == mergeableAttrName()) {
      tblgen_mergeable = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_untied;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == untiedAttrName()) {
      tblgen_untied = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 6)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 6 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps7(*this, tblgen_untied, "untied")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps7(*this, tblgen_mergeable, "mergeable")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_in_reductions, "in_reductions")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    if (valueGroup3.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup3.size();
    }

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult TaskOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TaskOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> if_exprOperands;
  ::llvm::SMLoc if_exprOperandsLoc;
  (void)if_exprOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> final_exprOperands;
  ::llvm::SMLoc final_exprOperandsLoc;
  (void)final_exprOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> in_reduction_varsOperands;
  ::llvm::SMLoc in_reduction_varsOperandsLoc;
  (void)in_reduction_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> in_reduction_varsTypes;
  ::mlir::ArrayAttr in_reductionsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> priorityOperands;
  ::llvm::SMLoc priorityOperandsLoc;
  (void)priorityOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocate_varsOperands;
  ::llvm::SMLoc allocate_varsOperandsLoc;
  (void)allocate_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocate_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allocators_varsOperands;
  ::llvm::SMLoc allocators_varsOperandsLoc;
  (void)allocators_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> allocators_varsTypes;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool ifClause = false;
  bool finalClause = false;
  bool untiedClause = false;
  bool mergeableClause = false;
  bool in_reductionClause = false;
  bool priorityClause = false;
  bool allocateClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    if_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      if_exprOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("final"))) {

  if (finalClause) {
    return parser.emitError(parser.getNameLoc())
          << "`final` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  finalClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    final_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      final_exprOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("untied"))) {

  if (untiedClause) {
    return parser.emitError(parser.getNameLoc())
          << "`untied` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  untiedClause = true;
  result.addAttribute("untied", UnitAttr::get(parser.getContext()));
    } else if (succeeded(parser.parseOptionalKeyword("mergeable"))) {

  if (mergeableClause) {
    return parser.emitError(parser.getNameLoc())
          << "`mergeable` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  mergeableClause = true;
  result.addAttribute("mergeable", UnitAttr::get(parser.getContext()));
    } else if (succeeded(parser.parseOptionalKeyword("in_reduction"))) {

  if (in_reductionClause) {
    return parser.emitError(parser.getNameLoc())
          << "`in_reduction` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  in_reductionClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    in_reduction_varsOperandsLoc = parser.getCurrentLocation();
    if (parseReductionVarList(parser, in_reduction_varsOperands, in_reduction_varsTypes, in_reductionsAttr))
      return ::mlir::failure();
    if (in_reductionsAttr)
      result.addAttribute("in_reductions", in_reductionsAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("priority"))) {

  if (priorityClause) {
    return parser.emitError(parser.getNameLoc())
          << "`priority` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  priorityClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    priorityOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::UnresolvedOperand operand;
    ::mlir::OptionalParseResult parseResult =
                                    parser.parseOptionalOperand(operand);
    if (parseResult.hasValue()) {
      if (failed(*parseResult))
        return ::mlir::failure();
      priorityOperands.push_back(operand);
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("allocate"))) {

  if (allocateClause) {
    return parser.emitError(parser.getNameLoc())
          << "`allocate` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  allocateClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    allocate_varsOperandsLoc = parser.getCurrentLocation();
    allocators_varsOperandsLoc = parser.getCurrentLocation();
    if (parseAllocateAndAllocator(parser, allocate_varsOperands, allocate_varsTypes, allocators_varsOperands, allocators_varsTypes))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}

  if (parser.parseRegion(*regionRegion))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(if_exprOperands.size()), static_cast<int32_t>(final_exprOperands.size()), static_cast<int32_t>(in_reduction_varsOperands.size()), static_cast<int32_t>(priorityOperands.size()), static_cast<int32_t>(allocate_varsOperands.size()), static_cast<int32_t>(allocators_varsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(if_exprOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(final_exprOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(in_reduction_varsOperands, in_reduction_varsTypes, in_reduction_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(priorityOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocate_varsOperands, allocate_varsTypes, allocate_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocators_varsOperands, allocators_varsTypes, allocators_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TaskOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || if_expr()) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = if_expr())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || final_expr()) {
  _odsPrinter << ' ' << "final";
  _odsPrinter << "(";
  if (::mlir::Value value = final_expr())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || untiedAttr()) {
  _odsPrinter << ' ' << "untied";
  }
  if (false || mergeableAttr()) {
  _odsPrinter << ' ' << "mergeable";
  }
  if (false || in_reduction_vars().size() || in_reductionsAttr()) {
  _odsPrinter << ' ' << "in_reduction";
  _odsPrinter << "(";
  printReductionVarList(_odsPrinter, *this, in_reduction_vars(), in_reduction_vars().getTypes(), in_reductionsAttr());
  _odsPrinter << ")";
  }
  if (false || priority()) {
  _odsPrinter << ' ' << "priority";
  _odsPrinter << "(";
  if (::mlir::Value value = priority())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if (false || allocate_vars().size() || allocators_vars().size()) {
  _odsPrinter << ' ' << "allocate";
  _odsPrinter << "(";
  printAllocateAndAllocator(_odsPrinter, *this, allocate_vars(), allocate_vars().getTypes(), allocators_vars(), allocators_vars().getTypes());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(region());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "untied", "mergeable", "in_reductions"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TaskOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskwaitOp definitions
//===----------------------------------------------------------------------===//

TaskwaitOpAdaptor::TaskwaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.taskwait", odsAttrs.getContext());
}

TaskwaitOpAdaptor::TaskwaitOpAdaptor(TaskwaitOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange TaskwaitOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TaskwaitOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TaskwaitOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr TaskwaitOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TaskwaitOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TaskwaitOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TaskwaitOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TaskwaitOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TaskwaitOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TaskwaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void TaskwaitOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TaskwaitOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TaskwaitOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult TaskwaitOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TaskwaitOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TaskwaitOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TaskwaitOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskyieldOp definitions
//===----------------------------------------------------------------------===//

TaskyieldOpAdaptor::TaskyieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.taskyield", odsAttrs.getContext());
}

TaskyieldOpAdaptor::TaskyieldOpAdaptor(TaskyieldOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange TaskyieldOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TaskyieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TaskyieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr TaskyieldOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TaskyieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TaskyieldOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TaskyieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TaskyieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TaskyieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TaskyieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void TaskyieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TaskyieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TaskyieldOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult TaskyieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TaskyieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TaskyieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TaskyieldOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TerminatorOp definitions
//===----------------------------------------------------------------------===//

TerminatorOpAdaptor::TerminatorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.terminator", odsAttrs.getContext());
}

TerminatorOpAdaptor::TerminatorOpAdaptor(TerminatorOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange TerminatorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TerminatorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TerminatorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr TerminatorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TerminatorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TerminatorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TerminatorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> TerminatorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TerminatorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TerminatorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void TerminatorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TerminatorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TerminatorOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult TerminatorOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TerminatorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void TerminatorOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TerminatorOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ThreadprivateOp definitions
//===----------------------------------------------------------------------===//

ThreadprivateOpAdaptor::ThreadprivateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.threadprivate", odsAttrs.getContext());
}

ThreadprivateOpAdaptor::ThreadprivateOpAdaptor(ThreadprivateOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ThreadprivateOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ThreadprivateOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ThreadprivateOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ThreadprivateOpAdaptor::sym_addr() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ThreadprivateOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ThreadprivateOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ThreadprivateOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ThreadprivateOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ThreadprivateOp::sym_addr() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ThreadprivateOp::sym_addrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ThreadprivateOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ThreadprivateOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ThreadprivateOp::tls_addr() {
  return *getODSResults(0).begin();
}

void ThreadprivateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type tls_addr, ::mlir::Value sym_addr) {
  odsState.addOperands(sym_addr);
  odsState.addTypes(tls_addr);
}

void ThreadprivateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value sym_addr) {
  odsState.addOperands(sym_addr);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ThreadprivateOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ThreadprivateOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ThreadprivateOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ThreadprivateOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sym_addrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sym_addrOperands(sym_addrRawOperands);  ::llvm::SMLoc sym_addrOperandsLoc;
  (void)sym_addrOperandsLoc;
  ::mlir::Type sym_addrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> sym_addrTypes(sym_addrRawTypes);
  ::mlir::Type tls_addrRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> tls_addrTypes(tls_addrRawTypes);

  sym_addrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sym_addrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::omp::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sym_addrRawTypes[0] = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::omp::PointerLikeType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    tls_addrRawTypes[0] = type;
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addTypes(tls_addrTypes);
  if (parser.resolveOperands(sym_addrOperands, sym_addrTypes, sym_addrOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ThreadprivateOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << sym_addr();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = sym_addr().getType();
    if (auto validType = type.dyn_cast<::mlir::omp::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = tls_addr().getType();
    if (auto validType = type.dyn_cast<::mlir::omp::PointerLikeType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ThreadprivateOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::WsLoopOp definitions
//===----------------------------------------------------------------------===//

WsLoopOpAdaptor::WsLoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.wsloop", odsAttrs.getContext());
}

WsLoopOpAdaptor::WsLoopOpAdaptor(WsLoopOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange WsLoopOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WsLoopOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, WsLoopOp::operand_segment_sizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange WsLoopOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange WsLoopOpAdaptor::lowerBound() {
  return getODSOperands(0);
}

::mlir::ValueRange WsLoopOpAdaptor::upperBound() {
  return getODSOperands(1);
}

::mlir::ValueRange WsLoopOpAdaptor::step() {
  return getODSOperands(2);
}

::mlir::ValueRange WsLoopOpAdaptor::linear_vars() {
  return getODSOperands(3);
}

::mlir::ValueRange WsLoopOpAdaptor::linear_step_vars() {
  return getODSOperands(4);
}

::mlir::ValueRange WsLoopOpAdaptor::reduction_vars() {
  return getODSOperands(5);
}

::mlir::Value WsLoopOpAdaptor::schedule_chunk_var() {
  auto operands = getODSOperands(6);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::DictionaryAttr WsLoopOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr WsLoopOpAdaptor::reductionsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, WsLoopOp::reductionsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > WsLoopOpAdaptor::reductions() {
  auto attr = reductionsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::omp::ClauseScheduleKindAttr WsLoopOpAdaptor::schedule_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, WsLoopOp::schedule_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseScheduleKindAttr>();
    return attr;
}

::llvm::Optional<::mlir::omp::ClauseScheduleKind> WsLoopOpAdaptor::schedule_val() {
  auto attr = schedule_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseScheduleKind>(attr.getValue()) : (::llvm::None);
}

::mlir::omp::ScheduleModifierAttr WsLoopOpAdaptor::schedule_modifierAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, WsLoopOp::schedule_modifierAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ScheduleModifierAttr>();
    return attr;
}

::llvm::Optional<::mlir::omp::ScheduleModifier> WsLoopOpAdaptor::schedule_modifier() {
  auto attr = schedule_modifierAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ScheduleModifier>(attr.getValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOpAdaptor::simd_modifierAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, WsLoopOp::simd_modifierAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool WsLoopOpAdaptor::simd_modifier() {
  auto attr = simd_modifierAttr();
  return attr != nullptr;
}

::mlir::IntegerAttr WsLoopOpAdaptor::collapse_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, WsLoopOp::collapse_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    return attr;
}

::llvm::Optional<uint64_t> WsLoopOpAdaptor::collapse_val() {
  auto attr = collapse_valAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOpAdaptor::nowaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, WsLoopOp::nowaitAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool WsLoopOpAdaptor::nowait() {
  auto attr = nowaitAttr();
  return attr != nullptr;
}

::mlir::IntegerAttr WsLoopOpAdaptor::ordered_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, WsLoopOp::ordered_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    return attr;
}

::llvm::Optional<uint64_t> WsLoopOpAdaptor::ordered_val() {
  auto attr = ordered_valAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::omp::ClauseOrderKindAttr WsLoopOpAdaptor::order_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, WsLoopOp::order_valAttrName(*odsOpName)).dyn_cast_or_null<::mlir::omp::ClauseOrderKindAttr>();
    return attr;
}

::llvm::Optional<::mlir::omp::ClauseOrderKind> WsLoopOpAdaptor::order_val() {
  auto attr = order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseOrderKind>(attr.getValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOpAdaptor::inclusiveAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, WsLoopOp::inclusiveAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool WsLoopOpAdaptor::inclusive() {
  auto attr = inclusiveAttr();
  return attr != nullptr;
}

::mlir::RegionRange WsLoopOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &WsLoopOpAdaptor::region() {
  return *odsRegions[0];
}

::mlir::LogicalResult WsLoopOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_collapse_val;
  ::mlir::Attribute tblgen_inclusive;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'omp.wsloop' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == WsLoopOp::operand_segment_sizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == WsLoopOp::collapse_valAttrName(*odsOpName)) {
      tblgen_collapse_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WsLoopOp::inclusiveAttrName(*odsOpName)) {
      tblgen_inclusive = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WsLoopOp::nowaitAttrName(*odsOpName)) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_order_val;
  ::mlir::Attribute tblgen_ordered_val;
  ::mlir::Attribute tblgen_reductions;
  ::mlir::Attribute tblgen_schedule_modifier;
  ::mlir::Attribute tblgen_schedule_val;
  ::mlir::Attribute tblgen_simd_modifier;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == WsLoopOp::order_valAttrName(*odsOpName)) {
      tblgen_order_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WsLoopOp::ordered_valAttrName(*odsOpName)) {
      tblgen_ordered_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WsLoopOp::reductionsAttrName(*odsOpName)) {
      tblgen_reductions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WsLoopOp::schedule_modifierAttrName(*odsOpName)) {
      tblgen_schedule_modifier = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WsLoopOp::schedule_valAttrName(*odsOpName)) {
      tblgen_schedule_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == WsLoopOp::simd_modifierAttrName(*odsOpName)) {
      tblgen_simd_modifier = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 7)
      return emitError(loc, "'omp.wsloop' op ""'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (tblgen_reductions && !(((tblgen_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'omp.wsloop' op ""attribute 'reductions' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_schedule_val && !((tblgen_schedule_val.isa<::mlir::omp::ClauseScheduleKindAttr>())))
    return emitError(loc, "'omp.wsloop' op ""attribute 'schedule_val' failed to satisfy constraint: ScheduleKind Clause");

  if (tblgen_schedule_modifier && !((tblgen_schedule_modifier.isa<::mlir::omp::ScheduleModifierAttr>())))
    return emitError(loc, "'omp.wsloop' op ""attribute 'schedule_modifier' failed to satisfy constraint: OpenMP Schedule Modifier");

  if (tblgen_simd_modifier && !((tblgen_simd_modifier.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.wsloop' op ""attribute 'simd_modifier' failed to satisfy constraint: unit attribute");

  if (tblgen_collapse_val && !((((tblgen_collapse_val.isa<::mlir::IntegerAttr>())) && ((tblgen_collapse_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_collapse_val.cast<::mlir::IntegerAttr>().getInt() >= 0))))
    return emitError(loc, "'omp.wsloop' op ""attribute 'collapse_val' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");

  if (tblgen_nowait && !((tblgen_nowait.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.wsloop' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");

  if (tblgen_ordered_val && !((((tblgen_ordered_val.isa<::mlir::IntegerAttr>())) && ((tblgen_ordered_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_ordered_val.cast<::mlir::IntegerAttr>().getInt() >= 0))))
    return emitError(loc, "'omp.wsloop' op ""attribute 'ordered_val' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");

  if (tblgen_order_val && !((tblgen_order_val.isa<::mlir::omp::ClauseOrderKindAttr>())))
    return emitError(loc, "'omp.wsloop' op ""attribute 'order_val' failed to satisfy constraint: OrderKind Clause");

  if (tblgen_inclusive && !((tblgen_inclusive.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'omp.wsloop' op ""attribute 'inclusive' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WsLoopOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range WsLoopOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range WsLoopOp::lowerBound() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range WsLoopOp::upperBound() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range WsLoopOp::step() {
  return getODSOperands(2);
}

::mlir::Operation::operand_range WsLoopOp::linear_vars() {
  return getODSOperands(3);
}

::mlir::Operation::operand_range WsLoopOp::linear_step_vars() {
  return getODSOperands(4);
}

::mlir::Operation::operand_range WsLoopOp::reduction_vars() {
  return getODSOperands(5);
}

::mlir::Value WsLoopOp::schedule_chunk_var() {
  auto operands = getODSOperands(6);
  return operands.empty() ? ::mlir::Value() : *operands.begin();
}

::mlir::MutableOperandRange WsLoopOp::lowerBoundMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::upperBoundMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::stepMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::linear_varsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::linear_step_varsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::reduction_varsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::schedule_chunk_varMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, operand_segment_sizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> WsLoopOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range WsLoopOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &WsLoopOp::region() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr WsLoopOp::reductionsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, reductionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > WsLoopOp::reductions() {
  auto attr = reductionsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::omp::ClauseScheduleKindAttr WsLoopOp::schedule_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, schedule_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseScheduleKindAttr>();
}

::llvm::Optional<::mlir::omp::ClauseScheduleKind> WsLoopOp::schedule_val() {
  auto attr = schedule_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseScheduleKind>(attr.getValue()) : (::llvm::None);
}

::mlir::omp::ScheduleModifierAttr WsLoopOp::schedule_modifierAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, schedule_modifierAttrName()).dyn_cast_or_null<::mlir::omp::ScheduleModifierAttr>();
}

::llvm::Optional<::mlir::omp::ScheduleModifier> WsLoopOp::schedule_modifier() {
  auto attr = schedule_modifierAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ScheduleModifier>(attr.getValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOp::simd_modifierAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, simd_modifierAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool WsLoopOp::simd_modifier() {
  auto attr = simd_modifierAttr();
  return attr != nullptr;
}

::mlir::IntegerAttr WsLoopOp::collapse_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, collapse_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> WsLoopOp::collapse_val() {
  auto attr = collapse_valAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOp::nowaitAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, nowaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool WsLoopOp::nowait() {
  auto attr = nowaitAttr();
  return attr != nullptr;
}

::mlir::IntegerAttr WsLoopOp::ordered_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, ordered_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> WsLoopOp::ordered_val() {
  auto attr = ordered_valAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::omp::ClauseOrderKindAttr WsLoopOp::order_valAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, order_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseOrderKindAttr>();
}

::llvm::Optional<::mlir::omp::ClauseOrderKind> WsLoopOp::order_val() {
  auto attr = order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseOrderKind>(attr.getValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOp::inclusiveAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, inclusiveAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool WsLoopOp::inclusive() {
  auto attr = inclusiveAttr();
  return attr != nullptr;
}

void WsLoopOp::reductionsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(reductionsAttrName(), attr);
}

void WsLoopOp::schedule_valAttr(::mlir::omp::ClauseScheduleKindAttr attr) {
  (*this)->setAttr(schedule_valAttrName(), attr);
}

void WsLoopOp::schedule_modifierAttr(::mlir::omp::ScheduleModifierAttr attr) {
  (*this)->setAttr(schedule_modifierAttrName(), attr);
}

void WsLoopOp::simd_modifierAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(simd_modifierAttrName(), attr);
}

void WsLoopOp::collapse_valAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(collapse_valAttrName(), attr);
}

void WsLoopOp::nowaitAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(nowaitAttrName(), attr);
}

void WsLoopOp::ordered_valAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(ordered_valAttrName(), attr);
}

void WsLoopOp::order_valAttr(::mlir::omp::ClauseOrderKindAttr attr) {
  (*this)->setAttr(order_valAttrName(), attr);
}

void WsLoopOp::inclusiveAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(inclusiveAttrName(), attr);
}

::mlir::Attribute WsLoopOp::removeReductionsAttr() {
  return (*this)->removeAttr(reductionsAttrName());
}

::mlir::Attribute WsLoopOp::removeSchedule_valAttr() {
  return (*this)->removeAttr(schedule_valAttrName());
}

::mlir::Attribute WsLoopOp::removeSchedule_modifierAttr() {
  return (*this)->removeAttr(schedule_modifierAttrName());
}

::mlir::Attribute WsLoopOp::removeSimd_modifierAttr() {
  return (*this)->removeAttr(simd_modifierAttrName());
}

::mlir::Attribute WsLoopOp::removeCollapse_valAttr() {
  return (*this)->removeAttr(collapse_valAttrName());
}

::mlir::Attribute WsLoopOp::removeNowaitAttr() {
  return (*this)->removeAttr(nowaitAttrName());
}

::mlir::Attribute WsLoopOp::removeOrdered_valAttr() {
  return (*this)->removeAttr(ordered_valAttrName());
}

::mlir::Attribute WsLoopOp::removeOrder_valAttr() {
  return (*this)->removeAttr(order_valAttrName());
}

::mlir::Attribute WsLoopOp::removeInclusiveAttr() {
  return (*this)->removeAttr(inclusiveAttrName());
}

void WsLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/::mlir::UnitAttr simd_modifier, /*optional*/::mlir::IntegerAttr collapse_val, /*optional*/::mlir::UnitAttr nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::UnitAttr inclusive) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addOperands(linear_vars);
  odsState.addOperands(linear_step_vars);
  odsState.addOperands(reduction_vars);
  if (schedule_chunk_var)
    odsState.addOperands(schedule_chunk_var);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), static_cast<int32_t>(linear_vars.size()), static_cast<int32_t>(linear_step_vars.size()), static_cast<int32_t>(reduction_vars.size()), (schedule_chunk_var ? 1 : 0)}));
  if (reductions) {
  odsState.addAttribute(reductionsAttrName(odsState.name), reductions);
  }
  if (schedule_val) {
  odsState.addAttribute(schedule_valAttrName(odsState.name), schedule_val);
  }
  if (schedule_modifier) {
  odsState.addAttribute(schedule_modifierAttrName(odsState.name), schedule_modifier);
  }
  if (simd_modifier) {
  odsState.addAttribute(simd_modifierAttrName(odsState.name), simd_modifier);
  }
  if (collapse_val) {
  odsState.addAttribute(collapse_valAttrName(odsState.name), collapse_val);
  }
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), nowait);
  }
  if (ordered_val) {
  odsState.addAttribute(ordered_valAttrName(odsState.name), ordered_val);
  }
  if (order_val) {
  odsState.addAttribute(order_valAttrName(odsState.name), order_val);
  }
  if (inclusive) {
  odsState.addAttribute(inclusiveAttrName(odsState.name), inclusive);
  }
  (void)odsState.addRegion();
}

void WsLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/::mlir::UnitAttr simd_modifier, /*optional*/::mlir::IntegerAttr collapse_val, /*optional*/::mlir::UnitAttr nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/::mlir::UnitAttr inclusive) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addOperands(linear_vars);
  odsState.addOperands(linear_step_vars);
  odsState.addOperands(reduction_vars);
  if (schedule_chunk_var)
    odsState.addOperands(schedule_chunk_var);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), static_cast<int32_t>(linear_vars.size()), static_cast<int32_t>(linear_step_vars.size()), static_cast<int32_t>(reduction_vars.size()), (schedule_chunk_var ? 1 : 0)}));
  if (reductions) {
  odsState.addAttribute(reductionsAttrName(odsState.name), reductions);
  }
  if (schedule_val) {
  odsState.addAttribute(schedule_valAttrName(odsState.name), schedule_val);
  }
  if (schedule_modifier) {
  odsState.addAttribute(schedule_modifierAttrName(odsState.name), schedule_modifier);
  }
  if (simd_modifier) {
  odsState.addAttribute(simd_modifierAttrName(odsState.name), simd_modifier);
  }
  if (collapse_val) {
  odsState.addAttribute(collapse_valAttrName(odsState.name), collapse_val);
  }
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), nowait);
  }
  if (ordered_val) {
  odsState.addAttribute(ordered_valAttrName(odsState.name), ordered_val);
  }
  if (order_val) {
  odsState.addAttribute(order_valAttrName(odsState.name), order_val);
  }
  if (inclusive) {
  odsState.addAttribute(inclusiveAttrName(odsState.name), inclusive);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WsLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/bool simd_modifier, /*optional*/::mlir::IntegerAttr collapse_val, /*optional*/bool nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/bool inclusive) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addOperands(linear_vars);
  odsState.addOperands(linear_step_vars);
  odsState.addOperands(reduction_vars);
  if (schedule_chunk_var)
    odsState.addOperands(schedule_chunk_var);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), static_cast<int32_t>(linear_vars.size()), static_cast<int32_t>(linear_step_vars.size()), static_cast<int32_t>(reduction_vars.size()), (schedule_chunk_var ? 1 : 0)}));
  if (reductions) {
  odsState.addAttribute(reductionsAttrName(odsState.name), reductions);
  }
  if (schedule_val) {
  odsState.addAttribute(schedule_valAttrName(odsState.name), schedule_val);
  }
  if (schedule_modifier) {
  odsState.addAttribute(schedule_modifierAttrName(odsState.name), schedule_modifier);
  }
  if (simd_modifier) {
  odsState.addAttribute(simd_modifierAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (collapse_val) {
  odsState.addAttribute(collapse_valAttrName(odsState.name), collapse_val);
  }
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (ordered_val) {
  odsState.addAttribute(ordered_valAttrName(odsState.name), ordered_val);
  }
  if (order_val) {
  odsState.addAttribute(order_valAttrName(odsState.name), order_val);
  }
  if (inclusive) {
  odsState.addAttribute(inclusiveAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  (void)odsState.addRegion();
}

void WsLoopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange lowerBound, ::mlir::ValueRange upperBound, ::mlir::ValueRange step, ::mlir::ValueRange linear_vars, ::mlir::ValueRange linear_step_vars, ::mlir::ValueRange reduction_vars, /*optional*/::mlir::ArrayAttr reductions, /*optional*/::mlir::omp::ClauseScheduleKindAttr schedule_val, /*optional*/::mlir::Value schedule_chunk_var, /*optional*/::mlir::omp::ScheduleModifierAttr schedule_modifier, /*optional*/bool simd_modifier, /*optional*/::mlir::IntegerAttr collapse_val, /*optional*/bool nowait, /*optional*/::mlir::IntegerAttr ordered_val, /*optional*/::mlir::omp::ClauseOrderKindAttr order_val, /*optional*/bool inclusive) {
  odsState.addOperands(lowerBound);
  odsState.addOperands(upperBound);
  odsState.addOperands(step);
  odsState.addOperands(linear_vars);
  odsState.addOperands(linear_step_vars);
  odsState.addOperands(reduction_vars);
  if (schedule_chunk_var)
    odsState.addOperands(schedule_chunk_var);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(lowerBound.size()), static_cast<int32_t>(upperBound.size()), static_cast<int32_t>(step.size()), static_cast<int32_t>(linear_vars.size()), static_cast<int32_t>(linear_step_vars.size()), static_cast<int32_t>(reduction_vars.size()), (schedule_chunk_var ? 1 : 0)}));
  if (reductions) {
  odsState.addAttribute(reductionsAttrName(odsState.name), reductions);
  }
  if (schedule_val) {
  odsState.addAttribute(schedule_valAttrName(odsState.name), schedule_val);
  }
  if (schedule_modifier) {
  odsState.addAttribute(schedule_modifierAttrName(odsState.name), schedule_modifier);
  }
  if (simd_modifier) {
  odsState.addAttribute(simd_modifierAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (collapse_val) {
  odsState.addAttribute(collapse_valAttrName(odsState.name), collapse_val);
  }
  if (nowait) {
  odsState.addAttribute(nowaitAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (ordered_val) {
  odsState.addAttribute(ordered_valAttrName(odsState.name), ordered_val);
  }
  if (order_val) {
  odsState.addAttribute(order_valAttrName(odsState.name), order_val);
  }
  if (inclusive) {
  odsState.addAttribute(inclusiveAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void WsLoopOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult WsLoopOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_collapse_val;
  ::mlir::Attribute tblgen_inclusive;
  ::mlir::Attribute tblgen_nowait;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == operand_segment_sizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == collapse_valAttrName()) {
      tblgen_collapse_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == inclusiveAttrName()) {
      tblgen_inclusive = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == nowaitAttrName()) {
      tblgen_nowait = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_order_val;
  ::mlir::Attribute tblgen_ordered_val;
  ::mlir::Attribute tblgen_reductions;
  ::mlir::Attribute tblgen_schedule_modifier;
  ::mlir::Attribute tblgen_schedule_val;
  ::mlir::Attribute tblgen_simd_modifier;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == order_valAttrName()) {
      tblgen_order_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == ordered_valAttrName()) {
      tblgen_ordered_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == reductionsAttrName()) {
      tblgen_reductions = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == schedule_modifierAttrName()) {
      tblgen_schedule_modifier = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == schedule_valAttrName()) {
      tblgen_schedule_val = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == simd_modifierAttrName()) {
      tblgen_simd_modifier = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 7)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_reductions, "reductions")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps11(*this, tblgen_schedule_val, "schedule_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps12(*this, tblgen_schedule_modifier, "schedule_modifier")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps7(*this, tblgen_simd_modifier, "simd_modifier")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps6(*this, tblgen_collapse_val, "collapse_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps7(*this, tblgen_nowait, "nowait")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps6(*this, tblgen_ordered_val, "ordered_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps13(*this, tblgen_order_val, "order_val")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps7(*this, tblgen_inclusive, "inclusive")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup5 = getODSOperands(5);

    for (auto v : valueGroup5) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup6 = getODSOperands(6);

    if (valueGroup6.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup6.size();
    }

    for (auto v : valueGroup6) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({this->getODSOperands(0).getType(), this->getODSOperands(1).getType(), this->getODSOperands(2).getType()})))))
    return emitOpError("failed to verify that all of {lowerBound, upperBound, step} have same type");
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult WsLoopOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult WsLoopOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> linear_varsOperands;
  ::llvm::SMLoc linear_varsOperandsLoc;
  (void)linear_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> linear_varsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> linear_step_varsOperands;
  ::llvm::SMLoc linear_step_varsOperandsLoc;
  (void)linear_step_varsOperandsLoc;
  ::mlir::omp::ClauseScheduleKindAttr schedule_valAttr;
  ::mlir::omp::ScheduleModifierAttr schedule_modifierAttr;
  ::mlir::UnitAttr simd_modifierAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> schedule_chunk_varOperands;
  ::llvm::SMLoc schedule_chunk_varOperandsLoc;
  (void)schedule_chunk_varOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> schedule_chunk_varTypes;
  ::mlir::IntegerAttr collapse_valAttr;
  ::mlir::IntegerAttr ordered_valAttr;
  ::mlir::omp::ClauseOrderKindAttr order_valAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> reduction_varsOperands;
  ::llvm::SMLoc reduction_varsOperandsLoc;
  (void)reduction_varsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> reduction_varsTypes;
  ::mlir::ArrayAttr reductionsAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> lowerBoundOperands;
  ::llvm::SMLoc lowerBoundOperandsLoc;
  (void)lowerBoundOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> upperBoundOperands;
  ::llvm::SMLoc upperBoundOperandsLoc;
  (void)upperBoundOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> stepOperands;
  ::llvm::SMLoc stepOperandsLoc;
  (void)stepOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> stepTypes;
  ::mlir::UnitAttr inclusiveAttr;
  bool linearClause = false;
  bool scheduleClause = false;
  bool collapseClause = false;
  bool nowaitClause = false;
  bool orderedClause = false;
  bool orderClause = false;
  bool reductionClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("linear"))) {

  if (linearClause) {
    return parser.emitError(parser.getNameLoc())
          << "`linear` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  linearClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    linear_varsOperandsLoc = parser.getCurrentLocation();
    linear_step_varsOperandsLoc = parser.getCurrentLocation();
    if (parseLinearClause(parser, linear_varsOperands, linear_varsTypes, linear_step_varsOperands))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("schedule"))) {

  if (scheduleClause) {
    return parser.emitError(parser.getNameLoc())
          << "`schedule` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  scheduleClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    schedule_chunk_varOperandsLoc = parser.getCurrentLocation();
    ::llvm::Optional<::mlir::OpAsmParser::UnresolvedOperand> schedule_chunk_varOperand;
    ::mlir::Type schedule_chunk_varType;
    if (parseScheduleClause(parser, schedule_valAttr, schedule_modifierAttr, simd_modifierAttr, schedule_chunk_varOperand, schedule_chunk_varType))
      return ::mlir::failure();
    if (schedule_valAttr)
      result.addAttribute("schedule_val", schedule_valAttr);
    if (schedule_modifierAttr)
      result.addAttribute("schedule_modifier", schedule_modifierAttr);
    if (simd_modifierAttr)
      result.addAttribute("simd_modifier", simd_modifierAttr);
    if (schedule_chunk_varOperand.hasValue())
      schedule_chunk_varOperands.push_back(*schedule_chunk_varOperand);
    if (schedule_chunk_varType)
      schedule_chunk_varTypes.push_back(schedule_chunk_varType);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("collapse"))) {

  if (collapseClause) {
    return parser.emitError(parser.getNameLoc())
          << "`collapse` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  collapseClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(collapse_valAttr, parser.getBuilder().getIntegerType(64), "collapse_val", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("nowait"))) {

  if (nowaitClause) {
    return parser.emitError(parser.getNameLoc())
          << "`nowait` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  nowaitClause = true;
  result.addAttribute("nowait", UnitAttr::get(parser.getContext()));
    } else if (succeeded(parser.parseOptionalKeyword("ordered"))) {

  if (orderedClause) {
    return parser.emitError(parser.getNameLoc())
          << "`ordered` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  orderedClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::OptionalParseResult parseResult =
      parser.parseOptionalAttribute(ordered_valAttr, parser.getBuilder().getIntegerType(64), "ordered_val", result.attributes);
    if (parseResult.hasValue() && failed(*parseResult))
      return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("order"))) {

  if (orderClause) {
    return parser.emitError(parser.getNameLoc())
          << "`order` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  orderClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    if (parseClauseAttr(parser, order_valAttr))
      return ::mlir::failure();
    if (order_valAttr)
      result.addAttribute("order_val", order_valAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else if (succeeded(parser.parseOptionalKeyword("reduction"))) {

  if (reductionClause) {
    return parser.emitError(parser.getNameLoc())
          << "`reduction` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  reductionClause = true;
  if (parser.parseLParen())
    return ::mlir::failure();
  {
    reduction_varsOperandsLoc = parser.getCurrentLocation();
    if (parseReductionVarList(parser, reduction_varsOperands, reduction_varsTypes, reductionsAttr))
      return ::mlir::failure();
    if (reductionsAttr)
      result.addAttribute("reductions", reductionsAttr);
  }
  if (parser.parseRParen())
    return ::mlir::failure();
    } else  {
    break;
  }
}
  if (parser.parseKeyword("for"))
    return ::mlir::failure();
  {
    lowerBoundOperandsLoc = parser.getCurrentLocation();
    upperBoundOperandsLoc = parser.getCurrentLocation();
    stepOperandsLoc = parser.getCurrentLocation();
    if (parseWsLoopControl(parser, *regionRegion, lowerBoundOperands, upperBoundOperands, stepOperands, stepTypes, inclusiveAttr))
      return ::mlir::failure();
    if (inclusiveAttr)
      result.addAttribute("inclusive", inclusiveAttr);
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addRegion(std::move(regionRegion));
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(lowerBoundOperands.size()), static_cast<int32_t>(upperBoundOperands.size()), static_cast<int32_t>(stepOperands.size()), static_cast<int32_t>(linear_varsOperands.size()), static_cast<int32_t>(linear_step_varsOperands.size()), static_cast<int32_t>(reduction_varsOperands.size()), static_cast<int32_t>(schedule_chunk_varOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  if (parser.resolveOperands(lowerBoundOperands, stepTypes, lowerBoundOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(upperBoundOperands, stepTypes, upperBoundOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(stepOperands, stepTypes, stepOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(linear_varsOperands, linear_varsTypes, linear_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(linear_step_varsOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(reduction_varsOperands, reduction_varsTypes, reduction_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(schedule_chunk_varOperands, schedule_chunk_varTypes, schedule_chunk_varOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void WsLoopOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if (false || linear_vars().size() || linear_step_vars().size()) {
  _odsPrinter << ' ' << "linear";
  _odsPrinter << "(";
  printLinearClause(_odsPrinter, *this, linear_vars(), linear_vars().getTypes(), linear_step_vars());
  _odsPrinter << ")";
  }
  if (false || schedule_valAttr() || schedule_modifierAttr() || simd_modifierAttr() || schedule_chunk_var()) {
  _odsPrinter << ' ' << "schedule";
  _odsPrinter << "(";
  printScheduleClause(_odsPrinter, *this, schedule_valAttr(), schedule_modifierAttr(), simd_modifierAttr(), schedule_chunk_var(), (schedule_chunk_var() ? schedule_chunk_var().getType() : Type()));
  _odsPrinter << ")";
  }
  if (false || collapse_valAttr()) {
  _odsPrinter << ' ' << "collapse";
  _odsPrinter << "(";
  _odsPrinter.printAttributeWithoutType(collapse_valAttr());
  _odsPrinter << ")";
  }
  if (false || nowaitAttr()) {
  _odsPrinter << ' ' << "nowait";
  }
  if (false || ordered_valAttr()) {
  _odsPrinter << ' ' << "ordered";
  _odsPrinter << "(";
  _odsPrinter.printAttributeWithoutType(ordered_valAttr());
  _odsPrinter << ")";
  }
  if (false || order_valAttr()) {
  _odsPrinter << ' ' << "order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, order_valAttr());
  _odsPrinter << ")";
  }
  if (false || reduction_vars().size() || reductionsAttr()) {
  _odsPrinter << ' ' << "reduction";
  _odsPrinter << "(";
  printReductionVarList(_odsPrinter, *this, reduction_vars(), reduction_vars().getTypes(), reductionsAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << "for";
  _odsPrinter << ' ';
  printWsLoopControl(_odsPrinter, *this, region(), lowerBound(), upperBound(), step(), step().getTypes(), inclusiveAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "schedule_val", "schedule_modifier", "simd_modifier", "collapse_val", "nowait", "ordered_val", "order_val", "reductions", "inclusive"});
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::WsLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::YieldOp definitions
//===----------------------------------------------------------------------===//

YieldOpAdaptor::YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("omp.yield", odsAttrs.getContext());
}

YieldOpAdaptor::YieldOpAdaptor(YieldOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange YieldOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> YieldOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange YieldOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange YieldOpAdaptor::results() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr YieldOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult YieldOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> YieldOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range YieldOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range YieldOp::results() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange YieldOp::resultsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> YieldOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range YieldOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 build(odsBuilder, odsState, {}); 
}

void YieldOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results) {
  odsState.addOperands(results);
}

void YieldOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult YieldOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult YieldOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult YieldOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> resultsOperands;
  ::llvm::SMLoc resultsOperandsLoc;
  (void)resultsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;
  if (succeeded(parser.parseOptionalLParen())) {

  resultsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(resultsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.resolveOperands(resultsOperands, resultsTypes, resultsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void YieldOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!results().empty()) {
  _odsPrinter << "(";
  _odsPrinter << results();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << results().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void YieldOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace omp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::YieldOp)


#endif  // GET_OP_CLASSES

