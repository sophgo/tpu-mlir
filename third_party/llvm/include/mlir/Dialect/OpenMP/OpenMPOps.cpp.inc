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
::mlir::omp::TargetOp,
::mlir::omp::TaskwaitOp,
::mlir::omp::TaskyieldOp,
::mlir::omp::TerminatorOp,
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
  if (!((type.isa<::mlir::IntegerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_OpenMPOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || ((type.isa<::mlir::IndexType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer or index, but got " << type;
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
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::FlatSymbolRefAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: flat symbol reference attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseDependAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: depend clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((attr.cast<::mlir::IntegerAttr>().getInt() >= 0)))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseProcBindKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: ProcBindKind Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any type attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::SymbolRefAttr>()); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: symbol ref array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ClauseScheduleKindAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: ScheduleKind Clause";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::omp::ScheduleModifierAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: OpenMP Schedule Modifier";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_OpenMPOps12(
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

AtomicCaptureOpAdaptor::AtomicCaptureOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AtomicCaptureOpAdaptor::AtomicCaptureOpAdaptor(AtomicCaptureOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
  ::mlir::IntegerAttr attr = odsAttrs.get("hint_val").dyn_cast_or_null<::mlir::IntegerAttr>();
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
  ::mlir::omp::ClauseMemoryOrderKindAttr attr = odsAttrs.get("memory_order_val").dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
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
  {
    auto tblgen_hint_val = odsAttrs.get("hint_val");
    if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
      return emitError(loc, "'omp.atomic.capture' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
    auto tblgen_memory_order_val = odsAttrs.get("memory_order_val");
    if (tblgen_memory_order_val && !((tblgen_memory_order_val.isa<::mlir::omp::ClauseMemoryOrderKindAttr>())))
      return emitError(loc, "'omp.atomic.capture' op ""attribute 'memory_order_val' failed to satisfy constraint: MemoryOrderKind Clause");
  }
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
  return (*this)->getAttr(hint_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t AtomicCaptureOp::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicCaptureOp::memory_order_valAttr() {
  return (*this)->getAttr(memory_order_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
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
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
}

void AtomicCaptureOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
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
  {
    auto tblgen_hint_val = (*this)->getAttr(hint_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
      return ::mlir::failure();
  }
  {
    auto tblgen_memory_order_val = (*this)->getAttr(memory_order_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps1(*this, tblgen_memory_order_val, "memory_order_val")))
      return ::mlir::failure();
  }
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
  result.addAttribute("memory_order", UnitAttr::get(parser.getContext()));
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
  result.addAttribute("hint", UnitAttr::get(parser.getContext()));
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
  if ((*this)->hasAttrOfType<UnitAttr>("memory_order")) {
  _odsPrinter << ' ' << "memory_order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, memory_order_valAttr());
  _odsPrinter << ")";
  }
  if ((*this)->hasAttrOfType<UnitAttr>("hint")) {
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
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"hint", "memory_order", "memory_order_val", "hint_val"});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicCaptureOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicReadOp definitions
//===----------------------------------------------------------------------===//

AtomicReadOpAdaptor::AtomicReadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AtomicReadOpAdaptor::AtomicReadOpAdaptor(AtomicReadOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
  ::mlir::IntegerAttr attr = odsAttrs.get("hint_val").dyn_cast_or_null<::mlir::IntegerAttr>();
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
  ::mlir::omp::ClauseMemoryOrderKindAttr attr = odsAttrs.get("memory_order_val").dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
  return attr;
}

::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> AtomicReadOpAdaptor::memory_order_val() {
  auto attr = memory_order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::llvm::None);
}

::mlir::LogicalResult AtomicReadOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_hint_val = odsAttrs.get("hint_val");
    if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
      return emitError(loc, "'omp.atomic.read' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
    auto tblgen_memory_order_val = odsAttrs.get("memory_order_val");
    if (tblgen_memory_order_val && !((tblgen_memory_order_val.isa<::mlir::omp::ClauseMemoryOrderKindAttr>())))
      return emitError(loc, "'omp.atomic.read' op ""attribute 'memory_order_val' failed to satisfy constraint: MemoryOrderKind Clause");
  }
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
  return (*this)->getAttr(hint_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t AtomicReadOp::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicReadOp::memory_order_valAttr() {
  return (*this)->getAttr(memory_order_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
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
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
}

void AtomicReadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::Value v, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addOperands(v);
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
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
  {
    auto tblgen_hint_val = (*this)->getAttr(hint_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
      return ::mlir::failure();
  }
  {
    auto tblgen_memory_order_val = (*this)->getAttr(memory_order_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps1(*this, tblgen_memory_order_val, "memory_order_val")))
      return ::mlir::failure();
  }
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
  ::mlir::OpAsmParser::OperandType vRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> vOperands(vRawOperands);  ::llvm::SMLoc vOperandsLoc;
  (void)vOperandsLoc;
  ::mlir::OpAsmParser::OperandType xRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> xOperands(xRawOperands);  ::llvm::SMLoc xOperandsLoc;
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
  result.addAttribute("memory_order", UnitAttr::get(parser.getContext()));
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
  result.addAttribute("hint", UnitAttr::get(parser.getContext()));
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
  if ((*this)->hasAttrOfType<UnitAttr>("memory_order")) {
  _odsPrinter << ' ' << "memory_order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, memory_order_valAttr());
  _odsPrinter << ")";
  }
  if ((*this)->hasAttrOfType<UnitAttr>("hint")) {
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
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"hint", "memory_order", "memory_order_val", "hint_val"});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicReadOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicUpdateOp definitions
//===----------------------------------------------------------------------===//

AtomicUpdateOpAdaptor::AtomicUpdateOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AtomicUpdateOpAdaptor::AtomicUpdateOpAdaptor(AtomicUpdateOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
  ::mlir::IntegerAttr attr = odsAttrs.get("hint_val").dyn_cast_or_null<::mlir::IntegerAttr>();
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
  ::mlir::omp::ClauseMemoryOrderKindAttr attr = odsAttrs.get("memory_order_val").dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
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
  {
    auto tblgen_hint_val = odsAttrs.get("hint_val");
    if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
      return emitError(loc, "'omp.atomic.update' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
    auto tblgen_memory_order_val = odsAttrs.get("memory_order_val");
    if (tblgen_memory_order_val && !((tblgen_memory_order_val.isa<::mlir::omp::ClauseMemoryOrderKindAttr>())))
      return emitError(loc, "'omp.atomic.update' op ""attribute 'memory_order_val' failed to satisfy constraint: MemoryOrderKind Clause");
  }
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
  return (*this)->getAttr(hint_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t AtomicUpdateOp::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicUpdateOp::memory_order_valAttr() {
  return (*this)->getAttr(memory_order_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
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
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
  (void)odsState.addRegion();
}

void AtomicUpdateOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value x, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(x);
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
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
  {
    auto tblgen_hint_val = (*this)->getAttr(hint_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
      return ::mlir::failure();
  }
  {
    auto tblgen_memory_order_val = (*this)->getAttr(memory_order_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps1(*this, tblgen_memory_order_val, "memory_order_val")))
      return ::mlir::failure();
  }
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
  ::mlir::OpAsmParser::OperandType xRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> xOperands(xRawOperands);  ::llvm::SMLoc xOperandsLoc;
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
  result.addAttribute("memory_order", UnitAttr::get(parser.getContext()));
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
  result.addAttribute("hint", UnitAttr::get(parser.getContext()));
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
  if ((*this)->hasAttrOfType<UnitAttr>("memory_order")) {
  _odsPrinter << ' ' << "memory_order";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, memory_order_valAttr());
  _odsPrinter << ")";
  }
  if ((*this)->hasAttrOfType<UnitAttr>("hint")) {
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
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"hint", "memory_order", "memory_order_val", "hint_val"});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicUpdateOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::AtomicWriteOp definitions
//===----------------------------------------------------------------------===//

AtomicWriteOpAdaptor::AtomicWriteOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

AtomicWriteOpAdaptor::AtomicWriteOpAdaptor(AtomicWriteOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
  ::mlir::IntegerAttr attr = odsAttrs.get("hint_val").dyn_cast_or_null<::mlir::IntegerAttr>();
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
  ::mlir::omp::ClauseMemoryOrderKindAttr attr = odsAttrs.get("memory_order_val").dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
  return attr;
}

::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind> AtomicWriteOpAdaptor::memory_order_val() {
  auto attr = memory_order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseMemoryOrderKind>(attr.getValue()) : (::llvm::None);
}

::mlir::LogicalResult AtomicWriteOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_hint_val = odsAttrs.get("hint_val");
    if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
      return emitError(loc, "'omp.atomic.write' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");
  }
  {
    auto tblgen_memory_order_val = odsAttrs.get("memory_order_val");
    if (tblgen_memory_order_val && !((tblgen_memory_order_val.isa<::mlir::omp::ClauseMemoryOrderKindAttr>())))
      return emitError(loc, "'omp.atomic.write' op ""attribute 'memory_order_val' failed to satisfy constraint: MemoryOrderKind Clause");
  }
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
  return (*this)->getAttr(hint_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint64_t AtomicWriteOp::hint_val() {
  auto attr = hint_valAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(64), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::omp::ClauseMemoryOrderKindAttr AtomicWriteOp::memory_order_valAttr() {
  return (*this)->getAttr(memory_order_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseMemoryOrderKindAttr>();
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
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
  if (memory_order_val) {
  odsState.addAttribute(memory_order_valAttrName(odsState.name), memory_order_val);
  }
}

void AtomicWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value address, ::mlir::Value value, ::mlir::IntegerAttr hint_val, /*optional*/::mlir::omp::ClauseMemoryOrderKindAttr memory_order_val) {
  odsState.addOperands(address);
  odsState.addOperands(value);
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
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
  {
    auto tblgen_hint_val = (*this)->getAttr(hint_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
      return ::mlir::failure();
  }
  {
    auto tblgen_memory_order_val = (*this)->getAttr(memory_order_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps1(*this, tblgen_memory_order_val, "memory_order_val")))
      return ::mlir::failure();
  }
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
  ::mlir::OpAsmParser::OperandType addressRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> addressOperands(addressRawOperands);  ::llvm::SMLoc addressOperandsLoc;
  (void)addressOperandsLoc;
  ::mlir::OpAsmParser::OperandType valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
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
  result.addAttribute("hint", UnitAttr::get(parser.getContext()));
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
  result.addAttribute("memory_order", UnitAttr::get(parser.getContext()));
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
  if ((*this)->hasAttrOfType<UnitAttr>("hint")) {
  _odsPrinter << ' ' << "hint";
  _odsPrinter << "(";
  printSynchronizationHint(_odsPrinter, *this, hint_valAttr());
  _odsPrinter << ")";
  }
  if ((*this)->hasAttrOfType<UnitAttr>("memory_order")) {
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
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"hint", "memory_order", "hint_val", "memory_order_val"});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::AtomicWriteOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::BarrierOp definitions
//===----------------------------------------------------------------------===//

BarrierOpAdaptor::BarrierOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

BarrierOpAdaptor::BarrierOpAdaptor(BarrierOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::BarrierOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalDeclareOp definitions
//===----------------------------------------------------------------------===//

CriticalDeclareOpAdaptor::CriticalDeclareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CriticalDeclareOpAdaptor::CriticalDeclareOpAdaptor(CriticalDeclareOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
  ::mlir::StringAttr attr = odsAttrs.get("sym_name").cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CriticalDeclareOpAdaptor::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

::mlir::IntegerAttr CriticalDeclareOpAdaptor::hint_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("hint_val").dyn_cast_or_null<::mlir::IntegerAttr>();
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
  {
    auto tblgen_sym_name = odsAttrs.get("sym_name");
    if (!tblgen_sym_name)
      return emitError(loc, "'omp.critical.declare' op ""requires attribute 'sym_name'");

    if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
      return emitError(loc, "'omp.critical.declare' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  }
  {
    auto tblgen_hint_val = odsAttrs.get("hint_val");
    if (tblgen_hint_val && !(((tblgen_hint_val.isa<::mlir::IntegerAttr>())) && ((tblgen_hint_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
      return emitError(loc, "'omp.critical.declare' op ""attribute 'hint_val' failed to satisfy constraint: 64-bit signless integer attribute");
  }
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
  return (*this)->getAttr(sym_nameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef CriticalDeclareOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

::mlir::IntegerAttr CriticalDeclareOp::hint_valAttr() {
  return (*this)->getAttr(hint_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
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
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
}

void CriticalDeclareOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::IntegerAttr hint_val) {
  odsState.addAttribute(sym_nameAttrName(odsState.name), sym_name);
  odsState.addAttribute(hint_valAttrName(odsState.name), hint_val);
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
  {
    auto tblgen_sym_name = (*this)->getAttr(sym_nameAttrName());
    if (!tblgen_sym_name)
      return emitOpError("requires attribute 'sym_name'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps2(*this, tblgen_sym_name, "sym_name")))
      return ::mlir::failure();
  }
  {
    auto tblgen_hint_val = (*this)->getAttr(hint_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps0(*this, tblgen_hint_val, "hint_val")))
      return ::mlir::failure();
  }
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
  result.addAttribute("hint", UnitAttr::get(parser.getContext()));
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
  if ((*this)->hasAttrOfType<UnitAttr>("hint")) {
  _odsPrinter << ' ' << "hint";
  _odsPrinter << "(";
  printSynchronizationHint(_odsPrinter, *this, hint_valAttr());
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"hint", "sym_name", "hint_val"});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::CriticalOp definitions
//===----------------------------------------------------------------------===//

CriticalOpAdaptor::CriticalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

CriticalOpAdaptor::CriticalOpAdaptor(CriticalOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
  ::mlir::FlatSymbolRefAttr attr = odsAttrs.get("name").dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
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
  {
    auto tblgen_name = odsAttrs.get("name");
    if (tblgen_name && !((tblgen_name.isa<::mlir::FlatSymbolRefAttr>())))
      return emitError(loc, "'omp.critical' op ""attribute 'name' failed to satisfy constraint: flat symbol reference attribute");
  }
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
  return (*this)->getAttr(nameAttrName()).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
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
  {
    auto tblgen_name = (*this)->getAttr(nameAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps3(*this, tblgen_name, "name")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_OpenMPOps1(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult CriticalOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::CriticalOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::FlushOp definitions
//===----------------------------------------------------------------------===//

FlushOpAdaptor::FlushOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

FlushOpAdaptor::FlushOpAdaptor(FlushOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> varListOperands;
  ::llvm::SMLoc varListOperandsLoc;
  (void)varListOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> varListTypes;
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::FlushOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::MasterOp definitions
//===----------------------------------------------------------------------===//

MasterOpAdaptor::MasterOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

MasterOpAdaptor::MasterOpAdaptor(MasterOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::MasterOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::OrderedOp definitions
//===----------------------------------------------------------------------===//

OrderedOpAdaptor::OrderedOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

OrderedOpAdaptor::OrderedOpAdaptor(OrderedOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
  ::mlir::omp::ClauseDependAttr attr = odsAttrs.get("depend_type_val").dyn_cast_or_null<::mlir::omp::ClauseDependAttr>();
  return attr;
}

::llvm::Optional<::mlir::omp::ClauseDepend> OrderedOpAdaptor::depend_type_val() {
  auto attr = depend_type_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseDepend>(attr.getValue()) : (::llvm::None);
}

::mlir::IntegerAttr OrderedOpAdaptor::num_loops_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("num_loops_val").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::llvm::Optional<uint64_t> OrderedOpAdaptor::num_loops_val() {
  auto attr = num_loops_valAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult OrderedOpAdaptor::verify(::mlir::Location loc) {
  {
    auto tblgen_depend_type_val = odsAttrs.get("depend_type_val");
    if (tblgen_depend_type_val && !((tblgen_depend_type_val.isa<::mlir::omp::ClauseDependAttr>())))
      return emitError(loc, "'omp.ordered' op ""attribute 'depend_type_val' failed to satisfy constraint: depend clause");
  }
  {
    auto tblgen_num_loops_val = odsAttrs.get("num_loops_val");
    if (tblgen_num_loops_val && !((((tblgen_num_loops_val.isa<::mlir::IntegerAttr>())) && ((tblgen_num_loops_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_num_loops_val.cast<::mlir::IntegerAttr>().getInt() >= 0))))
      return emitError(loc, "'omp.ordered' op ""attribute 'num_loops_val' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  }
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
  return (*this)->getAttr(depend_type_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseDependAttr>();
}

::llvm::Optional<::mlir::omp::ClauseDepend> OrderedOp::depend_type_val() {
  auto attr = depend_type_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseDepend>(attr.getValue()) : (::llvm::None);
}

::mlir::IntegerAttr OrderedOp::num_loops_valAttr() {
  return (*this)->getAttr(num_loops_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
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
  {
    auto tblgen_depend_type_val = (*this)->getAttr(depend_type_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps4(*this, tblgen_depend_type_val, "depend_type_val")))
      return ::mlir::failure();
  }
  {
    auto tblgen_num_loops_val = (*this)->getAttr(num_loops_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps5(*this, tblgen_num_loops_val, "num_loops_val")))
      return ::mlir::failure();
  }
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
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> depend_vec_varsOperands;
  ::llvm::SMLoc depend_vec_varsOperandsLoc;
  (void)depend_vec_varsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> depend_vec_varsTypes;
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::OrderedOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::OrderedRegionOp definitions
//===----------------------------------------------------------------------===//

OrderedRegionOpAdaptor::OrderedRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

OrderedRegionOpAdaptor::OrderedRegionOpAdaptor(OrderedRegionOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
  ::mlir::UnitAttr attr = odsAttrs.get("simd").dyn_cast_or_null<::mlir::UnitAttr>();
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
  {
    auto tblgen_simd = odsAttrs.get("simd");
    if (tblgen_simd && !((tblgen_simd.isa<::mlir::UnitAttr>())))
      return emitError(loc, "'omp.ordered_region' op ""attribute 'simd' failed to satisfy constraint: unit attribute");
  }
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
  return (*this)->getAttr(simdAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
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
  {
    auto tblgen_simd = (*this)->getAttr(simdAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps6(*this, tblgen_simd, "simd")))
      return ::mlir::failure();
  }
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::OrderedRegionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ParallelOp definitions
//===----------------------------------------------------------------------===//

ParallelOpAdaptor::ParallelOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ParallelOpAdaptor::ParallelOpAdaptor(ParallelOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange ParallelOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ParallelOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

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

::mlir::DictionaryAttr ParallelOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::omp::ClauseProcBindKindAttr ParallelOpAdaptor::proc_bind_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::omp::ClauseProcBindKindAttr attr = odsAttrs.get("proc_bind_val").dyn_cast_or_null<::mlir::omp::ClauseProcBindKindAttr>();
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
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'omp.parallel' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitError(loc, "'omp.parallel' op ""'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_proc_bind_val = odsAttrs.get("proc_bind_val");
    if (tblgen_proc_bind_val && !((tblgen_proc_bind_val.isa<::mlir::omp::ClauseProcBindKindAttr>())))
      return emitError(loc, "'omp.parallel' op ""attribute 'proc_bind_val' failed to satisfy constraint: ProcBindKind Clause");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ParallelOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

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

::mlir::MutableOperandRange ParallelOp::if_expr_varMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::num_threads_varMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::allocate_varsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange ParallelOp::allocators_varsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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

::mlir::omp::ClauseProcBindKindAttr ParallelOp::proc_bind_valAttr() {
  return (*this)->getAttr(proc_bind_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseProcBindKindAttr>();
}

::llvm::Optional<::mlir::omp::ClauseProcBindKind> ParallelOp::proc_bind_val() {
  auto attr = proc_bind_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseProcBindKind>(attr.getValue()) : (::llvm::None);
}

void ParallelOp::proc_bind_valAttr(::mlir::omp::ClauseProcBindKindAttr attr) {
  (*this)->setAttr(proc_bind_valAttrName(), attr);
}

::mlir::Attribute ParallelOp::removeProc_bind_valAttr() {
  return (*this)->removeAttr(proc_bind_valAttrName());
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::omp::ClauseProcBindKindAttr proc_bind_val) {
  if (if_expr_var)
    odsState.addOperands(if_expr_var);
  if (num_threads_var)
    odsState.addOperands(num_threads_var);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr_var ? 1 : 0), (num_threads_var ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
  if (proc_bind_val) {
  odsState.addAttribute(proc_bind_valAttrName(odsState.name), proc_bind_val);
  }
  (void)odsState.addRegion();
}

void ParallelOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::Value if_expr_var, /*optional*/::mlir::Value num_threads_var, ::mlir::ValueRange allocate_vars, ::mlir::ValueRange allocators_vars, /*optional*/::mlir::omp::ClauseProcBindKindAttr proc_bind_val) {
  if (if_expr_var)
    odsState.addOperands(if_expr_var);
  if (num_threads_var)
    odsState.addOperands(num_threads_var);
  odsState.addOperands(allocate_vars);
  odsState.addOperands(allocators_vars);
  odsState.addAttribute(operand_segment_sizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({(if_expr_var ? 1 : 0), (num_threads_var ? 1 : 0), static_cast<int32_t>(allocate_vars.size()), static_cast<int32_t>(allocators_vars.size())}));
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
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 4)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 4 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_proc_bind_val = (*this)->getAttr(proc_bind_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps7(*this, tblgen_proc_bind_val, "proc_bind_val")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    if (valueGroup0.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup0.size();
    }

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    if (valueGroup1.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup1.size();
    }

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
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
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> if_expr_varOperands;
  ::llvm::SMLoc if_expr_varOperandsLoc;
  (void)if_expr_varOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> if_expr_varTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> num_threads_varOperands;
  ::llvm::SMLoc num_threads_varOperandsLoc;
  (void)num_threads_varOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> num_threads_varTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allocate_varsOperands;
  ::llvm::SMLoc allocate_varsOperandsLoc;
  (void)allocate_varsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> allocate_varsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allocators_varsOperands;
  ::llvm::SMLoc allocators_varsOperandsLoc;
  (void)allocators_varsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> allocators_varsTypes;
  ::mlir::omp::ClauseProcBindKindAttr proc_bind_valAttr;
  std::unique_ptr<::mlir::Region> regionRegion = std::make_unique<::mlir::Region>();
  bool ifClause = false;
  bool num_threadsClause = false;
  bool allocateClause = false;
  bool proc_bindClause = false;
  while(true) {
if (succeeded(parser.parseOptionalKeyword("if"))) {

  if (ifClause) {
    return parser.emitError(parser.getNameLoc())
          << "`if` clause can appear at most once in the expansion of the "
             "oilist directive";
  }
  ifClause = true;
  result.addAttribute("if", UnitAttr::get(parser.getContext()));
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    if_expr_varOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
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
  result.addAttribute("num_threads", UnitAttr::get(parser.getContext()));
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    num_threads_varOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
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
  result.addAttribute("allocate", UnitAttr::get(parser.getContext()));
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
  result.addAttribute("proc_bind", UnitAttr::get(parser.getContext()));
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
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({static_cast<int32_t>(if_expr_varOperands.size()), static_cast<int32_t>(num_threads_varOperands.size()), static_cast<int32_t>(allocate_varsOperands.size()), static_cast<int32_t>(allocators_varsOperands.size())}));
  if (parser.resolveOperands(if_expr_varOperands, if_expr_varTypes, if_expr_varOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(num_threads_varOperands, num_threads_varTypes, num_threads_varOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocate_varsOperands, allocate_varsTypes, allocate_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(allocators_varsOperands, allocators_varsTypes, allocators_varsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ParallelOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << " ";
  if ((*this)->hasAttrOfType<UnitAttr>("if")) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = if_expr_var())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (if_expr_var() ? ::llvm::ArrayRef<::mlir::Type>(if_expr_var().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if ((*this)->hasAttrOfType<UnitAttr>("num_threads")) {
  _odsPrinter << ' ' << "num_threads";
  _odsPrinter << "(";
  if (::mlir::Value value = num_threads_var())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (num_threads_var() ? ::llvm::ArrayRef<::mlir::Type>(num_threads_var().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if ((*this)->hasAttrOfType<UnitAttr>("allocate")) {
  _odsPrinter << ' ' << "allocate";
  _odsPrinter << "(";
  printAllocateAndAllocator(_odsPrinter, *this, allocate_vars(), allocate_vars().getTypes(), allocators_vars(), allocators_vars().getTypes());
  _odsPrinter << ")";
  }
  if ((*this)->hasAttrOfType<UnitAttr>("proc_bind")) {
  _odsPrinter << ' ' << "proc_bind";
  _odsPrinter << "(";
  printClauseAttr(_odsPrinter, *this, proc_bind_valAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(region());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "proc_bind", "allocate", "num_threads", "if", "proc_bind_val"});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ParallelOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionDeclareOp definitions
//===----------------------------------------------------------------------===//

ReductionDeclareOpAdaptor::ReductionDeclareOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ReductionDeclareOpAdaptor::ReductionDeclareOpAdaptor(ReductionDeclareOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
  ::mlir::StringAttr attr = odsAttrs.get("sym_name").cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ReductionDeclareOpAdaptor::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

::mlir::TypeAttr ReductionDeclareOpAdaptor::typeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::TypeAttr attr = odsAttrs.get("type").cast<::mlir::TypeAttr>();
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
  {
    auto tblgen_sym_name = odsAttrs.get("sym_name");
    if (!tblgen_sym_name)
      return emitError(loc, "'omp.reduction.declare' op ""requires attribute 'sym_name'");

    if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
      return emitError(loc, "'omp.reduction.declare' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  }
  {
    auto tblgen_type = odsAttrs.get("type");
    if (!tblgen_type)
      return emitError(loc, "'omp.reduction.declare' op ""requires attribute 'type'");

    if (tblgen_type && !(((tblgen_type.isa<::mlir::TypeAttr>())) && ((tblgen_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
      return emitError(loc, "'omp.reduction.declare' op ""attribute 'type' failed to satisfy constraint: any type attribute");
  }
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
  return (*this)->getAttr(sym_nameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef ReductionDeclareOp::sym_name() {
  auto attr = sym_nameAttr();
  return attr.getValue();
}

::mlir::TypeAttr ReductionDeclareOp::typeAttr() {
  return (*this)->getAttr(typeAttrName()).cast<::mlir::TypeAttr>();
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
  {
    auto tblgen_sym_name = (*this)->getAttr(sym_nameAttrName());
    if (!tblgen_sym_name)
      return emitOpError("requires attribute 'sym_name'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps2(*this, tblgen_sym_name, "sym_name")))
      return ::mlir::failure();
  }
  {
    auto tblgen_type = (*this)->getAttr(typeAttrName());
    if (!tblgen_type)
      return emitOpError("requires attribute 'type'");

    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps8(*this, tblgen_type, "type")))
      return ::mlir::failure();
  }
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
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionDeclareOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::ReductionOp definitions
//===----------------------------------------------------------------------===//

ReductionOpAdaptor::ReductionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

ReductionOpAdaptor::ReductionOpAdaptor(ReductionOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
  ::mlir::OpAsmParser::OperandType operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::OpAsmParser::OperandType accumulatorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::OperandType> accumulatorOperands(accumulatorRawOperands);  ::llvm::SMLoc accumulatorOperandsLoc;
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
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::ReductionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SectionOp definitions
//===----------------------------------------------------------------------===//

SectionOpAdaptor::SectionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

SectionOpAdaptor::SectionOpAdaptor(SectionOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::SectionOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::SectionsOp definitions
//===----------------------------------------------------------------------===//

SectionsOpAdaptor::SectionsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

SectionsOpAdaptor::SectionsOpAdaptor(SectionsOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange SectionsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SectionsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

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
  ::mlir::ArrayAttr attr = odsAttrs.get("reductions").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > SectionsOpAdaptor::reductions() {
  auto attr = reductionsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::UnitAttr SectionsOpAdaptor::nowaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("nowait").dyn_cast_or_null<::mlir::UnitAttr>();
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
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'omp.sections' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'omp.sections' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_reductions = odsAttrs.get("reductions");
    if (tblgen_reductions && !(((tblgen_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::SymbolRefAttr>()); }))))
      return emitError(loc, "'omp.sections' op ""attribute 'reductions' failed to satisfy constraint: symbol ref array attribute");
  }
  {
    auto tblgen_nowait = odsAttrs.get("nowait");
    if (tblgen_nowait && !((tblgen_nowait.isa<::mlir::UnitAttr>())))
      return emitError(loc, "'omp.sections' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SectionsOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

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
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SectionsOp::allocate_varsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SectionsOp::allocators_varsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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
  return (*this)->getAttr(reductionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > SectionsOp::reductions() {
  auto attr = reductionsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::UnitAttr SectionsOp::nowaitAttr() {
  return (*this)->getAttr(nowaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
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
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_reductions = (*this)->getAttr(reductionsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps9(*this, tblgen_reductions, "reductions")))
      return ::mlir::failure();
  }
  {
    auto tblgen_nowait = (*this)->getAttr(nowaitAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps6(*this, tblgen_nowait, "nowait")))
      return ::mlir::failure();
  }
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
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> reduction_varsOperands;
  ::llvm::SMLoc reduction_varsOperandsLoc;
  (void)reduction_varsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> reduction_varsTypes;
  ::mlir::ArrayAttr reductionsAttr;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allocate_varsOperands;
  ::llvm::SMLoc allocate_varsOperandsLoc;
  (void)allocate_varsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> allocate_varsTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> allocators_varsOperands;
  ::llvm::SMLoc allocators_varsOperandsLoc;
  (void)allocators_varsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> allocators_varsTypes;
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
  result.addAttribute("reduction", UnitAttr::get(parser.getContext()));
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
  result.addAttribute("allocate", UnitAttr::get(parser.getContext()));
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
  if ((*this)->hasAttrOfType<UnitAttr>("reduction")) {
  _odsPrinter << ' ' << "reduction";
  _odsPrinter << "(";
  printReductionVarList(_odsPrinter, *this, reduction_vars(), reduction_vars().getTypes(), reductionsAttr());
  _odsPrinter << ")";
  }
  if ((*this)->hasAttrOfType<UnitAttr>("allocate")) {
  _odsPrinter << ' ' << "allocate";
  _odsPrinter << "(";
  printAllocateAndAllocator(_odsPrinter, *this, allocate_vars(), allocate_vars().getTypes(), allocators_vars(), allocators_vars().getTypes());
  _odsPrinter << ")";
  }
  if ((*this)->hasAttrOfType<UnitAttr>("nowait")) {
  _odsPrinter << ' ' << "nowait";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(region());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "nowait", "allocate", "reduction", "reductions"});
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::SectionsOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TargetOp definitions
//===----------------------------------------------------------------------===//

TargetOpAdaptor::TargetOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

TargetOpAdaptor::TargetOpAdaptor(TargetOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange TargetOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TargetOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

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
  ::mlir::UnitAttr attr = odsAttrs.get("nowait").dyn_cast_or_null<::mlir::UnitAttr>();
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
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'omp.target' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'omp.target' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_nowait = odsAttrs.get("nowait");
    if (tblgen_nowait && !((tblgen_nowait.isa<::mlir::UnitAttr>())))
      return emitError(loc, "'omp.target' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TargetOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

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
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TargetOp::deviceMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange TargetOp::thread_limitMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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
  return (*this)->getAttr(nowaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
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
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_nowait = (*this)->getAttr(nowaitAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps6(*this, tblgen_nowait, "nowait")))
      return ::mlir::failure();
  }
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

    if (valueGroup2.size() > 1) {
      return emitOpError("operand group starting at #") << index
          << " requires 0 or 1 element, but found " << valueGroup2.size();
    }

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps3(*this, v.getType(), "operand", index++)))
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
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> if_exprOperands;
  ::llvm::SMLoc if_exprOperandsLoc;
  (void)if_exprOperandsLoc;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> deviceOperands;
  ::llvm::SMLoc deviceOperandsLoc;
  (void)deviceOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> deviceTypes;
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> thread_limitOperands;
  ::llvm::SMLoc thread_limitOperandsLoc;
  (void)thread_limitOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> thread_limitTypes;
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
  result.addAttribute("if", UnitAttr::get(parser.getContext()));
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    if_exprOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
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
  result.addAttribute("device", UnitAttr::get(parser.getContext()));
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    deviceOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
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
  result.addAttribute("thread_limit", UnitAttr::get(parser.getContext()));
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    thread_limitOperandsLoc = parser.getCurrentLocation();
    ::mlir::OpAsmParser::OperandType operand;
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
  if ((*this)->hasAttrOfType<UnitAttr>("if")) {
  _odsPrinter << ' ' << "if";
  _odsPrinter << "(";
  if (::mlir::Value value = if_expr())
    _odsPrinter << value;
  _odsPrinter << ")";
  }
  if ((*this)->hasAttrOfType<UnitAttr>("device")) {
  _odsPrinter << ' ' << "device";
  _odsPrinter << "(";
  if (::mlir::Value value = device())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (device() ? ::llvm::ArrayRef<::mlir::Type>(device().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if ((*this)->hasAttrOfType<UnitAttr>("thread_limit")) {
  _odsPrinter << ' ' << "thread_limit";
  _odsPrinter << "(";
  if (::mlir::Value value = thread_limit())
    _odsPrinter << value;
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << (thread_limit() ? ::llvm::ArrayRef<::mlir::Type>(thread_limit().getType()) : ::llvm::ArrayRef<::mlir::Type>());
  _odsPrinter << ")";
  }
  if ((*this)->hasAttrOfType<UnitAttr>("nowait")) {
  _odsPrinter << ' ' << "nowait";
  }
  _odsPrinter << ' ';
  _odsPrinter.printRegion(region());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "device", "nowait", "thread_limit", "if", });
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TargetOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskwaitOp definitions
//===----------------------------------------------------------------------===//

TaskwaitOpAdaptor::TaskwaitOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

TaskwaitOpAdaptor::TaskwaitOpAdaptor(TaskwaitOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TaskwaitOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TaskyieldOp definitions
//===----------------------------------------------------------------------===//

TaskyieldOpAdaptor::TaskyieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

TaskyieldOpAdaptor::TaskyieldOpAdaptor(TaskyieldOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TaskyieldOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::TerminatorOp definitions
//===----------------------------------------------------------------------===//

TerminatorOpAdaptor::TerminatorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

TerminatorOpAdaptor::TerminatorOpAdaptor(TerminatorOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::TerminatorOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::WsLoopOp definitions
//===----------------------------------------------------------------------===//

WsLoopOpAdaptor::WsLoopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

WsLoopOpAdaptor::WsLoopOpAdaptor(WsLoopOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

::mlir::ValueRange WsLoopOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> WsLoopOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = odsAttrs.get("operand_segment_sizes").cast<::mlir::DenseIntElementsAttr>();

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
  ::mlir::ArrayAttr attr = odsAttrs.get("reductions").dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > WsLoopOpAdaptor::reductions() {
  auto attr = reductionsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::omp::ClauseScheduleKindAttr WsLoopOpAdaptor::schedule_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::omp::ClauseScheduleKindAttr attr = odsAttrs.get("schedule_val").dyn_cast_or_null<::mlir::omp::ClauseScheduleKindAttr>();
  return attr;
}

::llvm::Optional<::mlir::omp::ClauseScheduleKind> WsLoopOpAdaptor::schedule_val() {
  auto attr = schedule_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseScheduleKind>(attr.getValue()) : (::llvm::None);
}

::mlir::omp::ScheduleModifierAttr WsLoopOpAdaptor::schedule_modifierAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::omp::ScheduleModifierAttr attr = odsAttrs.get("schedule_modifier").dyn_cast_or_null<::mlir::omp::ScheduleModifierAttr>();
  return attr;
}

::llvm::Optional<::mlir::omp::ScheduleModifier> WsLoopOpAdaptor::schedule_modifier() {
  auto attr = schedule_modifierAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ScheduleModifier>(attr.getValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOpAdaptor::simd_modifierAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("simd_modifier").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool WsLoopOpAdaptor::simd_modifier() {
  auto attr = simd_modifierAttr();
  return attr != nullptr;
}

::mlir::IntegerAttr WsLoopOpAdaptor::collapse_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("collapse_val").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::llvm::Optional<uint64_t> WsLoopOpAdaptor::collapse_val() {
  auto attr = collapse_valAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOpAdaptor::nowaitAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("nowait").dyn_cast_or_null<::mlir::UnitAttr>();
  return attr;
}

bool WsLoopOpAdaptor::nowait() {
  auto attr = nowaitAttr();
  return attr != nullptr;
}

::mlir::IntegerAttr WsLoopOpAdaptor::ordered_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::IntegerAttr attr = odsAttrs.get("ordered_val").dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::llvm::Optional<uint64_t> WsLoopOpAdaptor::ordered_val() {
  auto attr = ordered_valAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::omp::ClauseOrderKindAttr WsLoopOpAdaptor::order_valAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::omp::ClauseOrderKindAttr attr = odsAttrs.get("order_val").dyn_cast_or_null<::mlir::omp::ClauseOrderKindAttr>();
  return attr;
}

::llvm::Optional<::mlir::omp::ClauseOrderKind> WsLoopOpAdaptor::order_val() {
  auto attr = order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseOrderKind>(attr.getValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOpAdaptor::inclusiveAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  ::mlir::UnitAttr attr = odsAttrs.get("inclusive").dyn_cast_or_null<::mlir::UnitAttr>();
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
  {
    auto sizeAttr = odsAttrs.get("operand_segment_sizes").dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitError(loc, "'omp.wsloop' op ""missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 7)
      return emitError(loc, "'omp.wsloop' op ""'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_reductions = odsAttrs.get("reductions");
    if (tblgen_reductions && !(((tblgen_reductions.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_reductions.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return (attr.isa<::mlir::SymbolRefAttr>()); }))))
      return emitError(loc, "'omp.wsloop' op ""attribute 'reductions' failed to satisfy constraint: symbol ref array attribute");
  }
  {
    auto tblgen_schedule_val = odsAttrs.get("schedule_val");
    if (tblgen_schedule_val && !((tblgen_schedule_val.isa<::mlir::omp::ClauseScheduleKindAttr>())))
      return emitError(loc, "'omp.wsloop' op ""attribute 'schedule_val' failed to satisfy constraint: ScheduleKind Clause");
  }
  {
    auto tblgen_schedule_modifier = odsAttrs.get("schedule_modifier");
    if (tblgen_schedule_modifier && !((tblgen_schedule_modifier.isa<::mlir::omp::ScheduleModifierAttr>())))
      return emitError(loc, "'omp.wsloop' op ""attribute 'schedule_modifier' failed to satisfy constraint: OpenMP Schedule Modifier");
  }
  {
    auto tblgen_simd_modifier = odsAttrs.get("simd_modifier");
    if (tblgen_simd_modifier && !((tblgen_simd_modifier.isa<::mlir::UnitAttr>())))
      return emitError(loc, "'omp.wsloop' op ""attribute 'simd_modifier' failed to satisfy constraint: unit attribute");
  }
  {
    auto tblgen_collapse_val = odsAttrs.get("collapse_val");
    if (tblgen_collapse_val && !((((tblgen_collapse_val.isa<::mlir::IntegerAttr>())) && ((tblgen_collapse_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_collapse_val.cast<::mlir::IntegerAttr>().getInt() >= 0))))
      return emitError(loc, "'omp.wsloop' op ""attribute 'collapse_val' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  }
  {
    auto tblgen_nowait = odsAttrs.get("nowait");
    if (tblgen_nowait && !((tblgen_nowait.isa<::mlir::UnitAttr>())))
      return emitError(loc, "'omp.wsloop' op ""attribute 'nowait' failed to satisfy constraint: unit attribute");
  }
  {
    auto tblgen_ordered_val = odsAttrs.get("ordered_val");
    if (tblgen_ordered_val && !((((tblgen_ordered_val.isa<::mlir::IntegerAttr>())) && ((tblgen_ordered_val.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))) && ((tblgen_ordered_val.cast<::mlir::IntegerAttr>().getInt() >= 0))))
      return emitError(loc, "'omp.wsloop' op ""attribute 'ordered_val' failed to satisfy constraint: 64-bit signless integer attribute whose minimum value is 0");
  }
  {
    auto tblgen_order_val = odsAttrs.get("order_val");
    if (tblgen_order_val && !((tblgen_order_val.isa<::mlir::omp::ClauseOrderKindAttr>())))
      return emitError(loc, "'omp.wsloop' op ""attribute 'order_val' failed to satisfy constraint: OrderKind Clause");
  }
  {
    auto tblgen_inclusive = odsAttrs.get("inclusive");
    if (tblgen_inclusive && !((tblgen_inclusive.isa<::mlir::UnitAttr>())))
      return emitError(loc, "'omp.wsloop' op ""attribute 'inclusive' failed to satisfy constraint: unit attribute");
  }
  return ::mlir::success();
}

std::pair<unsigned, unsigned> WsLoopOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr =
      (*this)->getAttr(operand_segment_sizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

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
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::upperBoundMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::stepMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::linear_varsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(3u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::linear_step_varsMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(4u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::reduction_varsMutable() {
  auto range = getODSOperandIndexAndLength(5);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(5u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange WsLoopOp::schedule_chunk_varMutable() {
  auto range = getODSOperandIndexAndLength(6);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(6u, *getOperation()->getAttrDictionary().getNamed(operand_segment_sizesAttrName())));
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
  return (*this)->getAttr(reductionsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > WsLoopOp::reductions() {
  auto attr = reductionsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::omp::ClauseScheduleKindAttr WsLoopOp::schedule_valAttr() {
  return (*this)->getAttr(schedule_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseScheduleKindAttr>();
}

::llvm::Optional<::mlir::omp::ClauseScheduleKind> WsLoopOp::schedule_val() {
  auto attr = schedule_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseScheduleKind>(attr.getValue()) : (::llvm::None);
}

::mlir::omp::ScheduleModifierAttr WsLoopOp::schedule_modifierAttr() {
  return (*this)->getAttr(schedule_modifierAttrName()).dyn_cast_or_null<::mlir::omp::ScheduleModifierAttr>();
}

::llvm::Optional<::mlir::omp::ScheduleModifier> WsLoopOp::schedule_modifier() {
  auto attr = schedule_modifierAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ScheduleModifier>(attr.getValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOp::simd_modifierAttr() {
  return (*this)->getAttr(simd_modifierAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool WsLoopOp::simd_modifier() {
  auto attr = simd_modifierAttr();
  return attr != nullptr;
}

::mlir::IntegerAttr WsLoopOp::collapse_valAttr() {
  return (*this)->getAttr(collapse_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> WsLoopOp::collapse_val() {
  auto attr = collapse_valAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOp::nowaitAttr() {
  return (*this)->getAttr(nowaitAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool WsLoopOp::nowait() {
  auto attr = nowaitAttr();
  return attr != nullptr;
}

::mlir::IntegerAttr WsLoopOp::ordered_valAttr() {
  return (*this)->getAttr(ordered_valAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> WsLoopOp::ordered_val() {
  auto attr = ordered_valAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::omp::ClauseOrderKindAttr WsLoopOp::order_valAttr() {
  return (*this)->getAttr(order_valAttrName()).dyn_cast_or_null<::mlir::omp::ClauseOrderKindAttr>();
}

::llvm::Optional<::mlir::omp::ClauseOrderKind> WsLoopOp::order_val() {
  auto attr = order_valAttr();
  return attr ? ::llvm::Optional<::mlir::omp::ClauseOrderKind>(attr.getValue()) : (::llvm::None);
}

::mlir::UnitAttr WsLoopOp::inclusiveAttr() {
  return (*this)->getAttr(inclusiveAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
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
  {
    auto sizeAttr = (*this)->getAttr(operand_segment_sizesAttrName()).dyn_cast<::mlir::DenseIntElementsAttr>();
    if (!sizeAttr)
      return emitOpError("missing segment sizes attribute 'operand_segment_sizes'");
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 7)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 7 "
                "elements, but got ") << numElements;
  }
    {
    auto tblgen_reductions = (*this)->getAttr(reductionsAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps9(*this, tblgen_reductions, "reductions")))
      return ::mlir::failure();
  }
  {
    auto tblgen_schedule_val = (*this)->getAttr(schedule_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps10(*this, tblgen_schedule_val, "schedule_val")))
      return ::mlir::failure();
  }
  {
    auto tblgen_schedule_modifier = (*this)->getAttr(schedule_modifierAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps11(*this, tblgen_schedule_modifier, "schedule_modifier")))
      return ::mlir::failure();
  }
  {
    auto tblgen_simd_modifier = (*this)->getAttr(simd_modifierAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps6(*this, tblgen_simd_modifier, "simd_modifier")))
      return ::mlir::failure();
  }
  {
    auto tblgen_collapse_val = (*this)->getAttr(collapse_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps5(*this, tblgen_collapse_val, "collapse_val")))
      return ::mlir::failure();
  }
  {
    auto tblgen_nowait = (*this)->getAttr(nowaitAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps6(*this, tblgen_nowait, "nowait")))
      return ::mlir::failure();
  }
  {
    auto tblgen_ordered_val = (*this)->getAttr(ordered_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps5(*this, tblgen_ordered_val, "ordered_val")))
      return ::mlir::failure();
  }
  {
    auto tblgen_order_val = (*this)->getAttr(order_valAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps12(*this, tblgen_order_val, "order_val")))
      return ::mlir::failure();
  }
  {
    auto tblgen_inclusive = (*this)->getAttr(inclusiveAttrName());
    if (::mlir::failed(__mlir_ods_local_attr_constraint_OpenMPOps6(*this, tblgen_inclusive, "inclusive")))
      return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_OpenMPOps1(*this, v.getType(), "operand", index++)))
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

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::WsLoopOp)

namespace mlir {
namespace omp {

//===----------------------------------------------------------------------===//
// ::mlir::omp::YieldOp definitions
//===----------------------------------------------------------------------===//

YieldOpAdaptor::YieldOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {}

YieldOpAdaptor::YieldOpAdaptor(YieldOp &op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()) {}

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
  ::mlir::SmallVector<::mlir::OpAsmParser::OperandType, 4> resultsOperands;
  ::llvm::SMLoc resultsOperandsLoc;
  (void)resultsOperandsLoc;
  ::mlir::SmallVector<::mlir::Type, 1> resultsTypes;
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

void YieldOp::getEffects(::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace omp
} // namespace mlir
DEFINE_EXPLICIT_TYPE_ID(::mlir::omp::YieldOp)


#endif  // GET_OP_CLASSES

