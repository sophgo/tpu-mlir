/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::quant::ConstFakeQuant,
::mlir::quant::ConstFakeQuantPerAxis,
::mlir::quant::CoupledRefOp,
::mlir::quant::DequantizeCastOp,
::mlir::quant::QuantizeCastOp,
::mlir::quant::QuantizeRegionOp,
::mlir::quant::ReturnOp,
::mlir::quant::StatisticsOp,
::mlir::quant::StatisticsRefOp,
::mlir::quant::StorageCastOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace quant {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_QuantOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (elementType.isF32()); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 32-bit float values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_QuantOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::FloatType>())) || ((type.isa<mlir::quant::QuantizedType>()))) || (((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return ((elementType.isa<::mlir::FloatType>())) || ((elementType.isa<mlir::quant::QuantizedType>())); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isa<::mlir::FloatType>())) || ((elementType.isa<mlir::quant::QuantizedType>())); }(type.cast<::mlir::ShapedType>().getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be primitive/tensor/vector of real valued primitive (float or quantized type), but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_QuantOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_QuantOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return (true); }(type.cast<::mlir::ShapedType>().getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of any type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_QuantOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((type.isa<::mlir::FloatType>())) || ((type.isa<mlir::quant::QuantizedType>()))) || (((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return ((elementType.isa<::mlir::FloatType>())) || ((elementType.isa<mlir::quant::QuantizedType>())); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isa<::mlir::FloatType>())) || ((elementType.isa<mlir::quant::QuantizedType>())); }(type.cast<::mlir::ShapedType>().getElementType())))) || ((((type.isSignlessInteger())) || ((type.isa<mlir::quant::QuantizedType>()))) || (((type.isa<::mlir::TensorType>())) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((elementType.isa<mlir::quant::QuantizedType>())); }(type.cast<::mlir::ShapedType>().getElementType()))) || ((((type.isa<::mlir::VectorType>())) && ((type.cast<::mlir::VectorType>().getRank() > 0))) && ([](::mlir::Type elementType) { return ((elementType.isSignlessInteger())) || ((elementType.isa<mlir::quant::QuantizedType>())); }(type.cast<::mlir::ShapedType>().getElementType())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be , but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_QuantOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::FloatAttr>())) && ((attr.cast<::mlir::FloatAttr>().getType().isF32())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit float attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_QuantOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_QuantOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_QuantOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::FloatAttr>())) && ((attr.cast<::mlir::FloatAttr>().getType().isF32()))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit float array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_QuantOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_QuantOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: type array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_QuantOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ElementsAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: constant vector/tensor attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_QuantOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}
} // namespace quant
} // namespace mlir
namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::ConstFakeQuant definitions
//===----------------------------------------------------------------------===//

ConstFakeQuantAdaptor::ConstFakeQuantAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("quant.const_fake_quant", odsAttrs.getContext());
}

ConstFakeQuantAdaptor::ConstFakeQuantAdaptor(ConstFakeQuant op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ConstFakeQuantAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConstFakeQuantAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstFakeQuantAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstFakeQuantAdaptor::inputs() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ConstFakeQuantAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::FloatAttr ConstFakeQuantAdaptor::minAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, ConstFakeQuant::minAttrName(*odsOpName)).cast<::mlir::FloatAttr>();
    return attr;
}

::llvm::APFloat ConstFakeQuantAdaptor::min() {
  auto attr = minAttr();
  return attr.getValue();
}

::mlir::FloatAttr ConstFakeQuantAdaptor::maxAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, ConstFakeQuant::maxAttrName(*odsOpName)).cast<::mlir::FloatAttr>();
    return attr;
}

::llvm::APFloat ConstFakeQuantAdaptor::max() {
  auto attr = maxAttr();
  return attr.getValue();
}

::mlir::IntegerAttr ConstFakeQuantAdaptor::num_bitsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, ConstFakeQuant::num_bitsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint64_t ConstFakeQuantAdaptor::num_bits() {
  auto attr = num_bitsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr ConstFakeQuantAdaptor::narrow_rangeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, ConstFakeQuant::narrow_rangeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
    return attr;
}

bool ConstFakeQuantAdaptor::narrow_range() {
  auto attr = narrow_rangeAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

::mlir::BoolAttr ConstFakeQuantAdaptor::is_signedAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, ConstFakeQuant::is_signedAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
    return attr;
}

bool ConstFakeQuantAdaptor::is_signed() {
  auto attr = is_signedAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

::mlir::LogicalResult ConstFakeQuantAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_max;
  ::mlir::Attribute tblgen_is_signed;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'quant.const_fake_quant' op ""requires attribute 'max'");
    if (namedAttrIt->getName() == ConstFakeQuant::maxAttrName(*odsOpName)) {
      tblgen_max = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == ConstFakeQuant::is_signedAttrName(*odsOpName)) {
      tblgen_is_signed = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_min;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'quant.const_fake_quant' op ""requires attribute 'min'");
    if (namedAttrIt->getName() == ConstFakeQuant::minAttrName(*odsOpName)) {
      tblgen_min = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_num_bits;
  ::mlir::Attribute tblgen_narrow_range;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'quant.const_fake_quant' op ""requires attribute 'num_bits'");
    if (namedAttrIt->getName() == ConstFakeQuant::num_bitsAttrName(*odsOpName)) {
      tblgen_num_bits = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == ConstFakeQuant::narrow_rangeAttrName(*odsOpName)) {
      tblgen_narrow_range = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_min && !(((tblgen_min.isa<::mlir::FloatAttr>())) && ((tblgen_min.cast<::mlir::FloatAttr>().getType().isF32()))))
    return emitError(loc, "'quant.const_fake_quant' op ""attribute 'min' failed to satisfy constraint: 32-bit float attribute");

  if (tblgen_max && !(((tblgen_max.isa<::mlir::FloatAttr>())) && ((tblgen_max.cast<::mlir::FloatAttr>().getType().isF32()))))
    return emitError(loc, "'quant.const_fake_quant' op ""attribute 'max' failed to satisfy constraint: 32-bit float attribute");

  if (tblgen_num_bits && !(((tblgen_num_bits.isa<::mlir::IntegerAttr>())) && ((tblgen_num_bits.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'quant.const_fake_quant' op ""attribute 'num_bits' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_narrow_range && !((tblgen_narrow_range.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'quant.const_fake_quant' op ""attribute 'narrow_range' failed to satisfy constraint: bool attribute");

  if (tblgen_is_signed && !((tblgen_is_signed.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'quant.const_fake_quant' op ""attribute 'is_signed' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConstFakeQuant::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstFakeQuant::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstFakeQuant::inputs() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ConstFakeQuant::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ConstFakeQuant::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstFakeQuant::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstFakeQuant::outputs() {
  return *getODSResults(0).begin();
}

::mlir::FloatAttr ConstFakeQuant::minAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, minAttrName()).cast<::mlir::FloatAttr>();
}

::llvm::APFloat ConstFakeQuant::min() {
  auto attr = minAttr();
  return attr.getValue();
}

::mlir::FloatAttr ConstFakeQuant::maxAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, maxAttrName()).cast<::mlir::FloatAttr>();
}

::llvm::APFloat ConstFakeQuant::max() {
  auto attr = maxAttr();
  return attr.getValue();
}

::mlir::IntegerAttr ConstFakeQuant::num_bitsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, num_bitsAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ConstFakeQuant::num_bits() {
  auto attr = num_bitsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr ConstFakeQuant::narrow_rangeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, narrow_rangeAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool ConstFakeQuant::narrow_range() {
  auto attr = narrow_rangeAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

::mlir::BoolAttr ConstFakeQuant::is_signedAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, is_signedAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool ConstFakeQuant::is_signed() {
  auto attr = is_signedAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

void ConstFakeQuant::minAttr(::mlir::FloatAttr attr) {
  (*this)->setAttr(minAttrName(), attr);
}

void ConstFakeQuant::maxAttr(::mlir::FloatAttr attr) {
  (*this)->setAttr(maxAttrName(), attr);
}

void ConstFakeQuant::num_bitsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(num_bitsAttrName(), attr);
}

void ConstFakeQuant::narrow_rangeAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(narrow_rangeAttrName(), attr);
}

void ConstFakeQuant::is_signedAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(is_signedAttrName(), attr);
}

void ConstFakeQuant::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputs, ::mlir::Value inputs, ::mlir::FloatAttr min, ::mlir::FloatAttr max, ::mlir::IntegerAttr num_bits, ::mlir::BoolAttr narrow_range, ::mlir::BoolAttr is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute(minAttrName(odsState.name), min);
  odsState.addAttribute(maxAttrName(odsState.name), max);
  odsState.addAttribute(num_bitsAttrName(odsState.name), num_bits);
  if (narrow_range) {
  odsState.addAttribute(narrow_rangeAttrName(odsState.name), narrow_range);
  }
  if (is_signed) {
  odsState.addAttribute(is_signedAttrName(odsState.name), is_signed);
  }
  odsState.addTypes(outputs);
}

void ConstFakeQuant::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputs, ::mlir::FloatAttr min, ::mlir::FloatAttr max, ::mlir::IntegerAttr num_bits, ::mlir::BoolAttr narrow_range, ::mlir::BoolAttr is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute(minAttrName(odsState.name), min);
  odsState.addAttribute(maxAttrName(odsState.name), max);
  odsState.addAttribute(num_bitsAttrName(odsState.name), num_bits);
  if (narrow_range) {
  odsState.addAttribute(narrow_rangeAttrName(odsState.name), narrow_range);
  }
  if (is_signed) {
  odsState.addAttribute(is_signedAttrName(odsState.name), is_signed);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ConstFakeQuant::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ConstFakeQuant::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputs, ::mlir::FloatAttr min, ::mlir::FloatAttr max, ::mlir::IntegerAttr num_bits, ::mlir::BoolAttr narrow_range, ::mlir::BoolAttr is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute(minAttrName(odsState.name), min);
  odsState.addAttribute(maxAttrName(odsState.name), max);
  odsState.addAttribute(num_bitsAttrName(odsState.name), num_bits);
  if (narrow_range) {
  odsState.addAttribute(narrow_rangeAttrName(odsState.name), narrow_range);
  }
  if (is_signed) {
  odsState.addAttribute(is_signedAttrName(odsState.name), is_signed);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstFakeQuant::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputs, ::mlir::Value inputs, ::llvm::APFloat min, ::llvm::APFloat max, uint64_t num_bits, bool narrow_range, bool is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute(minAttrName(odsState.name), odsBuilder.getFloatAttr(odsBuilder.getF32Type(), min));
  odsState.addAttribute(maxAttrName(odsState.name), odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max));
  odsState.addAttribute(num_bitsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_bits));
  odsState.addAttribute(narrow_rangeAttrName(odsState.name), odsBuilder.getBoolAttr(narrow_range));
  odsState.addAttribute(is_signedAttrName(odsState.name), odsBuilder.getBoolAttr(is_signed));
  odsState.addTypes(outputs);
}

void ConstFakeQuant::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputs, ::llvm::APFloat min, ::llvm::APFloat max, uint64_t num_bits, bool narrow_range, bool is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute(minAttrName(odsState.name), odsBuilder.getFloatAttr(odsBuilder.getF32Type(), min));
  odsState.addAttribute(maxAttrName(odsState.name), odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max));
  odsState.addAttribute(num_bitsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_bits));
  odsState.addAttribute(narrow_rangeAttrName(odsState.name), odsBuilder.getBoolAttr(narrow_range));
  odsState.addAttribute(is_signedAttrName(odsState.name), odsBuilder.getBoolAttr(is_signed));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ConstFakeQuant::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ConstFakeQuant::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputs, ::llvm::APFloat min, ::llvm::APFloat max, uint64_t num_bits, bool narrow_range, bool is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute(minAttrName(odsState.name), odsBuilder.getFloatAttr(odsBuilder.getF32Type(), min));
  odsState.addAttribute(maxAttrName(odsState.name), odsBuilder.getFloatAttr(odsBuilder.getF32Type(), max));
  odsState.addAttribute(num_bitsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_bits));
  odsState.addAttribute(narrow_rangeAttrName(odsState.name), odsBuilder.getBoolAttr(narrow_range));
  odsState.addAttribute(is_signedAttrName(odsState.name), odsBuilder.getBoolAttr(is_signed));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstFakeQuant::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ConstFakeQuant::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ConstFakeQuant::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ConstFakeQuant::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_max;
  ::mlir::Attribute tblgen_is_signed;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'max'");
    if (namedAttrIt->getName() == maxAttrName()) {
      tblgen_max = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == is_signedAttrName()) {
      tblgen_is_signed = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_min;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'min'");
    if (namedAttrIt->getName() == minAttrName()) {
      tblgen_min = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_num_bits;
  ::mlir::Attribute tblgen_narrow_range;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'num_bits'");
    if (namedAttrIt->getName() == num_bitsAttrName()) {
      tblgen_num_bits = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == narrow_rangeAttrName()) {
      tblgen_narrow_range = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps0(*this, tblgen_min, "min")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps0(*this, tblgen_max, "max")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps1(*this, tblgen_num_bits, "num_bits")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps2(*this, tblgen_narrow_range, "narrow_range")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps2(*this, tblgen_is_signed, "is_signed")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstFakeQuant::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult ConstFakeQuant::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void ConstFakeQuant::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace quant
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::ConstFakeQuant)

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::ConstFakeQuantPerAxis definitions
//===----------------------------------------------------------------------===//

ConstFakeQuantPerAxisAdaptor::ConstFakeQuantPerAxisAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("quant.const_fake_quant_per_axis", odsAttrs.getContext());
}

ConstFakeQuantPerAxisAdaptor::ConstFakeQuantPerAxisAdaptor(ConstFakeQuantPerAxis op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ConstFakeQuantPerAxisAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConstFakeQuantPerAxisAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstFakeQuantPerAxisAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstFakeQuantPerAxisAdaptor::inputs() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ConstFakeQuantPerAxisAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ConstFakeQuantPerAxisAdaptor::minAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, ConstFakeQuantPerAxis::minAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
    return attr;
}

::mlir::ArrayAttr ConstFakeQuantPerAxisAdaptor::min() {
  auto attr = minAttr();
  return attr;
}

::mlir::ArrayAttr ConstFakeQuantPerAxisAdaptor::maxAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 2, ConstFakeQuantPerAxis::maxAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
    return attr;
}

::mlir::ArrayAttr ConstFakeQuantPerAxisAdaptor::max() {
  auto attr = maxAttr();
  return attr;
}

::mlir::IntegerAttr ConstFakeQuantPerAxisAdaptor::axisAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, ConstFakeQuantPerAxis::axisAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint64_t ConstFakeQuantPerAxisAdaptor::axis() {
  auto attr = axisAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConstFakeQuantPerAxisAdaptor::num_bitsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, ConstFakeQuantPerAxis::num_bitsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint64_t ConstFakeQuantPerAxisAdaptor::num_bits() {
  auto attr = num_bitsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr ConstFakeQuantPerAxisAdaptor::narrow_rangeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 1, ConstFakeQuantPerAxis::narrow_rangeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
    return attr;
}

bool ConstFakeQuantPerAxisAdaptor::narrow_range() {
  auto attr = narrow_rangeAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

::mlir::BoolAttr ConstFakeQuantPerAxisAdaptor::is_signedAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 3, ConstFakeQuantPerAxis::is_signedAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
    return attr;
}

bool ConstFakeQuantPerAxisAdaptor::is_signed() {
  auto attr = is_signedAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

::mlir::LogicalResult ConstFakeQuantPerAxisAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'quant.const_fake_quant_per_axis' op ""requires attribute 'axis'");
    if (namedAttrIt->getName() == ConstFakeQuantPerAxis::axisAttrName(*odsOpName)) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_max;
  ::mlir::Attribute tblgen_is_signed;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'quant.const_fake_quant_per_axis' op ""requires attribute 'max'");
    if (namedAttrIt->getName() == ConstFakeQuantPerAxis::maxAttrName(*odsOpName)) {
      tblgen_max = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == ConstFakeQuantPerAxis::is_signedAttrName(*odsOpName)) {
      tblgen_is_signed = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_min;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'quant.const_fake_quant_per_axis' op ""requires attribute 'min'");
    if (namedAttrIt->getName() == ConstFakeQuantPerAxis::minAttrName(*odsOpName)) {
      tblgen_min = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_num_bits;
  ::mlir::Attribute tblgen_narrow_range;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'quant.const_fake_quant_per_axis' op ""requires attribute 'num_bits'");
    if (namedAttrIt->getName() == ConstFakeQuantPerAxis::num_bitsAttrName(*odsOpName)) {
      tblgen_num_bits = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == ConstFakeQuantPerAxis::narrow_rangeAttrName(*odsOpName)) {
      tblgen_narrow_range = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_min && !(((tblgen_min.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_min.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::FloatAttr>())) && ((attr.cast<::mlir::FloatAttr>().getType().isF32()))); }))))
    return emitError(loc, "'quant.const_fake_quant_per_axis' op ""attribute 'min' failed to satisfy constraint: 32-bit float array attribute");

  if (tblgen_max && !(((tblgen_max.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_max.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::FloatAttr>())) && ((attr.cast<::mlir::FloatAttr>().getType().isF32()))); }))))
    return emitError(loc, "'quant.const_fake_quant_per_axis' op ""attribute 'max' failed to satisfy constraint: 32-bit float array attribute");

  if (tblgen_axis && !(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'quant.const_fake_quant_per_axis' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_num_bits && !(((tblgen_num_bits.isa<::mlir::IntegerAttr>())) && ((tblgen_num_bits.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'quant.const_fake_quant_per_axis' op ""attribute 'num_bits' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_narrow_range && !((tblgen_narrow_range.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'quant.const_fake_quant_per_axis' op ""attribute 'narrow_range' failed to satisfy constraint: bool attribute");

  if (tblgen_is_signed && !((tblgen_is_signed.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'quant.const_fake_quant_per_axis' op ""attribute 'is_signed' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConstFakeQuantPerAxis::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstFakeQuantPerAxis::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstFakeQuantPerAxis::inputs() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ConstFakeQuantPerAxis::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ConstFakeQuantPerAxis::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstFakeQuantPerAxis::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstFakeQuantPerAxis::outputs() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr ConstFakeQuantPerAxis::minAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, minAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ConstFakeQuantPerAxis::min() {
  auto attr = minAttr();
  return attr;
}

::mlir::ArrayAttr ConstFakeQuantPerAxis::maxAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 2, maxAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ConstFakeQuantPerAxis::max() {
  auto attr = maxAttr();
  return attr;
}

::mlir::IntegerAttr ConstFakeQuantPerAxis::axisAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, axisAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ConstFakeQuantPerAxis::axis() {
  auto attr = axisAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr ConstFakeQuantPerAxis::num_bitsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, num_bitsAttrName()).cast<::mlir::IntegerAttr>();
}

uint64_t ConstFakeQuantPerAxis::num_bits() {
  auto attr = num_bitsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::BoolAttr ConstFakeQuantPerAxis::narrow_rangeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 1, narrow_rangeAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool ConstFakeQuantPerAxis::narrow_range() {
  auto attr = narrow_rangeAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

::mlir::BoolAttr ConstFakeQuantPerAxis::is_signedAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 3, is_signedAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool ConstFakeQuantPerAxis::is_signed() {
  auto attr = is_signedAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

void ConstFakeQuantPerAxis::minAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(minAttrName(), attr);
}

void ConstFakeQuantPerAxis::maxAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(maxAttrName(), attr);
}

void ConstFakeQuantPerAxis::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(axisAttrName(), attr);
}

void ConstFakeQuantPerAxis::num_bitsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(num_bitsAttrName(), attr);
}

void ConstFakeQuantPerAxis::narrow_rangeAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(narrow_rangeAttrName(), attr);
}

void ConstFakeQuantPerAxis::is_signedAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(is_signedAttrName(), attr);
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputs, ::mlir::Value inputs, ::mlir::ArrayAttr min, ::mlir::ArrayAttr max, ::mlir::IntegerAttr axis, ::mlir::IntegerAttr num_bits, ::mlir::BoolAttr narrow_range, ::mlir::BoolAttr is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute(minAttrName(odsState.name), min);
  odsState.addAttribute(maxAttrName(odsState.name), max);
  odsState.addAttribute(axisAttrName(odsState.name), axis);
  odsState.addAttribute(num_bitsAttrName(odsState.name), num_bits);
  if (narrow_range) {
  odsState.addAttribute(narrow_rangeAttrName(odsState.name), narrow_range);
  }
  if (is_signed) {
  odsState.addAttribute(is_signedAttrName(odsState.name), is_signed);
  }
  odsState.addTypes(outputs);
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputs, ::mlir::ArrayAttr min, ::mlir::ArrayAttr max, ::mlir::IntegerAttr axis, ::mlir::IntegerAttr num_bits, ::mlir::BoolAttr narrow_range, ::mlir::BoolAttr is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute(minAttrName(odsState.name), min);
  odsState.addAttribute(maxAttrName(odsState.name), max);
  odsState.addAttribute(axisAttrName(odsState.name), axis);
  odsState.addAttribute(num_bitsAttrName(odsState.name), num_bits);
  if (narrow_range) {
  odsState.addAttribute(narrow_rangeAttrName(odsState.name), narrow_range);
  }
  if (is_signed) {
  odsState.addAttribute(is_signedAttrName(odsState.name), is_signed);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ConstFakeQuantPerAxis::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputs, ::mlir::ArrayAttr min, ::mlir::ArrayAttr max, ::mlir::IntegerAttr axis, ::mlir::IntegerAttr num_bits, ::mlir::BoolAttr narrow_range, ::mlir::BoolAttr is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute(minAttrName(odsState.name), min);
  odsState.addAttribute(maxAttrName(odsState.name), max);
  odsState.addAttribute(axisAttrName(odsState.name), axis);
  odsState.addAttribute(num_bitsAttrName(odsState.name), num_bits);
  if (narrow_range) {
  odsState.addAttribute(narrow_rangeAttrName(odsState.name), narrow_range);
  }
  if (is_signed) {
  odsState.addAttribute(is_signedAttrName(odsState.name), is_signed);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type outputs, ::mlir::Value inputs, ::mlir::ArrayAttr min, ::mlir::ArrayAttr max, uint64_t axis, uint64_t num_bits, bool narrow_range, bool is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute(minAttrName(odsState.name), min);
  odsState.addAttribute(maxAttrName(odsState.name), max);
  odsState.addAttribute(axisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addAttribute(num_bitsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_bits));
  odsState.addAttribute(narrow_rangeAttrName(odsState.name), odsBuilder.getBoolAttr(narrow_range));
  odsState.addAttribute(is_signedAttrName(odsState.name), odsBuilder.getBoolAttr(is_signed));
  odsState.addTypes(outputs);
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputs, ::mlir::ArrayAttr min, ::mlir::ArrayAttr max, uint64_t axis, uint64_t num_bits, bool narrow_range, bool is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute(minAttrName(odsState.name), min);
  odsState.addAttribute(maxAttrName(odsState.name), max);
  odsState.addAttribute(axisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addAttribute(num_bitsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_bits));
  odsState.addAttribute(narrow_rangeAttrName(odsState.name), odsBuilder.getBoolAttr(narrow_range));
  odsState.addAttribute(is_signedAttrName(odsState.name), odsBuilder.getBoolAttr(is_signed));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ConstFakeQuantPerAxis::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputs, ::mlir::ArrayAttr min, ::mlir::ArrayAttr max, uint64_t axis, uint64_t num_bits, bool narrow_range, bool is_signed) {
  odsState.addOperands(inputs);
  odsState.addAttribute(minAttrName(odsState.name), min);
  odsState.addAttribute(maxAttrName(odsState.name), max);
  odsState.addAttribute(axisAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), axis));
  odsState.addAttribute(num_bitsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(64), num_bits));
  odsState.addAttribute(narrow_rangeAttrName(odsState.name), odsBuilder.getBoolAttr(narrow_range));
  odsState.addAttribute(is_signedAttrName(odsState.name), odsBuilder.getBoolAttr(is_signed));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ConstFakeQuantPerAxis::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ConstFakeQuantPerAxis::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ConstFakeQuantPerAxis::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_axis;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'axis'");
    if (namedAttrIt->getName() == axisAttrName()) {
      tblgen_axis = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_max;
  ::mlir::Attribute tblgen_is_signed;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'max'");
    if (namedAttrIt->getName() == maxAttrName()) {
      tblgen_max = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == is_signedAttrName()) {
      tblgen_is_signed = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_min;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'min'");
    if (namedAttrIt->getName() == minAttrName()) {
      tblgen_min = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_num_bits;
  ::mlir::Attribute tblgen_narrow_range;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'num_bits'");
    if (namedAttrIt->getName() == num_bitsAttrName()) {
      tblgen_num_bits = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == narrow_rangeAttrName()) {
      tblgen_narrow_range = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps3(*this, tblgen_min, "min")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps3(*this, tblgen_max, "max")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps1(*this, tblgen_axis, "axis")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps1(*this, tblgen_num_bits, "num_bits")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps2(*this, tblgen_narrow_range, "narrow_range")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps2(*this, tblgen_is_signed, "is_signed")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstFakeQuantPerAxis::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult ConstFakeQuantPerAxis::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void ConstFakeQuantPerAxis::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace quant
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::ConstFakeQuantPerAxis)

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::CoupledRefOp definitions
//===----------------------------------------------------------------------===//

CoupledRefOpAdaptor::CoupledRefOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("quant.coupled_ref", odsAttrs.getContext());
}

CoupledRefOpAdaptor::CoupledRefOpAdaptor(CoupledRefOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CoupledRefOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoupledRefOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoupledRefOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoupledRefOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CoupledRefOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CoupledRefOpAdaptor::coupledKeyAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CoupledRefOp::coupledKeyAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef CoupledRefOpAdaptor::coupledKey() {
  auto attr = coupledKeyAttr();
  return attr.getValue();
}

::mlir::LogicalResult CoupledRefOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_coupledKey;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'quant.coupled_ref' op ""requires attribute 'coupledKey'");
    if (namedAttrIt->getName() == CoupledRefOp::coupledKeyAttrName(*odsOpName)) {
      tblgen_coupledKey = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_coupledKey && !((tblgen_coupledKey.isa<::mlir::StringAttr>())))
    return emitError(loc, "'quant.coupled_ref' op ""attribute 'coupledKey' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CoupledRefOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoupledRefOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoupledRefOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CoupledRefOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CoupledRefOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoupledRefOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr CoupledRefOp::coupledKeyAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, coupledKeyAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef CoupledRefOp::coupledKey() {
  auto attr = coupledKeyAttr();
  return attr.getValue();
}

void CoupledRefOp::coupledKeyAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(coupledKeyAttrName(), attr);
}

void CoupledRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg, ::mlir::StringAttr coupledKey) {
  odsState.addOperands(arg);
  odsState.addAttribute(coupledKeyAttrName(odsState.name), coupledKey);
  odsState.addTypes(resultType0);
}

void CoupledRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::mlir::StringAttr coupledKey) {
  odsState.addOperands(arg);
  odsState.addAttribute(coupledKeyAttrName(odsState.name), coupledKey);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(CoupledRefOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void CoupledRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::mlir::StringAttr coupledKey) {
  odsState.addOperands(arg);
  odsState.addAttribute(coupledKeyAttrName(odsState.name), coupledKey);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoupledRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg, ::llvm::StringRef coupledKey) {
  odsState.addOperands(arg);
  odsState.addAttribute(coupledKeyAttrName(odsState.name), odsBuilder.getStringAttr(coupledKey));
  odsState.addTypes(resultType0);
}

void CoupledRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::llvm::StringRef coupledKey) {
  odsState.addOperands(arg);
  odsState.addAttribute(coupledKeyAttrName(odsState.name), odsBuilder.getStringAttr(coupledKey));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(CoupledRefOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void CoupledRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::llvm::StringRef coupledKey) {
  odsState.addOperands(arg);
  odsState.addAttribute(coupledKeyAttrName(odsState.name), odsBuilder.getStringAttr(coupledKey));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoupledRefOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CoupledRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(CoupledRefOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult CoupledRefOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_coupledKey;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'coupledKey'");
    if (namedAttrIt->getName() == coupledKeyAttrName()) {
      tblgen_coupledKey = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps4(*this, tblgen_coupledKey, "coupledKey")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CoupledRefOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult CoupledRefOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

} // namespace quant
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::CoupledRefOp)

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::DequantizeCastOp definitions
//===----------------------------------------------------------------------===//

DequantizeCastOpAdaptor::DequantizeCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("quant.dcast", odsAttrs.getContext());
}

DequantizeCastOpAdaptor::DequantizeCastOpAdaptor(DequantizeCastOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange DequantizeCastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> DequantizeCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange DequantizeCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DequantizeCastOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr DequantizeCastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult DequantizeCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> DequantizeCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range DequantizeCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value DequantizeCastOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange DequantizeCastOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> DequantizeCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range DequantizeCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void DequantizeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(resultType0);
}

void DequantizeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void DequantizeCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult DequantizeCastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult DequantizeCastOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void DequantizeCastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace quant
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::DequantizeCastOp)

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::QuantizeCastOp definitions
//===----------------------------------------------------------------------===//

QuantizeCastOpAdaptor::QuantizeCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("quant.qcast", odsAttrs.getContext());
}

QuantizeCastOpAdaptor::QuantizeCastOpAdaptor(QuantizeCastOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange QuantizeCastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> QuantizeCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange QuantizeCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value QuantizeCastOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr QuantizeCastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult QuantizeCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> QuantizeCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range QuantizeCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value QuantizeCastOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange QuantizeCastOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> QuantizeCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range QuantizeCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void QuantizeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(resultType0);
}

void QuantizeCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void QuantizeCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult QuantizeCastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult QuantizeCastOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void QuantizeCastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace quant
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::QuantizeCastOp)

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::QuantizeRegionOp definitions
//===----------------------------------------------------------------------===//

QuantizeRegionOpAdaptor::QuantizeRegionOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("quant.region", odsAttrs.getContext());
}

QuantizeRegionOpAdaptor::QuantizeRegionOpAdaptor(QuantizeRegionOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange QuantizeRegionOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> QuantizeRegionOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange QuantizeRegionOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange QuantizeRegionOpAdaptor::inputs() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr QuantizeRegionOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr QuantizeRegionOpAdaptor::input_specsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, QuantizeRegionOp::input_specsAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
    return attr;
}

::mlir::ArrayAttr QuantizeRegionOpAdaptor::input_specs() {
  auto attr = input_specsAttr();
  return attr;
}

::mlir::ArrayAttr QuantizeRegionOpAdaptor::output_specsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, QuantizeRegionOp::output_specsAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
    return attr;
}

::mlir::ArrayAttr QuantizeRegionOpAdaptor::output_specs() {
  auto attr = output_specsAttr();
  return attr;
}

::mlir::StringAttr QuantizeRegionOpAdaptor::logical_kernelAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, QuantizeRegionOp::logical_kernelAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef QuantizeRegionOpAdaptor::logical_kernel() {
  auto attr = logical_kernelAttr();
  return attr.getValue();
}

::mlir::RegionRange QuantizeRegionOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &QuantizeRegionOpAdaptor::body() {
  return *odsRegions[0];
}

::mlir::LogicalResult QuantizeRegionOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_input_specs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'quant.region' op ""requires attribute 'input_specs'");
    if (namedAttrIt->getName() == QuantizeRegionOp::input_specsAttrName(*odsOpName)) {
      tblgen_input_specs = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_logical_kernel;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'quant.region' op ""requires attribute 'logical_kernel'");
    if (namedAttrIt->getName() == QuantizeRegionOp::logical_kernelAttrName(*odsOpName)) {
      tblgen_logical_kernel = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_output_specs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'quant.region' op ""requires attribute 'output_specs'");
    if (namedAttrIt->getName() == QuantizeRegionOp::output_specsAttrName(*odsOpName)) {
      tblgen_output_specs = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_input_specs && !(((tblgen_input_specs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_input_specs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'quant.region' op ""attribute 'input_specs' failed to satisfy constraint: type array attribute");

  if (tblgen_output_specs && !(((tblgen_output_specs.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_output_specs.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))); }))))
    return emitError(loc, "'quant.region' op ""attribute 'output_specs' failed to satisfy constraint: type array attribute");

  if (tblgen_logical_kernel && !((tblgen_logical_kernel.isa<::mlir::StringAttr>())))
    return emitError(loc, "'quant.region' op ""attribute 'logical_kernel' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> QuantizeRegionOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range QuantizeRegionOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range QuantizeRegionOp::inputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange QuantizeRegionOp::inputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> QuantizeRegionOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range QuantizeRegionOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range QuantizeRegionOp::outputs() {
  return getODSResults(0);
}

::mlir::Region &QuantizeRegionOp::body() {
  return (*this)->getRegion(0);
}

::mlir::ArrayAttr QuantizeRegionOp::input_specsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, input_specsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr QuantizeRegionOp::input_specs() {
  auto attr = input_specsAttr();
  return attr;
}

::mlir::ArrayAttr QuantizeRegionOp::output_specsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, output_specsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr QuantizeRegionOp::output_specs() {
  auto attr = output_specsAttr();
  return attr;
}

::mlir::StringAttr QuantizeRegionOp::logical_kernelAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, logical_kernelAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef QuantizeRegionOp::logical_kernel() {
  auto attr = logical_kernelAttr();
  return attr.getValue();
}

void QuantizeRegionOp::input_specsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(input_specsAttrName(), attr);
}

void QuantizeRegionOp::output_specsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(output_specsAttrName(), attr);
}

void QuantizeRegionOp::logical_kernelAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(logical_kernelAttrName(), attr);
}

void QuantizeRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, ::mlir::ArrayAttr input_specs, ::mlir::ArrayAttr output_specs, ::mlir::StringAttr logical_kernel) {
  odsState.addOperands(inputs);
  odsState.addAttribute(input_specsAttrName(odsState.name), input_specs);
  odsState.addAttribute(output_specsAttrName(odsState.name), output_specs);
  odsState.addAttribute(logical_kernelAttrName(odsState.name), logical_kernel);
  (void)odsState.addRegion();
  odsState.addTypes(outputs);
}

void QuantizeRegionOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange outputs, ::mlir::ValueRange inputs, ::mlir::ArrayAttr input_specs, ::mlir::ArrayAttr output_specs, ::llvm::StringRef logical_kernel) {
  odsState.addOperands(inputs);
  odsState.addAttribute(input_specsAttrName(odsState.name), input_specs);
  odsState.addAttribute(output_specsAttrName(odsState.name), output_specs);
  odsState.addAttribute(logical_kernelAttrName(odsState.name), odsBuilder.getStringAttr(logical_kernel));
  (void)odsState.addRegion();
  odsState.addTypes(outputs);
}

void QuantizeRegionOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult QuantizeRegionOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_input_specs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'input_specs'");
    if (namedAttrIt->getName() == input_specsAttrName()) {
      tblgen_input_specs = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_logical_kernel;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'logical_kernel'");
    if (namedAttrIt->getName() == logical_kernelAttrName()) {
      tblgen_logical_kernel = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_output_specs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'output_specs'");
    if (namedAttrIt->getName() == output_specsAttrName()) {
      tblgen_output_specs = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps5(*this, tblgen_input_specs, "input_specs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps5(*this, tblgen_output_specs, "output_specs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps4(*this, tblgen_logical_kernel, "logical_kernel")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_QuantOps0(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult QuantizeRegionOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void QuantizeRegionOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace quant
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::QuantizeRegionOp)

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::ReturnOp definitions
//===----------------------------------------------------------------------===//

ReturnOpAdaptor::ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("quant.return", odsAttrs.getContext());
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

::mlir::ValueRange ReturnOpAdaptor::results() {
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

::mlir::Operation::operand_range ReturnOp::results() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::resultsMutable() {
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

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange results) {
  odsState.addOperands(results);
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReturnOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace quant
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::ReturnOp)

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::StatisticsOp definitions
//===----------------------------------------------------------------------===//

StatisticsOpAdaptor::StatisticsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("quant.stats", odsAttrs.getContext());
}

StatisticsOpAdaptor::StatisticsOpAdaptor(StatisticsOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatisticsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatisticsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StatisticsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatisticsOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr StatisticsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ElementsAttr StatisticsOpAdaptor::layerStatsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatisticsOp::layerStatsAttrName(*odsOpName)).cast<::mlir::ElementsAttr>();
    return attr;
}

::mlir::ElementsAttr StatisticsOpAdaptor::layerStats() {
  auto attr = layerStatsAttr();
  return attr;
}

::mlir::ElementsAttr StatisticsOpAdaptor::axisStatsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, StatisticsOp::axisStatsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ElementsAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ElementsAttr > StatisticsOpAdaptor::axisStats() {
  auto attr = axisStatsAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

::mlir::IntegerAttr StatisticsOpAdaptor::axisAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, StatisticsOp::axisAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    return attr;
}

::llvm::Optional<uint64_t> StatisticsOpAdaptor::axis() {
  auto attr = axisAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::LogicalResult StatisticsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_layerStats;
  ::mlir::Attribute tblgen_axis;
  ::mlir::Attribute tblgen_axisStats;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'quant.stats' op ""requires attribute 'layerStats'");
    if (namedAttrIt->getName() == StatisticsOp::layerStatsAttrName(*odsOpName)) {
      tblgen_layerStats = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == StatisticsOp::axisAttrName(*odsOpName)) {
      tblgen_axis = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StatisticsOp::axisStatsAttrName(*odsOpName)) {
      tblgen_axisStats = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_layerStats && !((tblgen_layerStats.isa<::mlir::ElementsAttr>())))
    return emitError(loc, "'quant.stats' op ""attribute 'layerStats' failed to satisfy constraint: constant vector/tensor attribute");

  if (tblgen_axisStats && !((tblgen_axisStats.isa<::mlir::ElementsAttr>())))
    return emitError(loc, "'quant.stats' op ""attribute 'axisStats' failed to satisfy constraint: constant vector/tensor attribute");

  if (tblgen_axis && !(((tblgen_axis.isa<::mlir::IntegerAttr>())) && ((tblgen_axis.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'quant.stats' op ""attribute 'axis' failed to satisfy constraint: 64-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatisticsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StatisticsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatisticsOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange StatisticsOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatisticsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StatisticsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ElementsAttr StatisticsOp::layerStatsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, layerStatsAttrName()).cast<::mlir::ElementsAttr>();
}

::mlir::ElementsAttr StatisticsOp::layerStats() {
  auto attr = layerStatsAttr();
  return attr;
}

::mlir::ElementsAttr StatisticsOp::axisStatsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, axisStatsAttrName()).dyn_cast_or_null<::mlir::ElementsAttr>();
}

::llvm::Optional< ::mlir::ElementsAttr > StatisticsOp::axisStats() {
  auto attr = axisStatsAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

::mlir::IntegerAttr StatisticsOp::axisAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, axisAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> StatisticsOp::axis() {
  auto attr = axisAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

void StatisticsOp::layerStatsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(layerStatsAttrName(), attr);
}

void StatisticsOp::axisStatsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(axisStatsAttrName(), attr);
}

void StatisticsOp::axisAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(axisAttrName(), attr);
}

::mlir::Attribute StatisticsOp::removeAxisStatsAttr() {
  return (*this)->removeAttr(axisStatsAttrName());
}

::mlir::Attribute StatisticsOp::removeAxisAttr() {
  return (*this)->removeAttr(axisAttrName());
}

void StatisticsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg, ::mlir::ElementsAttr layerStats, /*optional*/::mlir::ElementsAttr axisStats, /*optional*/::mlir::IntegerAttr axis) {
  odsState.addOperands(arg);
  odsState.addAttribute(layerStatsAttrName(odsState.name), layerStats);
  if (axisStats) {
  odsState.addAttribute(axisStatsAttrName(odsState.name), axisStats);
  }
  if (axis) {
  odsState.addAttribute(axisAttrName(odsState.name), axis);
  }
  odsState.addTypes(resultType0);
}

void StatisticsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::mlir::ElementsAttr layerStats, /*optional*/::mlir::ElementsAttr axisStats, /*optional*/::mlir::IntegerAttr axis) {
  odsState.addOperands(arg);
  odsState.addAttribute(layerStatsAttrName(odsState.name), layerStats);
  if (axisStats) {
  odsState.addAttribute(axisStatsAttrName(odsState.name), axisStats);
  }
  if (axis) {
  odsState.addAttribute(axisAttrName(odsState.name), axis);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(StatisticsOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void StatisticsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::mlir::ElementsAttr layerStats, /*optional*/::mlir::ElementsAttr axisStats, /*optional*/::mlir::IntegerAttr axis) {
  odsState.addOperands(arg);
  odsState.addAttribute(layerStatsAttrName(odsState.name), layerStats);
  if (axisStats) {
  odsState.addAttribute(axisStatsAttrName(odsState.name), axisStats);
  }
  if (axis) {
  odsState.addAttribute(axisAttrName(odsState.name), axis);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatisticsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void StatisticsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(StatisticsOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult StatisticsOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_layerStats;
  ::mlir::Attribute tblgen_axis;
  ::mlir::Attribute tblgen_axisStats;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'layerStats'");
    if (namedAttrIt->getName() == layerStatsAttrName()) {
      tblgen_layerStats = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == axisAttrName()) {
      tblgen_axis = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == axisStatsAttrName()) {
      tblgen_axisStats = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps6(*this, tblgen_layerStats, "layerStats")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps6(*this, tblgen_axisStats, "axisStats")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps1(*this, tblgen_axis, "axis")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatisticsOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::LogicalResult StatisticsOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

} // namespace quant
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::StatisticsOp)

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::StatisticsRefOp definitions
//===----------------------------------------------------------------------===//

StatisticsRefOpAdaptor::StatisticsRefOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("quant.stats_ref", odsAttrs.getContext());
}

StatisticsRefOpAdaptor::StatisticsRefOpAdaptor(StatisticsRefOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StatisticsRefOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StatisticsRefOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StatisticsRefOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatisticsRefOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr StatisticsRefOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr StatisticsRefOpAdaptor::statsKeyAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StatisticsRefOp::statsKeyAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef StatisticsRefOpAdaptor::statsKey() {
  auto attr = statsKeyAttr();
  return attr.getValue();
}

::mlir::LogicalResult StatisticsRefOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_statsKey;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'quant.stats_ref' op ""requires attribute 'statsKey'");
    if (namedAttrIt->getName() == StatisticsRefOp::statsKeyAttrName(*odsOpName)) {
      tblgen_statsKey = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_statsKey && !((tblgen_statsKey.isa<::mlir::StringAttr>())))
    return emitError(loc, "'quant.stats_ref' op ""attribute 'statsKey' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StatisticsRefOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StatisticsRefOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StatisticsRefOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange StatisticsRefOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StatisticsRefOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StatisticsRefOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr StatisticsRefOp::statsKeyAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, statsKeyAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef StatisticsRefOp::statsKey() {
  auto attr = statsKeyAttr();
  return attr.getValue();
}

void StatisticsRefOp::statsKeyAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(statsKeyAttrName(), attr);
}

void StatisticsRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg, ::mlir::StringAttr statsKey) {
  odsState.addOperands(arg);
  odsState.addAttribute(statsKeyAttrName(odsState.name), statsKey);
  odsState.addTypes(resultType0);
}

void StatisticsRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::mlir::StringAttr statsKey) {
  odsState.addOperands(arg);
  odsState.addAttribute(statsKeyAttrName(odsState.name), statsKey);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(StatisticsRefOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void StatisticsRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::mlir::StringAttr statsKey) {
  odsState.addOperands(arg);
  odsState.addAttribute(statsKeyAttrName(odsState.name), statsKey);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatisticsRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg, ::llvm::StringRef statsKey) {
  odsState.addOperands(arg);
  odsState.addAttribute(statsKeyAttrName(odsState.name), odsBuilder.getStringAttr(statsKey));
  odsState.addTypes(resultType0);
}

void StatisticsRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value arg, ::llvm::StringRef statsKey) {
  odsState.addOperands(arg);
  odsState.addAttribute(statsKeyAttrName(odsState.name), odsBuilder.getStringAttr(statsKey));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(StatisticsRefOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void StatisticsRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg, ::llvm::StringRef statsKey) {
  odsState.addOperands(arg);
  odsState.addAttribute(statsKeyAttrName(odsState.name), odsBuilder.getStringAttr(statsKey));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StatisticsRefOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void StatisticsRefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(StatisticsRefOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult StatisticsRefOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_statsKey;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'statsKey'");
    if (namedAttrIt->getName() == statsKeyAttrName()) {
      tblgen_statsKey = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_QuantOps4(*this, tblgen_statsKey, "statsKey")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StatisticsRefOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult StatisticsRefOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

} // namespace quant
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::StatisticsRefOp)

namespace mlir {
namespace quant {

//===----------------------------------------------------------------------===//
// ::mlir::quant::StorageCastOp definitions
//===----------------------------------------------------------------------===//

StorageCastOpAdaptor::StorageCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("quant.scast", odsAttrs.getContext());
}

StorageCastOpAdaptor::StorageCastOpAdaptor(StorageCastOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StorageCastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StorageCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StorageCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StorageCastOpAdaptor::arg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr StorageCastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult StorageCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StorageCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StorageCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StorageCastOp::arg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange StorageCastOp::argMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StorageCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StorageCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void StorageCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(resultType0);
}

void StorageCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StorageCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StorageCastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_QuantOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StorageCastOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void StorageCastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace quant
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::quant::StorageCastOp)


#endif  // GET_OP_CLASSES

