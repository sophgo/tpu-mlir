/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::pdl_interp::ApplyConstraintOp,
::mlir::pdl_interp::ApplyRewriteOp,
::mlir::pdl_interp::AreEqualOp,
::mlir::pdl_interp::BranchOp,
::mlir::pdl_interp::CheckAttributeOp,
::mlir::pdl_interp::CheckOperandCountOp,
::mlir::pdl_interp::CheckOperationNameOp,
::mlir::pdl_interp::CheckResultCountOp,
::mlir::pdl_interp::CheckTypeOp,
::mlir::pdl_interp::CheckTypesOp,
::mlir::pdl_interp::ContinueOp,
::mlir::pdl_interp::CreateAttributeOp,
::mlir::pdl_interp::CreateOperationOp,
::mlir::pdl_interp::CreateRangeOp,
::mlir::pdl_interp::CreateTypeOp,
::mlir::pdl_interp::CreateTypesOp,
::mlir::pdl_interp::EraseOp,
::mlir::pdl_interp::ExtractOp,
::mlir::pdl_interp::FinalizeOp,
::mlir::pdl_interp::ForEachOp,
::mlir::pdl_interp::FuncOp,
::mlir::pdl_interp::GetAttributeOp,
::mlir::pdl_interp::GetAttributeTypeOp,
::mlir::pdl_interp::GetDefiningOpOp,
::mlir::pdl_interp::GetOperandOp,
::mlir::pdl_interp::GetOperandsOp,
::mlir::pdl_interp::GetResultOp,
::mlir::pdl_interp::GetResultsOp,
::mlir::pdl_interp::GetUsersOp,
::mlir::pdl_interp::GetValueTypeOp,
::mlir::pdl_interp::IsNotNullOp,
::mlir::pdl_interp::RecordMatchOp,
::mlir::pdl_interp::ReplaceOp,
::mlir::pdl_interp::SwitchAttributeOp,
::mlir::pdl_interp::SwitchOperandCountOp,
::mlir::pdl_interp::SwitchOperationNameOp,
::mlir::pdl_interp::SwitchResultCountOp,
::mlir::pdl_interp::SwitchTypeOp,
::mlir::pdl_interp::SwitchTypesOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace pdl_interp {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::pdl::PDLType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be pdl type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::pdl::AttributeType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Attribute`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::pdl::OperationType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Operation *`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::pdl::TypeType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of PDL handle to an `mlir::Type` values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::ValueType>(type))) || (((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be single element or range of PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::TypeType>(type))) || (((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be single element or range of PDL handle to an `mlir::Type`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::RangeType>(type))) && (((::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType()))) || ((::llvm::isa<::mlir::pdl::ValueType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of PDL handle to an `mlir::Type` or PDL handle for an `mlir::Value` values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::PDLType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of pdl type values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::llvm::isa<::mlir::pdl::ValueType>(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be PDL handle for an `mlir::Value`, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_PDLInterpOps10(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::OperationType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be range of PDL handle to an `mlir::Operation *` values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps0(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps0(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((true)))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: any attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(attr).getValue().isNegative()))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<::mlir::UnitAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: unit attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: any type attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps5(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: type array attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps5(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps6(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::StringAttr>(attr))); }))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: string array attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps6(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps7(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(attr).getValue())))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: type attribute of function type";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps7(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps8(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: Array of dictionary attributes";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps8(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps9(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<::mlir::SymbolRefAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: symbol reference attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps9(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps10(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((((::llvm::isa<::mlir::IntegerAttr>(attr))) && ((::llvm::cast<::mlir::IntegerAttr>(attr).getType().isSignlessInteger(16)))) && ((!::llvm::cast<::mlir::IntegerAttr>(attr).getValue().isNegative()))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps10(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps11(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<::mlir::ArrayAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: array attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps11(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps12(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(attr))) && ((::llvm::cast<::mlir::DenseIntElementsAttr>(attr).getType().getElementType().isSignlessInteger(32)))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: 32-bit signless integer elements attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps12(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps12(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps13(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))); }))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: type-array array attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_PDLInterpOps13(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_PDLInterpOps13(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_PDLInterpOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_PDLInterpOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItemsOrMore(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with at least 1 blocks";
  }
  return ::mlir::success();
}
} // namespace pdl_interp
} // namespace mlir
namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ApplyConstraintOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ApplyConstraintOpGenericAdaptorBase::ApplyConstraintOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.apply_constraint", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ApplyConstraintOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr ApplyConstraintOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyConstraintOpGenericAdaptorBase::getNameAttr() {
  auto attr = getProperties().name.cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ApplyConstraintOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
ApplyConstraintOpAdaptor::ApplyConstraintOpAdaptor(ApplyConstraintOp op) : ApplyConstraintOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ApplyConstraintOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'pdl_interp.apply_constraint' op ""requires attribute 'name'");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'pdl_interp.apply_constraint' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ApplyConstraintOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ApplyConstraintOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ApplyConstraintOp::getArgs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyConstraintOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ApplyConstraintOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ApplyConstraintOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *ApplyConstraintOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *ApplyConstraintOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::LogicalResult ApplyConstraintOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.name;
    auto attr = dict.get("name");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for name in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `name` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ApplyConstraintOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ApplyConstraintOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ApplyConstraintOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "name")
      return prop.name;
  return std::nullopt;
}

void ApplyConstraintOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "name") {
       prop.name = dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }
}

void ApplyConstraintOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.name) attrs.append("name", prop.name);
}

::mlir::LogicalResult ApplyConstraintOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(attr, "name", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::StringAttr ApplyConstraintOp::getNameAttr() {
  return getProperties().name.cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyConstraintOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void ApplyConstraintOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void ApplyConstraintOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr name, ::mlir::ValueRange args, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange args, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef name, ::mlir::ValueRange args, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange args, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ApplyConstraintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyConstraintOp::verifyInvariantsImpl() {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult ApplyConstraintOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ApplyConstraintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.getOrAddProperties<ApplyConstraintOp::Properties>().name = nameAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyConstraintOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << "(";
  _odsPrinter << getArgs();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getArgs().getTypes();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ApplyConstraintOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ApplyRewriteOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ApplyRewriteOpGenericAdaptorBase::ApplyRewriteOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.apply_rewrite", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ApplyRewriteOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr ApplyRewriteOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr ApplyRewriteOpGenericAdaptorBase::getNameAttr() {
  auto attr = getProperties().name.cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef ApplyRewriteOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
ApplyRewriteOpAdaptor::ApplyRewriteOpAdaptor(ApplyRewriteOp op) : ApplyRewriteOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ApplyRewriteOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'pdl_interp.apply_rewrite' op ""requires attribute 'name'");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'pdl_interp.apply_rewrite' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ApplyRewriteOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ApplyRewriteOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ApplyRewriteOp::getArgs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ApplyRewriteOp::getArgsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ApplyRewriteOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range ApplyRewriteOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::result_range ApplyRewriteOp::getResults() {
  return getODSResults(0);
}

::mlir::LogicalResult ApplyRewriteOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.name;
    auto attr = dict.get("name");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for name in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `name` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ApplyRewriteOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ApplyRewriteOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ApplyRewriteOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "name")
      return prop.name;
  return std::nullopt;
}

void ApplyRewriteOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "name") {
       prop.name = dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }
}

void ApplyRewriteOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.name) attrs.append("name", prop.name);
}

::mlir::LogicalResult ApplyRewriteOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(attr, "name", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::StringAttr ApplyRewriteOp::getNameAttr() {
  return getProperties().name.cast<::mlir::StringAttr>();
}

::llvm::StringRef ApplyRewriteOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void ApplyRewriteOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void ApplyRewriteOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void ApplyRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::StringAttr name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.addTypes(results);
}

void ApplyRewriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::llvm::StringRef name, ::mlir::ValueRange args) {
  odsState.addOperands(args);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.addTypes(results);
}

void ApplyRewriteOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ApplyRewriteOp::verifyInvariantsImpl() {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ApplyRewriteOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ApplyRewriteOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;
  ::llvm::SmallVector<::mlir::Type, 1> resultsTypes;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.getOrAddProperties<ApplyRewriteOp::Properties>().name = nameAttr;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (::mlir::succeeded(parser.parseOptionalColon())) {

  if (parser.parseTypeList(resultsTypes))
    return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  result.addTypes(resultsTypes);
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ApplyRewriteOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  if (!getArgs().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getArgs();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getArgs().getTypes();
    _odsPrinter << ")";
  }
  if (!getResults().empty()) {
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getResults().getTypes();
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ApplyRewriteOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::AreEqualOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AreEqualOpGenericAdaptorBase::AreEqualOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.are_equal", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> AreEqualOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AreEqualOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AreEqualOpAdaptor::AreEqualOpAdaptor(AreEqualOp op) : AreEqualOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult AreEqualOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AreEqualOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AreEqualOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::PDLType> AreEqualOp::getLhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::pdl::PDLType> AreEqualOp::getRhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange AreEqualOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AreEqualOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AreEqualOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AreEqualOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *AreEqualOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *AreEqualOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

void AreEqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void AreEqualOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AreEqualOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AreEqualOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult AreEqualOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AreEqualOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allOperands;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  if (parser.resolveOperands(allOperands, ::llvm::concat<const ::mlir::Type>(::llvm::ArrayRef<::mlir::Type>(lhsTypes), ::llvm::ArrayRef<::mlir::Type>(lhsTypes[0])), allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AreEqualOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void AreEqualOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::AreEqualOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::BranchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
BranchOpGenericAdaptorBase::BranchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.branch", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> BranchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr BranchOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
BranchOpAdaptor::BranchOpAdaptor(BranchOp op) : BranchOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult BranchOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BranchOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BranchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> BranchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BranchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *BranchOp::getDest() {
  return (*this)->getSuccessor(0);
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Block *dest) {
  odsState.addSuccessors(dest);
}

void BranchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Block *dest) {
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BranchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BranchOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult BranchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BranchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Block *destSuccessor = nullptr;

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  result.addSuccessors(destSuccessor);
  return ::mlir::success();
}

void BranchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void BranchOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::BranchOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckAttributeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckAttributeOpGenericAdaptorBase::CheckAttributeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.check_attribute", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CheckAttributeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CheckAttributeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute CheckAttributeOpGenericAdaptorBase::getConstantValueAttr() {
  auto attr = getProperties().constantValue.cast<::mlir::Attribute>();
  return attr;
}

::mlir::Attribute CheckAttributeOpGenericAdaptorBase::getConstantValue() {
  auto attr = getConstantValueAttr();
  return attr;
}

} // namespace detail
CheckAttributeOpAdaptor::CheckAttributeOpAdaptor(CheckAttributeOp op) : CheckAttributeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult CheckAttributeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_constantValue = getProperties().constantValue; (void)tblgen_constantValue;
  if (!tblgen_constantValue) return emitError(loc, "'pdl_interp.check_attribute' op ""requires attribute 'constantValue'");

  if (tblgen_constantValue && !((true)))
    return emitError(loc, "'pdl_interp.check_attribute' op ""attribute 'constantValue' failed to satisfy constraint: any attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CheckAttributeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckAttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::AttributeType> CheckAttributeOp::getAttribute() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::AttributeType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange CheckAttributeOp::getAttributeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CheckAttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckAttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckAttributeOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckAttributeOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::LogicalResult CheckAttributeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.constantValue;
    auto attr = dict.get("constantValue");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for constantValue in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `constantValue` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CheckAttributeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.constantValue;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("constantValue",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code CheckAttributeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.constantValue.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CheckAttributeOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "constantValue")
      return prop.constantValue;
  return std::nullopt;
}

void CheckAttributeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "constantValue") {
       prop.constantValue = dyn_cast_or_null<std::remove_reference_t<decltype(prop.constantValue)>>(value);
       return;
    }
}

void CheckAttributeOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.constantValue) attrs.append("constantValue", prop.constantValue);
}

::mlir::LogicalResult CheckAttributeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getConstantValueAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(attr, "constantValue", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::Attribute CheckAttributeOp::getConstantValueAttr() {
  return getProperties().constantValue.cast<::mlir::Attribute>();
}

::mlir::Attribute CheckAttributeOp::getConstantValue() {
  auto attr = getConstantValueAttr();
  return attr;
}

void CheckAttributeOp::setConstantValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(getConstantValueAttrName(), attr);
}

void CheckAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value attribute, ::mlir::Attribute constantValue, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(attribute);
  odsState.getOrAddProperties<Properties>().constantValue = constantValue;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value attribute, ::mlir::Attribute constantValue, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(attribute);
  odsState.getOrAddProperties<Properties>().constantValue = constantValue;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckAttributeOp::verifyInvariantsImpl() {
  auto tblgen_constantValue = getProperties().constantValue; (void)tblgen_constantValue;
  if (!tblgen_constantValue) return emitOpError("requires attribute 'constantValue'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(*this, tblgen_constantValue, "constantValue")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CheckAttributeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand attributeRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> attributeOperands(attributeRawOperands);  ::llvm::SMLoc attributeOperandsLoc;
  (void)attributeOperandsLoc;
  ::mlir::Attribute constantValueAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  attributeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(attributeRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();

  if (parser.parseAttribute(constantValueAttr, ::mlir::Type{}))
    return ::mlir::failure();
  if (constantValueAttr) result.getOrAddProperties<CheckAttributeOp::Properties>().constantValue = constantValueAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  if (parser.resolveOperands(attributeOperands, odsBuildableType0, attributeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAttribute();
  _odsPrinter << ' ' << "is";
  _odsPrinter << ' ';
  _odsPrinter.printAttribute(getConstantValueAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("constantValue");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckAttributeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckOperandCountOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckOperandCountOpGenericAdaptorBase::CheckOperandCountOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.check_operand_count", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CheckOperandCountOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CheckOperandCountOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr CheckOperandCountOpGenericAdaptorBase::getCountAttr() {
  auto attr = getProperties().count.cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t CheckOperandCountOpGenericAdaptorBase::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckOperandCountOpGenericAdaptorBase::getCompareAtLeastAttr() {
  auto attr = getProperties().compareAtLeast.dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool CheckOperandCountOpGenericAdaptorBase::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
CheckOperandCountOpAdaptor::CheckOperandCountOpAdaptor(CheckOperandCountOp op) : CheckOperandCountOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult CheckOperandCountOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_compareAtLeast = getProperties().compareAtLeast; (void)tblgen_compareAtLeast;
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitError(loc, "'pdl_interp.check_operand_count' op ""requires attribute 'count'");

  if (tblgen_count && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_count))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_count).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_count).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.check_operand_count' op ""attribute 'count' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");

  if (tblgen_compareAtLeast && !((::llvm::isa<::mlir::UnitAttr>(tblgen_compareAtLeast))))
    return emitError(loc, "'pdl_interp.check_operand_count' op ""attribute 'compareAtLeast' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CheckOperandCountOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckOperandCountOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> CheckOperandCountOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange CheckOperandCountOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CheckOperandCountOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckOperandCountOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckOperandCountOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckOperandCountOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::LogicalResult CheckOperandCountOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.compareAtLeast;
    auto attr = dict.get("compareAtLeast");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for compareAtLeast in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `compareAtLeast` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.count;
    auto attr = dict.get("count");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for count in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `count` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CheckOperandCountOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.compareAtLeast;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("compareAtLeast",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.count;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("count",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code CheckOperandCountOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.compareAtLeast.getAsOpaquePointer()), 
    llvm::hash_value(prop.count.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CheckOperandCountOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "compareAtLeast")
      return prop.compareAtLeast;

    if (name == "count")
      return prop.count;
  return std::nullopt;
}

void CheckOperandCountOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "compareAtLeast") {
       prop.compareAtLeast = dyn_cast_or_null<std::remove_reference_t<decltype(prop.compareAtLeast)>>(value);
       return;
    }

    if (name == "count") {
       prop.count = dyn_cast_or_null<std::remove_reference_t<decltype(prop.count)>>(value);
       return;
    }
}

void CheckOperandCountOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.compareAtLeast) attrs.append("compareAtLeast", prop.compareAtLeast);

    if (prop.count) attrs.append("count", prop.count);
}

::mlir::LogicalResult CheckOperandCountOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getCompareAtLeastAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(attr, "compareAtLeast", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCountAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(attr, "count", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr CheckOperandCountOp::getCountAttr() {
  return getProperties().count.cast<::mlir::IntegerAttr>();
}

uint32_t CheckOperandCountOp::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckOperandCountOp::getCompareAtLeastAttr() {
  return getProperties().compareAtLeast.dyn_cast_or_null<::mlir::UnitAttr>();
}

bool CheckOperandCountOp::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void CheckOperandCountOp::setCountAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getCountAttrName(), attr);
}

void CheckOperandCountOp::setCount(uint32_t attrValue) {
  (*this)->setAttr(getCountAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void CheckOperandCountOp::setCompareAtLeastAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getCompareAtLeastAttrName(), attr);
}

void CheckOperandCountOp::setCompareAtLeast(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getCompareAtLeastAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getCompareAtLeastAttrName());
}

::mlir::Attribute CheckOperandCountOp::removeCompareAtLeastAttr() {
    auto &attr = getProperties().compareAtLeast;
    attr = {};
    return attr;
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = count;
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = compareAtLeast;
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = count;
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = compareAtLeast;
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = ((compareAtLeast) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = ((compareAtLeast) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperandCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckOperandCountOp::verifyInvariantsImpl() {
  auto tblgen_compareAtLeast = getProperties().compareAtLeast; (void)tblgen_compareAtLeast;
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitOpError("requires attribute 'count'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_count, "count")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(*this, tblgen_compareAtLeast, "compareAtLeast")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CheckOperandCountOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckOperandCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::IntegerAttr countAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("at_least"))) {
    result.getOrAddProperties<CheckOperandCountOp::Properties>().compareAtLeast = parser.getBuilder().getUnitAttr();  }

  if (parser.parseCustomAttributeWithFallback(countAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (countAttr) result.getOrAddProperties<CheckOperandCountOp::Properties>().count = countAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckOperandCountOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "is";
  if (getCompareAtLeastAttr()) {
    _odsPrinter << ' ' << "at_least";
  }
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCountAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("compareAtLeast");
  elidedAttrs.push_back("count");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getCompareAtLeastAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("compareAtLeast");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckOperandCountOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckOperandCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckOperationNameOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckOperationNameOpGenericAdaptorBase::CheckOperationNameOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.check_operation_name", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CheckOperationNameOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CheckOperationNameOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CheckOperationNameOpGenericAdaptorBase::getNameAttr() {
  auto attr = getProperties().name.cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CheckOperationNameOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
CheckOperationNameOpAdaptor::CheckOperationNameOpAdaptor(CheckOperationNameOp op) : CheckOperationNameOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult CheckOperationNameOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'pdl_interp.check_operation_name' op ""requires attribute 'name'");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'pdl_interp.check_operation_name' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CheckOperationNameOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckOperationNameOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> CheckOperationNameOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange CheckOperationNameOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CheckOperationNameOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckOperationNameOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckOperationNameOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckOperationNameOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::LogicalResult CheckOperationNameOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.name;
    auto attr = dict.get("name");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for name in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `name` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CheckOperationNameOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code CheckOperationNameOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CheckOperationNameOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "name")
      return prop.name;
  return std::nullopt;
}

void CheckOperationNameOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "name") {
       prop.name = dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }
}

void CheckOperationNameOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.name) attrs.append("name", prop.name);
}

::mlir::LogicalResult CheckOperationNameOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(attr, "name", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::StringAttr CheckOperationNameOp::getNameAttr() {
  return getProperties().name.cast<::mlir::StringAttr>();
}

::llvm::StringRef CheckOperationNameOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void CheckOperationNameOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void CheckOperationNameOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::StringAttr name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::StringAttr name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::llvm::StringRef name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::llvm::StringRef name, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckOperationNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckOperationNameOp::verifyInvariantsImpl() {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CheckOperationNameOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckOperationNameOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.getOrAddProperties<CheckOperationNameOp::Properties>().name = nameAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckOperationNameOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "is";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckOperationNameOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckOperationNameOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckResultCountOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckResultCountOpGenericAdaptorBase::CheckResultCountOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.check_result_count", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CheckResultCountOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CheckResultCountOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr CheckResultCountOpGenericAdaptorBase::getCountAttr() {
  auto attr = getProperties().count.cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t CheckResultCountOpGenericAdaptorBase::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckResultCountOpGenericAdaptorBase::getCompareAtLeastAttr() {
  auto attr = getProperties().compareAtLeast.dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool CheckResultCountOpGenericAdaptorBase::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
CheckResultCountOpAdaptor::CheckResultCountOpAdaptor(CheckResultCountOp op) : CheckResultCountOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult CheckResultCountOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_compareAtLeast = getProperties().compareAtLeast; (void)tblgen_compareAtLeast;
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitError(loc, "'pdl_interp.check_result_count' op ""requires attribute 'count'");

  if (tblgen_count && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_count))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_count).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_count).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.check_result_count' op ""attribute 'count' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");

  if (tblgen_compareAtLeast && !((::llvm::isa<::mlir::UnitAttr>(tblgen_compareAtLeast))))
    return emitError(loc, "'pdl_interp.check_result_count' op ""attribute 'compareAtLeast' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CheckResultCountOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckResultCountOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> CheckResultCountOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange CheckResultCountOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CheckResultCountOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckResultCountOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckResultCountOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckResultCountOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::LogicalResult CheckResultCountOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.compareAtLeast;
    auto attr = dict.get("compareAtLeast");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for compareAtLeast in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `compareAtLeast` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.count;
    auto attr = dict.get("count");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for count in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `count` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CheckResultCountOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.compareAtLeast;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("compareAtLeast",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.count;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("count",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code CheckResultCountOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.compareAtLeast.getAsOpaquePointer()), 
    llvm::hash_value(prop.count.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CheckResultCountOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "compareAtLeast")
      return prop.compareAtLeast;

    if (name == "count")
      return prop.count;
  return std::nullopt;
}

void CheckResultCountOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "compareAtLeast") {
       prop.compareAtLeast = dyn_cast_or_null<std::remove_reference_t<decltype(prop.compareAtLeast)>>(value);
       return;
    }

    if (name == "count") {
       prop.count = dyn_cast_or_null<std::remove_reference_t<decltype(prop.count)>>(value);
       return;
    }
}

void CheckResultCountOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.compareAtLeast) attrs.append("compareAtLeast", prop.compareAtLeast);

    if (prop.count) attrs.append("count", prop.count);
}

::mlir::LogicalResult CheckResultCountOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getCompareAtLeastAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(attr, "compareAtLeast", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getCountAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(attr, "count", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr CheckResultCountOp::getCountAttr() {
  return getProperties().count.cast<::mlir::IntegerAttr>();
}

uint32_t CheckResultCountOp::getCount() {
  auto attr = getCountAttr();
  return attr.getValue().getZExtValue();
}

::mlir::UnitAttr CheckResultCountOp::getCompareAtLeastAttr() {
  return getProperties().compareAtLeast.dyn_cast_or_null<::mlir::UnitAttr>();
}

bool CheckResultCountOp::getCompareAtLeast() {
  auto attr = getCompareAtLeastAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void CheckResultCountOp::setCountAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getCountAttrName(), attr);
}

void CheckResultCountOp::setCount(uint32_t attrValue) {
  (*this)->setAttr(getCountAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void CheckResultCountOp::setCompareAtLeastAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getCompareAtLeastAttrName(), attr);
}

void CheckResultCountOp::setCompareAtLeast(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getCompareAtLeastAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getCompareAtLeastAttrName());
}

::mlir::Attribute CheckResultCountOp::removeCompareAtLeastAttr() {
    auto &attr = getProperties().compareAtLeast;
    attr = {};
    return attr;
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = count;
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = compareAtLeast;
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::IntegerAttr count, /*optional*/::mlir::UnitAttr compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = count;
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = compareAtLeast;
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = ((compareAtLeast) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t count, /*optional*/bool compareAtLeast, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().count = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), count);
  if (compareAtLeast) {
    odsState.getOrAddProperties<Properties>().compareAtLeast = ((compareAtLeast) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckResultCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckResultCountOp::verifyInvariantsImpl() {
  auto tblgen_compareAtLeast = getProperties().compareAtLeast; (void)tblgen_compareAtLeast;
  auto tblgen_count = getProperties().count; (void)tblgen_count;
  if (!tblgen_count) return emitOpError("requires attribute 'count'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_count, "count")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(*this, tblgen_compareAtLeast, "compareAtLeast")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CheckResultCountOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckResultCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::IntegerAttr countAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("at_least"))) {
    result.getOrAddProperties<CheckResultCountOp::Properties>().compareAtLeast = parser.getBuilder().getUnitAttr();  }

  if (parser.parseCustomAttributeWithFallback(countAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (countAttr) result.getOrAddProperties<CheckResultCountOp::Properties>().count = countAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckResultCountOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "is";
  if (getCompareAtLeastAttr()) {
    _odsPrinter << ' ' << "at_least";
  }
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCountAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("compareAtLeast");
  elidedAttrs.push_back("count");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getCompareAtLeastAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("compareAtLeast");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckResultCountOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckResultCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckTypeOpGenericAdaptorBase::CheckTypeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.check_type", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CheckTypeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CheckTypeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr CheckTypeOpGenericAdaptorBase::getTypeAttr() {
  auto attr = getProperties().type.cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::Type CheckTypeOpGenericAdaptorBase::getType() {
  auto attr = getTypeAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

} // namespace detail
CheckTypeOpAdaptor::CheckTypeOpAdaptor(CheckTypeOp op) : CheckTypeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult CheckTypeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_type = getProperties().type; (void)tblgen_type;
  if (!tblgen_type) return emitError(loc, "'pdl_interp.check_type' op ""requires attribute 'type'");

  if (tblgen_type && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_type))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(tblgen_type).getValue()))) && ((true))))
    return emitError(loc, "'pdl_interp.check_type' op ""attribute 'type' failed to satisfy constraint: any type attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CheckTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::TypeType> CheckTypeOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::TypeType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange CheckTypeOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CheckTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckTypeOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckTypeOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::LogicalResult CheckTypeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.type;
    auto attr = dict.get("type");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for type in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `type` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CheckTypeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.type;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("type",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code CheckTypeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.type.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CheckTypeOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "type")
      return prop.type;
  return std::nullopt;
}

void CheckTypeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "type") {
       prop.type = dyn_cast_or_null<std::remove_reference_t<decltype(prop.type)>>(value);
       return;
    }
}

void CheckTypeOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.type) attrs.append("type", prop.type);
}

::mlir::LogicalResult CheckTypeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(attr, "type", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::TypeAttr CheckTypeOp::getTypeAttr() {
  return getProperties().type.cast<::mlir::TypeAttr>();
}

::mlir::Type CheckTypeOp::getType() {
  auto attr = getTypeAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

void CheckTypeOp::setTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getTypeAttrName(), attr);
}

void CheckTypeOp::setType(::mlir::Type attrValue) {
  (*this)->setAttr(getTypeAttrName(), ::mlir::TypeAttr::get(attrValue));
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::TypeAttr type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().type = type;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::TypeAttr type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().type = type;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Type type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().type = ::mlir::TypeAttr::get(type);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Type type, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().type = ::mlir::TypeAttr::get(type);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckTypeOp::verifyInvariantsImpl() {
  auto tblgen_type = getProperties().type; (void)tblgen_type;
  if (!tblgen_type) return emitOpError("requires attribute 'type'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(*this, tblgen_type, "type")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CheckTypeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::TypeAttr typeAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("is"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(typeAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (typeAttr) result.getOrAddProperties<CheckTypeOp::Properties>().type = typeAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "is";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getTypeAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("type");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CheckTypesOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CheckTypesOpGenericAdaptorBase::CheckTypesOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.check_types", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CheckTypesOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CheckTypesOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr CheckTypesOpGenericAdaptorBase::getTypesAttr() {
  auto attr = getProperties().types.cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CheckTypesOpGenericAdaptorBase::getTypes() {
  auto attr = getTypesAttr();
  return attr;
}

} // namespace detail
CheckTypesOpAdaptor::CheckTypesOpAdaptor(CheckTypesOp op) : CheckTypesOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult CheckTypesOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_types = getProperties().types; (void)tblgen_types;
  if (!tblgen_types) return emitError(loc, "'pdl_interp.check_types' op ""requires attribute 'types'");

  if (tblgen_types && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_types))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_types), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'pdl_interp.check_types' op ""attribute 'types' failed to satisfy constraint: type array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CheckTypesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CheckTypesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::RangeType> CheckTypesOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange CheckTypesOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CheckTypesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CheckTypesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CheckTypesOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CheckTypesOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::LogicalResult CheckTypesOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.types;
    auto attr = dict.get("types");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for types in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `types` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CheckTypesOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.types;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("types",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code CheckTypesOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.types.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CheckTypesOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "types")
      return prop.types;
  return std::nullopt;
}

void CheckTypesOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "types") {
       prop.types = dyn_cast_or_null<std::remove_reference_t<decltype(prop.types)>>(value);
       return;
    }
}

void CheckTypesOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.types) attrs.append("types", prop.types);
}

::mlir::LogicalResult CheckTypesOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getTypesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(attr, "types", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::ArrayAttr CheckTypesOp::getTypesAttr() {
  return getProperties().types.cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CheckTypesOp::getTypes() {
  auto attr = getTypesAttr();
  return attr;
}

void CheckTypesOp::setTypesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getTypesAttrName(), attr);
}

void CheckTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ArrayAttr types, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().types = types;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CheckTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ArrayAttr types, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().types = types;
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CheckTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CheckTypesOp::verifyInvariantsImpl() {
  auto tblgen_types = getProperties().types; (void)tblgen_types;
  if (!tblgen_types) return emitOpError("requires attribute 'types'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(*this, tblgen_types, "types")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CheckTypesOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CheckTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::ArrayAttr typesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("are"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(typesAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (typesAttr) result.getOrAddProperties<CheckTypesOp::Properties>().types = typesAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CheckTypesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "are";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getTypesAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("types");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void CheckTypesOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CheckTypesOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ContinueOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ContinueOpGenericAdaptorBase::ContinueOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.continue", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ContinueOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ContinueOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ContinueOpAdaptor::ContinueOpAdaptor(ContinueOp op) : ContinueOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ContinueOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ContinueOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ContinueOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ContinueOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ContinueOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ContinueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void ContinueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ContinueOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ContinueOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult ContinueOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ContinueOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void ContinueOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ContinueOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ContinueOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateAttributeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateAttributeOpGenericAdaptorBase::CreateAttributeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.create_attribute", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CreateAttributeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CreateAttributeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute CreateAttributeOpGenericAdaptorBase::getValueAttr() {
  auto attr = getProperties().value.cast<::mlir::Attribute>();
  return attr;
}

::mlir::Attribute CreateAttributeOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr;
}

} // namespace detail
CreateAttributeOpAdaptor::CreateAttributeOpAdaptor(CreateAttributeOp op) : CreateAttributeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult CreateAttributeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitError(loc, "'pdl_interp.create_attribute' op ""requires attribute 'value'");

  if (tblgen_value && !((true)))
    return emitError(loc, "'pdl_interp.create_attribute' op ""attribute 'value' failed to satisfy constraint: any attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateAttributeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CreateAttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CreateAttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateAttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::AttributeType> CreateAttributeOp::getAttribute() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::AttributeType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult CreateAttributeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.value;
    auto attr = dict.get("value");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for value in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `value` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CreateAttributeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.value;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("value",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code CreateAttributeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.value.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CreateAttributeOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "value")
      return prop.value;
  return std::nullopt;
}

void CreateAttributeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "value") {
       prop.value = dyn_cast_or_null<std::remove_reference_t<decltype(prop.value)>>(value);
       return;
    }
}

void CreateAttributeOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.value) attrs.append("value", prop.value);
}

::mlir::LogicalResult CreateAttributeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getValueAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(attr, "value", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::Attribute CreateAttributeOp::getValueAttr() {
  return getProperties().value.cast<::mlir::Attribute>();
}

::mlir::Attribute CreateAttributeOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void CreateAttributeOp::setValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void CreateAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Attribute value) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::AttributeType>(), value);
    
}

void CreateAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attribute, ::mlir::Attribute value) {
  odsState.getOrAddProperties<Properties>().value = value;
  odsState.addTypes(attribute);
}

void CreateAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value) {
  odsState.getOrAddProperties<Properties>().value = value;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateAttributeOp::verifyInvariantsImpl() {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitOpError("requires attribute 'value'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps1(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateAttributeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Attribute valueAttr;

  if (parser.parseAttribute(valueAttr, ::mlir::Type{}))
    return ::mlir::failure();
  if (valueAttr) result.getOrAddProperties<CreateAttributeOp::Properties>().value = valueAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void CreateAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttribute(getValueAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elidedAttrs);
}

void CreateAttributeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateOperationOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateOperationOpGenericAdaptorBase::CreateOperationOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.create_operation", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CreateOperationOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(getProperties().operand_segment_sizes && "missing segment size attribute for op");
  auto sizeAttr = getProperties().operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr CreateOperationOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr CreateOperationOpGenericAdaptorBase::getNameAttr() {
  auto attr = getProperties().name.cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef CreateOperationOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CreateOperationOpGenericAdaptorBase::getInputAttributeNamesAttr() {
  auto attr = getProperties().inputAttributeNames.cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CreateOperationOpGenericAdaptorBase::getInputAttributeNames() {
  auto attr = getInputAttributeNamesAttr();
  return attr;
}

::mlir::UnitAttr CreateOperationOpGenericAdaptorBase::getInferredResultTypesAttr() {
  auto attr = getProperties().inferredResultTypes.dyn_cast_or_null<::mlir::UnitAttr>();
  if (!attr)
    attr = ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr);
  return attr;
}

bool CreateOperationOpGenericAdaptorBase::getInferredResultTypes() {
  auto attr = getInferredResultTypesAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder(odsAttrs.getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

} // namespace detail
CreateOperationOpAdaptor::CreateOperationOpAdaptor(CreateOperationOp op) : CreateOperationOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult CreateOperationOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_inferredResultTypes = getProperties().inferredResultTypes; (void)tblgen_inferredResultTypes;
  auto tblgen_inputAttributeNames = getProperties().inputAttributeNames; (void)tblgen_inputAttributeNames;
  if (!tblgen_inputAttributeNames) return emitError(loc, "'pdl_interp.create_operation' op ""requires attribute 'inputAttributeNames'");
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'pdl_interp.create_operation' op ""requires attribute 'name'");
  auto tblgen_operand_segment_sizes = getProperties().operand_segment_sizes; (void)tblgen_operand_segment_sizes;
  if (!tblgen_operand_segment_sizes) return emitError(loc, "'pdl_interp.create_operation' op ""requires attribute 'operand_segment_sizes'");

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitError(loc, "'pdl_interp.create_operation' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'pdl_interp.create_operation' op ""attribute 'name' failed to satisfy constraint: string attribute");

  if (tblgen_inputAttributeNames && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_inputAttributeNames))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_inputAttributeNames), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::StringAttr>(attr))); }))))
    return emitError(loc, "'pdl_interp.create_operation' op ""attribute 'inputAttributeNames' failed to satisfy constraint: string array attribute");

  if (tblgen_inferredResultTypes && !((::llvm::isa<::mlir::UnitAttr>(tblgen_inferredResultTypes))))
    return emitError(loc, "'pdl_interp.create_operation' op ""attribute 'inferredResultTypes' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateOperationOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getProperties().operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range CreateOperationOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range CreateOperationOp::getInputOperands() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range CreateOperationOp::getInputAttributes() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range CreateOperationOp::getInputResultTypes() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange CreateOperationOp::getInputOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, {getOperandSegmentSizesAttrName(), getProperties().operand_segment_sizes}));
  return mutableRange;
}

::mlir::MutableOperandRange CreateOperationOp::getInputAttributesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, {getOperandSegmentSizesAttrName(), getProperties().operand_segment_sizes}));
  return mutableRange;
}

::mlir::MutableOperandRange CreateOperationOp::getInputResultTypesMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, {getOperandSegmentSizesAttrName(), getProperties().operand_segment_sizes}));
  return mutableRange;
}

std::pair<unsigned, unsigned> CreateOperationOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateOperationOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> CreateOperationOp::getResultOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult CreateOperationOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.inferredResultTypes;
    auto attr = dict.get("inferredResultTypes");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for inferredResultTypes in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `inferredResultTypes` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.inputAttributeNames;
    auto attr = dict.get("inputAttributeNames");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for inputAttributeNames in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `inputAttributeNames` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.name;
    auto attr = dict.get("name");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for name in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `name` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.operand_segment_sizes;
    auto attr = dict.get("operand_segment_sizes");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for operand_segment_sizes in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `operand_segment_sizes` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CreateOperationOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.inferredResultTypes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("inferredResultTypes",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.inputAttributeNames;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("inputAttributeNames",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("name",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.operand_segment_sizes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("operand_segment_sizes",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code CreateOperationOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.inferredResultTypes.getAsOpaquePointer()), 
    llvm::hash_value(prop.inputAttributeNames.getAsOpaquePointer()), 
    llvm::hash_value(prop.name.getAsOpaquePointer()), 
    llvm::hash_value(prop.operand_segment_sizes.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CreateOperationOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "inferredResultTypes")
      return prop.inferredResultTypes;

    if (name == "inputAttributeNames")
      return prop.inputAttributeNames;

    if (name == "name")
      return prop.name;

    if (name == "operand_segment_sizes")
      return prop.operand_segment_sizes;
  return std::nullopt;
}

void CreateOperationOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "inferredResultTypes") {
       prop.inferredResultTypes = dyn_cast_or_null<std::remove_reference_t<decltype(prop.inferredResultTypes)>>(value);
       return;
    }

    if (name == "inputAttributeNames") {
       prop.inputAttributeNames = dyn_cast_or_null<std::remove_reference_t<decltype(prop.inputAttributeNames)>>(value);
       return;
    }

    if (name == "name") {
       prop.name = dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }

    if (name == "operand_segment_sizes") {
       prop.operand_segment_sizes = dyn_cast_or_null<std::remove_reference_t<decltype(prop.operand_segment_sizes)>>(value);
       return;
    }
}

void CreateOperationOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.inferredResultTypes) attrs.append("inferredResultTypes", prop.inferredResultTypes);

    if (prop.inputAttributeNames) attrs.append("inputAttributeNames", prop.inputAttributeNames);

    if (prop.name) attrs.append("name", prop.name);

    if (prop.operand_segment_sizes) attrs.append("operand_segment_sizes", prop.operand_segment_sizes);
}

::mlir::LogicalResult CreateOperationOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getInferredResultTypesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(attr, "inferredResultTypes", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getInputAttributeNamesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(attr, "inputAttributeNames", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(attr, "name", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::StringAttr CreateOperationOp::getNameAttr() {
  return getProperties().name.cast<::mlir::StringAttr>();
}

::llvm::StringRef CreateOperationOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

::mlir::ArrayAttr CreateOperationOp::getInputAttributeNamesAttr() {
  return getProperties().inputAttributeNames.cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CreateOperationOp::getInputAttributeNames() {
  auto attr = getInputAttributeNamesAttr();
  return attr;
}

::mlir::UnitAttr CreateOperationOp::getInferredResultTypesAttr() {
  return getProperties().inferredResultTypes.dyn_cast_or_null<::mlir::UnitAttr>();
}

bool CreateOperationOp::getInferredResultTypes() {
  auto attr = getInferredResultTypesAttr();
    if (!attr)
      return ((false) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr) != nullptr;
  return attr != nullptr;
}

void CreateOperationOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void CreateOperationOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void CreateOperationOp::setInputAttributeNamesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getInputAttributeNamesAttrName(), attr);
}

void CreateOperationOp::setInferredResultTypesAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getInferredResultTypesAttrName(), attr);
}

void CreateOperationOp::setInferredResultTypes(bool attrValue) {
    if (attrValue)
      return (*this)->setAttr(getInferredResultTypesAttrName(), ((attrValue) ? ::mlir::Builder((*this)->getContext()).getUnitAttr() : nullptr));
    (*this)->removeAttr(getInferredResultTypesAttrName());
}

::mlir::Attribute CreateOperationOp::removeInferredResultTypesAttr() {
    auto &attr = getProperties().inferredResultTypes;
    attr = {};
    return attr;
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, ValueRange types, bool inferredResultTypes, ValueRange operands, ValueRange attributes, ArrayAttr attributeNames) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::OperationType>(), name,
            operands, attributes, attributeNames, types, inferredResultTypes);
    
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultOp, ::mlir::StringAttr name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes, /*optional*/::mlir::UnitAttr inferredResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
   (odsState.getOrAddProperties<Properties>().operand_segment_sizes= odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}));
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.getOrAddProperties<Properties>().inputAttributeNames = inputAttributeNames;
  if (inferredResultTypes) {
    odsState.getOrAddProperties<Properties>().inferredResultTypes = inferredResultTypes;
  }
  odsState.addTypes(resultOp);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes, /*optional*/::mlir::UnitAttr inferredResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
   (odsState.getOrAddProperties<Properties>().operand_segment_sizes= odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}));
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.getOrAddProperties<Properties>().inputAttributeNames = inputAttributeNames;
  if (inferredResultTypes) {
    odsState.getOrAddProperties<Properties>().inferredResultTypes = inferredResultTypes;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultOp, ::llvm::StringRef name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes, /*optional*/bool inferredResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
   (odsState.getOrAddProperties<Properties>().operand_segment_sizes= odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}));
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.getOrAddProperties<Properties>().inputAttributeNames = inputAttributeNames;
  if (inferredResultTypes) {
    odsState.getOrAddProperties<Properties>().inferredResultTypes = ((inferredResultTypes) ? odsBuilder.getUnitAttr() : nullptr);
  }
  odsState.addTypes(resultOp);
}

void CreateOperationOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef name, ::mlir::ValueRange inputOperands, ::mlir::ValueRange inputAttributes, ::mlir::ArrayAttr inputAttributeNames, ::mlir::ValueRange inputResultTypes, /*optional*/bool inferredResultTypes) {
  odsState.addOperands(inputOperands);
  odsState.addOperands(inputAttributes);
  odsState.addOperands(inputResultTypes);
   (odsState.getOrAddProperties<Properties>().operand_segment_sizes= odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputOperands.size()), static_cast<int32_t>(inputAttributes.size()), static_cast<int32_t>(inputResultTypes.size())}));
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.getOrAddProperties<Properties>().inputAttributeNames = inputAttributeNames;
  if (inferredResultTypes) {
    odsState.getOrAddProperties<Properties>().inferredResultTypes = ((inferredResultTypes) ? odsBuilder.getUnitAttr() : nullptr);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateOperationOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateOperationOp::verifyInvariantsImpl() {
  auto tblgen_inferredResultTypes = getProperties().inferredResultTypes; (void)tblgen_inferredResultTypes;
  auto tblgen_inputAttributeNames = getProperties().inputAttributeNames; (void)tblgen_inputAttributeNames;
  if (!tblgen_inputAttributeNames) return emitOpError("requires attribute 'inputAttributeNames'");
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");
  auto tblgen_operand_segment_sizes = getProperties().operand_segment_sizes; (void)tblgen_operand_segment_sizes;
  if (!tblgen_operand_segment_sizes) return emitOpError("requires attribute 'operand_segment_sizes'");

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 3)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(*this, tblgen_inputAttributeNames, "inputAttributeNames")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps3(*this, tblgen_inferredResultTypes, "inferredResultTypes")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateOperationOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CreateOperationOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputOperandsOperands;
  ::llvm::SMLoc inputOperandsOperandsLoc;
  (void)inputOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputOperandsTypes;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputAttributesOperands;
  ::llvm::SMLoc inputAttributesOperandsLoc;
  (void)inputAttributesOperandsLoc;
  ::mlir::ArrayAttr inputAttributeNamesAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputResultTypesOperands;
  ::llvm::SMLoc inputResultTypesOperandsLoc;
  (void)inputResultTypesOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputResultTypesTypes;
  ::mlir::UnitAttr inferredResultTypesAttr;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.getOrAddProperties<CreateOperationOp::Properties>().name = nameAttr;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  inputOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  {
    inputAttributesOperandsLoc = parser.getCurrentLocation();
    if (parseCreateOperationOpAttributes(parser, inputAttributesOperands, inputAttributeNamesAttr))
      return ::mlir::failure();
    result.getOrAddProperties<CreateOperationOp::Properties>().inputAttributeNames = inputAttributeNamesAttr;
  }
  {
    inputResultTypesOperandsLoc = parser.getCurrentLocation();
    if (parseCreateOperationOpResults(parser, inputResultTypesOperands, inputResultTypesTypes, inferredResultTypesAttr))
      return ::mlir::failure();
    if (inferredResultTypesAttr)
      result.getOrAddProperties<CreateOperationOp::Properties>().inferredResultTypes = inferredResultTypesAttr;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  result.getOrAddProperties<CreateOperationOp::Properties>().operand_segment_sizes = (parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(inputOperandsOperands.size()), static_cast<int32_t>(inputAttributesOperands.size()), static_cast<int32_t>(inputResultTypesOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputOperandsOperands, inputOperandsTypes, inputOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(inputAttributesOperands, odsBuildableType1, inputAttributesOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(inputResultTypesOperands, inputResultTypesTypes, inputResultTypesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateOperationOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  if (!getInputOperands().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getInputOperands();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getInputOperands().getTypes();
    _odsPrinter << ")";
  }
  printCreateOperationOpAttributes(_odsPrinter, *this, getInputAttributes(), getInputAttributeNamesAttr());
  _odsPrinter << ' ';
  printCreateOperationOpResults(_odsPrinter, *this, getInputResultTypes(), getInputResultTypes().getTypes(), getInferredResultTypesAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("name");
  elidedAttrs.push_back("inputAttributeNames");
  elidedAttrs.push_back("inferredResultTypes");
  {
     ::mlir::Builder odsBuilder(getContext());
     ::mlir::Attribute attr = getInferredResultTypesAttr();
     if(attr && (attr == ((false) ? odsBuilder.getUnitAttr() : nullptr)))
       elidedAttrs.push_back("inferredResultTypes");
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateOperationOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateRangeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateRangeOpGenericAdaptorBase::CreateRangeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.create_range", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CreateRangeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr CreateRangeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
CreateRangeOpAdaptor::CreateRangeOpAdaptor(CreateRangeOp op) : CreateRangeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult CreateRangeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateRangeOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range CreateRangeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range CreateRangeOp::getArguments() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange CreateRangeOp::getArgumentsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CreateRangeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateRangeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::RangeType> CreateRangeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSResults(0).begin());
}

void CreateRangeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ValueRange arguments) {
  odsState.addOperands(arguments);
  odsState.addTypes(result);
}

void CreateRangeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateRangeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps7(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateRangeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult CreateRangeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argumentsOperands;
  ::llvm::SMLoc argumentsOperandsLoc;
  (void)argumentsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argumentsTypes;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  argumentsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argumentsOperands))
    return ::mlir::failure();
  if (!argumentsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argumentsTypes))
    return ::mlir::failure();
  }
  {
    if (parseRangeType(parser, argumentsTypes, resultRawTypes[0]))
      return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(argumentsOperands, argumentsTypes, argumentsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateRangeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getArguments().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << getArguments();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getArguments().getTypes();
  }
  _odsPrinter << ' ';
  printRangeType(_odsPrinter, *this, getArguments().getTypes(), getResult().getType());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void CreateRangeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateRangeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateTypeOpGenericAdaptorBase::CreateTypeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.create_type", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CreateTypeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CreateTypeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr CreateTypeOpGenericAdaptorBase::getValueAttr() {
  auto attr = getProperties().value.cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::Type CreateTypeOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

} // namespace detail
CreateTypeOpAdaptor::CreateTypeOpAdaptor(CreateTypeOp op) : CreateTypeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult CreateTypeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitError(loc, "'pdl_interp.create_type' op ""requires attribute 'value'");

  if (tblgen_value && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_value))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(tblgen_value).getValue()))) && ((true))))
    return emitError(loc, "'pdl_interp.create_type' op ""attribute 'value' failed to satisfy constraint: any type attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CreateTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CreateTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::TypeType> CreateTypeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::TypeType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult CreateTypeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.value;
    auto attr = dict.get("value");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for value in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `value` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CreateTypeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.value;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("value",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code CreateTypeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.value.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CreateTypeOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "value")
      return prop.value;
  return std::nullopt;
}

void CreateTypeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "value") {
       prop.value = dyn_cast_or_null<std::remove_reference_t<decltype(prop.value)>>(value);
       return;
    }
}

void CreateTypeOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.value) attrs.append("value", prop.value);
}

::mlir::LogicalResult CreateTypeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getValueAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(attr, "value", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::TypeAttr CreateTypeOp::getValueAttr() {
  return getProperties().value.cast<::mlir::TypeAttr>();
}

::mlir::Type CreateTypeOp::getValue() {
  auto attr = getValueAttr();
  return ::llvm::cast<::mlir::Type>(attr.getValue());
}

void CreateTypeOp::setValueAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void CreateTypeOp::setValue(::mlir::Type attrValue) {
  (*this)->setAttr(getValueAttrName(), ::mlir::TypeAttr::get(attrValue));
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeAttr type) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::TypeType>(), type);
    
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::TypeAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  odsState.addTypes(result);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::TypeAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Type value) {
  odsState.getOrAddProperties<Properties>().value = ::mlir::TypeAttr::get(value);
  odsState.addTypes(result);
}

void CreateTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Type value) {
  odsState.getOrAddProperties<Properties>().value = ::mlir::TypeAttr::get(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateTypeOp::verifyInvariantsImpl() {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitOpError("requires attribute 'value'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps4(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateTypeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::TypeAttr valueAttr;

  if (parser.parseCustomAttributeWithFallback(valueAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (valueAttr) result.getOrAddProperties<CreateTypeOp::Properties>().value = valueAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void CreateTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getValueAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void CreateTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::CreateTypesOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
CreateTypesOpGenericAdaptorBase::CreateTypesOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.create_types", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> CreateTypesOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr CreateTypesOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr CreateTypesOpGenericAdaptorBase::getValueAttr() {
  auto attr = getProperties().value.cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr CreateTypesOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr;
}

} // namespace detail
CreateTypesOpAdaptor::CreateTypesOpAdaptor(CreateTypesOp op) : CreateTypesOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult CreateTypesOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitError(loc, "'pdl_interp.create_types' op ""requires attribute 'value'");

  if (tblgen_value && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_value))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_value), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'pdl_interp.create_types' op ""attribute 'value' failed to satisfy constraint: type array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CreateTypesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CreateTypesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CreateTypesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CreateTypesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::RangeType> CreateTypesOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult CreateTypesOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.value;
    auto attr = dict.get("value");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for value in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `value` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute CreateTypesOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.value;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("value",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code CreateTypesOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.value.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> CreateTypesOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "value")
      return prop.value;
  return std::nullopt;
}

void CreateTypesOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "value") {
       prop.value = dyn_cast_or_null<std::remove_reference_t<decltype(prop.value)>>(value);
       return;
    }
}

void CreateTypesOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.value) attrs.append("value", prop.value);
}

::mlir::LogicalResult CreateTypesOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getValueAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(attr, "value", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::ArrayAttr CreateTypesOp::getValueAttr() {
  return getProperties().value.cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr CreateTypesOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void CreateTypesOp::setValueAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void CreateTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ArrayAttr type) {
      build(odsBuilder, odsState,
            pdl::RangeType::get(odsBuilder.getType<pdl::TypeType>()), type);
    
}

void CreateTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::ArrayAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  odsState.addTypes(result);
}

void CreateTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ArrayAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CreateTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CreateTypesOp::verifyInvariantsImpl() {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitOpError("requires attribute 'value'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CreateTypesOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CreateTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::ArrayAttr valueAttr;

  if (parser.parseCustomAttributeWithFallback(valueAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (valueAttr) result.getOrAddProperties<CreateTypesOp::Properties>().value = valueAttr;
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  result.addTypes(odsBuildableType0);
  return ::mlir::success();
}

void CreateTypesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getValueAttr());
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("value");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void CreateTypesOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::CreateTypesOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::EraseOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
EraseOpGenericAdaptorBase::EraseOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.erase", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> EraseOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr EraseOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
EraseOpAdaptor::EraseOpAdaptor(EraseOp op) : EraseOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult EraseOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> EraseOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range EraseOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> EraseOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange EraseOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> EraseOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EraseOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp) {
  odsState.addOperands(inputOp);
}

void EraseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp) {
  odsState.addOperands(inputOp);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EraseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult EraseOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult EraseOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult EraseOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EraseOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::EraseOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ExtractOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ExtractOpGenericAdaptorBase::ExtractOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.extract", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ExtractOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ExtractOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr ExtractOpGenericAdaptorBase::getIndexAttr() {
  auto attr = getProperties().index.cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t ExtractOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
ExtractOpAdaptor::ExtractOpAdaptor(ExtractOp op) : ExtractOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ExtractOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_index = getProperties().index; (void)tblgen_index;
  if (!tblgen_index) return emitError(loc, "'pdl_interp.extract' op ""requires attribute 'index'");

  if (tblgen_index && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_index))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.extract' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExtractOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExtractOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::RangeType> ExtractOp::getRange() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ExtractOp::getRangeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::PDLType> ExtractOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult ExtractOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.index;
    auto attr = dict.get("index");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for index in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `index` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ExtractOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.index;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("index",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code ExtractOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.index.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ExtractOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "index")
      return prop.index;
  return std::nullopt;
}

void ExtractOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "index") {
       prop.index = dyn_cast_or_null<std::remove_reference_t<decltype(prop.index)>>(value);
       return;
    }
}

void ExtractOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.index) attrs.append("index", prop.index);
}

::mlir::LogicalResult ExtractOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getIndexAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(attr, "index", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr ExtractOp::getIndexAttr() {
  return getProperties().index.cast<::mlir::IntegerAttr>();
}

uint32_t ExtractOp::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

void ExtractOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void ExtractOp::setIndex(uint32_t attrValue) {
  (*this)->setAttr(getIndexAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value range, unsigned index) {
      build(odsBuilder, odsState,
            ::llvm::cast<pdl::RangeType>(range.getType()).getElementType(),
            range, index);
    
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value range, ::mlir::IntegerAttr index) {
  odsState.addOperands(range);
  odsState.getOrAddProperties<Properties>().index = index;
  odsState.addTypes(result);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value range, ::mlir::IntegerAttr index) {
  odsState.addOperands(range);
  odsState.getOrAddProperties<Properties>().index = index;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value range, uint32_t index) {
  odsState.addOperands(range);
  odsState.getOrAddProperties<Properties>().index = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index);
  odsState.addTypes(result);
}

void ExtractOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value range, uint32_t index) {
  odsState.addOperands(range);
  odsState.getOrAddProperties<Properties>().index = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExtractOp::verifyInvariantsImpl() {
  auto tblgen_index = getProperties().index; (void)tblgen_index;
  if (!tblgen_index) return emitOpError("requires attribute 'index'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()(pdl::RangeType::get((*this->getODSResults(0).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that `range` is a PDL range whose element type matches type of `result`");
  return ::mlir::success();
}

::mlir::LogicalResult ExtractOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ExtractOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand rangeRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rangeOperands(rangeRawOperands);  ::llvm::SMLoc rangeOperandsLoc;
  (void)rangeOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);

  if (parser.parseCustomAttributeWithFallback(indexAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (indexAttr) result.getOrAddProperties<ExtractOp::Properties>().index = indexAttr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  rangeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rangeRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  for (::mlir::Type type : resultTypes) {
    (void)type;
    if (!((::llvm::isa<::mlir::pdl::PDLType>(type)))) {
      return parser.emitError(parser.getNameLoc()) << "'result' must be pdl type, but got " << type;
    }
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(rangeOperands, pdl::RangeType::get(resultTypes[0]), rangeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ExtractOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getRange();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ExtractOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ExtractOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::FinalizeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FinalizeOpGenericAdaptorBase::FinalizeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.finalize", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FinalizeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FinalizeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
FinalizeOpAdaptor::FinalizeOpAdaptor(FinalizeOp op) : FinalizeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult FinalizeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FinalizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FinalizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> FinalizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FinalizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void FinalizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void FinalizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FinalizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FinalizeOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult FinalizeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FinalizeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  return ::mlir::success();
}

void FinalizeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void FinalizeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::FinalizeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ForEachOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ForEachOpGenericAdaptorBase::ForEachOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.foreach", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ForEachOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ForEachOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::Region &ForEachOpGenericAdaptorBase::getRegion() {
  return *odsRegions[0];
}

::mlir::RegionRange ForEachOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
ForEachOpAdaptor::ForEachOpAdaptor(ForEachOp op) : ForEachOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ForEachOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ForEachOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ForEachOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::RangeType> ForEachOp::getValues() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ForEachOp::getValuesMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ForEachOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ForEachOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &ForEachOp::getRegion() {
  return (*this)->getRegion(0);
}

::mlir::Block *ForEachOp::getSuccessor() {
  return (*this)->getSuccessor(0);
}

void ForEachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value values, ::mlir::Block *successor) {
  odsState.addOperands(values);
  (void)odsState.addRegion();
  odsState.addSuccessors(successor);
}

void ForEachOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value values, ::mlir::Block *successor) {
  odsState.addOperands(values);
  (void)odsState.addRegion();
  odsState.addSuccessors(successor);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ForEachOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ForEachOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_PDLInterpOps0(*this, region, "region", index++)))
        return ::mlir::failure();
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult ForEachOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ForEachOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::FuncOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FuncOpGenericAdaptorBase::FuncOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.func", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> FuncOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FuncOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr FuncOpGenericAdaptorBase::getSymNameAttr() {
  auto attr = getProperties().sym_name.cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef FuncOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr FuncOpGenericAdaptorBase::getFunctionTypeAttr() {
  auto attr = getProperties().function_type.cast<::mlir::TypeAttr>();
  return attr;
}

::mlir::FunctionType FuncOpGenericAdaptorBase::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return ::llvm::cast<::mlir::FunctionType>(attr.getValue());
}

::mlir::ArrayAttr FuncOpGenericAdaptorBase::getArgAttrsAttr() {
  auto attr = getProperties().arg_attrs.dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > FuncOpGenericAdaptorBase::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr FuncOpGenericAdaptorBase::getResAttrsAttr() {
  auto attr = getProperties().res_attrs.dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > FuncOpGenericAdaptorBase::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::Region &FuncOpGenericAdaptorBase::getBody() {
  return *odsRegions[0];
}

::mlir::RegionRange FuncOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
FuncOpAdaptor::FuncOpAdaptor(FuncOp op) : FuncOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult FuncOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_arg_attrs = getProperties().arg_attrs; (void)tblgen_arg_attrs;
  auto tblgen_function_type = getProperties().function_type; (void)tblgen_function_type;
  if (!tblgen_function_type) return emitError(loc, "'pdl_interp.func' op ""requires attribute 'function_type'");
  auto tblgen_res_attrs = getProperties().res_attrs; (void)tblgen_res_attrs;
  auto tblgen_sym_name = getProperties().sym_name; (void)tblgen_sym_name;
  if (!tblgen_sym_name) return emitError(loc, "'pdl_interp.func' op ""requires attribute 'sym_name'");

  if (tblgen_sym_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_sym_name))))
    return emitError(loc, "'pdl_interp.func' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_function_type && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_function_type))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(tblgen_function_type).getValue()))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(tblgen_function_type).getValue())))))
    return emitError(loc, "'pdl_interp.func' op ""attribute 'function_type' failed to satisfy constraint: type attribute of function type");

  if (tblgen_arg_attrs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_arg_attrs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_arg_attrs), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return emitError(loc, "'pdl_interp.func' op ""attribute 'arg_attrs' failed to satisfy constraint: Array of dictionary attributes");

  if (tblgen_res_attrs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_res_attrs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_res_attrs), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return emitError(loc, "'pdl_interp.func' op ""attribute 'res_attrs' failed to satisfy constraint: Array of dictionary attributes");
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

::mlir::LogicalResult FuncOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.arg_attrs;
    auto attr = dict.get("arg_attrs");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for arg_attrs in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `arg_attrs` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.function_type;
    auto attr = dict.get("function_type");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for function_type in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `function_type` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.res_attrs;
    auto attr = dict.get("res_attrs");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for res_attrs in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `res_attrs` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.sym_name;
    auto attr = dict.get("sym_name");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for sym_name in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `sym_name` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute FuncOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.arg_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("arg_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.function_type;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("function_type",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.res_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("res_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.sym_name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("sym_name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code FuncOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.arg_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.function_type.getAsOpaquePointer()), 
    llvm::hash_value(prop.res_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.sym_name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> FuncOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "arg_attrs")
      return prop.arg_attrs;

    if (name == "function_type")
      return prop.function_type;

    if (name == "res_attrs")
      return prop.res_attrs;

    if (name == "sym_name")
      return prop.sym_name;
  return std::nullopt;
}

void FuncOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "arg_attrs") {
       prop.arg_attrs = dyn_cast_or_null<std::remove_reference_t<decltype(prop.arg_attrs)>>(value);
       return;
    }

    if (name == "function_type") {
       prop.function_type = dyn_cast_or_null<std::remove_reference_t<decltype(prop.function_type)>>(value);
       return;
    }

    if (name == "res_attrs") {
       prop.res_attrs = dyn_cast_or_null<std::remove_reference_t<decltype(prop.res_attrs)>>(value);
       return;
    }

    if (name == "sym_name") {
       prop.sym_name = dyn_cast_or_null<std::remove_reference_t<decltype(prop.sym_name)>>(value);
       return;
    }
}

void FuncOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.arg_attrs) attrs.append("arg_attrs", prop.arg_attrs);

    if (prop.function_type) attrs.append("function_type", prop.function_type);

    if (prop.res_attrs) attrs.append("res_attrs", prop.res_attrs);

    if (prop.sym_name) attrs.append("sym_name", prop.sym_name);
}

::mlir::LogicalResult FuncOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getArgAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps8(attr, "arg_attrs", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getFunctionTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps7(attr, "function_type", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getResAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps8(attr, "res_attrs", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSymNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(attr, "sym_name", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::StringAttr FuncOp::getSymNameAttr() {
  return getProperties().sym_name.cast<::mlir::StringAttr>();
}

::llvm::StringRef FuncOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr FuncOp::getFunctionTypeAttr() {
  return getProperties().function_type.cast<::mlir::TypeAttr>();
}

::mlir::FunctionType FuncOp::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return ::llvm::cast<::mlir::FunctionType>(attr.getValue());
}

::mlir::ArrayAttr FuncOp::getArgAttrsAttr() {
  return getProperties().arg_attrs.dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > FuncOp::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr FuncOp::getResAttrsAttr() {
  return getProperties().res_attrs.dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > FuncOp::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void FuncOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void FuncOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void FuncOp::setFunctionTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getFunctionTypeAttrName(), attr);
}

void FuncOp::setFunctionType(::mlir::FunctionType attrValue) {
  (*this)->setAttr(getFunctionTypeAttrName(), ::mlir::TypeAttr::get(attrValue));
}

void FuncOp::setArgAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getArgAttrsAttrName(), attr);
}

void FuncOp::setResAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getResAttrsAttrName(), attr);
}

::mlir::Attribute FuncOp::removeArgAttrsAttr() {
    auto &attr = getProperties().arg_attrs;
    attr = {};
    return attr;
}

::mlir::Attribute FuncOp::removeResAttrsAttr() {
    auto &attr = getProperties().res_attrs;
    attr = {};
    return attr;
}

::mlir::LogicalResult FuncOp::verifyInvariantsImpl() {
  auto tblgen_arg_attrs = getProperties().arg_attrs; (void)tblgen_arg_attrs;
  auto tblgen_function_type = getProperties().function_type; (void)tblgen_function_type;
  if (!tblgen_function_type) return emitOpError("requires attribute 'function_type'");
  auto tblgen_res_attrs = getProperties().res_attrs; (void)tblgen_res_attrs;
  auto tblgen_sym_name = getProperties().sym_name; (void)tblgen_sym_name;
  if (!tblgen_sym_name) return emitOpError("requires attribute 'sym_name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps7(*this, tblgen_function_type, "function_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps8(*this, tblgen_arg_attrs, "arg_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps8(*this, tblgen_res_attrs, "res_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_PDLInterpOps1(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult FuncOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::FuncOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetAttributeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetAttributeOpGenericAdaptorBase::GetAttributeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_attribute", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetAttributeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetAttributeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr GetAttributeOpGenericAdaptorBase::getNameAttr() {
  auto attr = getProperties().name.cast<::mlir::StringAttr>();
  return attr;
}

::llvm::StringRef GetAttributeOpGenericAdaptorBase::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

} // namespace detail
GetAttributeOpAdaptor::GetAttributeOpAdaptor(GetAttributeOp op) : GetAttributeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult GetAttributeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitError(loc, "'pdl_interp.get_attribute' op ""requires attribute 'name'");

  if (tblgen_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_name))))
    return emitError(loc, "'pdl_interp.get_attribute' op ""attribute 'name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetAttributeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetAttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> GetAttributeOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange GetAttributeOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetAttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetAttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::AttributeType> GetAttributeOp::getAttribute() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::AttributeType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult GetAttributeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.name;
    auto attr = dict.get("name");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for name in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `name` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GetAttributeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code GetAttributeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GetAttributeOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "name")
      return prop.name;
  return std::nullopt;
}

void GetAttributeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "name") {
       prop.name = dyn_cast_or_null<std::remove_reference_t<decltype(prop.name)>>(value);
       return;
    }
}

void GetAttributeOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.name) attrs.append("name", prop.name);
}

::mlir::LogicalResult GetAttributeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(attr, "name", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::StringAttr GetAttributeOp::getNameAttr() {
  return getProperties().name.cast<::mlir::StringAttr>();
}

::llvm::StringRef GetAttributeOp::getName() {
  auto attr = getNameAttr();
  return attr.getValue();
}

void GetAttributeOp::setNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getNameAttrName(), attr);
}

void GetAttributeOp::setName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attribute, ::mlir::Value inputOp, ::mlir::StringAttr name) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = name;
  odsState.addTypes(attribute);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::StringAttr name) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = name;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type attribute, ::mlir::Value inputOp, ::llvm::StringRef name) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  odsState.addTypes(attribute);
}

void GetAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::llvm::StringRef name) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().name = odsBuilder.getStringAttr(name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetAttributeOp::verifyInvariantsImpl() {
  auto tblgen_name = getProperties().name; (void)tblgen_name;
  if (!tblgen_name) return emitOpError("requires attribute 'name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_name, "name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetAttributeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr nameAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;

  if (parser.parseCustomAttributeWithFallback(nameAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (nameAttr) result.getOrAddProperties<GetAttributeOp::Properties>().name = nameAttr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType1, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getNameAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("name");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetAttributeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetAttributeTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetAttributeTypeOpGenericAdaptorBase::GetAttributeTypeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_attribute_type", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetAttributeTypeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetAttributeTypeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
GetAttributeTypeOpAdaptor::GetAttributeTypeOpAdaptor(GetAttributeTypeOp op) : GetAttributeTypeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult GetAttributeTypeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetAttributeTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetAttributeTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::AttributeType> GetAttributeTypeOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::AttributeType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange GetAttributeTypeOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetAttributeTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetAttributeTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::TypeType> GetAttributeTypeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::TypeType>>(*getODSResults(0).begin());
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value) {
      build(odsBuilder, odsState, odsBuilder.getType<pdl::TypeType>(), value);
    
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(result);
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetAttributeTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetAttributeTypeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetAttributeTypeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetAttributeTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueOperands, odsBuildableType1, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetAttributeTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetAttributeTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetAttributeTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetDefiningOpOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetDefiningOpOpGenericAdaptorBase::GetDefiningOpOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_defining_op", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetDefiningOpOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetDefiningOpOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
GetDefiningOpOpAdaptor::GetDefiningOpOpAdaptor(GetDefiningOpOp op) : GetDefiningOpOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult GetDefiningOpOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetDefiningOpOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetDefiningOpOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::PDLType> GetDefiningOpOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange GetDefiningOpOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetDefiningOpOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetDefiningOpOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> GetDefiningOpOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSResults(0).begin());
}

void GetDefiningOpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type inputOp, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(inputOp);
}

void GetDefiningOpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetDefiningOpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetDefiningOpOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetDefiningOpOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetDefiningOpOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetDefiningOpOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetDefiningOpOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetDefiningOpOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetOperandOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetOperandOpGenericAdaptorBase::GetOperandOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_operand", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetOperandOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetOperandOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetOperandOpGenericAdaptorBase::getIndexAttr() {
  auto attr = getProperties().index.cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t GetOperandOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
GetOperandOpAdaptor::GetOperandOpAdaptor(GetOperandOp op) : GetOperandOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult GetOperandOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_index = getProperties().index; (void)tblgen_index;
  if (!tblgen_index) return emitError(loc, "'pdl_interp.get_operand' op ""requires attribute 'index'");

  if (tblgen_index && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_index))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.get_operand' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetOperandOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetOperandOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> GetOperandOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange GetOperandOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetOperandOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetOperandOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::ValueType> GetOperandOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::ValueType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult GetOperandOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.index;
    auto attr = dict.get("index");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for index in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `index` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GetOperandOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.index;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("index",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code GetOperandOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.index.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GetOperandOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "index")
      return prop.index;
  return std::nullopt;
}

void GetOperandOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "index") {
       prop.index = dyn_cast_or_null<std::remove_reference_t<decltype(prop.index)>>(value);
       return;
    }
}

void GetOperandOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.index) attrs.append("index", prop.index);
}

::mlir::LogicalResult GetOperandOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getIndexAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(attr, "index", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr GetOperandOp::getIndexAttr() {
  return getProperties().index.cast<::mlir::IntegerAttr>();
}

uint32_t GetOperandOp::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

void GetOperandOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void GetOperandOp::setIndex(uint32_t attrValue) {
  (*this)->setAttr(getIndexAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, ::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = index;
  odsState.addTypes(value);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = index;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, uint32_t index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index);
  odsState.addTypes(value);
}

void GetOperandOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetOperandOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetOperandOp::verifyInvariantsImpl() {
  auto tblgen_index = getProperties().index; (void)tblgen_index;
  if (!tblgen_index) return emitOpError("requires attribute 'index'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetOperandOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetOperandOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;

  if (parser.parseCustomAttributeWithFallback(indexAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (indexAttr) result.getOrAddProperties<GetOperandOp::Properties>().index = indexAttr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType1, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetOperandOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetOperandOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetOperandOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetOperandsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetOperandsOpGenericAdaptorBase::GetOperandsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_operands", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetOperandsOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetOperandsOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetOperandsOpGenericAdaptorBase::getIndexAttr() {
  auto attr = getProperties().index.dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint32_t> GetOperandsOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
GetOperandsOpAdaptor::GetOperandsOpAdaptor(GetOperandsOp op) : GetOperandsOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult GetOperandsOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_index = getProperties().index; (void)tblgen_index;

  if (tblgen_index && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_index))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.get_operands' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetOperandsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetOperandsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> GetOperandsOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange GetOperandsOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetOperandsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetOperandsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::PDLType> GetOperandsOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult GetOperandsOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.index;
    auto attr = dict.get("index");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for index in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `index` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GetOperandsOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.index;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("index",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code GetOperandsOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.index.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GetOperandsOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "index")
      return prop.index;
  return std::nullopt;
}

void GetOperandsOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "index") {
       prop.index = dyn_cast_or_null<std::remove_reference_t<decltype(prop.index)>>(value);
       return;
    }
}

void GetOperandsOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.index) attrs.append("index", prop.index);
}

::mlir::LogicalResult GetOperandsOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getIndexAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(attr, "index", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr GetOperandsOp::getIndexAttr() {
  return getProperties().index.dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint32_t> GetOperandsOp::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void GetOperandsOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void GetOperandsOp::setIndex(::std::optional<uint32_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getIndexAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), *attrValue));
    (*this)->removeAttr(getIndexAttrName());
}

::mlir::Attribute GetOperandsOp::removeIndexAttr() {
    auto &attr = getProperties().index;
    attr = {};
    return attr;
}

void GetOperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value inputOp, std::optional<unsigned> index) {
      build(odsBuilder, odsState, resultType, inputOp,
            index ? odsBuilder.getI32IntegerAttr(*index) : IntegerAttr());
    
}

void GetOperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  if (index) {
    odsState.getOrAddProperties<Properties>().index = index;
  }
  odsState.addTypes(value);
}

void GetOperandsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  if (index) {
    odsState.getOrAddProperties<Properties>().index = index;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetOperandsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetOperandsOp::verifyInvariantsImpl() {
  auto tblgen_index = getProperties().index; (void)tblgen_index;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetOperandsOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetOperandsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  ::mlir::OptionalParseResult parseResultindexAttr =
    parser.parseOptionalAttribute(indexAttr, parser.getBuilder().getIntegerType(32));
  if (parseResultindexAttr.has_value() && failed(*parseResultindexAttr))
    return ::mlir::failure();
  if (parseResultindexAttr.has_value() && succeeded(*parseResultindexAttr))
  if (indexAttr) result.getOrAddProperties<GetOperandsOp::Properties>().index = indexAttr;
  if (indexAttr) {
  }
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(valueTypes);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetOperandsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (getIndexAttr()) {
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getIndexAttr());
  }
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetOperandsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetOperandsOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetResultOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetResultOpGenericAdaptorBase::GetResultOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_result", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetResultOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetResultOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetResultOpGenericAdaptorBase::getIndexAttr() {
  auto attr = getProperties().index.cast<::mlir::IntegerAttr>();
  return attr;
}

uint32_t GetResultOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
GetResultOpAdaptor::GetResultOpAdaptor(GetResultOp op) : GetResultOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult GetResultOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_index = getProperties().index; (void)tblgen_index;
  if (!tblgen_index) return emitError(loc, "'pdl_interp.get_result' op ""requires attribute 'index'");

  if (tblgen_index && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_index))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.get_result' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetResultOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetResultOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> GetResultOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange GetResultOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetResultOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetResultOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::ValueType> GetResultOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::ValueType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult GetResultOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.index;
    auto attr = dict.get("index");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for index in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `index` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GetResultOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.index;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("index",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code GetResultOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.index.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GetResultOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "index")
      return prop.index;
  return std::nullopt;
}

void GetResultOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "index") {
       prop.index = dyn_cast_or_null<std::remove_reference_t<decltype(prop.index)>>(value);
       return;
    }
}

void GetResultOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.index) attrs.append("index", prop.index);
}

::mlir::LogicalResult GetResultOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getIndexAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(attr, "index", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr GetResultOp::getIndexAttr() {
  return getProperties().index.cast<::mlir::IntegerAttr>();
}

uint32_t GetResultOp::getIndex() {
  auto attr = getIndexAttr();
  return attr.getValue().getZExtValue();
}

void GetResultOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void GetResultOp::setIndex(uint32_t attrValue) {
  (*this)->setAttr(getIndexAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), attrValue));
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, ::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = index;
  odsState.addTypes(value);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = index;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, uint32_t index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index);
  odsState.addTypes(value);
}

void GetResultOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, uint32_t index) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().index = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), index);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetResultOp::verifyInvariantsImpl() {
  auto tblgen_index = getProperties().index; (void)tblgen_index;
  if (!tblgen_index) return emitOpError("requires attribute 'index'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetResultOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetResultOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;

  if (parser.parseCustomAttributeWithFallback(indexAttr, parser.getBuilder().getIntegerType(32))) {
    return ::mlir::failure();
  }
  if (indexAttr) result.getOrAddProperties<GetResultOp::Properties>().index = indexAttr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::ValueType>();
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType1, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getIndexAttr());
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetResultOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetResultOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetResultsOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetResultsOpGenericAdaptorBase::GetResultsOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_results", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetResultsOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetResultsOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr GetResultsOpGenericAdaptorBase::getIndexAttr() {
  auto attr = getProperties().index.dyn_cast_or_null<::mlir::IntegerAttr>();
  return attr;
}

::std::optional<uint32_t> GetResultsOpGenericAdaptorBase::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

} // namespace detail
GetResultsOpAdaptor::GetResultsOpAdaptor(GetResultsOp op) : GetResultsOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult GetResultsOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_index = getProperties().index; (void)tblgen_index;

  if (tblgen_index && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_index))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getType().isSignlessInteger(32)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_index).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.get_results' op ""attribute 'index' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetResultsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetResultsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> GetResultsOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange GetResultsOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetResultsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetResultsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::PDLType> GetResultsOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSResults(0).begin());
}

::mlir::LogicalResult GetResultsOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.index;
    auto attr = dict.get("index");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for index in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `index` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GetResultsOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.index;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("index",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code GetResultsOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.index.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GetResultsOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "index")
      return prop.index;
  return std::nullopt;
}

void GetResultsOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "index") {
       prop.index = dyn_cast_or_null<std::remove_reference_t<decltype(prop.index)>>(value);
       return;
    }
}

void GetResultsOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.index) attrs.append("index", prop.index);
}

::mlir::LogicalResult GetResultsOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getIndexAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(attr, "index", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::IntegerAttr GetResultsOp::getIndexAttr() {
  return getProperties().index.dyn_cast_or_null<::mlir::IntegerAttr>();
}

::std::optional<uint32_t> GetResultsOp::getIndex() {
  auto attr = getIndexAttr();
  return attr ? ::std::optional<uint32_t>(attr.getValue().getZExtValue()) : (::std::nullopt);
}

void GetResultsOp::setIndexAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIndexAttrName(), attr);
}

void GetResultsOp::setIndex(::std::optional<uint32_t> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getIndexAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), *attrValue));
    (*this)->removeAttr(getIndexAttrName());
}

::mlir::Attribute GetResultsOp::removeIndexAttr() {
    auto &attr = getProperties().index;
    attr = {};
    return attr;
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value inputOp, std::optional<unsigned> index) {
      build(odsBuilder, odsState, resultType, inputOp,
            index ? odsBuilder.getI32IntegerAttr(*index) : IntegerAttr());
    
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value inputOp) {
      build(odsBuilder, odsState,
            pdl::RangeType::get(odsBuilder.getType<pdl::ValueType>()), inputOp,
            IntegerAttr());
    
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value inputOp, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  if (index) {
    odsState.getOrAddProperties<Properties>().index = index;
  }
  odsState.addTypes(value);
}

void GetResultsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, /*optional*/::mlir::IntegerAttr index) {
  odsState.addOperands(inputOp);
  if (index) {
    odsState.getOrAddProperties<Properties>().index = index;
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetResultsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetResultsOp::verifyInvariantsImpl() {
  auto tblgen_index = getProperties().index; (void)tblgen_index;

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps2(*this, tblgen_index, "index")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetResultsOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetResultsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::IntegerAttr indexAttr;
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  ::mlir::OptionalParseResult parseResultindexAttr =
    parser.parseOptionalAttribute(indexAttr, parser.getBuilder().getIntegerType(32));
  if (parseResultindexAttr.has_value() && failed(*parseResultindexAttr))
    return ::mlir::failure();
  if (parseResultindexAttr.has_value() && succeeded(*parseResultindexAttr))
  if (indexAttr) result.getOrAddProperties<GetResultsOp::Properties>().index = indexAttr;
  if (indexAttr) {
  }
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  result.addTypes(valueTypes);
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetResultsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (getIndexAttr()) {
    _odsPrinter << ' ';
    _odsPrinter.printAttributeWithoutType(getIndexAttr());
  }
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("index");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetResultsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetResultsOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetUsersOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetUsersOpGenericAdaptorBase::GetUsersOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_users", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetUsersOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetUsersOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
GetUsersOpAdaptor::GetUsersOpAdaptor(GetUsersOp op) : GetUsersOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult GetUsersOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetUsersOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetUsersOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::PDLType> GetUsersOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange GetUsersOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetUsersOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetUsersOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::RangeType> GetUsersOp::getOperations() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSResults(0).begin());
}

void GetUsersOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value) {
      build(odsBuilder, odsState,
            pdl::RangeType::get(odsBuilder.getType<pdl::OperationType>()),
            value);
    
}

void GetUsersOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type operations, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(operations);
}

void GetUsersOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetUsersOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetUsersOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetUsersOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetUsersOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::OperationType>());
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetUsersOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetUsersOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetUsersOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::GetValueTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GetValueTypeOpGenericAdaptorBase::GetValueTypeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.get_value_type", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> GetValueTypeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr GetValueTypeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
GetValueTypeOpAdaptor::GetValueTypeOpAdaptor(GetValueTypeOp op) : GetValueTypeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult GetValueTypeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetValueTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetValueTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::PDLType> GetValueTypeOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange GetValueTypeOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetValueTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetValueTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::PDLType> GetValueTypeOp::getResult() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSResults(0).begin());
}

void GetValueTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value value) {
      Type valType = value.getType();
      Type typeType = odsBuilder.getType<pdl::TypeType>();
      build(odsBuilder, odsState,
            ::llvm::isa<pdl::RangeType>(valType) ? pdl::RangeType::get(typeType)
                                          : typeType,
            value);
    
}

void GetValueTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type result, ::mlir::Value value) {
  odsState.addOperands(value);
  odsState.addTypes(result);
}

void GetValueTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetValueTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetValueTypeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((std::equal_to<>()(getGetValueTypeOpValueType((*this->getODSResults(0).begin()).getType()), (*this->getODSOperands(0).begin()).getType()))))
    return emitOpError("failed to verify that `value` type matches arity of `result`");
  return ::mlir::success();
}

::mlir::LogicalResult GetValueTypeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetValueTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type resultRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resultTypes(resultRawTypes);
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resultRawTypes[0] = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  for (::mlir::Type type : resultTypes) {
    (void)type;
    if (!(((::llvm::isa<::mlir::pdl::TypeType>(type))) || (((::llvm::isa<::mlir::pdl::RangeType>(type))) && ((::llvm::isa<::mlir::pdl::TypeType>(::llvm::cast<::mlir::pdl::RangeType>(type).getElementType())))))) {
      return parser.emitError(parser.getNameLoc()) << "'result' must be single element or range of PDL handle to an `mlir::Type`, but got " << type;
    }
  }
  result.addTypes(resultTypes);
  if (parser.resolveOperands(valueOperands, getGetValueTypeOpValueType(resultTypes[0]), valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetValueTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getResult().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void GetValueTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::GetValueTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::IsNotNullOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
IsNotNullOpGenericAdaptorBase::IsNotNullOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.is_not_null", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> IsNotNullOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr IsNotNullOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
IsNotNullOpAdaptor::IsNotNullOpAdaptor(IsNotNullOp op) : IsNotNullOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult IsNotNullOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> IsNotNullOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IsNotNullOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::PDLType> IsNotNullOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::PDLType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange IsNotNullOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> IsNotNullOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IsNotNullOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *IsNotNullOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *IsNotNullOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

void IsNotNullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void IsNotNullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(value);
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IsNotNullOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IsNotNullOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult IsNotNullOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult IsNotNullOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::llvm::SmallVector<::mlir::Block *, 2> fullSuccessors;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::pdl::PDLType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      fullSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        fullSuccessors.emplace_back(succ);
      }
    }
  }
  result.addSuccessors(fullSuccessors);
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void IsNotNullOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::pdl::PDLType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  ::llvm::interleaveComma(getOperation()->getSuccessors(), _odsPrinter);
}

void IsNotNullOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::IsNotNullOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::RecordMatchOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
RecordMatchOpGenericAdaptorBase::RecordMatchOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.record_match", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> RecordMatchOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  assert(getProperties().operand_segment_sizes && "missing segment size attribute for op");
  auto sizeAttr = getProperties().operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::DictionaryAttr RecordMatchOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::SymbolRefAttr RecordMatchOpGenericAdaptorBase::getRewriterAttr() {
  auto attr = getProperties().rewriter.cast<::mlir::SymbolRefAttr>();
  return attr;
}

::mlir::SymbolRefAttr RecordMatchOpGenericAdaptorBase::getRewriter() {
  auto attr = getRewriterAttr();
  return attr;
}

::mlir::StringAttr RecordMatchOpGenericAdaptorBase::getRootKindAttr() {
  auto attr = getProperties().rootKind.dyn_cast_or_null<::mlir::StringAttr>();
  return attr;
}

::std::optional< ::llvm::StringRef > RecordMatchOpGenericAdaptorBase::getRootKind() {
  auto attr = getRootKindAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr RecordMatchOpGenericAdaptorBase::getGeneratedOpsAttr() {
  auto attr = getProperties().generatedOps.dyn_cast_or_null<::mlir::ArrayAttr>();
  return attr;
}

::std::optional< ::mlir::ArrayAttr > RecordMatchOpGenericAdaptorBase::getGeneratedOps() {
  auto attr = getGeneratedOpsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::IntegerAttr RecordMatchOpGenericAdaptorBase::getBenefitAttr() {
  auto attr = getProperties().benefit.cast<::mlir::IntegerAttr>();
  return attr;
}

uint16_t RecordMatchOpGenericAdaptorBase::getBenefit() {
  auto attr = getBenefitAttr();
  return attr.getValue().getZExtValue();
}

} // namespace detail
RecordMatchOpAdaptor::RecordMatchOpAdaptor(RecordMatchOp op) : RecordMatchOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult RecordMatchOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_benefit = getProperties().benefit; (void)tblgen_benefit;
  if (!tblgen_benefit) return emitError(loc, "'pdl_interp.record_match' op ""requires attribute 'benefit'");
  auto tblgen_generatedOps = getProperties().generatedOps; (void)tblgen_generatedOps;
  auto tblgen_operand_segment_sizes = getProperties().operand_segment_sizes; (void)tblgen_operand_segment_sizes;
  if (!tblgen_operand_segment_sizes) return emitError(loc, "'pdl_interp.record_match' op ""requires attribute 'operand_segment_sizes'");
  auto tblgen_rewriter = getProperties().rewriter; (void)tblgen_rewriter;
  if (!tblgen_rewriter) return emitError(loc, "'pdl_interp.record_match' op ""requires attribute 'rewriter'");
  auto tblgen_rootKind = getProperties().rootKind; (void)tblgen_rootKind;

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitError(loc, "'pdl_interp.record_match' op ""'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (tblgen_rewriter && !((::llvm::isa<::mlir::SymbolRefAttr>(tblgen_rewriter))))
    return emitError(loc, "'pdl_interp.record_match' op ""attribute 'rewriter' failed to satisfy constraint: symbol reference attribute");

  if (tblgen_rootKind && !((::llvm::isa<::mlir::StringAttr>(tblgen_rootKind))))
    return emitError(loc, "'pdl_interp.record_match' op ""attribute 'rootKind' failed to satisfy constraint: string attribute");

  if (tblgen_generatedOps && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_generatedOps))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_generatedOps), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::StringAttr>(attr))); }))))
    return emitError(loc, "'pdl_interp.record_match' op ""attribute 'generatedOps' failed to satisfy constraint: string array attribute");

  if (tblgen_benefit && !((((::llvm::isa<::mlir::IntegerAttr>(tblgen_benefit))) && ((::llvm::cast<::mlir::IntegerAttr>(tblgen_benefit).getType().isSignlessInteger(16)))) && ((!::llvm::cast<::mlir::IntegerAttr>(tblgen_benefit).getValue().isNegative()))))
    return emitError(loc, "'pdl_interp.record_match' op ""attribute 'benefit' failed to satisfy constraint: 16-bit signless integer attribute whose value is non-negative");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> RecordMatchOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = getProperties().operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttr[i];
  return {start, sizeAttr[index]};
}

::mlir::Operation::operand_range RecordMatchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range RecordMatchOp::getInputs() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range RecordMatchOp::getMatchedOps() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange RecordMatchOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, {getOperandSegmentSizesAttrName(), getProperties().operand_segment_sizes}));
  return mutableRange;
}

::mlir::MutableOperandRange RecordMatchOp::getMatchedOpsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, {getOperandSegmentSizesAttrName(), getProperties().operand_segment_sizes}));
  return mutableRange;
}

std::pair<unsigned, unsigned> RecordMatchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range RecordMatchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *RecordMatchOp::getDest() {
  return (*this)->getSuccessor(0);
}

::mlir::LogicalResult RecordMatchOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.benefit;
    auto attr = dict.get("benefit");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for benefit in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `benefit` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.generatedOps;
    auto attr = dict.get("generatedOps");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for generatedOps in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `generatedOps` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.operand_segment_sizes;
    auto attr = dict.get("operand_segment_sizes");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for operand_segment_sizes in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `operand_segment_sizes` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.rewriter;
    auto attr = dict.get("rewriter");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for rewriter in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `rewriter` in property conversion: " << attr;
        return failure();
      }
    }
  }

  {
    auto &propStorage = prop.rootKind;
    auto attr = dict.get("rootKind");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for rootKind in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `rootKind` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute RecordMatchOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.benefit;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("benefit",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.generatedOps;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("generatedOps",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.operand_segment_sizes;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("operand_segment_sizes",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.rewriter;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("rewriter",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.rootKind;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("rootKind",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code RecordMatchOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.benefit.getAsOpaquePointer()), 
    llvm::hash_value(prop.generatedOps.getAsOpaquePointer()), 
    llvm::hash_value(prop.operand_segment_sizes.getAsOpaquePointer()), 
    llvm::hash_value(prop.rewriter.getAsOpaquePointer()), 
    llvm::hash_value(prop.rootKind.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> RecordMatchOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "benefit")
      return prop.benefit;

    if (name == "generatedOps")
      return prop.generatedOps;

    if (name == "operand_segment_sizes")
      return prop.operand_segment_sizes;

    if (name == "rewriter")
      return prop.rewriter;

    if (name == "rootKind")
      return prop.rootKind;
  return std::nullopt;
}

void RecordMatchOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "benefit") {
       prop.benefit = dyn_cast_or_null<std::remove_reference_t<decltype(prop.benefit)>>(value);
       return;
    }

    if (name == "generatedOps") {
       prop.generatedOps = dyn_cast_or_null<std::remove_reference_t<decltype(prop.generatedOps)>>(value);
       return;
    }

    if (name == "operand_segment_sizes") {
       prop.operand_segment_sizes = dyn_cast_or_null<std::remove_reference_t<decltype(prop.operand_segment_sizes)>>(value);
       return;
    }

    if (name == "rewriter") {
       prop.rewriter = dyn_cast_or_null<std::remove_reference_t<decltype(prop.rewriter)>>(value);
       return;
    }

    if (name == "rootKind") {
       prop.rootKind = dyn_cast_or_null<std::remove_reference_t<decltype(prop.rootKind)>>(value);
       return;
    }
}

void RecordMatchOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.benefit) attrs.append("benefit", prop.benefit);

    if (prop.generatedOps) attrs.append("generatedOps", prop.generatedOps);

    if (prop.operand_segment_sizes) attrs.append("operand_segment_sizes", prop.operand_segment_sizes);

    if (prop.rewriter) attrs.append("rewriter", prop.rewriter);

    if (prop.rootKind) attrs.append("rootKind", prop.rootKind);
}

::mlir::LogicalResult RecordMatchOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getBenefitAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps10(attr, "benefit", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getGeneratedOpsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(attr, "generatedOps", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getRewriterAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps9(attr, "rewriter", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getRootKindAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(attr, "rootKind", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::SymbolRefAttr RecordMatchOp::getRewriterAttr() {
  return getProperties().rewriter.cast<::mlir::SymbolRefAttr>();
}

::mlir::SymbolRefAttr RecordMatchOp::getRewriter() {
  auto attr = getRewriterAttr();
  return attr;
}

::mlir::StringAttr RecordMatchOp::getRootKindAttr() {
  return getProperties().rootKind.dyn_cast_or_null<::mlir::StringAttr>();
}

::std::optional< ::llvm::StringRef > RecordMatchOp::getRootKind() {
  auto attr = getRootKindAttr();
  return attr ? ::std::optional< ::llvm::StringRef >(attr.getValue()) : (::std::nullopt);
}

::mlir::ArrayAttr RecordMatchOp::getGeneratedOpsAttr() {
  return getProperties().generatedOps.dyn_cast_or_null<::mlir::ArrayAttr>();
}

::std::optional< ::mlir::ArrayAttr > RecordMatchOp::getGeneratedOps() {
  auto attr = getGeneratedOpsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::IntegerAttr RecordMatchOp::getBenefitAttr() {
  return getProperties().benefit.cast<::mlir::IntegerAttr>();
}

uint16_t RecordMatchOp::getBenefit() {
  auto attr = getBenefitAttr();
  return attr.getValue().getZExtValue();
}

void RecordMatchOp::setRewriterAttr(::mlir::SymbolRefAttr attr) {
  (*this)->setAttr(getRewriterAttrName(), attr);
}

void RecordMatchOp::setRootKindAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getRootKindAttrName(), attr);
}

void RecordMatchOp::setRootKind(::std::optional<::llvm::StringRef> attrValue) {
    if (attrValue)
      return (*this)->setAttr(getRootKindAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(*attrValue));
    (*this)->removeAttr(getRootKindAttrName());
}

void RecordMatchOp::setGeneratedOpsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getGeneratedOpsAttrName(), attr);
}

void RecordMatchOp::setBenefitAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getBenefitAttrName(), attr);
}

void RecordMatchOp::setBenefit(uint16_t attrValue) {
  (*this)->setAttr(getBenefitAttrName(), ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(16), attrValue));
}

::mlir::Attribute RecordMatchOp::removeRootKindAttr() {
    auto &attr = getProperties().rootKind;
    attr = {};
    return attr;
}

::mlir::Attribute RecordMatchOp::removeGeneratedOpsAttr() {
    auto &attr = getProperties().generatedOps;
    attr = {};
    return attr;
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, ::mlir::IntegerAttr benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
   (odsState.getOrAddProperties<Properties>().operand_segment_sizes= odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
  odsState.getOrAddProperties<Properties>().rewriter = rewriter;
  if (rootKind) {
    odsState.getOrAddProperties<Properties>().rootKind = rootKind;
  }
  if (generatedOps) {
    odsState.getOrAddProperties<Properties>().generatedOps = generatedOps;
  }
  odsState.getOrAddProperties<Properties>().benefit = benefit;
  odsState.addSuccessors(dest);
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, ::mlir::IntegerAttr benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
   (odsState.getOrAddProperties<Properties>().operand_segment_sizes= odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
  odsState.getOrAddProperties<Properties>().rewriter = rewriter;
  if (rootKind) {
    odsState.getOrAddProperties<Properties>().rootKind = rootKind;
  }
  if (generatedOps) {
    odsState.getOrAddProperties<Properties>().generatedOps = generatedOps;
  }
  odsState.getOrAddProperties<Properties>().benefit = benefit;
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, uint16_t benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
   (odsState.getOrAddProperties<Properties>().operand_segment_sizes= odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
  odsState.getOrAddProperties<Properties>().rewriter = rewriter;
  if (rootKind) {
    odsState.getOrAddProperties<Properties>().rootKind = rootKind;
  }
  if (generatedOps) {
    odsState.getOrAddProperties<Properties>().generatedOps = generatedOps;
  }
  odsState.getOrAddProperties<Properties>().benefit = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit);
  odsState.addSuccessors(dest);
}

void RecordMatchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange inputs, ::mlir::ValueRange matchedOps, ::mlir::SymbolRefAttr rewriter, /*optional*/::mlir::StringAttr rootKind, /*optional*/::mlir::ArrayAttr generatedOps, uint16_t benefit, ::mlir::Block *dest) {
  odsState.addOperands(inputs);
  odsState.addOperands(matchedOps);
   (odsState.getOrAddProperties<Properties>().operand_segment_sizes= odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()), static_cast<int32_t>(matchedOps.size())}));
  odsState.getOrAddProperties<Properties>().rewriter = rewriter;
  if (rootKind) {
    odsState.getOrAddProperties<Properties>().rootKind = rootKind;
  }
  if (generatedOps) {
    odsState.getOrAddProperties<Properties>().generatedOps = generatedOps;
  }
  odsState.getOrAddProperties<Properties>().benefit = odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(16), benefit);
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void RecordMatchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult RecordMatchOp::verifyInvariantsImpl() {
  auto tblgen_benefit = getProperties().benefit; (void)tblgen_benefit;
  if (!tblgen_benefit) return emitOpError("requires attribute 'benefit'");
  auto tblgen_generatedOps = getProperties().generatedOps; (void)tblgen_generatedOps;
  auto tblgen_operand_segment_sizes = getProperties().operand_segment_sizes; (void)tblgen_operand_segment_sizes;
  if (!tblgen_operand_segment_sizes) return emitOpError("requires attribute 'operand_segment_sizes'");
  auto tblgen_rewriter = getProperties().rewriter; (void)tblgen_rewriter;
  if (!tblgen_rewriter) return emitOpError("requires attribute 'rewriter'");
  auto tblgen_rootKind = getProperties().rootKind; (void)tblgen_rootKind;

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseI32ArrayAttr>();
    auto numElements = sizeAttr.asArrayRef().size();
    if (numElements != 2)
      return emitOpError("'operand_segment_sizes' attribute for specifying operand segments must have 2 "
                "elements, but got ") << numElements;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps9(*this, tblgen_rewriter, "rewriter")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps0(*this, tblgen_rootKind, "rootKind")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(*this, tblgen_generatedOps, "generatedOps")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps10(*this, tblgen_benefit, "benefit")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult RecordMatchOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult RecordMatchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::SymbolRefAttr rewriterAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  ::llvm::SMLoc inputsOperandsLoc;
  (void)inputsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputsTypes;
  ::mlir::IntegerAttr benefitAttr;
  ::mlir::ArrayAttr generatedOpsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> matchedOpsOperands;
  ::llvm::SMLoc matchedOpsOperandsLoc;
  (void)matchedOpsOperandsLoc;
  ::mlir::StringAttr rootKindAttr;
  ::mlir::Block *destSuccessor = nullptr;

  if (parser.parseCustomAttributeWithFallback(rewriterAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (rewriterAttr) result.getOrAddProperties<RecordMatchOp::Properties>().rewriter = rewriterAttr;
  if (::mlir::succeeded(parser.parseOptionalLParen())) {

  inputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseKeyword("benefit"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(benefitAttr, parser.getBuilder().getIntegerType(16))) {
    return ::mlir::failure();
  }
  if (benefitAttr) result.getOrAddProperties<RecordMatchOp::Properties>().benefit = benefitAttr;
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalKeyword("generatedOps"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(generatedOpsAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (generatedOpsAttr) result.getOrAddProperties<RecordMatchOp::Properties>().generatedOps = generatedOpsAttr;
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  }
  if (parser.parseKeyword("loc"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();

  matchedOpsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(matchedOpsOperands))
    return ::mlir::failure();
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalComma())) {
  if (parser.parseKeyword("root"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(rootKindAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (rootKindAttr) result.getOrAddProperties<RecordMatchOp::Properties>().rootKind = rootKindAttr;
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  result.addSuccessors(destSuccessor);
  result.getOrAddProperties<RecordMatchOp::Properties>().operand_segment_sizes = (parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(inputsOperands.size()), static_cast<int32_t>(matchedOpsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(matchedOpsOperands, odsBuildableType0, matchedOpsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void RecordMatchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getRewriterAttr());
  if (!getInputs().empty()) {
    _odsPrinter << "(";
    _odsPrinter << getInputs();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getInputs().getTypes();
    _odsPrinter << ")";
  }
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "benefit";
  _odsPrinter << "(";
  _odsPrinter.printAttributeWithoutType(getBenefitAttr());
  _odsPrinter << ")";
  _odsPrinter << ",";
  if (getGeneratedOpsAttr()) {
    _odsPrinter << ' ' << "generatedOps";
    _odsPrinter << "(";
    _odsPrinter.printAttributeWithoutType(getGeneratedOpsAttr());
    _odsPrinter << ")";
    _odsPrinter << ",";
  }
  _odsPrinter << ' ' << "loc";
  _odsPrinter << "(";
  _odsPrinter << "[";
  _odsPrinter << getMatchedOps();
  _odsPrinter << "]";
  _odsPrinter << ")";
  if (getRootKindAttr()) {
    _odsPrinter << ",";
    _odsPrinter << ' ' << "root";
    _odsPrinter << "(";
    _odsPrinter.printAttributeWithoutType(getRootKindAttr());
    _odsPrinter << ")";
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operand_segment_sizes");
  elidedAttrs.push_back("rewriter");
  elidedAttrs.push_back("benefit");
  elidedAttrs.push_back("generatedOps");
  elidedAttrs.push_back("rootKind");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDest();
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::RecordMatchOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::ReplaceOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReplaceOpGenericAdaptorBase::ReplaceOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.replace", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> ReplaceOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 1) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr ReplaceOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ReplaceOpAdaptor::ReplaceOpAdaptor(ReplaceOp op) : ReplaceOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult ReplaceOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReplaceOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range ReplaceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> ReplaceOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
}

::mlir::Operation::operand_range ReplaceOp::getReplValues() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange ReplaceOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ReplaceOp::getReplValuesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReplaceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReplaceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::ValueRange replValues) {
  odsState.addOperands(inputOp);
  odsState.addOperands(replValues);
}

void ReplaceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::ValueRange replValues) {
  odsState.addOperands(inputOp);
  odsState.addOperands(replValues);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReplaceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReplaceOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReplaceOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ReplaceOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> replValuesOperands;
  ::llvm::SMLoc replValuesOperandsLoc;
  (void)replValuesOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> replValuesTypes;

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("with"))
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  replValuesOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(replValuesOperands))
    return ::mlir::failure();
  if (!replValuesOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(replValuesTypes))
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(replValuesOperands, replValuesTypes, replValuesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReplaceOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "with";
  _odsPrinter << ' ';
  _odsPrinter << "(";
  if (!getReplValues().empty()) {
    _odsPrinter << getReplValues();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getReplValues().getTypes();
  }
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::ReplaceOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchAttributeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchAttributeOpGenericAdaptorBase::SwitchAttributeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.switch_attribute", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchAttributeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SwitchAttributeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchAttributeOpGenericAdaptorBase::getCaseValuesAttr() {
  auto attr = getProperties().caseValues.cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SwitchAttributeOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchAttributeOpAdaptor::SwitchAttributeOpAdaptor(SwitchAttributeOp op) : SwitchAttributeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult SwitchAttributeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_attribute' op ""requires attribute 'caseValues'");

  if (tblgen_caseValues && !((::llvm::isa<::mlir::ArrayAttr>(tblgen_caseValues))))
    return emitError(loc, "'pdl_interp.switch_attribute' op ""attribute 'caseValues' failed to satisfy constraint: array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchAttributeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchAttributeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::AttributeType> SwitchAttributeOp::getAttribute() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::AttributeType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange SwitchAttributeOp::getAttributeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SwitchAttributeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchAttributeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchAttributeOp::getDefaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchAttributeOp::getCases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::LogicalResult SwitchAttributeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.caseValues;
    auto attr = dict.get("caseValues");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for caseValues in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `caseValues` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SwitchAttributeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.caseValues;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("caseValues",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code SwitchAttributeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.caseValues.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SwitchAttributeOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "caseValues")
      return prop.caseValues;
  return std::nullopt;
}

void SwitchAttributeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "caseValues") {
       prop.caseValues = dyn_cast_or_null<std::remove_reference_t<decltype(prop.caseValues)>>(value);
       return;
    }
}

void SwitchAttributeOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.caseValues) attrs.append("caseValues", prop.caseValues);
}

::mlir::LogicalResult SwitchAttributeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getCaseValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps11(attr, "caseValues", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::ArrayAttr SwitchAttributeOp::getCaseValuesAttr() {
  return getProperties().caseValues.cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SwitchAttributeOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchAttributeOp::setCaseValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value attribute, ArrayRef<Attribute> caseValues, Block *defaultDest, BlockRange dests) {
    build(odsBuilder, odsState, attribute, odsBuilder.getArrayAttr(caseValues),
          defaultDest, dests);
  
}

void SwitchAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value attribute, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(attribute);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchAttributeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value attribute, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(attribute);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchAttributeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchAttributeOp::verifyInvariantsImpl() {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitOpError("requires attribute 'caseValues'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps11(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchAttributeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchAttributeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand attributeRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> attributeOperands(attributeRawOperands);  ::llvm::SMLoc attributeOperandsLoc;
  (void)attributeOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;

  attributeOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(attributeRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (caseValuesAttr) result.getOrAddProperties<SwitchAttributeOp::Properties>().caseValues = caseValuesAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::AttributeType>();
  if (parser.resolveOperands(attributeOperands, odsBuildableType0, attributeOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchAttributeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAttribute();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchAttributeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchAttributeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchOperandCountOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchOperandCountOpGenericAdaptorBase::SwitchOperandCountOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.switch_operand_count", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchOperandCountOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SwitchOperandCountOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr SwitchOperandCountOpGenericAdaptorBase::getCaseValuesAttr() {
  auto attr = getProperties().caseValues.cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr SwitchOperandCountOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchOperandCountOpAdaptor::SwitchOperandCountOpAdaptor(SwitchOperandCountOp op) : SwitchOperandCountOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult SwitchOperandCountOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_operand_count' op ""requires attribute 'caseValues'");

  if (tblgen_caseValues && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(tblgen_caseValues))) && ((::llvm::cast<::mlir::DenseIntElementsAttr>(tblgen_caseValues).getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'pdl_interp.switch_operand_count' op ""attribute 'caseValues' failed to satisfy constraint: 32-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchOperandCountOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchOperandCountOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> SwitchOperandCountOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange SwitchOperandCountOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SwitchOperandCountOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchOperandCountOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchOperandCountOp::getDefaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchOperandCountOp::getCases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::LogicalResult SwitchOperandCountOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.caseValues;
    auto attr = dict.get("caseValues");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for caseValues in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `caseValues` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SwitchOperandCountOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.caseValues;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("caseValues",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code SwitchOperandCountOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.caseValues.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SwitchOperandCountOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "caseValues")
      return prop.caseValues;
  return std::nullopt;
}

void SwitchOperandCountOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "caseValues") {
       prop.caseValues = dyn_cast_or_null<std::remove_reference_t<decltype(prop.caseValues)>>(value);
       return;
    }
}

void SwitchOperandCountOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.caseValues) attrs.append("caseValues", prop.caseValues);
}

::mlir::LogicalResult SwitchOperandCountOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getCaseValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps12(attr, "caseValues", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::DenseIntElementsAttr SwitchOperandCountOp::getCaseValuesAttr() {
  return getProperties().caseValues.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr SwitchOperandCountOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchOperandCountOp::setCaseValuesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value inputOp, ArrayRef<int32_t> counts, Block *defaultDest, BlockRange dests) {
    build(odsBuilder, odsState, inputOp, odsBuilder.getI32VectorAttr(counts),
          defaultDest, dests);
  
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOperandCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchOperandCountOp::verifyInvariantsImpl() {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitOpError("requires attribute 'caseValues'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps12(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchOperandCountOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchOperandCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::DenseIntElementsAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, ::mlir::Type{})) {
    return ::mlir::failure();
  }
  if (caseValuesAttr) result.getOrAddProperties<SwitchOperandCountOp::Properties>().caseValues = caseValuesAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchOperandCountOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchOperandCountOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchOperandCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchOperationNameOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchOperationNameOpGenericAdaptorBase::SwitchOperationNameOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.switch_operation_name", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchOperationNameOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SwitchOperationNameOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchOperationNameOpGenericAdaptorBase::getCaseValuesAttr() {
  auto attr = getProperties().caseValues.cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SwitchOperationNameOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchOperationNameOpAdaptor::SwitchOperationNameOpAdaptor(SwitchOperationNameOp op) : SwitchOperationNameOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult SwitchOperationNameOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_operation_name' op ""requires attribute 'caseValues'");

  if (tblgen_caseValues && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_caseValues))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_caseValues), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::StringAttr>(attr))); }))))
    return emitError(loc, "'pdl_interp.switch_operation_name' op ""attribute 'caseValues' failed to satisfy constraint: string array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchOperationNameOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchOperationNameOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> SwitchOperationNameOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange SwitchOperationNameOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SwitchOperationNameOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchOperationNameOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchOperationNameOp::getDefaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchOperationNameOp::getCases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::LogicalResult SwitchOperationNameOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.caseValues;
    auto attr = dict.get("caseValues");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for caseValues in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `caseValues` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SwitchOperationNameOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.caseValues;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("caseValues",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code SwitchOperationNameOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.caseValues.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SwitchOperationNameOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "caseValues")
      return prop.caseValues;
  return std::nullopt;
}

void SwitchOperationNameOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "caseValues") {
       prop.caseValues = dyn_cast_or_null<std::remove_reference_t<decltype(prop.caseValues)>>(value);
       return;
    }
}

void SwitchOperationNameOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.caseValues) attrs.append("caseValues", prop.caseValues);
}

::mlir::LogicalResult SwitchOperationNameOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getCaseValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(attr, "caseValues", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::ArrayAttr SwitchOperationNameOp::getCaseValuesAttr() {
  return getProperties().caseValues.cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SwitchOperationNameOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchOperationNameOp::setCaseValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value inputOp, ArrayRef<OperationName> names, Block *defaultDest, BlockRange dests) {
      auto stringNames = llvm::to_vector<8>(llvm::map_range(names,
          [](OperationName name) { return name.getStringRef(); }));
      build(odsBuilder, odsState, inputOp, odsBuilder.getStrArrayAttr(stringNames),
            defaultDest, dests);
    
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOperationNameOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchOperationNameOp::verifyInvariantsImpl() {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitOpError("requires attribute 'caseValues'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps6(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchOperationNameOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchOperationNameOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (caseValuesAttr) result.getOrAddProperties<SwitchOperationNameOp::Properties>().caseValues = caseValuesAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchOperationNameOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchOperationNameOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchOperationNameOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchResultCountOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchResultCountOpGenericAdaptorBase::SwitchResultCountOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.switch_result_count", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchResultCountOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SwitchResultCountOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr SwitchResultCountOpGenericAdaptorBase::getCaseValuesAttr() {
  auto attr = getProperties().caseValues.cast<::mlir::DenseIntElementsAttr>();
  return attr;
}

::mlir::DenseIntElementsAttr SwitchResultCountOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchResultCountOpAdaptor::SwitchResultCountOpAdaptor(SwitchResultCountOp op) : SwitchResultCountOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult SwitchResultCountOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_result_count' op ""requires attribute 'caseValues'");

  if (tblgen_caseValues && !(((::llvm::isa<::mlir::DenseIntElementsAttr>(tblgen_caseValues))) && ((::llvm::cast<::mlir::DenseIntElementsAttr>(tblgen_caseValues).getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'pdl_interp.switch_result_count' op ""attribute 'caseValues' failed to satisfy constraint: 32-bit signless integer elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchResultCountOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchResultCountOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::OperationType> SwitchResultCountOp::getInputOp() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::OperationType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange SwitchResultCountOp::getInputOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SwitchResultCountOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchResultCountOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchResultCountOp::getDefaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchResultCountOp::getCases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::LogicalResult SwitchResultCountOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.caseValues;
    auto attr = dict.get("caseValues");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for caseValues in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `caseValues` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SwitchResultCountOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.caseValues;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("caseValues",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code SwitchResultCountOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.caseValues.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SwitchResultCountOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "caseValues")
      return prop.caseValues;
  return std::nullopt;
}

void SwitchResultCountOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "caseValues") {
       prop.caseValues = dyn_cast_or_null<std::remove_reference_t<decltype(prop.caseValues)>>(value);
       return;
    }
}

void SwitchResultCountOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.caseValues) attrs.append("caseValues", prop.caseValues);
}

::mlir::LogicalResult SwitchResultCountOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getCaseValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps12(attr, "caseValues", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::DenseIntElementsAttr SwitchResultCountOp::getCaseValuesAttr() {
  return getProperties().caseValues.cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr SwitchResultCountOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchResultCountOp::setCaseValuesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value inputOp, ArrayRef<int32_t> counts, Block *defaultDest, BlockRange dests) {
    build(odsBuilder, odsState, inputOp, odsBuilder.getI32VectorAttr(counts),
          defaultDest, dests);
  
}

void SwitchResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value inputOp, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchResultCountOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value inputOp, ::mlir::DenseIntElementsAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(inputOp);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchResultCountOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchResultCountOp::verifyInvariantsImpl() {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitOpError("requires attribute 'caseValues'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps12(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchResultCountOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchResultCountOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputOpRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOpOperands(inputOpRawOperands);  ::llvm::SMLoc inputOpOperandsLoc;
  (void)inputOpOperandsLoc;
  ::mlir::DenseIntElementsAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;
  if (parser.parseKeyword("of"))
    return ::mlir::failure();

  inputOpOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputOpRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, ::mlir::Type{})) {
    return ::mlir::failure();
  }
  if (caseValuesAttr) result.getOrAddProperties<SwitchResultCountOp::Properties>().caseValues = caseValuesAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::OperationType>();
  if (parser.resolveOperands(inputOpOperands, odsBuildableType0, inputOpOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchResultCountOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ' << "of";
  _odsPrinter << ' ';
  _odsPrinter << getInputOp();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
_odsPrinter.printStrippedAttrOrType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchResultCountOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchResultCountOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchTypeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchTypeOpGenericAdaptorBase::SwitchTypeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.switch_type", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchTypeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SwitchTypeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchTypeOpGenericAdaptorBase::getCaseValuesAttr() {
  auto attr = getProperties().caseValues.cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SwitchTypeOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchTypeOpAdaptor::SwitchTypeOpAdaptor(SwitchTypeOp op) : SwitchTypeOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult SwitchTypeOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_type' op ""requires attribute 'caseValues'");

  if (tblgen_caseValues && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_caseValues))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_caseValues), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))))
    return emitError(loc, "'pdl_interp.switch_type' op ""attribute 'caseValues' failed to satisfy constraint: type array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchTypeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchTypeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::TypeType> SwitchTypeOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::TypeType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange SwitchTypeOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SwitchTypeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchTypeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchTypeOp::getDefaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchTypeOp::getCases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::LogicalResult SwitchTypeOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.caseValues;
    auto attr = dict.get("caseValues");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for caseValues in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `caseValues` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SwitchTypeOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.caseValues;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("caseValues",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code SwitchTypeOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.caseValues.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SwitchTypeOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "caseValues")
      return prop.caseValues;
  return std::nullopt;
}

void SwitchTypeOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "caseValues") {
       prop.caseValues = dyn_cast_or_null<std::remove_reference_t<decltype(prop.caseValues)>>(value);
       return;
    }
}

void SwitchTypeOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.caseValues) attrs.append("caseValues", prop.caseValues);
}

::mlir::LogicalResult SwitchTypeOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getCaseValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(attr, "caseValues", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::ArrayAttr SwitchTypeOp::getCaseValuesAttr() {
  return getProperties().caseValues.cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SwitchTypeOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchTypeOp::setCaseValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value edge, ArrayRef<Attribute> types, Block *defaultDest, BlockRange dests) {
      build(odsBuilder, odsState, edge, odsBuilder.getArrayAttr(types),
            defaultDest, dests);
    
}

void SwitchTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchTypeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchTypeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchTypeOp::verifyInvariantsImpl() {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitOpError("requires attribute 'caseValues'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps5(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchTypeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchTypeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (caseValuesAttr) result.getOrAddProperties<SwitchTypeOp::Properties>().caseValues = caseValuesAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getType<::mlir::pdl::TypeType>();
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchTypeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchTypeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchTypeOp)

namespace mlir {
namespace pdl_interp {

//===----------------------------------------------------------------------===//
// ::mlir::pdl_interp::SwitchTypesOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
SwitchTypesOpGenericAdaptorBase::SwitchTypesOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("pdl_interp.switch_types", odsAttrs.getContext());
}

std::pair<unsigned, unsigned> SwitchTypesOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr SwitchTypesOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr SwitchTypesOpGenericAdaptorBase::getCaseValuesAttr() {
  auto attr = getProperties().caseValues.cast<::mlir::ArrayAttr>();
  return attr;
}

::mlir::ArrayAttr SwitchTypesOpGenericAdaptorBase::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

} // namespace detail
SwitchTypesOpAdaptor::SwitchTypesOpAdaptor(SwitchTypesOp op) : SwitchTypesOpAdaptor(op->getOperands(), op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

::mlir::LogicalResult SwitchTypesOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitError(loc, "'pdl_interp.switch_types' op ""requires attribute 'caseValues'");

  if (tblgen_caseValues && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_caseValues))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_caseValues), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && (((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::Type>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((true))); }))); }))))
    return emitError(loc, "'pdl_interp.switch_types' op ""attribute 'caseValues' failed to satisfy constraint: type-array array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchTypesOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SwitchTypesOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::pdl::RangeType> SwitchTypesOp::getValue() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::pdl::RangeType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange SwitchTypesOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SwitchTypesOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchTypesOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchTypesOp::getDefaultDest() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchTypesOp::getCases() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::LogicalResult SwitchTypesOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::mlir::InFlightDiagnostic *diag) {
  ::mlir::DictionaryAttr dict = dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    if (diag)
      *diag << "expected DictionaryAttr to set properties";
    return failure();
  }

  {
    auto &propStorage = prop.caseValues;
    auto attr = dict.get("caseValues");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        if (diag)
          *diag << "expected key entry for caseValues in DictionaryAttr to set "
                   "Properties.";
        return failure();
      }
      auto convertedAttr = dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        if (diag)
          *diag << "Invalid attribute `caseValues` in property conversion: " << attr;
        return failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute SwitchTypesOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.caseValues;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("caseValues",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};    
}

llvm::hash_code SwitchTypesOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.caseValues.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> SwitchTypesOp::getInherentAttr(const Properties &prop, llvm::StringRef name) {
    if (name == "caseValues")
      return prop.caseValues;
  return std::nullopt;
}

void SwitchTypesOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "caseValues") {
       prop.caseValues = dyn_cast_or_null<std::remove_reference_t<decltype(prop.caseValues)>>(value);
       return;
    }
}

void SwitchTypesOp::populateInherentAttrs(const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.caseValues) attrs.append("caseValues", prop.caseValues);
}

::mlir::LogicalResult SwitchTypesOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getCaseValuesAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps13(attr, "caseValues", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::ArrayAttr SwitchTypesOp::getCaseValuesAttr() {
  return getProperties().caseValues.cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr SwitchTypesOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr;
}

void SwitchTypesOp::setCaseValuesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value edge, ArrayRef<Attribute> types, Block *defaultDest, BlockRange dests) {
      build(odsBuilder, odsState, edge, odsBuilder.getArrayAttr(types),
            defaultDest, dests);
    
}

void SwitchTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
}

void SwitchTypesOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ArrayAttr caseValues, ::mlir::Block *defaultDest, ::mlir::BlockRange cases) {
  odsState.addOperands(value);
  odsState.getOrAddProperties<Properties>().caseValues = caseValues;
  odsState.addSuccessors(defaultDest);
  odsState.addSuccessors(cases);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchTypesOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchTypesOp::verifyInvariantsImpl() {
  auto tblgen_caseValues = getProperties().caseValues; (void)tblgen_caseValues;
  if (!tblgen_caseValues) return emitOpError("requires attribute 'caseValues'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_PDLInterpOps13(*this, tblgen_caseValues, "caseValues")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_PDLInterpOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchTypesOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchTypesOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::ArrayAttr caseValuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> casesSuccessors;
  ::mlir::Block *defaultDestSuccessor = nullptr;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(caseValuesAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (caseValuesAttr) result.getOrAddProperties<SwitchTypesOp::Properties>().caseValues = caseValuesAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Block *succ;
    auto firstSucc = parser.parseOptionalSuccessor(succ);
    if (firstSucc.has_value()) {
      if (failed(*firstSucc))
        return ::mlir::failure();
      casesSuccessors.emplace_back(succ);

      // Parse any trailing successors.
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseSuccessor(succ))
          return ::mlir::failure();
        casesSuccessors.emplace_back(succ);
      }
    }
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestSuccessor))
    return ::mlir::failure();
  result.addSuccessors(defaultDestSuccessor);
  result.addSuccessors(casesSuccessors);
  ::mlir::Type odsBuildableType0 = ::mlir::pdl::RangeType::get(parser.getBuilder().getType<::mlir::pdl::TypeType>());
  if (parser.resolveOperands(valueOperands, odsBuildableType0, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchTypesOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCaseValuesAttr());
  _odsPrinter << "(";
  ::llvm::interleaveComma(getCases(), _odsPrinter);
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("caseValues");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDest();
}

void SwitchTypesOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace pdl_interp
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::pdl_interp::SwitchTypesOp)


#endif  // GET_OP_CLASSES

