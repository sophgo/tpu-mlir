/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::LLVM::AShrOp,
::mlir::LLVM::AccessGroupMetadataOp,
::mlir::LLVM::AddOp,
::mlir::LLVM::AddrSpaceCastOp,
::mlir::LLVM::AddressOfOp,
::mlir::LLVM::AliasScopeDomainMetadataOp,
::mlir::LLVM::AliasScopeMetadataOp,
::mlir::LLVM::AllocaOp,
::mlir::LLVM::AndOp,
::mlir::LLVM::AtomicCmpXchgOp,
::mlir::LLVM::AtomicRMWOp,
::mlir::LLVM::BitcastOp,
::mlir::LLVM::BrOp,
::mlir::LLVM::CallOp,
::mlir::LLVM::CondBrOp,
::mlir::LLVM::ConstantOp,
::mlir::LLVM::ExtractElementOp,
::mlir::LLVM::ExtractValueOp,
::mlir::LLVM::FAddOp,
::mlir::LLVM::FCmpOp,
::mlir::LLVM::FDivOp,
::mlir::LLVM::FMulOp,
::mlir::LLVM::FNegOp,
::mlir::LLVM::FPExtOp,
::mlir::LLVM::FPToSIOp,
::mlir::LLVM::FPToUIOp,
::mlir::LLVM::FPTruncOp,
::mlir::LLVM::FRemOp,
::mlir::LLVM::FSubOp,
::mlir::LLVM::FenceOp,
::mlir::LLVM::FreezeOp,
::mlir::LLVM::GEPOp,
::mlir::LLVM::GlobalCtorsOp,
::mlir::LLVM::GlobalDtorsOp,
::mlir::LLVM::GlobalOp,
::mlir::LLVM::ICmpOp,
::mlir::LLVM::InlineAsmOp,
::mlir::LLVM::InsertElementOp,
::mlir::LLVM::InsertValueOp,
::mlir::LLVM::IntToPtrOp,
::mlir::LLVM::InvokeOp,
::mlir::LLVM::LLVMFuncOp,
::mlir::LLVM::LShrOp,
::mlir::LLVM::LandingpadOp,
::mlir::LLVM::LoadOp,
::mlir::LLVM::MetadataOp,
::mlir::LLVM::MulOp,
::mlir::LLVM::NullOp,
::mlir::LLVM::OrOp,
::mlir::LLVM::PtrToIntOp,
::mlir::LLVM::ResumeOp,
::mlir::LLVM::ReturnOp,
::mlir::LLVM::SDivOp,
::mlir::LLVM::SExtOp,
::mlir::LLVM::SIToFPOp,
::mlir::LLVM::SRemOp,
::mlir::LLVM::SelectOp,
::mlir::LLVM::ShlOp,
::mlir::LLVM::ShuffleVectorOp,
::mlir::LLVM::StoreOp,
::mlir::LLVM::SubOp,
::mlir::LLVM::SwitchOp,
::mlir::LLVM::TruncOp,
::mlir::LLVM::UDivOp,
::mlir::LLVM::UIToFPOp,
::mlir::LLVM::URemOp,
::mlir::LLVM::UndefOp,
::mlir::LLVM::UnreachableOp,
::mlir::LLVM::XOrOp,
::mlir::LLVM::ZExtOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace LLVM {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::IntegerType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer or LLVM dialect-compatible vector of integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::LLVM::LLVMPointerType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMPointerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IntegerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) && (((type.cast<::mlir::LLVM::LLVMPointerType>().isOpaque())) || (((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::IntegerType>())) || ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::LLVM::LLVMPointerType>())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer to integer or LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::IntegerType>())) || ((type.isa<::mlir::LLVM::LLVMPointerType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer or LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMStructType>())) && ((type.cast<::mlir::LLVM::LLVMStructType>().getBody().size() == 2)) && (((type.cast<::mlir::LLVM::LLVMStructType>().getBody()[0].isa<::mlir::IntegerType>())) || ((type.cast<::mlir::LLVM::LLVMStructType>().getBody()[0].isa<::mlir::LLVM::LLVMPointerType>()))) && ((type.cast<::mlir::LLVM::LLVMStructType>().getBody()[1].isSignlessInteger(1))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be an LLVM struct type with any integer or pointer followed by a single-bit integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) && (((type.cast<::mlir::LLVM::LLVMPointerType>().isOpaque())) || (((::mlir::LLVM::isCompatibleFloatingPointType(type.cast<::mlir::LLVM::LLVMPointerType>().getElementType()))) || ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::IntegerType>())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer to floating point LLVM type or integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::mlir::LLVM::isCompatibleFloatingPointType(type))) || ((type.isa<::mlir::IntegerType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be floating point LLVM type or integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::mlir::LLVM::isCompatibleOuterType(type))) && (!((type.isa<::mlir::LLVM::LLVMStructType, ::mlir::LLVM::LLVMArrayType>()))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM-compatible non-aggregate type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps10(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::mlir::LLVM::isCompatibleOuterType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM dialect-compatible type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps11(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps12(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::mlir::LLVM::isCompatibleVectorType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM dialect-compatible vector type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps13(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMStructType, ::mlir::LLVM::LLVMArrayType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM aggregate type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps14(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::mlir::LLVM::isCompatibleFloatingPointType(type))) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::isCompatibleFloatingPointType(::mlir::LLVM::getVectorElementType(type))))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be floating point LLVM type or LLVM dialect-compatible vector of floating point LLVM type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps15(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isSignlessInteger(1))) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isSignlessInteger(1)))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer or LLVM dialect-compatible vector of 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps16(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((type.isa<::mlir::IntegerType>())) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::IntegerType>())))) || (((type.isa<::mlir::LLVM::LLVMPointerType>())) || (((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::LLVM::LLVMPointerType>())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer or LLVM dialect-compatible vector of integer or LLVM pointer type or LLVM dialect-compatible vector of LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps17(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::mlir::LLVM::isCompatibleOuterType(type))) && ((!type.isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be primitive LLVM type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps18(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) && (((type.cast<::mlir::LLVM::LLVMPointerType>().isOpaque())) || (((((::mlir::LLVM::isCompatibleOuterType(type.cast<::mlir::LLVM::LLVMPointerType>().getElementType()))) && ((!type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>()))) && (!(((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::LLVM::LLVMStructType>())) && ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().cast<::mlir::LLVM::LLVMStructType>().isOpaque()))))) || ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isa<::mlir::LLVM::PointerElementTypeInterface>())))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer to LLVM type with size, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps19(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((((::mlir::LLVM::isCompatibleOuterType(type))) && ((!type.isa<::mlir::LLVM::LLVMVoidType, ::mlir::LLVM::LLVMFunctionType>()))) && (!(((type.isa<::mlir::LLVM::LLVMStructType>())) && ((type.cast<::mlir::LLVM::LLVMStructType>().isOpaque()))))) || ((type.isa<::mlir::LLVM::PointerElementTypeInterface>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM type with size, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMOps20(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((true))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be any type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::StringAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: string attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::FlatSymbolRefAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: flat symbol reference attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 64-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any type attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::AtomicOrderingAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: Atomic ordering for LLVM's memory model";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps5(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::AtomicBinOpAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: llvm.atomicrmw binary operations";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps6(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::FMFAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: LLVM fastmath flags";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps7(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ElementsAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: constant vector/tensor attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps8(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((true))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: any attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps9(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::ArrayAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps10(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::FCmpPredicateAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: llvm.fcmp comparison predicate";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps11(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::DenseIntElementsAttr>())) && ((attr.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer elements attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps12(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::FlatSymbolRefAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: flat symbol ref array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps13(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit integer array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps14(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::UnitAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: unit attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps15(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::LinkageAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: LLVM Linkage specification";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps16(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!attr.cast<::mlir::IntegerAttr>().getValue().isNegative())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps17(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::UnnamedAddrAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: LLVM GlobalValue UnnamedAddr";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps18(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::ICmpPredicateAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: llvm.icmp comparison predicate";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps19(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::AsmDialectAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: ATT (0) or Intel (1) asm dialect";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps20(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::TypeAttr>())) && ((attr.cast<::mlir::TypeAttr>().getValue().isa<::mlir::LLVM::LLVMFunctionType>())))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: type attribute of LLVM function type";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps21(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::LLVM::CConvAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: LLVM Calling Convention specification";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMOps22(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(attr.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); })))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: symbol ref array attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_LLVMOps0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_LLVMOps1(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((::llvm::hasNItems(region, 1)))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: region with 1 blocks";
  }
  return ::mlir::success();
}
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AShrOp definitions
//===----------------------------------------------------------------------===//

AShrOpAdaptor::AShrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.ashr", odsAttrs.getContext());
}

AShrOpAdaptor::AShrOpAdaptor(AShrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AShrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AShrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AShrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AShrOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AShrOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr AShrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AShrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AShrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AShrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AShrOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AShrOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AShrOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AShrOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AShrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AShrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AShrOp::getRes() {
  return *getODSResults(0).begin();
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AShrOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AShrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AShrOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AShrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AShrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult AShrOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult AShrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AShrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AShrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AShrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AccessGroupMetadataOp definitions
//===----------------------------------------------------------------------===//

AccessGroupMetadataOpAdaptor::AccessGroupMetadataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.access_group", odsAttrs.getContext());
}

AccessGroupMetadataOpAdaptor::AccessGroupMetadataOpAdaptor(AccessGroupMetadataOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AccessGroupMetadataOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AccessGroupMetadataOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AccessGroupMetadataOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr AccessGroupMetadataOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr AccessGroupMetadataOpAdaptor::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AccessGroupMetadataOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef AccessGroupMetadataOpAdaptor::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::LogicalResult AccessGroupMetadataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.access_group' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == AccessGroupMetadataOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.access_group' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AccessGroupMetadataOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AccessGroupMetadataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AccessGroupMetadataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AccessGroupMetadataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr AccessGroupMetadataOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef AccessGroupMetadataOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

void AccessGroupMetadataOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AccessGroupMetadataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AccessGroupMetadataOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult AccessGroupMetadataOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AccessGroupMetadataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void AccessGroupMetadataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"sym_name"});
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AccessGroupMetadataOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AddOp definitions
//===----------------------------------------------------------------------===//

AddOpAdaptor::AddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.add", odsAttrs.getContext());
}

AddOpAdaptor::AddOpAdaptor(AddOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AddOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr AddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AddOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AddOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AddOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddOp::getRes() {
  return *getODSResults(0).begin();
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AddOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AddOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AddOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult AddOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult AddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AddOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AddOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AddrSpaceCastOp definitions
//===----------------------------------------------------------------------===//

AddrSpaceCastOpAdaptor::AddrSpaceCastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.addrspacecast", odsAttrs.getContext());
}

AddrSpaceCastOpAdaptor::AddrSpaceCastOpAdaptor(AddrSpaceCastOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AddrSpaceCastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AddrSpaceCastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AddrSpaceCastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddrSpaceCastOpAdaptor::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AddrSpaceCastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AddrSpaceCastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AddrSpaceCastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddrSpaceCastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddrSpaceCastOp::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AddrSpaceCastOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AddrSpaceCastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddrSpaceCastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddrSpaceCastOp::getRes() {
  return *getODSResults(0).begin();
}

void AddrSpaceCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void AddrSpaceCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void AddrSpaceCastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddrSpaceCastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AddrSpaceCastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AddrSpaceCastOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AddrSpaceCastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AddrSpaceCastOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AddrSpaceCastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AddrSpaceCastOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AddressOfOp definitions
//===----------------------------------------------------------------------===//

AddressOfOpAdaptor::AddressOfOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.addressof", odsAttrs.getContext());
}

AddressOfOpAdaptor::AddressOfOpAdaptor(AddressOfOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AddressOfOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AddressOfOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AddressOfOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr AddressOfOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr AddressOfOpAdaptor::getGlobalNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AddressOfOp::getGlobalNameAttrName(*odsOpName)).cast<::mlir::FlatSymbolRefAttr>();
    return attr;
}

::llvm::StringRef AddressOfOpAdaptor::getGlobalName() {
  auto attr = getGlobalNameAttr();
  return attr.getValue();
}

::mlir::LogicalResult AddressOfOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.addressof' op ""requires attribute 'global_name'");
    if (namedAttrIt->getName() == AddressOfOp::getGlobalNameAttrName(*odsOpName)) {
      tblgen_global_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_global_name && !((tblgen_global_name.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'llvm.mlir.addressof' op ""attribute 'global_name' failed to satisfy constraint: flat symbol reference attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AddressOfOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddressOfOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AddressOfOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddressOfOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AddressOfOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::FlatSymbolRefAttr AddressOfOp::getGlobalNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getGlobalNameAttrName()).cast<::mlir::FlatSymbolRefAttr>();
}

::llvm::StringRef AddressOfOp::getGlobalName() {
  auto attr = getGlobalNameAttr();
  return attr.getValue();
}

void AddressOfOp::setGlobalNameAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getGlobalNameAttrName(), attr);
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, GlobalOp global, ArrayRef<NamedAttribute> attrs) {
      build(odsBuilder, odsState,
            LLVM::LLVMPointerType::get(global.getType(), global.getAddrSpace()),
            global.getSymName());
      odsState.addAttributes(attrs);
    
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, LLVMFuncOp func, ArrayRef<NamedAttribute> attrs) {
      build(odsBuilder, odsState,
            LLVM::LLVMPointerType::get(func.getFunctionType()), func.getName());
      odsState.addAttributes(attrs);
    
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::FlatSymbolRefAttr global_name) {
  odsState.addAttribute(getGlobalNameAttrName(odsState.name), global_name);
  odsState.addTypes(res);
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::FlatSymbolRefAttr global_name) {
  odsState.addAttribute(getGlobalNameAttrName(odsState.name), global_name);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::llvm::StringRef global_name) {
  odsState.addAttribute(getGlobalNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), global_name));
  odsState.addTypes(res);
}

void AddressOfOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef global_name) {
  odsState.addAttribute(getGlobalNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), global_name));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddressOfOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AddressOfOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'global_name'");
    if (namedAttrIt->getName() == getGlobalNameAttrName()) {
      tblgen_global_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps1(*this, tblgen_global_name, "global_name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AddressOfOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult AddressOfOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::FlatSymbolRefAttr global_nameAttr;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  if (parser.parseCustomAttributeWithFallback(global_nameAttr, parser.getBuilder().getType<::mlir::NoneType>(), "global_name",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::LLVM::LLVMPointerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void AddressOfOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getGlobalNameAttr());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"global_name"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::LLVM::LLVMPointerType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AddressOfOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AddressOfOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AliasScopeDomainMetadataOp definitions
//===----------------------------------------------------------------------===//

AliasScopeDomainMetadataOpAdaptor::AliasScopeDomainMetadataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.alias_scope_domain", odsAttrs.getContext());
}

AliasScopeDomainMetadataOpAdaptor::AliasScopeDomainMetadataOpAdaptor(AliasScopeDomainMetadataOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AliasScopeDomainMetadataOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AliasScopeDomainMetadataOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AliasScopeDomainMetadataOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr AliasScopeDomainMetadataOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr AliasScopeDomainMetadataOpAdaptor::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AliasScopeDomainMetadataOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef AliasScopeDomainMetadataOpAdaptor::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::StringAttr AliasScopeDomainMetadataOpAdaptor::getDescriptionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AliasScopeDomainMetadataOp::getDescriptionAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > AliasScopeDomainMetadataOpAdaptor::getDescription() {
  auto attr = getDescriptionAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::LogicalResult AliasScopeDomainMetadataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_description;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.alias_scope_domain' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == AliasScopeDomainMetadataOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == AliasScopeDomainMetadataOp::getDescriptionAttrName(*odsOpName)) {
      tblgen_description = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.alias_scope_domain' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_description && !((tblgen_description.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.alias_scope_domain' op ""attribute 'description' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AliasScopeDomainMetadataOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AliasScopeDomainMetadataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AliasScopeDomainMetadataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AliasScopeDomainMetadataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr AliasScopeDomainMetadataOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef AliasScopeDomainMetadataOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::StringAttr AliasScopeDomainMetadataOp::getDescriptionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDescriptionAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > AliasScopeDomainMetadataOp::getDescription() {
  auto attr = getDescriptionAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void AliasScopeDomainMetadataOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void AliasScopeDomainMetadataOp::setDescriptionAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getDescriptionAttrName(), attr);
}

::mlir::Attribute AliasScopeDomainMetadataOp::removeDescriptionAttr() {
  return (*this)->removeAttr(getDescriptionAttrName());
}

void AliasScopeDomainMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  if (description) {
  odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
}

void AliasScopeDomainMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  if (description) {
  odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AliasScopeDomainMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  if (description) {
  odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
}

void AliasScopeDomainMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  if (description) {
  odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AliasScopeDomainMetadataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AliasScopeDomainMetadataOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_description;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDescriptionAttrName()) {
      tblgen_description = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_description, "description")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult AliasScopeDomainMetadataOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AliasScopeDomainMetadataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void AliasScopeDomainMetadataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"sym_name"});
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AliasScopeDomainMetadataOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AliasScopeMetadataOp definitions
//===----------------------------------------------------------------------===//

AliasScopeMetadataOpAdaptor::AliasScopeMetadataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.alias_scope", odsAttrs.getContext());
}

AliasScopeMetadataOpAdaptor::AliasScopeMetadataOpAdaptor(AliasScopeMetadataOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AliasScopeMetadataOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AliasScopeMetadataOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AliasScopeMetadataOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr AliasScopeMetadataOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr AliasScopeMetadataOpAdaptor::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, AliasScopeMetadataOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef AliasScopeMetadataOpAdaptor::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::FlatSymbolRefAttr AliasScopeMetadataOpAdaptor::getDomainAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AliasScopeMetadataOp::getDomainAttrName(*odsOpName)).cast<::mlir::FlatSymbolRefAttr>();
    return attr;
}

::llvm::StringRef AliasScopeMetadataOpAdaptor::getDomain() {
  auto attr = getDomainAttr();
  return attr.getValue();
}

::mlir::StringAttr AliasScopeMetadataOpAdaptor::getDescriptionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, AliasScopeMetadataOp::getDescriptionAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > AliasScopeMetadataOpAdaptor::getDescription() {
  auto attr = getDescriptionAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::LogicalResult AliasScopeMetadataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_domain;
  ::mlir::Attribute tblgen_description;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.alias_scope' op ""requires attribute 'domain'");
    if (namedAttrIt->getName() == AliasScopeMetadataOp::getDomainAttrName(*odsOpName)) {
      tblgen_domain = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == AliasScopeMetadataOp::getDescriptionAttrName(*odsOpName)) {
      tblgen_description = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.alias_scope' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == AliasScopeMetadataOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.alias_scope' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_domain && !((tblgen_domain.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'llvm.alias_scope' op ""attribute 'domain' failed to satisfy constraint: flat symbol reference attribute");

  if (tblgen_description && !((tblgen_description.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.alias_scope' op ""attribute 'description' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AliasScopeMetadataOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AliasScopeMetadataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> AliasScopeMetadataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AliasScopeMetadataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::StringAttr AliasScopeMetadataOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef AliasScopeMetadataOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::FlatSymbolRefAttr AliasScopeMetadataOp::getDomainAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDomainAttrName()).cast<::mlir::FlatSymbolRefAttr>();
}

::llvm::StringRef AliasScopeMetadataOp::getDomain() {
  auto attr = getDomainAttr();
  return attr.getValue();
}

::mlir::StringAttr AliasScopeMetadataOp::getDescriptionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getDescriptionAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > AliasScopeMetadataOp::getDescription() {
  auto attr = getDescriptionAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void AliasScopeMetadataOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void AliasScopeMetadataOp::setDomainAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getDomainAttrName(), attr);
}

void AliasScopeMetadataOp::setDescriptionAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getDescriptionAttrName(), attr);
}

::mlir::Attribute AliasScopeMetadataOp::removeDescriptionAttr() {
  return (*this)->removeAttr(getDescriptionAttrName());
}

void AliasScopeMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name, ::mlir::FlatSymbolRefAttr domain, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getDomainAttrName(odsState.name), domain);
  if (description) {
  odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
}

void AliasScopeMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name, ::mlir::FlatSymbolRefAttr domain, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getDomainAttrName(odsState.name), domain);
  if (description) {
  odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AliasScopeMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name, ::llvm::StringRef domain, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getDomainAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), domain));
  if (description) {
  odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
}

void AliasScopeMetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name, ::llvm::StringRef domain, /*optional*/::mlir::StringAttr description) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getDomainAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), domain));
  if (description) {
  odsState.addAttribute(getDescriptionAttrName(odsState.name), description);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AliasScopeMetadataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AliasScopeMetadataOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_domain;
  ::mlir::Attribute tblgen_description;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'domain'");
    if (namedAttrIt->getName() == getDomainAttrName()) {
      tblgen_domain = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getDescriptionAttrName()) {
      tblgen_description = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps1(*this, tblgen_domain, "domain")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_description, "description")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult AliasScopeMetadataOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult AliasScopeMetadataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void AliasScopeMetadataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"sym_name"});
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AliasScopeMetadataOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AllocaOp definitions
//===----------------------------------------------------------------------===//

AllocaOpAdaptor::AllocaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.alloca", odsAttrs.getContext());
}

AllocaOpAdaptor::AllocaOpAdaptor(AllocaOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AllocaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AllocaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AllocaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AllocaOpAdaptor::getArraySize() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AllocaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr AllocaOpAdaptor::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AllocaOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    return attr;
}

::llvm::Optional<uint64_t> AllocaOpAdaptor::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::TypeAttr AllocaOpAdaptor::getElemTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, AllocaOp::getElemTypeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::TypeAttr>();
    return attr;
}

::llvm::Optional<::mlir::Type> AllocaOpAdaptor::getElemType() {
  auto attr = getElemTypeAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

::mlir::LogicalResult AllocaOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_elem_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == AllocaOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == AllocaOp::getElemTypeAttrName(*odsOpName)) {
      tblgen_elem_type = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'llvm.alloca' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_elem_type && !(((tblgen_elem_type.isa<::mlir::TypeAttr>())) && ((tblgen_elem_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return emitError(loc, "'llvm.alloca' op ""attribute 'elem_type' failed to satisfy constraint: any type attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AllocaOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AllocaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AllocaOp::getArraySize() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AllocaOp::getArraySizeMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AllocaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AllocaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AllocaOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr AllocaOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> AllocaOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::TypeAttr AllocaOp::getElemTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getElemTypeAttrName()).dyn_cast_or_null<::mlir::TypeAttr>();
}

::llvm::Optional<::mlir::Type> AllocaOp::getElemType() {
  auto attr = getElemTypeAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

void AllocaOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void AllocaOp::setElemTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getElemTypeAttrName(), attr);
}

::mlir::Attribute AllocaOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute AllocaOp::removeElem_typeAttr() {
  return (*this)->removeAttr(getElemTypeAttrName());
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Value arraySize, unsigned alignment) {
      assert(!resultType.cast<LLVMPointerType>().isOpaque() &&
             "pass the allocated type explicitly if opaque pointers are used");
      if (alignment == 0)
        return build(odsBuilder, odsState, resultType, arraySize, IntegerAttr(),
                     TypeAttr());
      build(odsBuilder, odsState, resultType, arraySize,
        odsBuilder.getI64IntegerAttr(alignment), TypeAttr());
    
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, Type elementType, Value arraySize, unsigned alignment) {
      TypeAttr elemTypeAttr =
          resultType.cast<LLVMPointerType>().isOpaque() ?
          TypeAttr::get(elementType) : TypeAttr();
      build(odsBuilder, odsState, resultType, arraySize,
            alignment == 0 ? IntegerAttr()
                           : odsBuilder.getI64IntegerAttr(alignment),
            elemTypeAttr);

    
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arraySize, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::TypeAttr elem_type) {
  odsState.addOperands(arraySize);
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (elem_type) {
  odsState.addAttribute(getElemTypeAttrName(odsState.name), elem_type);
  }
  odsState.addTypes(res);
}

void AllocaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arraySize, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::TypeAttr elem_type) {
  odsState.addOperands(arraySize);
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (elem_type) {
  odsState.addAttribute(getElemTypeAttrName(odsState.name), elem_type);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AllocaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AllocaOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_elem_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getElemTypeAttrName()) {
      tblgen_elem_type = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps2(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps3(*this, tblgen_elem_type, "elem_type")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AllocaOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void AllocaOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSResults(0))
    effects.emplace_back(::mlir::MemoryEffects::Allocate::get(), value, ::mlir::SideEffects::AutomaticAllocationScopeResource::get());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AllocaOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AndOp definitions
//===----------------------------------------------------------------------===//

AndOpAdaptor::AndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.and", odsAttrs.getContext());
}

AndOpAdaptor::AndOpAdaptor(AndOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AndOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AndOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AndOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr AndOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AndOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value AndOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AndOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AndOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AndOp::getRes() {
  return *getODSResults(0).begin();
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(AndOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void AndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(AndOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult AndOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AndOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult AndOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult AndOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void AndOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void AndOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AndOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AtomicCmpXchgOp definitions
//===----------------------------------------------------------------------===//

AtomicCmpXchgOpAdaptor::AtomicCmpXchgOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.cmpxchg", odsAttrs.getContext());
}

AtomicCmpXchgOpAdaptor::AtomicCmpXchgOpAdaptor(AtomicCmpXchgOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AtomicCmpXchgOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AtomicCmpXchgOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AtomicCmpXchgOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicCmpXchgOpAdaptor::getPtr() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicCmpXchgOpAdaptor::getCmp() {
  return *getODSOperands(1).begin();
}

::mlir::Value AtomicCmpXchgOpAdaptor::getVal() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr AtomicCmpXchgOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::AtomicOrderingAttr AtomicCmpXchgOpAdaptor::getSuccessOrderingAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, AtomicCmpXchgOp::getSuccessOrderingAttrName(*odsOpName)).cast<::mlir::LLVM::AtomicOrderingAttr>();
    return attr;
}

::mlir::LLVM::AtomicOrdering AtomicCmpXchgOpAdaptor::getSuccessOrdering() {
  auto attr = getSuccessOrderingAttr();
  return attr.getValue();
}

::mlir::LLVM::AtomicOrderingAttr AtomicCmpXchgOpAdaptor::getFailureOrderingAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AtomicCmpXchgOp::getFailureOrderingAttrName(*odsOpName)).cast<::mlir::LLVM::AtomicOrderingAttr>();
    return attr;
}

::mlir::LLVM::AtomicOrdering AtomicCmpXchgOpAdaptor::getFailureOrdering() {
  auto attr = getFailureOrderingAttr();
  return attr.getValue();
}

::mlir::LogicalResult AtomicCmpXchgOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_failure_ordering;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.cmpxchg' op ""requires attribute 'failure_ordering'");
    if (namedAttrIt->getName() == AtomicCmpXchgOp::getFailureOrderingAttrName(*odsOpName)) {
      tblgen_failure_ordering = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_success_ordering;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.cmpxchg' op ""requires attribute 'success_ordering'");
    if (namedAttrIt->getName() == AtomicCmpXchgOp::getSuccessOrderingAttrName(*odsOpName)) {
      tblgen_success_ordering = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_success_ordering && !((tblgen_success_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>())))
    return emitError(loc, "'llvm.cmpxchg' op ""attribute 'success_ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");

  if (tblgen_failure_ordering && !((tblgen_failure_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>())))
    return emitError(loc, "'llvm.cmpxchg' op ""attribute 'failure_ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicCmpXchgOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicCmpXchgOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicCmpXchgOp::getPtr() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicCmpXchgOp::getCmp() {
  return *getODSOperands(1).begin();
}

::mlir::Value AtomicCmpXchgOp::getVal() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange AtomicCmpXchgOp::getPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtomicCmpXchgOp::getCmpMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtomicCmpXchgOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtomicCmpXchgOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicCmpXchgOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicCmpXchgOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::AtomicOrderingAttr AtomicCmpXchgOp::getSuccessOrderingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSuccessOrderingAttrName()).cast<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering AtomicCmpXchgOp::getSuccessOrdering() {
  auto attr = getSuccessOrderingAttr();
  return attr.getValue();
}

::mlir::LLVM::AtomicOrderingAttr AtomicCmpXchgOp::getFailureOrderingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFailureOrderingAttrName()).cast<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering AtomicCmpXchgOp::getFailureOrdering() {
  auto attr = getFailureOrderingAttr();
  return attr.getValue();
}

void AtomicCmpXchgOp::setSuccessOrderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr(getSuccessOrderingAttrName(), attr);
}

void AtomicCmpXchgOp::setFailureOrderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr(getFailureOrderingAttrName(), attr);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr success_ordering, ::mlir::LLVM::AtomicOrderingAttr failure_ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute(getSuccessOrderingAttrName(odsState.name), success_ordering);
  odsState.addAttribute(getFailureOrderingAttrName(odsState.name), failure_ordering);
  odsState.addTypes(res);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr success_ordering, ::mlir::LLVM::AtomicOrderingAttr failure_ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute(getSuccessOrderingAttrName(odsState.name), success_ordering);
  odsState.addAttribute(getFailureOrderingAttrName(odsState.name), failure_ordering);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering success_ordering, ::mlir::LLVM::AtomicOrdering failure_ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute(getSuccessOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), success_ordering));
  odsState.addAttribute(getFailureOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), failure_ordering));
  odsState.addTypes(res);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value cmp, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering success_ordering, ::mlir::LLVM::AtomicOrdering failure_ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(cmp);
  odsState.addOperands(val);
  odsState.addAttribute(getSuccessOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), success_ordering));
  odsState.addAttribute(getFailureOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), failure_ordering));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicCmpXchgOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AtomicCmpXchgOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_failure_ordering;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'failure_ordering'");
    if (namedAttrIt->getName() == getFailureOrderingAttrName()) {
      tblgen_failure_ordering = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_success_ordering;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'success_ordering'");
    if (namedAttrIt->getName() == getSuccessOrderingAttrName()) {
      tblgen_success_ordering = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_success_ordering, "success_ordering")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_failure_ordering, "failure_ordering")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps6(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AtomicCmpXchgOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AtomicCmpXchgOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AtomicRMWOp definitions
//===----------------------------------------------------------------------===//

AtomicRMWOpAdaptor::AtomicRMWOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.atomicrmw", odsAttrs.getContext());
}

AtomicRMWOpAdaptor::AtomicRMWOpAdaptor(AtomicRMWOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AtomicRMWOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AtomicRMWOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AtomicRMWOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicRMWOpAdaptor::getPtr() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicRMWOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr AtomicRMWOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::AtomicBinOpAttr AtomicRMWOpAdaptor::getBinOpAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, AtomicRMWOp::getBinOpAttrName(*odsOpName)).cast<::mlir::LLVM::AtomicBinOpAttr>();
    return attr;
}

::mlir::LLVM::AtomicBinOp AtomicRMWOpAdaptor::getBinOp() {
  auto attr = getBinOpAttr();
  return attr.getValue();
}

::mlir::LLVM::AtomicOrderingAttr AtomicRMWOpAdaptor::getOrderingAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, AtomicRMWOp::getOrderingAttrName(*odsOpName)).cast<::mlir::LLVM::AtomicOrderingAttr>();
    return attr;
}

::mlir::LLVM::AtomicOrdering AtomicRMWOpAdaptor::getOrdering() {
  auto attr = getOrderingAttr();
  return attr.getValue();
}

::mlir::LogicalResult AtomicRMWOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_bin_op;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.atomicrmw' op ""requires attribute 'bin_op'");
    if (namedAttrIt->getName() == AtomicRMWOp::getBinOpAttrName(*odsOpName)) {
      tblgen_bin_op = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_ordering;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.atomicrmw' op ""requires attribute 'ordering'");
    if (namedAttrIt->getName() == AtomicRMWOp::getOrderingAttrName(*odsOpName)) {
      tblgen_ordering = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_bin_op && !((tblgen_bin_op.isa<::mlir::LLVM::AtomicBinOpAttr>())))
    return emitError(loc, "'llvm.atomicrmw' op ""attribute 'bin_op' failed to satisfy constraint: llvm.atomicrmw binary operations");

  if (tblgen_ordering && !((tblgen_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>())))
    return emitError(loc, "'llvm.atomicrmw' op ""attribute 'ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AtomicRMWOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AtomicRMWOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicRMWOp::getPtr() {
  return *getODSOperands(0).begin();
}

::mlir::Value AtomicRMWOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange AtomicRMWOp::getPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AtomicRMWOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AtomicRMWOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AtomicRMWOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AtomicRMWOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::AtomicBinOpAttr AtomicRMWOp::getBinOpAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getBinOpAttrName()).cast<::mlir::LLVM::AtomicBinOpAttr>();
}

::mlir::LLVM::AtomicBinOp AtomicRMWOp::getBinOp() {
  auto attr = getBinOpAttr();
  return attr.getValue();
}

::mlir::LLVM::AtomicOrderingAttr AtomicRMWOp::getOrderingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOrderingAttrName()).cast<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering AtomicRMWOp::getOrdering() {
  auto attr = getOrderingAttr();
  return attr.getValue();
}

void AtomicRMWOp::setBinOpAttr(::mlir::LLVM::AtomicBinOpAttr attr) {
  (*this)->setAttr(getBinOpAttrName(), attr);
}

void AtomicRMWOp::setOrderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr(getOrderingAttrName(), attr);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::AtomicBinOpAttr bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute(getBinOpAttrName(odsState.name), bin_op);
  odsState.addAttribute(getOrderingAttrName(odsState.name), ordering);
  odsState.addTypes(res);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::AtomicBinOpAttr bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrderingAttr ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute(getBinOpAttrName(odsState.name), bin_op);
  odsState.addAttribute(getOrderingAttrName(odsState.name), ordering);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::AtomicBinOp bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute(getBinOpAttrName(odsState.name), ::mlir::LLVM::AtomicBinOpAttr::get(odsBuilder.getContext(), bin_op));
  odsState.addAttribute(getOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  odsState.addTypes(res);
}

void AtomicRMWOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::AtomicBinOp bin_op, ::mlir::Value ptr, ::mlir::Value val, ::mlir::LLVM::AtomicOrdering ordering) {
  odsState.addOperands(ptr);
  odsState.addOperands(val);
  odsState.addAttribute(getBinOpAttrName(odsState.name), ::mlir::LLVM::AtomicBinOpAttr::get(odsBuilder.getContext(), bin_op));
  odsState.addAttribute(getOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AtomicRMWOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AtomicRMWOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_bin_op;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'bin_op'");
    if (namedAttrIt->getName() == getBinOpAttrName()) {
      tblgen_bin_op = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_ordering;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'ordering'");
    if (namedAttrIt->getName() == getOrderingAttrName()) {
      tblgen_ordering = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps5(*this, tblgen_bin_op, "bin_op")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_ordering, "ordering")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps8(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AtomicRMWOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AtomicRMWOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::BitcastOp definitions
//===----------------------------------------------------------------------===//

BitcastOpAdaptor::BitcastOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.bitcast", odsAttrs.getContext());
}

BitcastOpAdaptor::BitcastOpAdaptor(BitcastOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BitcastOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BitcastOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BitcastOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitcastOpAdaptor::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr BitcastOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BitcastOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BitcastOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitcastOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitcastOp::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange BitcastOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BitcastOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitcastOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitcastOp::getRes() {
  return *getODSResults(0).begin();
}

void BitcastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void BitcastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void BitcastOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitcastOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BitcastOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps9(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BitcastOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BitcastOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BitcastOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void BitcastOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::BitcastOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::BrOp definitions
//===----------------------------------------------------------------------===//

BrOpAdaptor::BrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.br", odsAttrs.getContext());
}

BrOpAdaptor::BrOpAdaptor(BrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange BrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange BrOpAdaptor::getDestOperands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr BrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BrOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range BrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range BrOp::getDestOperands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange BrOp::getDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *BrOp::getDest() {
  return (*this)->getSuccessor(0);
}

void BrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, SuccessorRange destinations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(operands);
    odsState.addSuccessors(destinations);
    odsState.addAttributes(attributes);
  
}

void BrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange destOperands, ::mlir::Block *dest) {
  odsState.addOperands(destOperands);
  odsState.addSuccessors(dest);
}

void BrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange destOperands, ::mlir::Block *dest) {
  odsState.addOperands(destOperands);
  odsState.addSuccessors(dest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult BrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult BrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult BrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Block *destSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> destOperandsOperands;
  ::llvm::SMLoc destOperandsOperandsLoc;
  (void)destOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> destOperandsTypes;

  if (parser.parseSuccessor(destSuccessor))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  destOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(destOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(destOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addSuccessors(destSuccessor);
  if (parser.resolveOperands(destOperandsOperands, destOperandsTypes, destOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void BrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getDest();
  if (!getDestOperands().empty()) {
  _odsPrinter << "(";
  _odsPrinter << getDestOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDestOperands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

void BrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::BrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CallOp definitions
//===----------------------------------------------------------------------===//

CallOpAdaptor::CallOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.call", odsAttrs.getContext());
}

CallOpAdaptor::CallOpAdaptor(CallOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CallOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CallOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange CallOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CallOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr CallOpAdaptor::getCalleeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CallOp::getCalleeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > CallOpAdaptor::getCallee() {
  auto attr = getCalleeAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::LLVM::FMFAttr CallOpAdaptor::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CallOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
    if (!attr)
      attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
    return attr;
}

::mlir::LLVM::FastmathFlags CallOpAdaptor::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

::mlir::LogicalResult CallOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_callee;
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == CallOp::getCalleeAttrName(*odsOpName)) {
      tblgen_callee = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == CallOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_callee && !((tblgen_callee.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'llvm.call' op ""attribute 'callee' failed to satisfy constraint: flat symbol reference attribute");

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>())))
    return emitError(loc, "'llvm.call' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CallOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range CallOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CallOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range CallOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::FlatSymbolRefAttr CallOp::getCalleeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCalleeAttrName()).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
}

::llvm::Optional< ::llvm::StringRef > CallOp::getCallee() {
  auto attr = getCalleeAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::LLVM::FMFAttr CallOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags CallOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void CallOp::setCalleeAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getCalleeAttrName(), attr);
}

void CallOp::setFastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

::mlir::Attribute CallOp::removeCalleeAttr() {
  return (*this)->removeAttr(getCalleeAttrName());
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, LLVMFuncOp func, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
      Type resultType = func.getFunctionType().getReturnType();
      if (!resultType.isa<LLVM::LLVMVoidType>())
        odsState.addTypes(resultType);
      odsState.addAttribute("callee", SymbolRefAttr::get(func));
      odsState.addAttributes(attributes);
      odsState.addOperands(operands);
    
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, StringAttr callee, ValueRange operands) {
      build(odsBuilder, odsState, results, SymbolRefAttr::get(callee), operands);
    
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange results, StringRef callee, ValueRange operands) {
      build(odsBuilder, odsState, results,
            StringAttr::get(odsBuilder.getContext(), callee), operands);
    
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, /*optional*/::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange odsArg_0, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(odsArg_0);
  if (callee) {
  odsState.addAttribute(getCalleeAttrName(odsState.name), callee);
  }
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(resultType0);
}

void CallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, /*optional*/::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange odsArg_0, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(odsArg_0);
  if (callee) {
  odsState.addAttribute(getCalleeAttrName(odsState.name), callee);
  }
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(resultType0);
}

void CallOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CallOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_callee;
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getCalleeAttrName()) {
      tblgen_callee = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps1(*this, tblgen_callee, "callee")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CallOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CallOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CondBrOp definitions
//===----------------------------------------------------------------------===//

CondBrOpAdaptor::CondBrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.cond_br", odsAttrs.getContext());
}

CondBrOpAdaptor::CondBrOpAdaptor(CondBrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CondBrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CondBrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, CondBrOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange CondBrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CondBrOpAdaptor::getCondition() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange CondBrOpAdaptor::getTrueDestOperands() {
  return getODSOperands(1);
}

::mlir::ValueRange CondBrOpAdaptor::getFalseDestOperands() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr CondBrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ElementsAttr CondBrOpAdaptor::getBranchWeightsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, CondBrOp::getBranchWeightsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ElementsAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ElementsAttr > CondBrOpAdaptor::getBranchWeights() {
  auto attr = getBranchWeightsAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult CondBrOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_branch_weights;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.cond_br' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == CondBrOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == CondBrOp::getBranchWeightsAttrName(*odsOpName)) {
      tblgen_branch_weights = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'llvm.cond_br' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_branch_weights && !((tblgen_branch_weights.isa<::mlir::ElementsAttr>())))
    return emitError(loc, "'llvm.cond_br' op ""attribute 'branch_weights' failed to satisfy constraint: constant vector/tensor attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CondBrOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range CondBrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CondBrOp::getCondition() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range CondBrOp::getTrueDestOperands() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range CondBrOp::getFalseDestOperands() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange CondBrOp::getConditionMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CondBrOp::getTrueDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange CondBrOp::getFalseDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> CondBrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CondBrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *CondBrOp::getTrueDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *CondBrOp::getFalseDest() {
  return (*this)->getSuccessor(1);
}

::mlir::ElementsAttr CondBrOp::getBranchWeightsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getBranchWeightsAttrName()).dyn_cast_or_null<::mlir::ElementsAttr>();
}

::llvm::Optional< ::mlir::ElementsAttr > CondBrOp::getBranchWeights() {
  auto attr = getBranchWeightsAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

void CondBrOp::setBranchWeightsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(getBranchWeightsAttrName(), attr);
}

::mlir::Attribute CondBrOp::removeBranch_weightsAttr() {
  return (*this)->removeAttr(getBranchWeightsAttrName());
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, Block *trueDest, ValueRange trueOperands, Block *falseDest, ValueRange falseOperands, Optional<std::pair<uint32_t, uint32_t>> weights) {
        ElementsAttr weightsAttr;
        if (weights) {
          weightsAttr =
              odsBuilder.getI32VectorAttr({static_cast<int32_t>(weights->first),
                                       static_cast<int32_t>(weights->second)});
        }
        build(odsBuilder, odsState, condition, trueOperands, falseOperands, weightsAttr,
              trueDest, falseDest);
  
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value condition, Block *trueDest, Block *falseDest, ValueRange falseOperands) {
      build(odsBuilder, odsState, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
  
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, SuccessorRange destinations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(operands);
    odsState.addSuccessors(destinations);
    odsState.addAttributes(attributes);
  
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::mlir::ValueRange trueDestOperands, ::mlir::ValueRange falseDestOperands, /*optional*/::mlir::ElementsAttr branch_weights, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}));
  if (branch_weights) {
  odsState.addAttribute(getBranchWeightsAttrName(odsState.name), branch_weights);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
}

void CondBrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::ValueRange trueDestOperands, ::mlir::ValueRange falseDestOperands, /*optional*/::mlir::ElementsAttr branch_weights, ::mlir::Block *trueDest, ::mlir::Block *falseDest) {
  odsState.addOperands(condition);
  odsState.addOperands(trueDestOperands);
  odsState.addOperands(falseDestOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(trueDestOperands.size()), static_cast<int32_t>(falseDestOperands.size())}));
  if (branch_weights) {
  odsState.addAttribute(getBranchWeightsAttrName(odsState.name), branch_weights);
  }
  odsState.addSuccessors(trueDest);
  odsState.addSuccessors(falseDest);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CondBrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CondBrOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_branch_weights;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBranchWeightsAttrName()) {
      tblgen_branch_weights = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps7(*this, tblgen_branch_weights, "branch_weights")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult CondBrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CondBrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand conditionRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> conditionOperands(conditionRawOperands);  ::llvm::SMLoc conditionOperandsLoc;
  (void)conditionOperandsLoc;
  ::mlir::ElementsAttr branch_weightsAttr;
  ::mlir::Block *trueDestSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> trueDestOperandsOperands;
  ::llvm::SMLoc trueDestOperandsOperandsLoc;
  (void)trueDestOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> trueDestOperandsTypes;
  ::mlir::Block *falseDestSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> falseDestOperandsOperands;
  ::llvm::SMLoc falseDestOperandsOperandsLoc;
  (void)falseDestOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> falseDestOperandsTypes;

  conditionOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(conditionRawOperands[0]))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("weights"))) {
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(branch_weightsAttr, ::mlir::Type{}, "branch_weights",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(trueDestSuccessor))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  trueDestOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(trueDestOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(trueDestOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(falseDestSuccessor))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  falseDestOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(falseDestOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(falseDestOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addSuccessors(trueDestSuccessor);
  result.addSuccessors(falseDestSuccessor);
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, static_cast<int32_t>(trueDestOperandsOperands.size()), static_cast<int32_t>(falseDestOperandsOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(1);
  if (parser.resolveOperands(conditionOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(trueDestOperandsOperands, trueDestOperandsTypes, trueDestOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(falseDestOperandsOperands, falseDestOperandsTypes, falseDestOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CondBrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getCondition();
  if ((*this)->getAttr("branch_weights")) {
  _odsPrinter << ' ' << "weights";
  _odsPrinter << "(";
_odsPrinter.printStrippedAttrOrType(getBranchWeightsAttr());
  _odsPrinter << ")";
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getTrueDest();
  if (!getTrueDestOperands().empty()) {
  _odsPrinter << "(";
  _odsPrinter << getTrueDestOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getTrueDestOperands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getFalseDest();
  if (!getFalseDestOperands().empty()) {
  _odsPrinter << "(";
  _odsPrinter << getFalseDestOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getFalseDestOperands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "branch_weights"});
}

void CondBrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CondBrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ConstantOp definitions
//===----------------------------------------------------------------------===//

ConstantOpAdaptor::ConstantOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.constant", odsAttrs.getContext());
}

ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ConstantOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ConstantOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ConstantOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr ConstantOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::Attribute ConstantOpAdaptor::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ConstantOp::getValueAttrName(*odsOpName)).cast<::mlir::Attribute>();
    return attr;
}

::mlir::Attribute ConstantOpAdaptor::getValue() {
  auto attr = getValueAttr();
  return attr;
}

::mlir::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.constant' op ""requires attribute 'value'");
    if (namedAttrIt->getName() == ConstantOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_value && !((true)))
    return emitError(loc, "'llvm.mlir.constant' op ""attribute 'value' failed to satisfy constraint: any attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConstantOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstantOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstantOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstantOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ConstantOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::Attribute ConstantOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getValueAttrName()).cast<::mlir::Attribute>();
}

::mlir::Attribute ConstantOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void ConstantOp::setValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Attribute value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  odsState.addTypes(res);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Attribute value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConstantOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'value'");
    if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps8(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstantOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ConstantOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Attribute valueAttr;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseLParen())
    return ::mlir::failure();

  if (parser.parseAttribute(valueAttr, ::mlir::Type{}, "value", result.attributes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void ConstantOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter.printAttribute(getValueAttr());
  _odsPrinter << ")";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ConstantOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ConstantOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ExtractElementOp definitions
//===----------------------------------------------------------------------===//

ExtractElementOpAdaptor::ExtractElementOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.extractelement", odsAttrs.getContext());
}

ExtractElementOpAdaptor::ExtractElementOpAdaptor(ExtractElementOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ExtractElementOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExtractElementOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ExtractElementOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOpAdaptor::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractElementOpAdaptor::getPosition() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ExtractElementOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ExtractElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExtractElementOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExtractElementOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOp::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value ExtractElementOp::getPosition() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ExtractElementOp::getVectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ExtractElementOp::getPositionMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractElementOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractElementOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractElementOp::getRes() {
  return *getODSResults(0).begin();
}

void ExtractElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value vector, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(position);
  odsState.addTypes(res);
}

void ExtractElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractElementOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExtractElementOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExtractElementOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void ExtractElementOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ExtractElementOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ExtractValueOp definitions
//===----------------------------------------------------------------------===//

ExtractValueOpAdaptor::ExtractValueOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.extractvalue", odsAttrs.getContext());
}

ExtractValueOpAdaptor::ExtractValueOpAdaptor(ExtractValueOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ExtractValueOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExtractValueOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ExtractValueOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractValueOpAdaptor::getContainer() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ExtractValueOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ExtractValueOpAdaptor::getPositionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ExtractValueOp::getPositionAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
    return attr;
}

::mlir::ArrayAttr ExtractValueOpAdaptor::getPosition() {
  auto attr = getPositionAttr();
  return attr;
}

::mlir::LogicalResult ExtractValueOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_position;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.extractvalue' op ""requires attribute 'position'");
    if (namedAttrIt->getName() == ExtractValueOp::getPositionAttrName(*odsOpName)) {
      tblgen_position = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_position && !((tblgen_position.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'llvm.extractvalue' op ""attribute 'position' failed to satisfy constraint: array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExtractValueOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExtractValueOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractValueOp::getContainer() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ExtractValueOp::getContainerMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExtractValueOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExtractValueOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExtractValueOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr ExtractValueOp::getPositionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPositionAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ExtractValueOp::getPosition() {
  auto attr = getPositionAttr();
  return attr;
}

void ExtractValueOp::setPositionAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPositionAttrName(), attr);
}

void ExtractValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void ExtractValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value container, ::mlir::ArrayAttr position) {
  odsState.addOperands(container);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);
  odsState.addTypes(res);
}

void ExtractValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value container, ::mlir::ArrayAttr position) {
  odsState.addOperands(container);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExtractValueOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ExtractValueOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_position;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'position'");
    if (namedAttrIt->getName() == getPositionAttrName()) {
      tblgen_position = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_position, "position")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExtractValueOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void ExtractValueOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ExtractValueOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FAddOp definitions
//===----------------------------------------------------------------------===//

FAddOpAdaptor::FAddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fadd", odsAttrs.getContext());
}

FAddOpAdaptor::FAddOpAdaptor(FAddOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FAddOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FAddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FAddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAddOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FAddOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FAddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FMFAttr FAddOpAdaptor::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FAddOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
    if (!attr)
      attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
    return attr;
}

::mlir::LLVM::FastmathFlags FAddOpAdaptor::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

::mlir::LogicalResult FAddOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FAddOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>())))
    return emitError(loc, "'llvm.fadd' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FAddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FAddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAddOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FAddOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FAddOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FAddOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FAddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FAddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAddOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FMFAttr FAddOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FAddOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FAddOp::setFastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FAddOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FAddOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FAddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FAddOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FAddOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FAddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FAddOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FAddOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FAddOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FAddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FAddOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FCmpOp definitions
//===----------------------------------------------------------------------===//

FCmpOpAdaptor::FCmpOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fcmp", odsAttrs.getContext());
}

FCmpOpAdaptor::FCmpOpAdaptor(FCmpOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FCmpOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FCmpOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FCmpOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCmpOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FCmpOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FCmpOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FCmpPredicateAttr FCmpOpAdaptor::getPredicateAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FCmpOp::getPredicateAttrName(*odsOpName)).cast<::mlir::LLVM::FCmpPredicateAttr>();
    return attr;
}

::mlir::LLVM::FCmpPredicate FCmpOpAdaptor::getPredicate() {
  auto attr = getPredicateAttr();
  return attr.getValue();
}

::mlir::LLVM::FMFAttr FCmpOpAdaptor::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, FCmpOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
    if (!attr)
      attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
    return attr;
}

::mlir::LLVM::FastmathFlags FCmpOpAdaptor::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

::mlir::LogicalResult FCmpOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_predicate;
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.fcmp' op ""requires attribute 'predicate'");
    if (namedAttrIt->getName() == FCmpOp::getPredicateAttrName(*odsOpName)) {
      tblgen_predicate = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == FCmpOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_predicate && !((tblgen_predicate.isa<::mlir::LLVM::FCmpPredicateAttr>())))
    return emitError(loc, "'llvm.fcmp' op ""attribute 'predicate' failed to satisfy constraint: llvm.fcmp comparison predicate");

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>())))
    return emitError(loc, "'llvm.fcmp' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FCmpOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FCmpOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCmpOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FCmpOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FCmpOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FCmpOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FCmpOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FCmpOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCmpOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FCmpPredicateAttr FCmpOp::getPredicateAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPredicateAttrName()).cast<::mlir::LLVM::FCmpPredicateAttr>();
}

::mlir::LLVM::FCmpPredicate FCmpOp::getPredicate() {
  auto attr = getPredicateAttr();
  return attr.getValue();
}

::mlir::LLVM::FMFAttr FCmpOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FCmpOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FCmpOp::setPredicateAttr(::mlir::LLVM::FCmpPredicateAttr attr) {
  (*this)->setAttr(getPredicateAttrName(), attr);
}

void FCmpOp::setFastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::FCmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), predicate);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::FCmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), predicate);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::FCmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), ::mlir::LLVM::FCmpPredicateAttr::get(odsBuilder.getContext(), predicate));
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FCmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::FCmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), ::mlir::LLVM::FCmpPredicateAttr::get(odsBuilder.getContext(), predicate));
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FCmpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FCmpOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_predicate;
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'predicate'");
    if (namedAttrIt->getName() == getPredicateAttrName()) {
      tblgen_predicate = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps10(*this, tblgen_predicate, "predicate")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps15(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FCmpOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void FCmpOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FCmpOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FDivOp definitions
//===----------------------------------------------------------------------===//

FDivOpAdaptor::FDivOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fdiv", odsAttrs.getContext());
}

FDivOpAdaptor::FDivOpAdaptor(FDivOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FDivOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FDivOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FDivOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FDivOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FDivOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FDivOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FMFAttr FDivOpAdaptor::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FDivOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
    if (!attr)
      attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
    return attr;
}

::mlir::LLVM::FastmathFlags FDivOpAdaptor::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

::mlir::LogicalResult FDivOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FDivOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>())))
    return emitError(loc, "'llvm.fdiv' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FDivOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FDivOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FDivOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FDivOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FDivOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FMFAttr FDivOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FDivOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FDivOp::setFastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FDivOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FDivOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FDivOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FDivOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FDivOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FDivOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FDivOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FDivOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FDivOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FDivOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FMulOp definitions
//===----------------------------------------------------------------------===//

FMulOpAdaptor::FMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fmul", odsAttrs.getContext());
}

FMulOpAdaptor::FMulOpAdaptor(FMulOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FMulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FMulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FMulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMulOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FMulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FMFAttr FMulOpAdaptor::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FMulOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
    if (!attr)
      attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
    return attr;
}

::mlir::LLVM::FastmathFlags FMulOpAdaptor::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

::mlir::LogicalResult FMulOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FMulOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>())))
    return emitError(loc, "'llvm.fmul' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FMulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FMulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMulOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FMulOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FMulOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FMulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FMulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FMFAttr FMulOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FMulOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FMulOp::setFastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FMulOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FMulOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FMulOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FMulOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FMulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FMulOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FMulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FMulOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FMulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FMulOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FNegOp definitions
//===----------------------------------------------------------------------===//

FNegOpAdaptor::FNegOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fneg", odsAttrs.getContext());
}

FNegOpAdaptor::FNegOpAdaptor(FNegOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FNegOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FNegOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FNegOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FNegOpAdaptor::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FNegOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FMFAttr FNegOpAdaptor::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FNegOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
    if (!attr)
      attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
    return attr;
}

::mlir::LLVM::FastmathFlags FNegOpAdaptor::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

::mlir::LogicalResult FNegOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FNegOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>())))
    return emitError(loc, "'llvm.fneg' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FNegOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FNegOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FNegOp::getOperand() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FNegOp::getOperandMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FNegOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FNegOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FNegOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FMFAttr FNegOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FNegOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FNegOp::setFastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value operand, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(operand);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(operand);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FNegOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(operand);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value operand, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(operand);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value operand, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(operand);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FNegOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value operand, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(operand);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FNegOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FNegOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FNegOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FNegOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FNegOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FNegOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand operandRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> operandOperands(operandRawOperands);  ::llvm::SMLoc operandOperandsLoc;
  (void)operandOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  operandOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(operandRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(operandOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FNegOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperand();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FNegOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FNegOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FPExtOp definitions
//===----------------------------------------------------------------------===//

FPExtOpAdaptor::FPExtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fpext", odsAttrs.getContext());
}

FPExtOpAdaptor::FPExtOpAdaptor(FPExtOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FPExtOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FPExtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FPExtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPExtOpAdaptor::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FPExtOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FPExtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FPExtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPExtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPExtOp::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FPExtOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FPExtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPExtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPExtOp::getRes() {
  return *getODSResults(0).begin();
}

void FPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void FPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPExtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FPExtOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FPExtOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FPExtOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FPExtOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FPExtOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FPExtOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FPToSIOp definitions
//===----------------------------------------------------------------------===//

FPToSIOpAdaptor::FPToSIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fptosi", odsAttrs.getContext());
}

FPToSIOpAdaptor::FPToSIOpAdaptor(FPToSIOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FPToSIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FPToSIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FPToSIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToSIOpAdaptor::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FPToSIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FPToSIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FPToSIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPToSIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToSIOp::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FPToSIOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FPToSIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPToSIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToSIOp::getRes() {
  return *getODSResults(0).begin();
}

void FPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void FPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPToSIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FPToSIOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FPToSIOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FPToSIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FPToSIOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FPToSIOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FPToSIOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FPToUIOp definitions
//===----------------------------------------------------------------------===//

FPToUIOpAdaptor::FPToUIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fptoui", odsAttrs.getContext());
}

FPToUIOpAdaptor::FPToUIOpAdaptor(FPToUIOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FPToUIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FPToUIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FPToUIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToUIOpAdaptor::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FPToUIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FPToUIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FPToUIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPToUIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToUIOp::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FPToUIOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FPToUIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPToUIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPToUIOp::getRes() {
  return *getODSResults(0).begin();
}

void FPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void FPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPToUIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FPToUIOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FPToUIOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FPToUIOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FPToUIOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FPToUIOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FPToUIOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FPTruncOp definitions
//===----------------------------------------------------------------------===//

FPTruncOpAdaptor::FPTruncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fptrunc", odsAttrs.getContext());
}

FPTruncOpAdaptor::FPTruncOpAdaptor(FPTruncOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FPTruncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FPTruncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FPTruncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPTruncOpAdaptor::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FPTruncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FPTruncOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FPTruncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FPTruncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPTruncOp::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FPTruncOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FPTruncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FPTruncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FPTruncOp::getRes() {
  return *getODSResults(0).begin();
}

void FPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void FPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FPTruncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FPTruncOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FPTruncOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult FPTruncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FPTruncOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FPTruncOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FPTruncOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FRemOp definitions
//===----------------------------------------------------------------------===//

FRemOpAdaptor::FRemOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.frem", odsAttrs.getContext());
}

FRemOpAdaptor::FRemOpAdaptor(FRemOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FRemOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FRemOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FRemOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FRemOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FRemOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FRemOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FMFAttr FRemOpAdaptor::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FRemOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
    if (!attr)
      attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
    return attr;
}

::mlir::LLVM::FastmathFlags FRemOpAdaptor::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

::mlir::LogicalResult FRemOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FRemOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>())))
    return emitError(loc, "'llvm.frem' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FRemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FRemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FRemOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FRemOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FRemOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FRemOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FRemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FRemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FRemOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FMFAttr FRemOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FRemOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FRemOp::setFastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FRemOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FRemOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FRemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FRemOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FRemOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FRemOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FRemOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FRemOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FRemOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FRemOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FRemOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FSubOp definitions
//===----------------------------------------------------------------------===//

FSubOpAdaptor::FSubOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fsub", odsAttrs.getContext());
}

FSubOpAdaptor::FSubOpAdaptor(FSubOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FSubOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FSubOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FSubOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FSubOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FSubOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr FSubOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::FMFAttr FSubOpAdaptor::getFastmathFlagsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, FSubOp::getFastmathFlagsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
    if (!attr)
      attr = ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {});
    return attr;
}

::mlir::LLVM::FastmathFlags FSubOpAdaptor::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

::mlir::LogicalResult FSubOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == FSubOp::getFastmathFlagsAttrName(*odsOpName)) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_fastmathFlags && !((tblgen_fastmathFlags.isa<::mlir::LLVM::FMFAttr>())))
    return emitError(loc, "'llvm.fsub' op ""attribute 'fastmathFlags' failed to satisfy constraint: LLVM fastmath flags");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FSubOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FSubOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FSubOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value FSubOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange FSubOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FSubOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FSubOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FSubOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FSubOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::FMFAttr FSubOp::getFastmathFlagsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFastmathFlagsAttrName()).dyn_cast_or_null<::mlir::LLVM::FMFAttr>();
}

::mlir::LLVM::FastmathFlags FSubOp::getFastmathFlags() {
  auto attr = getFastmathFlagsAttr();
    if (!attr)
      return ::mlir::LLVM::FMFAttr::get(::mlir::Builder((*this)->getContext()).getContext(), {}).getFlags();
  return attr.getFlags();
}

void FSubOp::setFastmathFlagsAttr(::mlir::LLVM::FMFAttr attr) {
  (*this)->setAttr(getFastmathFlagsAttrName(), attr);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  odsState.addTypes(res);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FSubOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FMFAttr fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  if (fastmathFlags) {
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), fastmathFlags);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  odsState.addTypes(res);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FSubOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::LLVM::FastmathFlags fastmathFlags) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getFastmathFlagsAttrName(odsState.name), ::mlir::LLVM::FMFAttr::get(odsBuilder.getContext(), fastmathFlags));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FSubOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FSubOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FSubOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_fastmathFlags;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getFastmathFlagsAttrName()) {
      tblgen_fastmathFlags = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps6(*this, tblgen_fastmathFlags, "fastmathFlags")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FSubOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FSubOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FSubOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FSubOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void FSubOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FSubOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FenceOp definitions
//===----------------------------------------------------------------------===//

FenceOpAdaptor::FenceOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.fence", odsAttrs.getContext());
}

FenceOpAdaptor::FenceOpAdaptor(FenceOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FenceOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FenceOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FenceOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr FenceOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::AtomicOrderingAttr FenceOpAdaptor::getOrderingAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, FenceOp::getOrderingAttrName(*odsOpName)).cast<::mlir::LLVM::AtomicOrderingAttr>();
    return attr;
}

::mlir::LLVM::AtomicOrdering FenceOpAdaptor::getOrdering() {
  auto attr = getOrderingAttr();
  return attr.getValue();
}

::mlir::StringAttr FenceOpAdaptor::getSyncscopeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, FenceOp::getSyncscopeAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef FenceOpAdaptor::getSyncscope() {
  auto attr = getSyncscopeAttr();
  return attr.getValue();
}

::mlir::LogicalResult FenceOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_ordering;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.fence' op ""requires attribute 'ordering'");
    if (namedAttrIt->getName() == FenceOp::getOrderingAttrName(*odsOpName)) {
      tblgen_ordering = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_syncscope;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.fence' op ""requires attribute 'syncscope'");
    if (namedAttrIt->getName() == FenceOp::getSyncscopeAttrName(*odsOpName)) {
      tblgen_syncscope = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_ordering && !((tblgen_ordering.isa<::mlir::LLVM::AtomicOrderingAttr>())))
    return emitError(loc, "'llvm.fence' op ""attribute 'ordering' failed to satisfy constraint: Atomic ordering for LLVM's memory model");

  if (tblgen_syncscope && !((tblgen_syncscope.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.fence' op ""attribute 'syncscope' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FenceOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FenceOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> FenceOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FenceOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::LLVM::AtomicOrderingAttr FenceOp::getOrderingAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getOrderingAttrName()).cast<::mlir::LLVM::AtomicOrderingAttr>();
}

::mlir::LLVM::AtomicOrdering FenceOp::getOrdering() {
  auto attr = getOrderingAttr();
  return attr.getValue();
}

::mlir::StringAttr FenceOp::getSyncscopeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getSyncscopeAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef FenceOp::getSyncscope() {
  auto attr = getSyncscopeAttr();
  return attr.getValue();
}

void FenceOp::setOrderingAttr(::mlir::LLVM::AtomicOrderingAttr attr) {
  (*this)->setAttr(getOrderingAttrName(), attr);
}

void FenceOp::setSyncscopeAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSyncscopeAttrName(), attr);
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    assert(isCompatibleType(resultType) && "result must be an LLVM type");
    assert(resultType.isa<LLVMVoidType>() &&
           "for zero-result operands, only 'void' is accepted as result type");
    build(odsBuilder, odsState, operands, attributes);
  
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::LLVM::AtomicOrderingAttr ordering, ::mlir::StringAttr syncscope) {
  odsState.addAttribute(getOrderingAttrName(odsState.name), ordering);
  odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::AtomicOrderingAttr ordering, ::mlir::StringAttr syncscope) {
  odsState.addAttribute(getOrderingAttrName(odsState.name), ordering);
  odsState.addAttribute(getSyncscopeAttrName(odsState.name), syncscope);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::LLVM::AtomicOrdering ordering, ::llvm::StringRef syncscope) {
  odsState.addAttribute(getOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  odsState.addAttribute(getSyncscopeAttrName(odsState.name), odsBuilder.getStringAttr(syncscope));
}

void FenceOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::AtomicOrdering ordering, ::llvm::StringRef syncscope) {
  odsState.addAttribute(getOrderingAttrName(odsState.name), ::mlir::LLVM::AtomicOrderingAttr::get(odsBuilder.getContext(), ordering));
  odsState.addAttribute(getSyncscopeAttrName(odsState.name), odsBuilder.getStringAttr(syncscope));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FenceOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult FenceOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_ordering;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'ordering'");
    if (namedAttrIt->getName() == getOrderingAttrName()) {
      tblgen_ordering = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_syncscope;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'syncscope'");
    if (namedAttrIt->getName() == getSyncscopeAttrName()) {
      tblgen_syncscope = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps4(*this, tblgen_ordering, "ordering")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_syncscope, "syncscope")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult FenceOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FenceOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FreezeOp definitions
//===----------------------------------------------------------------------===//

FreezeOpAdaptor::FreezeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.freeze", odsAttrs.getContext());
}

FreezeOpAdaptor::FreezeOpAdaptor(FreezeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FreezeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FreezeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FreezeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FreezeOpAdaptor::getVal() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FreezeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FreezeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FreezeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FreezeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FreezeOp::getVal() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FreezeOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FreezeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FreezeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FreezeOp::getRes() {
  return *getODSResults(0).begin();
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value val) {
  odsState.addOperands(val);
  odsState.addTypes(res);
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value val) {
  odsState.addOperands(val);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FreezeOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value val) {
  odsState.addOperands(val);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FreezeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FreezeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FreezeOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FreezeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FreezeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FreezeOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult FreezeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valOperands(valRawOperands);  ::llvm::SMLoc valOperandsLoc;
  (void)valOperandsLoc;
  ::mlir::Type valRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valTypes(valRawTypes);

  valOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valRawTypes[0] = type;
  }
  result.addTypes(valTypes);
  if (parser.resolveOperands(valOperands, valTypes, valOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void FreezeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getVal();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getVal().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FreezeOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::GEPOp definitions
//===----------------------------------------------------------------------===//

GEPOpAdaptor::GEPOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.getelementptr", odsAttrs.getContext());
}

GEPOpAdaptor::GEPOpAdaptor(GEPOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GEPOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GEPOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange GEPOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GEPOpAdaptor::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange GEPOpAdaptor::getIndices() {
  return getODSOperands(1);
}

::mlir::DictionaryAttr GEPOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::DenseIntElementsAttr GEPOpAdaptor::getStructIndicesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, GEPOp::getStructIndicesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();
    return attr;
}

::mlir::DenseIntElementsAttr GEPOpAdaptor::getStructIndices() {
  auto attr = getStructIndicesAttr();
  return attr;
}

::mlir::TypeAttr GEPOpAdaptor::getElemTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GEPOp::getElemTypeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::TypeAttr>();
    return attr;
}

::llvm::Optional<::mlir::Type> GEPOpAdaptor::getElemType() {
  auto attr = getElemTypeAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

::mlir::LogicalResult GEPOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_structIndices;
  ::mlir::Attribute tblgen_elem_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.getelementptr' op ""requires attribute 'structIndices'");
    if (namedAttrIt->getName() == GEPOp::getStructIndicesAttrName(*odsOpName)) {
      tblgen_structIndices = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GEPOp::getElemTypeAttrName(*odsOpName)) {
      tblgen_elem_type = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_structIndices && !(((tblgen_structIndices.isa<::mlir::DenseIntElementsAttr>())) && ((tblgen_structIndices.cast<::mlir::DenseIntElementsAttr>().getType().getElementType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.getelementptr' op ""attribute 'structIndices' failed to satisfy constraint: 32-bit signless integer elements attribute");

  if (tblgen_elem_type && !(((tblgen_elem_type.isa<::mlir::TypeAttr>())) && ((tblgen_elem_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return emitError(loc, "'llvm.getelementptr' op ""attribute 'elem_type' failed to satisfy constraint: any type attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GEPOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range GEPOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GEPOp::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range GEPOp::getIndices() {
  return getODSOperands(1);
}

::mlir::MutableOperandRange GEPOp::getBaseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GEPOp::getIndicesMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GEPOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GEPOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GEPOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::DenseIntElementsAttr GEPOp::getStructIndicesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getStructIndicesAttrName()).cast<::mlir::DenseIntElementsAttr>();
}

::mlir::DenseIntElementsAttr GEPOp::getStructIndices() {
  auto attr = getStructIndicesAttr();
  return attr;
}

::mlir::TypeAttr GEPOp::getElemTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getElemTypeAttrName()).dyn_cast_or_null<::mlir::TypeAttr>();
}

::llvm::Optional<::mlir::Type> GEPOp::getElemType() {
  auto attr = getElemTypeAttr();
  return attr ? ::llvm::Optional<::mlir::Type>(attr.getValue().cast<::mlir::Type>()) : (::llvm::None);
}

void GEPOp::setStructIndicesAttr(::mlir::DenseIntElementsAttr attr) {
  (*this)->setAttr(getStructIndicesAttrName(), attr);
}

void GEPOp::setElemTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getElemTypeAttrName(), attr);
}

::mlir::Attribute GEPOp::removeElem_typeAttr() {
  return (*this)->removeAttr(getElemTypeAttrName());
}

::mlir::LogicalResult GEPOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_structIndices;
  ::mlir::Attribute tblgen_elem_type;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'structIndices'");
    if (namedAttrIt->getName() == getStructIndicesAttrName()) {
      tblgen_structIndices = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getElemTypeAttrName()) {
      tblgen_elem_type = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps11(*this, tblgen_structIndices, "structIndices")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps3(*this, tblgen_elem_type, "elem_type")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GEPOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GEPOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> indicesOperands;
  ::llvm::SMLoc indicesOperandsLoc;
  (void)indicesOperandsLoc;
  ::mlir::DenseIntElementsAttr structIndicesAttr;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;
  ::mlir::TypeAttr elem_typeAttr;

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseLSquare())
    return ::mlir::failure();
  {
    indicesOperandsLoc = parser.getCurrentLocation();
    if (parseGEPIndices(parser, indicesOperands, structIndicesAttr))
      return ::mlir::failure();
    result.addAttribute("structIndices", structIndicesAttr);
  }
  if (parser.parseRSquare())
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
  if (succeeded(parser.parseOptionalComma())) {

  if (parser.parseCustomAttributeWithFallback(elem_typeAttr, parser.getBuilder().getType<::mlir::NoneType>(), "elem_type",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(::llvm::concat<const ::mlir::OpAsmParser::UnresolvedOperand>(baseOperands, indicesOperands), allOperandTypes, parser.getNameLoc(), result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GEPOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBase();
  _odsPrinter << "[";
  printGEPIndices(_odsPrinter, *this, getIndices(), getStructIndicesAttr());
  _odsPrinter << "]";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"structIndices", "elem_type"});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
  if ((*this)->getAttr("elem_type")) {
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getElemTypeAttr());
  }
}

void GEPOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::GEPOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::GlobalCtorsOp definitions
//===----------------------------------------------------------------------===//

GlobalCtorsOpAdaptor::GlobalCtorsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.global_ctors", odsAttrs.getContext());
}

GlobalCtorsOpAdaptor::GlobalCtorsOpAdaptor(GlobalCtorsOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GlobalCtorsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GlobalCtorsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GlobalCtorsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GlobalCtorsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr GlobalCtorsOpAdaptor::getCtorsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GlobalCtorsOp::getCtorsAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
    return attr;
}

::mlir::ArrayAttr GlobalCtorsOpAdaptor::getCtors() {
  auto attr = getCtorsAttr();
  return attr;
}

::mlir::ArrayAttr GlobalCtorsOpAdaptor::getPrioritiesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, GlobalCtorsOp::getPrioritiesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
    return attr;
}

::mlir::ArrayAttr GlobalCtorsOpAdaptor::getPriorities() {
  auto attr = getPrioritiesAttr();
  return attr;
}

::mlir::LogicalResult GlobalCtorsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_ctors;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global_ctors' op ""requires attribute 'ctors'");
    if (namedAttrIt->getName() == GlobalCtorsOp::getCtorsAttrName(*odsOpName)) {
      tblgen_ctors = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_priorities;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global_ctors' op ""requires attribute 'priorities'");
    if (namedAttrIt->getName() == GlobalCtorsOp::getPrioritiesAttrName(*odsOpName)) {
      tblgen_priorities = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_ctors && !(((tblgen_ctors.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_ctors.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::FlatSymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.mlir.global_ctors' op ""attribute 'ctors' failed to satisfy constraint: flat symbol ref array attribute");

  if (tblgen_priorities && !(((tblgen_priorities.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_priorities.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))); }))))
    return emitError(loc, "'llvm.mlir.global_ctors' op ""attribute 'priorities' failed to satisfy constraint: 32-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GlobalCtorsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GlobalCtorsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GlobalCtorsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GlobalCtorsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr GlobalCtorsOp::getCtorsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCtorsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GlobalCtorsOp::getCtors() {
  auto attr = getCtorsAttr();
  return attr;
}

::mlir::ArrayAttr GlobalCtorsOp::getPrioritiesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getPrioritiesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GlobalCtorsOp::getPriorities() {
  auto attr = getPrioritiesAttr();
  return attr;
}

void GlobalCtorsOp::setCtorsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getCtorsAttrName(), attr);
}

void GlobalCtorsOp::setPrioritiesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPrioritiesAttrName(), attr);
}

void GlobalCtorsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ArrayAttr ctors, ::mlir::ArrayAttr priorities) {
  odsState.addAttribute(getCtorsAttrName(odsState.name), ctors);
  odsState.addAttribute(getPrioritiesAttrName(odsState.name), priorities);
}

void GlobalCtorsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ArrayAttr ctors, ::mlir::ArrayAttr priorities) {
  odsState.addAttribute(getCtorsAttrName(odsState.name), ctors);
  odsState.addAttribute(getPrioritiesAttrName(odsState.name), priorities);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalCtorsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GlobalCtorsOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_ctors;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'ctors'");
    if (namedAttrIt->getName() == getCtorsAttrName()) {
      tblgen_ctors = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_priorities;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'priorities'");
    if (namedAttrIt->getName() == getPrioritiesAttrName()) {
      tblgen_priorities = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps12(*this, tblgen_ctors, "ctors")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps13(*this, tblgen_priorities, "priorities")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult GlobalCtorsOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GlobalCtorsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void GlobalCtorsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::GlobalCtorsOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::GlobalDtorsOp definitions
//===----------------------------------------------------------------------===//

GlobalDtorsOpAdaptor::GlobalDtorsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.global_dtors", odsAttrs.getContext());
}

GlobalDtorsOpAdaptor::GlobalDtorsOpAdaptor(GlobalDtorsOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GlobalDtorsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GlobalDtorsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GlobalDtorsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr GlobalDtorsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr GlobalDtorsOpAdaptor::getDtorsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, GlobalDtorsOp::getDtorsAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
    return attr;
}

::mlir::ArrayAttr GlobalDtorsOpAdaptor::getDtors() {
  auto attr = getDtorsAttr();
  return attr;
}

::mlir::ArrayAttr GlobalDtorsOpAdaptor::getPrioritiesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, GlobalDtorsOp::getPrioritiesAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
    return attr;
}

::mlir::ArrayAttr GlobalDtorsOpAdaptor::getPriorities() {
  auto attr = getPrioritiesAttr();
  return attr;
}

::mlir::LogicalResult GlobalDtorsOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dtors;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global_dtors' op ""requires attribute 'dtors'");
    if (namedAttrIt->getName() == GlobalDtorsOp::getDtorsAttrName(*odsOpName)) {
      tblgen_dtors = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_priorities;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global_dtors' op ""requires attribute 'priorities'");
    if (namedAttrIt->getName() == GlobalDtorsOp::getPrioritiesAttrName(*odsOpName)) {
      tblgen_priorities = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_dtors && !(((tblgen_dtors.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_dtors.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::FlatSymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.mlir.global_dtors' op ""attribute 'dtors' failed to satisfy constraint: flat symbol ref array attribute");

  if (tblgen_priorities && !(((tblgen_priorities.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_priorities.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && (((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))); }))))
    return emitError(loc, "'llvm.mlir.global_dtors' op ""attribute 'priorities' failed to satisfy constraint: 32-bit integer array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GlobalDtorsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GlobalDtorsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> GlobalDtorsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GlobalDtorsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr GlobalDtorsOp::getDtorsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDtorsAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GlobalDtorsOp::getDtors() {
  auto attr = getDtorsAttr();
  return attr;
}

::mlir::ArrayAttr GlobalDtorsOp::getPrioritiesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getPrioritiesAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr GlobalDtorsOp::getPriorities() {
  auto attr = getPrioritiesAttr();
  return attr;
}

void GlobalDtorsOp::setDtorsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getDtorsAttrName(), attr);
}

void GlobalDtorsOp::setPrioritiesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPrioritiesAttrName(), attr);
}

void GlobalDtorsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ArrayAttr dtors, ::mlir::ArrayAttr priorities) {
  odsState.addAttribute(getDtorsAttrName(odsState.name), dtors);
  odsState.addAttribute(getPrioritiesAttrName(odsState.name), priorities);
}

void GlobalDtorsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ArrayAttr dtors, ::mlir::ArrayAttr priorities) {
  odsState.addAttribute(getDtorsAttrName(odsState.name), dtors);
  odsState.addAttribute(getPrioritiesAttrName(odsState.name), priorities);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalDtorsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GlobalDtorsOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_dtors;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'dtors'");
    if (namedAttrIt->getName() == getDtorsAttrName()) {
      tblgen_dtors = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_priorities;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'priorities'");
    if (namedAttrIt->getName() == getPrioritiesAttrName()) {
      tblgen_priorities = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps12(*this, tblgen_dtors, "dtors")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps13(*this, tblgen_priorities, "priorities")))
    return ::mlir::failure();
  return ::mlir::success();
}

::mlir::LogicalResult GlobalDtorsOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult GlobalDtorsOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void GlobalDtorsOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::GlobalDtorsOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::GlobalOp definitions
//===----------------------------------------------------------------------===//

GlobalOpAdaptor::GlobalOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.global", odsAttrs.getContext());
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

::mlir::TypeAttr GlobalOpAdaptor::getGlobalTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, GlobalOp::getGlobalTypeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
    return attr;
}

::mlir::Type GlobalOpAdaptor::getGlobalType() {
  auto attr = getGlobalTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::UnitAttr GlobalOpAdaptor::getConstantAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, GlobalOp::getConstantAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool GlobalOpAdaptor::getConstant() {
  auto attr = getConstantAttr();
  return attr != nullptr;
}

::mlir::StringAttr GlobalOpAdaptor::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, GlobalOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef GlobalOpAdaptor::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::LLVM::LinkageAttr GlobalOpAdaptor::getLinkageAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, GlobalOp::getLinkageAttrName(*odsOpName)).cast<::mlir::LLVM::LinkageAttr>();
    return attr;
}

::mlir::LLVM::Linkage GlobalOpAdaptor::getLinkage() {
  auto attr = getLinkageAttr();
  return attr.getLinkage();
}

::mlir::UnitAttr GlobalOpAdaptor::getDsoLocalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, GlobalOp::getDsoLocalAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool GlobalOpAdaptor::getDsoLocal() {
  auto attr = getDsoLocalAttr();
  return attr != nullptr;
}

::mlir::UnitAttr GlobalOpAdaptor::getThreadLocal_Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, GlobalOp::getThreadLocal_AttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool GlobalOpAdaptor::getThreadLocal_() {
  auto attr = getThreadLocal_Attr();
  return attr != nullptr;
}

::mlir::Attribute GlobalOpAdaptor::getValueAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, GlobalOp::getValueAttrName(*odsOpName)).dyn_cast_or_null<::mlir::Attribute>();
    return attr;
}

::llvm::Optional<::mlir::Attribute> GlobalOpAdaptor::getValue() {
  auto attr = getValueAttr();
  return attr ? ::llvm::Optional<::mlir::Attribute>(attr) : (::llvm::None);
}

::mlir::IntegerAttr GlobalOpAdaptor::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, GlobalOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    return attr;
}

::llvm::Optional<uint64_t> GlobalOpAdaptor::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::IntegerAttr GlobalOpAdaptor::getAddrSpaceAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 3, GlobalOp::getAddrSpaceAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(32), 0);
    return attr;
}

uint32_t GlobalOpAdaptor::getAddrSpace() {
  auto attr = getAddrSpaceAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getIntegerAttr(::mlir::Builder(odsAttrs.getContext()).getIntegerType(32), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::LLVM::UnnamedAddrAttr GlobalOpAdaptor::getUnnamedAddrAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 3, odsAttrs.end() - 0, GlobalOp::getUnnamedAddrAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::UnnamedAddrAttr>();
    return attr;
}

::llvm::Optional<::mlir::LLVM::UnnamedAddr> GlobalOpAdaptor::getUnnamedAddr() {
  auto attr = getUnnamedAddrAttr();
  return attr ? ::llvm::Optional<::mlir::LLVM::UnnamedAddr>(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr GlobalOpAdaptor::getSectionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 1, GlobalOp::getSectionAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > GlobalOpAdaptor::getSection() {
  auto attr = getSectionAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::RegionRange GlobalOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &GlobalOpAdaptor::getInitializer() {
  return *odsRegions[0];
}

::mlir::LogicalResult GlobalOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global_type;
  ::mlir::Attribute tblgen_addr_space;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_constant;
  ::mlir::Attribute tblgen_dso_local;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global' op ""requires attribute 'global_type'");
    if (namedAttrIt->getName() == GlobalOp::getGlobalTypeAttrName(*odsOpName)) {
      tblgen_global_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GlobalOp::getAddrSpaceAttrName(*odsOpName)) {
      tblgen_addr_space = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GlobalOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GlobalOp::getConstantAttrName(*odsOpName)) {
      tblgen_constant = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GlobalOp::getDsoLocalAttrName(*odsOpName)) {
      tblgen_dso_local = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_linkage;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global' op ""requires attribute 'linkage'");
    if (namedAttrIt->getName() == GlobalOp::getLinkageAttrName(*odsOpName)) {
      tblgen_linkage = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_section;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.mlir.global' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == GlobalOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == GlobalOp::getSectionAttrName(*odsOpName)) {
      tblgen_section = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_thread_local_;
  ::mlir::Attribute tblgen_unnamed_addr;
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == GlobalOp::getThreadLocal_AttrName(*odsOpName)) {
      tblgen_thread_local_ = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GlobalOp::getUnnamedAddrAttrName(*odsOpName)) {
      tblgen_unnamed_addr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == GlobalOp::getValueAttrName(*odsOpName)) {
      tblgen_value = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_global_type && !(((tblgen_global_type.isa<::mlir::TypeAttr>())) && ((tblgen_global_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::Type>()))))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'global_type' failed to satisfy constraint: any type attribute");

  if (tblgen_constant && !((tblgen_constant.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'constant' failed to satisfy constraint: unit attribute");

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_linkage && !((tblgen_linkage.isa<::mlir::LLVM::LinkageAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'linkage' failed to satisfy constraint: LLVM Linkage specification");

  if (tblgen_dso_local && !((tblgen_dso_local.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'dso_local' failed to satisfy constraint: unit attribute");

  if (tblgen_thread_local_ && !((tblgen_thread_local_.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'thread_local_' failed to satisfy constraint: unit attribute");

  if (tblgen_value && !((true)))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'value' failed to satisfy constraint: any attribute");

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_addr_space && !((((tblgen_addr_space.isa<::mlir::IntegerAttr>())) && ((tblgen_addr_space.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))) && ((!tblgen_addr_space.cast<::mlir::IntegerAttr>().getValue().isNegative()))))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'addr_space' failed to satisfy constraint: 32-bit signless integer attribute whose value is non-negative");

  if (tblgen_unnamed_addr && !((tblgen_unnamed_addr.isa<::mlir::LLVM::UnnamedAddrAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'unnamed_addr' failed to satisfy constraint: LLVM GlobalValue UnnamedAddr");

  if (tblgen_section && !((tblgen_section.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.mlir.global' op ""attribute 'section' failed to satisfy constraint: string attribute");
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

::mlir::Region &GlobalOp::getInitializer() {
  return (*this)->getRegion(0);
}

::mlir::TypeAttr GlobalOp::getGlobalTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getGlobalTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::Type GlobalOp::getGlobalType() {
  auto attr = getGlobalTypeAttr();
  return attr.getValue().cast<::mlir::Type>();
}

::mlir::UnitAttr GlobalOp::getConstantAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getConstantAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool GlobalOp::getConstant() {
  auto attr = getConstantAttr();
  return attr != nullptr;
}

::mlir::StringAttr GlobalOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef GlobalOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::LLVM::LinkageAttr GlobalOp::getLinkageAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getLinkageAttrName()).cast<::mlir::LLVM::LinkageAttr>();
}

::mlir::LLVM::Linkage GlobalOp::getLinkage() {
  auto attr = getLinkageAttr();
  return attr.getLinkage();
}

::mlir::UnitAttr GlobalOp::getDsoLocalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getDsoLocalAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool GlobalOp::getDsoLocal() {
  auto attr = getDsoLocalAttr();
  return attr != nullptr;
}

::mlir::UnitAttr GlobalOp::getThreadLocal_Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getThreadLocal_AttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool GlobalOp::getThreadLocal_() {
  auto attr = getThreadLocal_Attr();
  return attr != nullptr;
}

::mlir::Attribute GlobalOp::getValueAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getValueAttrName()).dyn_cast_or_null<::mlir::Attribute>();
}

::llvm::Optional<::mlir::Attribute> GlobalOp::getValue() {
  auto attr = getValueAttr();
  return attr ? ::llvm::Optional<::mlir::Attribute>(attr) : (::llvm::None);
}

::mlir::IntegerAttr GlobalOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> GlobalOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::IntegerAttr GlobalOp::getAddrSpaceAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 3, getAddrSpaceAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

uint32_t GlobalOp::getAddrSpace() {
  auto attr = getAddrSpaceAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getIntegerAttr(::mlir::Builder((*this)->getContext()).getIntegerType(32), 0).getValue().getZExtValue();
  return attr.getValue().getZExtValue();
}

::mlir::LLVM::UnnamedAddrAttr GlobalOp::getUnnamedAddrAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 3, (*this)->getAttrs().end() - 0, getUnnamedAddrAttrName()).dyn_cast_or_null<::mlir::LLVM::UnnamedAddrAttr>();
}

::llvm::Optional<::mlir::LLVM::UnnamedAddr> GlobalOp::getUnnamedAddr() {
  auto attr = getUnnamedAddrAttr();
  return attr ? ::llvm::Optional<::mlir::LLVM::UnnamedAddr>(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr GlobalOp::getSectionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 1, getSectionAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > GlobalOp::getSection() {
  auto attr = getSectionAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void GlobalOp::setGlobalTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getGlobalTypeAttrName(), attr);
}

void GlobalOp::setConstantAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getConstantAttrName(), attr);
}

void GlobalOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void GlobalOp::setLinkageAttr(::mlir::LLVM::LinkageAttr attr) {
  (*this)->setAttr(getLinkageAttrName(), attr);
}

void GlobalOp::setDsoLocalAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getDsoLocalAttrName(), attr);
}

void GlobalOp::setThreadLocal_Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getThreadLocal_AttrName(), attr);
}

void GlobalOp::setValueAttr(::mlir::Attribute attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void GlobalOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void GlobalOp::setAddrSpaceAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAddrSpaceAttrName(), attr);
}

void GlobalOp::setUnnamedAddrAttr(::mlir::LLVM::UnnamedAddrAttr attr) {
  (*this)->setAttr(getUnnamedAddrAttrName(), attr);
}

void GlobalOp::setSectionAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSectionAttrName(), attr);
}

::mlir::Attribute GlobalOp::removeConstantAttr() {
  return (*this)->removeAttr(getConstantAttrName());
}

::mlir::Attribute GlobalOp::removeDso_localAttr() {
  return (*this)->removeAttr(getDsoLocalAttrName());
}

::mlir::Attribute GlobalOp::removeThread_local_Attr() {
  return (*this)->removeAttr(getThreadLocal_AttrName());
}

::mlir::Attribute GlobalOp::removeValueAttr() {
  return (*this)->removeAttr(getValueAttrName());
}

::mlir::Attribute GlobalOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute GlobalOp::removeUnnamed_addrAttr() {
  return (*this)->removeAttr(getUnnamedAddrAttrName());
}

::mlir::Attribute GlobalOp::removeSectionAttr() {
  return (*this)->removeAttr(getSectionAttrName());
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeAttr global_type, /*optional*/::mlir::UnitAttr constant, ::mlir::StringAttr sym_name, ::mlir::LLVM::LinkageAttr linkage, /*optional*/::mlir::UnitAttr dso_local, /*optional*/::mlir::UnitAttr thread_local_, /*optional*/::mlir::Attribute value, /*optional*/::mlir::IntegerAttr alignment, ::mlir::IntegerAttr addr_space, /*optional*/::mlir::LLVM::UnnamedAddrAttr unnamed_addr, /*optional*/::mlir::StringAttr section) {
  odsState.addAttribute(getGlobalTypeAttrName(odsState.name), global_type);
  if (constant) {
  odsState.addAttribute(getConstantAttrName(odsState.name), constant);
  }
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getLinkageAttrName(odsState.name), linkage);
  if (dso_local) {
  odsState.addAttribute(getDsoLocalAttrName(odsState.name), dso_local);
  }
  if (thread_local_) {
  odsState.addAttribute(getThreadLocal_AttrName(odsState.name), thread_local_);
  }
  if (value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (addr_space) {
  odsState.addAttribute(getAddrSpaceAttrName(odsState.name), addr_space);
  }
  if (unnamed_addr) {
  odsState.addAttribute(getUnnamedAddrAttrName(odsState.name), unnamed_addr);
  }
  if (section) {
  odsState.addAttribute(getSectionAttrName(odsState.name), section);
  }
  (void)odsState.addRegion();
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::TypeAttr global_type, /*optional*/::mlir::UnitAttr constant, ::mlir::StringAttr sym_name, ::mlir::LLVM::LinkageAttr linkage, /*optional*/::mlir::UnitAttr dso_local, /*optional*/::mlir::UnitAttr thread_local_, /*optional*/::mlir::Attribute value, /*optional*/::mlir::IntegerAttr alignment, ::mlir::IntegerAttr addr_space, /*optional*/::mlir::LLVM::UnnamedAddrAttr unnamed_addr, /*optional*/::mlir::StringAttr section) {
  odsState.addAttribute(getGlobalTypeAttrName(odsState.name), global_type);
  if (constant) {
  odsState.addAttribute(getConstantAttrName(odsState.name), constant);
  }
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  odsState.addAttribute(getLinkageAttrName(odsState.name), linkage);
  if (dso_local) {
  odsState.addAttribute(getDsoLocalAttrName(odsState.name), dso_local);
  }
  if (thread_local_) {
  odsState.addAttribute(getThreadLocal_AttrName(odsState.name), thread_local_);
  }
  if (value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (addr_space) {
  odsState.addAttribute(getAddrSpaceAttrName(odsState.name), addr_space);
  }
  if (unnamed_addr) {
  odsState.addAttribute(getUnnamedAddrAttrName(odsState.name), unnamed_addr);
  }
  if (section) {
  odsState.addAttribute(getSectionAttrName(odsState.name), section);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type global_type, /*optional*/bool constant, ::llvm::StringRef sym_name, ::mlir::LLVM::Linkage linkage, /*optional*/bool dso_local, /*optional*/bool thread_local_, /*optional*/::mlir::Attribute value, /*optional*/::mlir::IntegerAttr alignment, uint32_t addr_space, /*optional*/::mlir::LLVM::UnnamedAddrAttr unnamed_addr, /*optional*/::mlir::StringAttr section) {
  odsState.addAttribute(getGlobalTypeAttrName(odsState.name), ::mlir::TypeAttr::get(global_type));
  if (constant) {
  odsState.addAttribute(getConstantAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getLinkageAttrName(odsState.name), ::mlir::LLVM::LinkageAttr::get(odsBuilder.getContext(), linkage));
  if (dso_local) {
  odsState.addAttribute(getDsoLocalAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (thread_local_) {
  odsState.addAttribute(getThreadLocal_AttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  odsState.addAttribute(getAddrSpaceAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), addr_space));
  if (unnamed_addr) {
  odsState.addAttribute(getUnnamedAddrAttrName(odsState.name), unnamed_addr);
  }
  if (section) {
  odsState.addAttribute(getSectionAttrName(odsState.name), section);
  }
  (void)odsState.addRegion();
}

void GlobalOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Type global_type, /*optional*/bool constant, ::llvm::StringRef sym_name, ::mlir::LLVM::Linkage linkage, /*optional*/bool dso_local, /*optional*/bool thread_local_, /*optional*/::mlir::Attribute value, /*optional*/::mlir::IntegerAttr alignment, uint32_t addr_space, /*optional*/::mlir::LLVM::UnnamedAddrAttr unnamed_addr, /*optional*/::mlir::StringAttr section) {
  odsState.addAttribute(getGlobalTypeAttrName(odsState.name), ::mlir::TypeAttr::get(global_type));
  if (constant) {
  odsState.addAttribute(getConstantAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  odsState.addAttribute(getLinkageAttrName(odsState.name), ::mlir::LLVM::LinkageAttr::get(odsBuilder.getContext(), linkage));
  if (dso_local) {
  odsState.addAttribute(getDsoLocalAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (thread_local_) {
  odsState.addAttribute(getThreadLocal_AttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (value) {
  odsState.addAttribute(getValueAttrName(odsState.name), value);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  odsState.addAttribute(getAddrSpaceAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), addr_space));
  if (unnamed_addr) {
  odsState.addAttribute(getUnnamedAddrAttrName(odsState.name), unnamed_addr);
  }
  if (section) {
  odsState.addAttribute(getSectionAttrName(odsState.name), section);
  }
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GlobalOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GlobalOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_global_type;
  ::mlir::Attribute tblgen_addr_space;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_constant;
  ::mlir::Attribute tblgen_dso_local;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'global_type'");
    if (namedAttrIt->getName() == getGlobalTypeAttrName()) {
      tblgen_global_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAddrSpaceAttrName()) {
      tblgen_addr_space = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getConstantAttrName()) {
      tblgen_constant = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getDsoLocalAttrName()) {
      tblgen_dso_local = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_linkage;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'linkage'");
    if (namedAttrIt->getName() == getLinkageAttrName()) {
      tblgen_linkage = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_sym_name;
  ::mlir::Attribute tblgen_section;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'sym_name'");
    if (namedAttrIt->getName() == getSymNameAttrName()) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getSectionAttrName()) {
      tblgen_section = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_thread_local_;
  ::mlir::Attribute tblgen_unnamed_addr;
  ::mlir::Attribute tblgen_value;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getThreadLocal_AttrName()) {
      tblgen_thread_local_ = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getUnnamedAddrAttrName()) {
      tblgen_unnamed_addr = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getValueAttrName()) {
      tblgen_value = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps3(*this, tblgen_global_type, "global_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_constant, "constant")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps15(*this, tblgen_linkage, "linkage")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_dso_local, "dso_local")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_thread_local_, "thread_local_")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps8(*this, tblgen_value, "value")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps2(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps16(*this, tblgen_addr_space, "addr_space")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps17(*this, tblgen_unnamed_addr, "unnamed_addr")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_section, "section")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LLVMOps0(*this, region, "initializer", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult GlobalOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::GlobalOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ICmpOp definitions
//===----------------------------------------------------------------------===//

ICmpOpAdaptor::ICmpOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.icmp", odsAttrs.getContext());
}

ICmpOpAdaptor::ICmpOpAdaptor(ICmpOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ICmpOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ICmpOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ICmpOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ICmpOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ICmpOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ICmpOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LLVM::ICmpPredicateAttr ICmpOpAdaptor::getPredicateAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ICmpOp::getPredicateAttrName(*odsOpName)).cast<::mlir::LLVM::ICmpPredicateAttr>();
    return attr;
}

::mlir::LLVM::ICmpPredicate ICmpOpAdaptor::getPredicate() {
  auto attr = getPredicateAttr();
  return attr.getValue();
}

::mlir::LogicalResult ICmpOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_predicate;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.icmp' op ""requires attribute 'predicate'");
    if (namedAttrIt->getName() == ICmpOp::getPredicateAttrName(*odsOpName)) {
      tblgen_predicate = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_predicate && !((tblgen_predicate.isa<::mlir::LLVM::ICmpPredicateAttr>())))
    return emitError(loc, "'llvm.icmp' op ""attribute 'predicate' failed to satisfy constraint: llvm.icmp comparison predicate");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ICmpOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ICmpOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ICmpOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ICmpOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ICmpOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ICmpOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ICmpOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ICmpOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ICmpOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::LLVM::ICmpPredicateAttr ICmpOp::getPredicateAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPredicateAttrName()).cast<::mlir::LLVM::ICmpPredicateAttr>();
}

::mlir::LLVM::ICmpPredicate ICmpOp::getPredicate() {
  auto attr = getPredicateAttr();
  return attr.getValue();
}

void ICmpOp::setPredicateAttr(::mlir::LLVM::ICmpPredicateAttr attr) {
  (*this)->setAttr(getPredicateAttrName(), attr);
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ICmpPredicate predicate, Value lhs, Value rhs) {
      build(odsBuilder, odsState, IntegerType::get(lhs.getType().getContext(), 1),
            predicate, lhs, rhs);
    
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::ICmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), predicate);
  odsState.addTypes(res);
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::ICmpPredicateAttr predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), predicate);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::LLVM::ICmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), ::mlir::LLVM::ICmpPredicateAttr::get(odsBuilder.getContext(), predicate));
  odsState.addTypes(res);
}

void ICmpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::LLVM::ICmpPredicate predicate, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getPredicateAttrName(odsState.name), ::mlir::LLVM::ICmpPredicateAttr::get(odsBuilder.getContext(), predicate));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ICmpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ICmpOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_predicate;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'predicate'");
    if (namedAttrIt->getName() == getPredicateAttrName()) {
      tblgen_predicate = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps18(*this, tblgen_predicate, "predicate")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps16(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps16(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps15(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ICmpOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void ICmpOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ICmpOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::InlineAsmOp definitions
//===----------------------------------------------------------------------===//

InlineAsmOpAdaptor::InlineAsmOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.inline_asm", odsAttrs.getContext());
}

InlineAsmOpAdaptor::InlineAsmOpAdaptor(InlineAsmOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange InlineAsmOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InlineAsmOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange InlineAsmOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange InlineAsmOpAdaptor::operands() {
  return getODSOperands(0);
}

::mlir::DictionaryAttr InlineAsmOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr InlineAsmOpAdaptor::getAsmStringAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, InlineAsmOp::getAsmStringAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef InlineAsmOpAdaptor::getAsmString() {
  auto attr = getAsmStringAttr();
  return attr.getValue();
}

::mlir::StringAttr InlineAsmOpAdaptor::getConstraintsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, InlineAsmOp::getConstraintsAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef InlineAsmOpAdaptor::getConstraints() {
  auto attr = getConstraintsAttr();
  return attr.getValue();
}

::mlir::UnitAttr InlineAsmOpAdaptor::getHasSideEffectsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, InlineAsmOp::getHasSideEffectsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool InlineAsmOpAdaptor::getHasSideEffects() {
  auto attr = getHasSideEffectsAttr();
  return attr != nullptr;
}

::mlir::UnitAttr InlineAsmOpAdaptor::getIsAlignStackAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, InlineAsmOp::getIsAlignStackAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool InlineAsmOpAdaptor::getIsAlignStack() {
  auto attr = getIsAlignStackAttr();
  return attr != nullptr;
}

::mlir::LLVM::AsmDialectAttr InlineAsmOpAdaptor::getAsmDialectAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, InlineAsmOp::getAsmDialectAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::AsmDialectAttr>();
    return attr;
}

::llvm::Optional<::mlir::LLVM::AsmDialect> InlineAsmOpAdaptor::getAsmDialect() {
  auto attr = getAsmDialectAttr();
  return attr ? ::llvm::Optional<::mlir::LLVM::AsmDialect>(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr InlineAsmOpAdaptor::getOperandAttrsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, InlineAsmOp::getOperandAttrsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > InlineAsmOpAdaptor::getOperandAttrs() {
  auto attr = getOperandAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult InlineAsmOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_asm_string;
  ::mlir::Attribute tblgen_asm_dialect;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.inline_asm' op ""requires attribute 'asm_string'");
    if (namedAttrIt->getName() == InlineAsmOp::getAsmStringAttrName(*odsOpName)) {
      tblgen_asm_string = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == InlineAsmOp::getAsmDialectAttrName(*odsOpName)) {
      tblgen_asm_dialect = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_constraints;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.inline_asm' op ""requires attribute 'constraints'");
    if (namedAttrIt->getName() == InlineAsmOp::getConstraintsAttrName(*odsOpName)) {
      tblgen_constraints = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_has_side_effects;
  ::mlir::Attribute tblgen_is_align_stack;
  ::mlir::Attribute tblgen_operand_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == InlineAsmOp::getHasSideEffectsAttrName(*odsOpName)) {
      tblgen_has_side_effects = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == InlineAsmOp::getIsAlignStackAttrName(*odsOpName)) {
      tblgen_is_align_stack = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == InlineAsmOp::getOperandAttrsAttrName(*odsOpName)) {
      tblgen_operand_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_asm_string && !((tblgen_asm_string.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.inline_asm' op ""attribute 'asm_string' failed to satisfy constraint: string attribute");

  if (tblgen_constraints && !((tblgen_constraints.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.inline_asm' op ""attribute 'constraints' failed to satisfy constraint: string attribute");

  if (tblgen_has_side_effects && !((tblgen_has_side_effects.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.inline_asm' op ""attribute 'has_side_effects' failed to satisfy constraint: unit attribute");

  if (tblgen_is_align_stack && !((tblgen_is_align_stack.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.inline_asm' op ""attribute 'is_align_stack' failed to satisfy constraint: unit attribute");

  if (tblgen_asm_dialect && !((tblgen_asm_dialect.isa<::mlir::LLVM::AsmDialectAttr>())))
    return emitError(loc, "'llvm.inline_asm' op ""attribute 'asm_dialect' failed to satisfy constraint: ATT (0) or Intel (1) asm dialect");

  if (tblgen_operand_attrs && !((tblgen_operand_attrs.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'llvm.inline_asm' op ""attribute 'operand_attrs' failed to satisfy constraint: array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InlineAsmOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range InlineAsmOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range InlineAsmOp::operands() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange InlineAsmOp::operandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> InlineAsmOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range InlineAsmOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InlineAsmOp::getRes() {
  auto results = getODSResults(0);
  return results.empty() ? ::mlir::Value() : *results.begin();
}

::mlir::StringAttr InlineAsmOp::getAsmStringAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getAsmStringAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef InlineAsmOp::getAsmString() {
  auto attr = getAsmStringAttr();
  return attr.getValue();
}

::mlir::StringAttr InlineAsmOp::getConstraintsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getConstraintsAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef InlineAsmOp::getConstraints() {
  auto attr = getConstraintsAttr();
  return attr.getValue();
}

::mlir::UnitAttr InlineAsmOp::getHasSideEffectsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getHasSideEffectsAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool InlineAsmOp::getHasSideEffects() {
  auto attr = getHasSideEffectsAttr();
  return attr != nullptr;
}

::mlir::UnitAttr InlineAsmOp::getIsAlignStackAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getIsAlignStackAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool InlineAsmOp::getIsAlignStack() {
  auto attr = getIsAlignStackAttr();
  return attr != nullptr;
}

::mlir::LLVM::AsmDialectAttr InlineAsmOp::getAsmDialectAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getAsmDialectAttrName()).dyn_cast_or_null<::mlir::LLVM::AsmDialectAttr>();
}

::llvm::Optional<::mlir::LLVM::AsmDialect> InlineAsmOp::getAsmDialect() {
  auto attr = getAsmDialectAttr();
  return attr ? ::llvm::Optional<::mlir::LLVM::AsmDialect>(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr InlineAsmOp::getOperandAttrsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getOperandAttrsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > InlineAsmOp::getOperandAttrs() {
  auto attr = getOperandAttrsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void InlineAsmOp::setAsmStringAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getAsmStringAttrName(), attr);
}

void InlineAsmOp::setConstraintsAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getConstraintsAttrName(), attr);
}

void InlineAsmOp::setHasSideEffectsAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getHasSideEffectsAttrName(), attr);
}

void InlineAsmOp::setIsAlignStackAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getIsAlignStackAttrName(), attr);
}

void InlineAsmOp::setAsmDialectAttr(::mlir::LLVM::AsmDialectAttr attr) {
  (*this)->setAttr(getAsmDialectAttrName(), attr);
}

void InlineAsmOp::setOperandAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getOperandAttrsAttrName(), attr);
}

::mlir::Attribute InlineAsmOp::removeHas_side_effectsAttr() {
  return (*this)->removeAttr(getHasSideEffectsAttrName());
}

::mlir::Attribute InlineAsmOp::removeIs_align_stackAttr() {
  return (*this)->removeAttr(getIsAlignStackAttrName());
}

::mlir::Attribute InlineAsmOp::removeAsm_dialectAttr() {
  return (*this)->removeAttr(getAsmDialectAttrName());
}

::mlir::Attribute InlineAsmOp::removeOperand_attrsAttr() {
  return (*this)->removeAttr(getOperandAttrsAttrName());
}

void InlineAsmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type res, ::mlir::ValueRange operands, ::mlir::StringAttr asm_string, ::mlir::StringAttr constraints, /*optional*/::mlir::UnitAttr has_side_effects, /*optional*/::mlir::UnitAttr is_align_stack, /*optional*/::mlir::LLVM::AsmDialectAttr asm_dialect, /*optional*/::mlir::ArrayAttr operand_attrs) {
  odsState.addOperands(operands);
  odsState.addAttribute(getAsmStringAttrName(odsState.name), asm_string);
  odsState.addAttribute(getConstraintsAttrName(odsState.name), constraints);
  if (has_side_effects) {
  odsState.addAttribute(getHasSideEffectsAttrName(odsState.name), has_side_effects);
  }
  if (is_align_stack) {
  odsState.addAttribute(getIsAlignStackAttrName(odsState.name), is_align_stack);
  }
  if (asm_dialect) {
  odsState.addAttribute(getAsmDialectAttrName(odsState.name), asm_dialect);
  }
  if (operand_attrs) {
  odsState.addAttribute(getOperandAttrsAttrName(odsState.name), operand_attrs);
  }
  if (res)
    odsState.addTypes(res);
}

void InlineAsmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::mlir::StringAttr asm_string, ::mlir::StringAttr constraints, /*optional*/::mlir::UnitAttr has_side_effects, /*optional*/::mlir::UnitAttr is_align_stack, /*optional*/::mlir::LLVM::AsmDialectAttr asm_dialect, /*optional*/::mlir::ArrayAttr operand_attrs) {
  odsState.addOperands(operands);
  odsState.addAttribute(getAsmStringAttrName(odsState.name), asm_string);
  odsState.addAttribute(getConstraintsAttrName(odsState.name), constraints);
  if (has_side_effects) {
  odsState.addAttribute(getHasSideEffectsAttrName(odsState.name), has_side_effects);
  }
  if (is_align_stack) {
  odsState.addAttribute(getIsAlignStackAttrName(odsState.name), is_align_stack);
  }
  if (asm_dialect) {
  odsState.addAttribute(getAsmDialectAttrName(odsState.name), asm_dialect);
  }
  if (operand_attrs) {
  odsState.addAttribute(getOperandAttrsAttrName(odsState.name), operand_attrs);
  }
  odsState.addTypes(resultTypes);
}

void InlineAsmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, /*optional*/::mlir::Type res, ::mlir::ValueRange operands, ::llvm::StringRef asm_string, ::llvm::StringRef constraints, /*optional*/bool has_side_effects, /*optional*/bool is_align_stack, /*optional*/::mlir::LLVM::AsmDialectAttr asm_dialect, /*optional*/::mlir::ArrayAttr operand_attrs) {
  odsState.addOperands(operands);
  odsState.addAttribute(getAsmStringAttrName(odsState.name), odsBuilder.getStringAttr(asm_string));
  odsState.addAttribute(getConstraintsAttrName(odsState.name), odsBuilder.getStringAttr(constraints));
  if (has_side_effects) {
  odsState.addAttribute(getHasSideEffectsAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (is_align_stack) {
  odsState.addAttribute(getIsAlignStackAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (asm_dialect) {
  odsState.addAttribute(getAsmDialectAttrName(odsState.name), asm_dialect);
  }
  if (operand_attrs) {
  odsState.addAttribute(getOperandAttrsAttrName(odsState.name), operand_attrs);
  }
  if (res)
    odsState.addTypes(res);
}

void InlineAsmOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::StringRef asm_string, ::llvm::StringRef constraints, /*optional*/bool has_side_effects, /*optional*/bool is_align_stack, /*optional*/::mlir::LLVM::AsmDialectAttr asm_dialect, /*optional*/::mlir::ArrayAttr operand_attrs) {
  odsState.addOperands(operands);
  odsState.addAttribute(getAsmStringAttrName(odsState.name), odsBuilder.getStringAttr(asm_string));
  odsState.addAttribute(getConstraintsAttrName(odsState.name), odsBuilder.getStringAttr(constraints));
  if (has_side_effects) {
  odsState.addAttribute(getHasSideEffectsAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (is_align_stack) {
  odsState.addAttribute(getIsAlignStackAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (asm_dialect) {
  odsState.addAttribute(getAsmDialectAttrName(odsState.name), asm_dialect);
  }
  if (operand_attrs) {
  odsState.addAttribute(getOperandAttrsAttrName(odsState.name), operand_attrs);
  }
  odsState.addTypes(resultTypes);
}

void InlineAsmOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InlineAsmOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_asm_string;
  ::mlir::Attribute tblgen_asm_dialect;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'asm_string'");
    if (namedAttrIt->getName() == getAsmStringAttrName()) {
      tblgen_asm_string = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getAsmDialectAttrName()) {
      tblgen_asm_dialect = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_constraints;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'constraints'");
    if (namedAttrIt->getName() == getConstraintsAttrName()) {
      tblgen_constraints = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_has_side_effects;
  ::mlir::Attribute tblgen_is_align_stack;
  ::mlir::Attribute tblgen_operand_attrs;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getHasSideEffectsAttrName()) {
      tblgen_has_side_effects = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getIsAlignStackAttrName()) {
      tblgen_is_align_stack = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getOperandAttrsAttrName()) {
      tblgen_operand_attrs = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_asm_string, "asm_string")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_constraints, "constraints")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_has_side_effects, "has_side_effects")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_is_align_stack, "is_align_stack")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps19(*this, tblgen_asm_dialect, "asm_dialect")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_operand_attrs, "operand_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult InlineAsmOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult InlineAsmOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::LLVM::AsmDialectAttr asm_dialectAttr;
  ::mlir::ArrayAttr operand_attrsAttr;
  ::mlir::StringAttr asm_stringAttr;
  ::mlir::StringAttr constraintsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allOperands;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;
  if (succeeded(parser.parseOptionalKeyword("has_side_effects"))) {
    result.addAttribute("has_side_effects", parser.getBuilder().getUnitAttr());
  }
  if (succeeded(parser.parseOptionalKeyword("is_align_stack"))) {
    result.addAttribute("is_align_stack", parser.getBuilder().getUnitAttr());
  }
  if (succeeded(parser.parseOptionalKeyword("asm_dialect"))) {
  if (parser.parseEqual())
    return ::mlir::failure();

  {
    ::llvm::StringRef attrStr;
    ::mlir::NamedAttrList attrStorage;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeyword(&attrStr, {"att","intel"})) {
      ::mlir::StringAttr attrVal;
      ::mlir::OptionalParseResult parseResult =
        parser.parseOptionalAttribute(attrVal,
                                      parser.getBuilder().getNoneType(),
                                      "asm_dialect", attrStorage);
      if (parseResult.hasValue()) {
        if (failed(*parseResult))
          return ::mlir::failure();
        attrStr = attrVal.getValue();
      } else {

      }
    }
    if (!attrStr.empty()) {
      auto attrOptional = ::mlir::LLVM::symbolizeAsmDialect(attrStr);
      if (!attrOptional)
        return parser.emitError(loc, "invalid ")
               << "asm_dialect attribute specification: \"" << attrStr << '"';;

      asm_dialectAttr = ::mlir::LLVM::AsmDialectAttr::get(parser.getBuilder().getContext(), attrOptional.getValue());
      result.addAttribute("asm_dialect", asm_dialectAttr);
    }
  }
  }
  if (succeeded(parser.parseOptionalKeyword("operand_attrs"))) {
  if (parser.parseEqual())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(operand_attrsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "operand_attrs",
          result.attributes)) {
    return ::mlir::failure();
  }
  }
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(asm_stringAttr, parser.getBuilder().getType<::mlir::NoneType>(), "asm_string",
          result.attributes)) {
    return ::mlir::failure();
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseCustomAttributeWithFallback(constraintsAttr, parser.getBuilder().getType<::mlir::NoneType>(), "constraints",
          result.attributes)) {
    return ::mlir::failure();
  }
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType allOperand__allResult_functionType;
  if (parser.parseType(allOperand__allResult_functionType))
    return ::mlir::failure();
  allOperandTypes = allOperand__allResult_functionType.getInputs();
  allResultTypes = allOperand__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void InlineAsmOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if ((*this)->getAttr("has_side_effects")) {
  _odsPrinter << ' ' << "has_side_effects";
  }
  if ((*this)->getAttr("is_align_stack")) {
  _odsPrinter << ' ' << "is_align_stack";
  }
  if ((*this)->getAttr("asm_dialect")) {
  _odsPrinter << ' ' << "asm_dialect";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';

  {
    auto caseValue = *getAsmDialect();
    auto caseValueStr = stringifyAsmDialect(caseValue);
    _odsPrinter << caseValueStr;
  }
  }
  if ((*this)->getAttr("operand_attrs")) {
  _odsPrinter << ' ' << "operand_attrs";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getOperandAttrsAttr());
  }
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"has_side_effects", "is_align_stack", "asm_dialect", "operand_attrs", "asm_string", "constraints"});
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getAsmStringAttr());
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getConstraintsAttr());
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::InlineAsmOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::InsertElementOp definitions
//===----------------------------------------------------------------------===//

InsertElementOpAdaptor::InsertElementOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.insertelement", odsAttrs.getContext());
}

InsertElementOpAdaptor::InsertElementOpAdaptor(InsertElementOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange InsertElementOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InsertElementOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange InsertElementOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertElementOpAdaptor::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertElementOpAdaptor::getValue() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertElementOpAdaptor::getPosition() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr InsertElementOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult InsertElementOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InsertElementOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range InsertElementOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertElementOp::getVector() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertElementOp::getValue() {
  return *getODSOperands(1).begin();
}

::mlir::Value InsertElementOp::getPosition() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange InsertElementOp::getVectorMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertElementOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertElementOp::getPositionMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> InsertElementOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertElementOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertElementOp::getRes() {
  return *getODSResults(0).begin();
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value vector, ::mlir::Value value, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(value);
  odsState.addOperands(position);
  odsState.addTypes(res);
}

void InsertElementOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value vector, ::mlir::Value value, ::mlir::Value position) {
  odsState.addOperands(vector);
  odsState.addOperands(value);
  odsState.addOperands(position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertElementOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InsertElementOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps17(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult InsertElementOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void InsertElementOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::InsertElementOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::InsertValueOp definitions
//===----------------------------------------------------------------------===//

InsertValueOpAdaptor::InsertValueOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.insertvalue", odsAttrs.getContext());
}

InsertValueOpAdaptor::InsertValueOpAdaptor(InsertValueOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange InsertValueOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InsertValueOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange InsertValueOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertValueOpAdaptor::getContainer() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertValueOpAdaptor::getValue() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr InsertValueOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr InsertValueOpAdaptor::getPositionAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, InsertValueOp::getPositionAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
    return attr;
}

::mlir::ArrayAttr InsertValueOpAdaptor::getPosition() {
  auto attr = getPositionAttr();
  return attr;
}

::mlir::LogicalResult InsertValueOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_position;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.insertvalue' op ""requires attribute 'position'");
    if (namedAttrIt->getName() == InsertValueOp::getPositionAttrName(*odsOpName)) {
      tblgen_position = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_position && !((tblgen_position.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'llvm.insertvalue' op ""attribute 'position' failed to satisfy constraint: array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InsertValueOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range InsertValueOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertValueOp::getContainer() {
  return *getODSOperands(0).begin();
}

::mlir::Value InsertValueOp::getValue() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange InsertValueOp::getContainerMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange InsertValueOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> InsertValueOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range InsertValueOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value InsertValueOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr InsertValueOp::getPositionAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getPositionAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr InsertValueOp::getPosition() {
  auto attr = getPositionAttr();
  return attr;
}

void InsertValueOp::setPositionAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPositionAttrName(), attr);
}

void InsertValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value container, Value value, ArrayAttr position) {
      build(odsBuilder, odsState, container.getType(), container, value, position);
    
}

void InsertValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value container, ::mlir::Value value, ::mlir::ArrayAttr position) {
  odsState.addOperands(container);
  odsState.addOperands(value);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);
  odsState.addTypes(res);
}

void InsertValueOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value container, ::mlir::Value value, ::mlir::ArrayAttr position) {
  odsState.addOperands(container);
  odsState.addOperands(value);
  odsState.addAttribute(getPositionAttrName(odsState.name), position);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void InsertValueOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InsertValueOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_position;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'position'");
    if (namedAttrIt->getName() == getPositionAttrName()) {
      tblgen_position = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_position, "position")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps13(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps17(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps13(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult InsertValueOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void InsertValueOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::InsertValueOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::IntToPtrOp definitions
//===----------------------------------------------------------------------===//

IntToPtrOpAdaptor::IntToPtrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.inttoptr", odsAttrs.getContext());
}

IntToPtrOpAdaptor::IntToPtrOpAdaptor(IntToPtrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange IntToPtrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> IntToPtrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange IntToPtrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IntToPtrOpAdaptor::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr IntToPtrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult IntToPtrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> IntToPtrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range IntToPtrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IntToPtrOp::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange IntToPtrOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> IntToPtrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range IntToPtrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value IntToPtrOp::getRes() {
  return *getODSResults(0).begin();
}

void IntToPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void IntToPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void IntToPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void IntToPtrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult IntToPtrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult IntToPtrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult IntToPtrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void IntToPtrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void IntToPtrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::IntToPtrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::InvokeOp definitions
//===----------------------------------------------------------------------===//

InvokeOpAdaptor::InvokeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.invoke", odsAttrs.getContext());
}

InvokeOpAdaptor::InvokeOpAdaptor(InvokeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange InvokeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> InvokeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, InvokeOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange InvokeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::ValueRange InvokeOpAdaptor::getCalleeOperands() {
  return getODSOperands(0);
}

::mlir::ValueRange InvokeOpAdaptor::getNormalDestOperands() {
  return getODSOperands(1);
}

::mlir::ValueRange InvokeOpAdaptor::getUnwindDestOperands() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr InvokeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr InvokeOpAdaptor::getCalleeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, InvokeOp::getCalleeAttrName(*odsOpName)).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > InvokeOpAdaptor::getCallee() {
  auto attr = getCalleeAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::LogicalResult InvokeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_callee;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.invoke' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == InvokeOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == InvokeOp::getCalleeAttrName(*odsOpName)) {
      tblgen_callee = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'llvm.invoke' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_callee && !((tblgen_callee.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'llvm.invoke' op ""attribute 'callee' failed to satisfy constraint: flat symbol reference attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> InvokeOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range InvokeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range InvokeOp::getCalleeOperands() {
  return getODSOperands(0);
}

::mlir::Operation::operand_range InvokeOp::getNormalDestOperands() {
  return getODSOperands(1);
}

::mlir::Operation::operand_range InvokeOp::getUnwindDestOperands() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange InvokeOp::getCalleeOperandsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InvokeOp::getNormalDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange InvokeOp::getUnwindDestOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

std::pair<unsigned, unsigned> InvokeOp::getODSResultIndexAndLength(unsigned index) {
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

::mlir::Operation::result_range InvokeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *InvokeOp::getNormalDest() {
  return (*this)->getSuccessor(0);
}

::mlir::Block *InvokeOp::getUnwindDest() {
  return (*this)->getSuccessor(1);
}

::mlir::FlatSymbolRefAttr InvokeOp::getCalleeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCalleeAttrName()).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
}

::llvm::Optional< ::llvm::StringRef > InvokeOp::getCallee() {
  auto attr = getCalleeAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

void InvokeOp::setCalleeAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getCalleeAttrName(), attr);
}

::mlir::Attribute InvokeOp::removeCalleeAttr() {
  return (*this)->removeAttr(getCalleeAttrName());
}

void InvokeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange tys, FlatSymbolRefAttr callee, ValueRange ops, Block*normal, ValueRange normalOps, Block*unwind, ValueRange unwindOps) {
      odsState.addAttribute("callee", callee);
      build(odsBuilder, odsState, tys, ops, normal, normalOps, unwind, unwindOps);
    
}

void InvokeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, TypeRange tys, ValueRange ops, Block*normal, ValueRange normalOps, Block*unwind, ValueRange unwindOps) {
      build(odsBuilder, odsState, tys, /*callee=*/FlatSymbolRefAttr(), ops, normalOps,
            unwindOps, normal, unwind);
    
}

void InvokeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultType0, /*optional*/::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange callee_operands, ::mlir::ValueRange normalDestOperands, ::mlir::ValueRange unwindDestOperands, ::mlir::Block *normalDest, ::mlir::Block *unwindDest) {
  odsState.addOperands(callee_operands);
  odsState.addOperands(normalDestOperands);
  odsState.addOperands(unwindDestOperands);
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({static_cast<int32_t>(callee_operands.size()), static_cast<int32_t>(normalDestOperands.size()), static_cast<int32_t>(unwindDestOperands.size())}));
  if (callee) {
  odsState.addAttribute(getCalleeAttrName(odsState.name), callee);
  }
  odsState.addSuccessors(normalDest);
  odsState.addSuccessors(unwindDest);
  odsState.addTypes(resultType0);
}

void InvokeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult InvokeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_callee;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCalleeAttrName()) {
      tblgen_callee = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps1(*this, tblgen_callee, "callee")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult InvokeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::InvokeOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LLVMFuncOp definitions
//===----------------------------------------------------------------------===//

LLVMFuncOpAdaptor::LLVMFuncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.func", odsAttrs.getContext());
}

LLVMFuncOpAdaptor::LLVMFuncOpAdaptor(LLVMFuncOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LLVMFuncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LLVMFuncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LLVMFuncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr LLVMFuncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::TypeAttr LLVMFuncOpAdaptor::getFunctionTypeAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LLVMFuncOp::getFunctionTypeAttrName(*odsOpName)).cast<::mlir::TypeAttr>();
    return attr;
}

::mlir::LLVM::LLVMFunctionType LLVMFuncOpAdaptor::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::LLVM::LLVMFunctionType>();
}

::mlir::LLVM::LinkageAttr LLVMFuncOpAdaptor::getLinkageAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, LLVMFuncOp::getLinkageAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::LinkageAttr>();
    if (!attr)
      attr = ::mlir::LLVM::LinkageAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), Linkage::External);
    return attr;
}

::mlir::LLVM::Linkage LLVMFuncOpAdaptor::getLinkage() {
  auto attr = getLinkageAttr();
    if (!attr)
      return ::mlir::LLVM::LinkageAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), Linkage::External).getLinkage();
  return attr.getLinkage();
}

::mlir::UnitAttr LLVMFuncOpAdaptor::getDsoLocalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LLVMFuncOp::getDsoLocalAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool LLVMFuncOpAdaptor::getDsoLocal() {
  auto attr = getDsoLocalAttr();
  return attr != nullptr;
}

::mlir::LLVM::CConvAttr LLVMFuncOpAdaptor::getCConvAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, LLVMFuncOp::getCConvAttrName(*odsOpName)).dyn_cast_or_null<::mlir::LLVM::CConvAttr>();
    if (!attr)
      attr = ::mlir::LLVM::CConvAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), CConv::C);
    return attr;
}

::mlir::LLVM::cconv::CConv LLVMFuncOpAdaptor::getCConv() {
  auto attr = getCConvAttr();
    if (!attr)
      return ::mlir::LLVM::CConvAttr::get(::mlir::Builder(odsAttrs.getContext()).getContext(), CConv::C).getCallingConv();
  return attr.getCallingConv();
}

::mlir::FlatSymbolRefAttr LLVMFuncOpAdaptor::getPersonalityAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, LLVMFuncOp::getPersonalityAttrName(*odsOpName)).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > LLVMFuncOpAdaptor::getPersonality() {
  auto attr = getPersonalityAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr LLVMFuncOpAdaptor::getGarbageCollectorAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, LLVMFuncOp::getGarbageCollectorAttrName(*odsOpName)).dyn_cast_or_null<::mlir::StringAttr>();
    return attr;
}

::llvm::Optional< ::llvm::StringRef > LLVMFuncOpAdaptor::getGarbageCollector() {
  auto attr = getGarbageCollectorAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr LLVMFuncOpAdaptor::getPassthroughAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, LLVMFuncOp::getPassthroughAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > LLVMFuncOpAdaptor::getPassthrough() {
  auto attr = getPassthroughAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::RegionRange LLVMFuncOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &LLVMFuncOpAdaptor::getBody() {
  return *odsRegions[0];
}

::mlir::LogicalResult LLVMFuncOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  ::mlir::Attribute tblgen_CConv;
  ::mlir::Attribute tblgen_dso_local;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.func' op ""requires attribute 'function_type'");
    if (namedAttrIt->getName() == LLVMFuncOp::getFunctionTypeAttrName(*odsOpName)) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getCConvAttrName(*odsOpName)) {
      tblgen_CConv = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getDsoLocalAttrName(*odsOpName)) {
      tblgen_dso_local = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_garbageCollector;
  ::mlir::Attribute tblgen_linkage;
  ::mlir::Attribute tblgen_passthrough;
  ::mlir::Attribute tblgen_personality;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getGarbageCollectorAttrName(*odsOpName)) {
      tblgen_garbageCollector = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getLinkageAttrName(*odsOpName)) {
      tblgen_linkage = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getPassthroughAttrName(*odsOpName)) {
      tblgen_passthrough = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LLVMFuncOp::getPersonalityAttrName(*odsOpName)) {
      tblgen_personality = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_function_type && !(((tblgen_function_type.isa<::mlir::TypeAttr>())) && ((tblgen_function_type.cast<::mlir::TypeAttr>().getValue().isa<::mlir::LLVM::LLVMFunctionType>()))))
    return emitError(loc, "'llvm.func' op ""attribute 'function_type' failed to satisfy constraint: type attribute of LLVM function type");

  if (tblgen_linkage && !((tblgen_linkage.isa<::mlir::LLVM::LinkageAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'linkage' failed to satisfy constraint: LLVM Linkage specification");

  if (tblgen_dso_local && !((tblgen_dso_local.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'dso_local' failed to satisfy constraint: unit attribute");

  if (tblgen_CConv && !((tblgen_CConv.isa<::mlir::LLVM::CConvAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'CConv' failed to satisfy constraint: LLVM Calling Convention specification");

  if (tblgen_personality && !((tblgen_personality.isa<::mlir::FlatSymbolRefAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'personality' failed to satisfy constraint: flat symbol reference attribute");

  if (tblgen_garbageCollector && !((tblgen_garbageCollector.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'garbageCollector' failed to satisfy constraint: string attribute");

  if (tblgen_passthrough && !((tblgen_passthrough.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'llvm.func' op ""attribute 'passthrough' failed to satisfy constraint: array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LLVMFuncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LLVMFuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> LLVMFuncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LLVMFuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &LLVMFuncOp::getBody() {
  return (*this)->getRegion(0);
}

::mlir::TypeAttr LLVMFuncOp::getFunctionTypeAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getFunctionTypeAttrName()).cast<::mlir::TypeAttr>();
}

::mlir::LLVM::LLVMFunctionType LLVMFuncOp::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return attr.getValue().cast<::mlir::LLVM::LLVMFunctionType>();
}

::mlir::LLVM::LinkageAttr LLVMFuncOp::getLinkageAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getLinkageAttrName()).dyn_cast_or_null<::mlir::LLVM::LinkageAttr>();
}

::mlir::LLVM::Linkage LLVMFuncOp::getLinkage() {
  auto attr = getLinkageAttr();
    if (!attr)
      return ::mlir::LLVM::LinkageAttr::get(::mlir::Builder((*this)->getContext()).getContext(), Linkage::External).getLinkage();
  return attr.getLinkage();
}

::mlir::UnitAttr LLVMFuncOp::getDsoLocalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getDsoLocalAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LLVMFuncOp::getDsoLocal() {
  auto attr = getDsoLocalAttr();
  return attr != nullptr;
}

::mlir::LLVM::CConvAttr LLVMFuncOp::getCConvAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCConvAttrName()).dyn_cast_or_null<::mlir::LLVM::CConvAttr>();
}

::mlir::LLVM::cconv::CConv LLVMFuncOp::getCConv() {
  auto attr = getCConvAttr();
    if (!attr)
      return ::mlir::LLVM::CConvAttr::get(::mlir::Builder((*this)->getContext()).getContext(), CConv::C).getCallingConv();
  return attr.getCallingConv();
}

::mlir::FlatSymbolRefAttr LLVMFuncOp::getPersonalityAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getPersonalityAttrName()).dyn_cast_or_null<::mlir::FlatSymbolRefAttr>();
}

::llvm::Optional< ::llvm::StringRef > LLVMFuncOp::getPersonality() {
  auto attr = getPersonalityAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::StringAttr LLVMFuncOp::getGarbageCollectorAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getGarbageCollectorAttrName()).dyn_cast_or_null<::mlir::StringAttr>();
}

::llvm::Optional< ::llvm::StringRef > LLVMFuncOp::getGarbageCollector() {
  auto attr = getGarbageCollectorAttr();
  return attr ? ::llvm::Optional< ::llvm::StringRef >(attr.getValue()) : (::llvm::None);
}

::mlir::ArrayAttr LLVMFuncOp::getPassthroughAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getPassthroughAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > LLVMFuncOp::getPassthrough() {
  auto attr = getPassthroughAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

void LLVMFuncOp::setFunctionTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getFunctionTypeAttrName(), attr);
}

void LLVMFuncOp::setLinkageAttr(::mlir::LLVM::LinkageAttr attr) {
  (*this)->setAttr(getLinkageAttrName(), attr);
}

void LLVMFuncOp::setDsoLocalAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getDsoLocalAttrName(), attr);
}

void LLVMFuncOp::setCConvAttr(::mlir::LLVM::CConvAttr attr) {
  (*this)->setAttr(getCConvAttrName(), attr);
}

void LLVMFuncOp::setPersonalityAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getPersonalityAttrName(), attr);
}

void LLVMFuncOp::setGarbageCollectorAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getGarbageCollectorAttrName(), attr);
}

void LLVMFuncOp::setPassthroughAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getPassthroughAttrName(), attr);
}

::mlir::Attribute LLVMFuncOp::removeDso_localAttr() {
  return (*this)->removeAttr(getDsoLocalAttrName());
}

::mlir::Attribute LLVMFuncOp::removePersonalityAttr() {
  return (*this)->removeAttr(getPersonalityAttrName());
}

::mlir::Attribute LLVMFuncOp::removeGarbageCollectorAttr() {
  return (*this)->removeAttr(getGarbageCollectorAttrName());
}

::mlir::Attribute LLVMFuncOp::removePassthroughAttr() {
  return (*this)->removeAttr(getPassthroughAttrName());
}

::mlir::LogicalResult LLVMFuncOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_function_type;
  ::mlir::Attribute tblgen_CConv;
  ::mlir::Attribute tblgen_dso_local;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'function_type'");
    if (namedAttrIt->getName() == getFunctionTypeAttrName()) {
      tblgen_function_type = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCConvAttrName()) {
      tblgen_CConv = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getDsoLocalAttrName()) {
      tblgen_dso_local = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_garbageCollector;
  ::mlir::Attribute tblgen_linkage;
  ::mlir::Attribute tblgen_passthrough;
  ::mlir::Attribute tblgen_personality;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getGarbageCollectorAttrName()) {
      tblgen_garbageCollector = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getLinkageAttrName()) {
      tblgen_linkage = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getPassthroughAttrName()) {
      tblgen_passthrough = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getPersonalityAttrName()) {
      tblgen_personality = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps20(*this, tblgen_function_type, "function_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps15(*this, tblgen_linkage, "linkage")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_dso_local, "dso_local")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps21(*this, tblgen_CConv, "CConv")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps1(*this, tblgen_personality, "personality")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_garbageCollector, "garbageCollector")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_passthrough, "passthrough")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LLVMOps0(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult LLVMFuncOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LLVMFuncOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LShrOp definitions
//===----------------------------------------------------------------------===//

LShrOpAdaptor::LShrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.lshr", odsAttrs.getContext());
}

LShrOpAdaptor::LShrOpAdaptor(LShrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LShrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LShrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LShrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LShrOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value LShrOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr LShrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LShrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LShrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LShrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LShrOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value LShrOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange LShrOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange LShrOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LShrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LShrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LShrOp::getRes() {
  return *getODSResults(0).begin();
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(LShrOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LShrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(LShrOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult LShrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LShrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult LShrOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult LShrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void LShrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void LShrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LShrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LandingpadOp definitions
//===----------------------------------------------------------------------===//

LandingpadOpAdaptor::LandingpadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.landingpad", odsAttrs.getContext());
}

LandingpadOpAdaptor::LandingpadOpAdaptor(LandingpadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LandingpadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LandingpadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::ValueRange LandingpadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr LandingpadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::UnitAttr LandingpadOpAdaptor::getCleanupAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LandingpadOp::getCleanupAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool LandingpadOpAdaptor::getCleanup() {
  auto attr = getCleanupAttr();
  return attr != nullptr;
}

::mlir::LogicalResult LandingpadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_cleanup;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == LandingpadOp::getCleanupAttrName(*odsOpName)) {
      tblgen_cleanup = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_cleanup && !((tblgen_cleanup.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.landingpad' op ""attribute 'cleanup' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LandingpadOp::getODSOperandIndexAndLength(unsigned index) {
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

::mlir::Operation::operand_range LandingpadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> LandingpadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LandingpadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LandingpadOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::UnitAttr LandingpadOp::getCleanupAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getCleanupAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LandingpadOp::getCleanup() {
  auto attr = getCleanupAttr();
  return attr != nullptr;
}

void LandingpadOp::setCleanupAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getCleanupAttrName(), attr);
}

::mlir::Attribute LandingpadOp::removeCleanupAttr() {
  return (*this)->removeAttr(getCleanupAttrName());
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, /*optional*/::mlir::UnitAttr cleanup, ::mlir::ValueRange odsArg_0) {
  odsState.addOperands(odsArg_0);
  if (cleanup) {
  odsState.addAttribute(getCleanupAttrName(odsState.name), cleanup);
  }
  odsState.addTypes(res);
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/::mlir::UnitAttr cleanup, ::mlir::ValueRange odsArg_0) {
  odsState.addOperands(odsArg_0);
  if (cleanup) {
  odsState.addAttribute(getCleanupAttrName(odsState.name), cleanup);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, /*optional*/bool cleanup, ::mlir::ValueRange odsArg_0) {
  odsState.addOperands(odsArg_0);
  if (cleanup) {
  odsState.addAttribute(getCleanupAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  odsState.addTypes(res);
}

void LandingpadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, /*optional*/bool cleanup, ::mlir::ValueRange odsArg_0) {
  odsState.addOperands(odsArg_0);
  if (cleanup) {
  odsState.addAttribute(getCleanupAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LandingpadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LandingpadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_cleanup;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getCleanupAttrName()) {
      tblgen_cleanup = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_cleanup, "cleanup")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LandingpadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LandingpadOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LoadOp definitions
//===----------------------------------------------------------------------===//

LoadOpAdaptor::LoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.load", odsAttrs.getContext());
}

LoadOpAdaptor::LoadOpAdaptor(LoadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOpAdaptor::getAddr() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr LoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr LoadOpAdaptor::getAccessGroupsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getAccessGroupsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > LoadOpAdaptor::getAccessGroups() {
  auto attr = getAccessGroupsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr LoadOpAdaptor::getAliasScopesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getAliasScopesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > LoadOpAdaptor::getAliasScopes() {
  auto attr = getAliasScopesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr LoadOpAdaptor::getNoaliasScopesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getNoaliasScopesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > LoadOpAdaptor::getNoaliasScopes() {
  auto attr = getNoaliasScopesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::IntegerAttr LoadOpAdaptor::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    return attr;
}

::llvm::Optional<uint64_t> LoadOpAdaptor::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr LoadOpAdaptor::getVolatile_Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getVolatile_AttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool LoadOpAdaptor::getVolatile_() {
  auto attr = getVolatile_Attr();
  return attr != nullptr;
}

::mlir::UnitAttr LoadOpAdaptor::getNontemporalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, LoadOp::getNontemporalAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool LoadOpAdaptor::getNontemporal() {
  auto attr = getNontemporalAttr();
  return attr != nullptr;
}

::mlir::LogicalResult LoadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_access_groups;
  ::mlir::Attribute tblgen_alias_scopes;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_noalias_scopes;
  ::mlir::Attribute tblgen_nontemporal;
  ::mlir::Attribute tblgen_volatile_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == LoadOp::getAccessGroupsAttrName(*odsOpName)) {
      tblgen_access_groups = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoadOp::getAliasScopesAttrName(*odsOpName)) {
      tblgen_alias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoadOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoadOp::getNoaliasScopesAttrName(*odsOpName)) {
      tblgen_noalias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoadOp::getNontemporalAttrName(*odsOpName)) {
      tblgen_nontemporal = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == LoadOp::getVolatile_AttrName(*odsOpName)) {
      tblgen_volatile_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_access_groups && !(((tblgen_access_groups.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_access_groups.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.load' op ""attribute 'access_groups' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_alias_scopes && !(((tblgen_alias_scopes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_alias_scopes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.load' op ""attribute 'alias_scopes' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_noalias_scopes && !(((tblgen_noalias_scopes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_noalias_scopes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.load' op ""attribute 'noalias_scopes' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'llvm.load' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_volatile_ && !((tblgen_volatile_.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.load' op ""attribute 'volatile_' failed to satisfy constraint: unit attribute");

  if (tblgen_nontemporal && !((tblgen_nontemporal.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.load' op ""attribute 'nontemporal' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LoadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOp::getAddr() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange LoadOp::getAddrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LoadOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr LoadOp::getAccessGroupsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAccessGroupsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > LoadOp::getAccessGroups() {
  auto attr = getAccessGroupsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr LoadOp::getAliasScopesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAliasScopesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > LoadOp::getAliasScopes() {
  auto attr = getAliasScopesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr LoadOp::getNoaliasScopesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNoaliasScopesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > LoadOp::getNoaliasScopes() {
  auto attr = getNoaliasScopesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::IntegerAttr LoadOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> LoadOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr LoadOp::getVolatile_Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getVolatile_AttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoadOp::getVolatile_() {
  auto attr = getVolatile_Attr();
  return attr != nullptr;
}

::mlir::UnitAttr LoadOp::getNontemporalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNontemporalAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool LoadOp::getNontemporal() {
  auto attr = getNontemporalAttr();
  return attr != nullptr;
}

void LoadOp::setAccessGroupsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAccessGroupsAttrName(), attr);
}

void LoadOp::setAliasScopesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAliasScopesAttrName(), attr);
}

void LoadOp::setNoaliasScopesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getNoaliasScopesAttrName(), attr);
}

void LoadOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void LoadOp::setVolatile_Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getVolatile_AttrName(), attr);
}

void LoadOp::setNontemporalAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNontemporalAttrName(), attr);
}

::mlir::Attribute LoadOp::removeAccess_groupsAttr() {
  return (*this)->removeAttr(getAccessGroupsAttrName());
}

::mlir::Attribute LoadOp::removeAlias_scopesAttr() {
  return (*this)->removeAttr(getAliasScopesAttrName());
}

::mlir::Attribute LoadOp::removeNoalias_scopesAttr() {
  return (*this)->removeAttr(getNoaliasScopesAttrName());
}

::mlir::Attribute LoadOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute LoadOp::removeVolatile_Attr() {
  return (*this)->removeAttr(getVolatile_AttrName());
}

::mlir::Attribute LoadOp::removeNontemporalAttr() {
  return (*this)->removeAttr(getNontemporalAttrName());
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value addr, unsigned alignment, bool isVolatile, bool isNonTemporal) {
      auto type = addr.getType().cast<LLVMPointerType>().getElementType();
      assert(type && "must provide explicit element type to the constructor "
                     "when the pointer type is opaque");
      build(odsBuilder, odsState, type, addr, alignment, isVolatile, isNonTemporal);
    
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::UnitAttr nontemporal) {
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
  odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
  odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
  odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (nontemporal) {
  odsState.addAttribute(getNontemporalAttrName(odsState.name), nontemporal);
  }
  odsState.addTypes(res);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::UnitAttr nontemporal) {
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
  odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
  odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
  odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (nontemporal) {
  odsState.addAttribute(getNontemporalAttrName(odsState.name), nontemporal);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/bool nontemporal) {
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
  odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
  odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
  odsState.addAttribute(getVolatile_AttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (nontemporal) {
  odsState.addAttribute(getNontemporalAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  odsState.addTypes(res);
}

void LoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/bool nontemporal) {
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
  odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
  odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
  odsState.addAttribute(getVolatile_AttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (nontemporal) {
  odsState.addAttribute(getNontemporalAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult LoadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_access_groups;
  ::mlir::Attribute tblgen_alias_scopes;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_noalias_scopes;
  ::mlir::Attribute tblgen_nontemporal;
  ::mlir::Attribute tblgen_volatile_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getAccessGroupsAttrName()) {
      tblgen_access_groups = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAliasScopesAttrName()) {
      tblgen_alias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNoaliasScopesAttrName()) {
      tblgen_noalias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNontemporalAttrName()) {
      tblgen_nontemporal = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getVolatile_AttrName()) {
      tblgen_volatile_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps22(*this, tblgen_access_groups, "access_groups")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps22(*this, tblgen_alias_scopes, "alias_scopes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps22(*this, tblgen_noalias_scopes, "noalias_scopes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps2(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_volatile_, "volatile_")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_nontemporal, "nontemporal")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps18(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LoadOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void LoadOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LoadOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MetadataOp definitions
//===----------------------------------------------------------------------===//

MetadataOpAdaptor::MetadataOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.metadata", odsAttrs.getContext());
}

MetadataOpAdaptor::MetadataOpAdaptor(MetadataOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MetadataOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MetadataOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MetadataOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr MetadataOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr MetadataOpAdaptor::getSymNameAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MetadataOp::getSymNameAttrName(*odsOpName)).cast<::mlir::StringAttr>();
    return attr;
}

::llvm::StringRef MetadataOpAdaptor::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::RegionRange MetadataOpAdaptor::getRegions() {
  return odsRegions;
}

::mlir::Region &MetadataOpAdaptor::getBody() {
  return *odsRegions[0];
}

::mlir::LogicalResult MetadataOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_sym_name;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.metadata' op ""requires attribute 'sym_name'");
    if (namedAttrIt->getName() == MetadataOp::getSymNameAttrName(*odsOpName)) {
      tblgen_sym_name = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_sym_name && !((tblgen_sym_name.isa<::mlir::StringAttr>())))
    return emitError(loc, "'llvm.metadata' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MetadataOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MetadataOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> MetadataOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MetadataOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &MetadataOp::getBody() {
  return (*this)->getRegion(0);
}

::mlir::StringAttr MetadataOp::getSymNameAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getSymNameAttrName()).cast<::mlir::StringAttr>();
}

::llvm::StringRef MetadataOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

void MetadataOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void MetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::StringAttr sym_name) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  (void)odsState.addRegion();
}

void MetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::StringAttr sym_name) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef sym_name) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  (void)odsState.addRegion();
}

void MetadataOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef sym_name) {
  odsState.addAttribute(getSymNameAttrName(odsState.name), odsBuilder.getStringAttr(sym_name));
  (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MetadataOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  for (unsigned i = 0; i != 1; ++i)
    (void)odsState.addRegion();
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MetadataOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps0(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::makeMutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_LLVMOps1(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult MetadataOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MetadataOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::StringAttr sym_nameAttr;
  std::unique_ptr<::mlir::Region> bodyRegion = std::make_unique<::mlir::Region>();

  if (parser.parseSymbolName(sym_nameAttr, "sym_name", result.attributes))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return ::mlir::failure();

  if (parser.parseRegion(*bodyRegion))
    return ::mlir::failure();
  result.addRegion(std::move(bodyRegion));
  return ::mlir::success();
}

void MetadataOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printSymbolName(getSymNameAttr().getValue());
  _odsPrinter.printOptionalAttrDictWithKeyword((*this)->getAttrs(), /*elidedAttrs=*/{"sym_name"});
  _odsPrinter << ' ';
  _odsPrinter.printRegion(getBody());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MetadataOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MulOp definitions
//===----------------------------------------------------------------------===//

MulOpAdaptor::MulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mul", odsAttrs.getContext());
}

MulOpAdaptor::MulOpAdaptor(MulOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MulOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MulOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MulOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MulOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MulOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MulOp::getRes() {
  return *getODSResults(0).begin();
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(MulOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(MulOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult MulOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult MulOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult MulOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MulOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void MulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MulOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::NullOp definitions
//===----------------------------------------------------------------------===//

NullOpAdaptor::NullOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.null", odsAttrs.getContext());
}

NullOpAdaptor::NullOpAdaptor(NullOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange NullOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> NullOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange NullOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr NullOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult NullOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> NullOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range NullOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> NullOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range NullOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value NullOp::getRes() {
  return *getODSResults(0).begin();
}

void NullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void NullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void NullOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void NullOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult NullOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps2(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult NullOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult NullOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::LLVM::LLVMPointerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void NullOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::LLVM::LLVMPointerType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void NullOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::NullOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::OrOp definitions
//===----------------------------------------------------------------------===//

OrOpAdaptor::OrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.or", odsAttrs.getContext());
}

OrOpAdaptor::OrOpAdaptor(OrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange OrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> OrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange OrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OrOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value OrOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr OrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult OrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> OrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range OrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OrOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value OrOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange OrOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange OrOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> OrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range OrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value OrOp::getRes() {
  return *getODSResults(0).begin();
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(OrOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void OrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void OrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(OrOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult OrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult OrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult OrOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult OrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void OrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void OrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::OrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::PtrToIntOp definitions
//===----------------------------------------------------------------------===//

PtrToIntOpAdaptor::PtrToIntOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.ptrtoint", odsAttrs.getContext());
}

PtrToIntOpAdaptor::PtrToIntOpAdaptor(PtrToIntOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PtrToIntOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PtrToIntOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PtrToIntOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PtrToIntOpAdaptor::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr PtrToIntOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult PtrToIntOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PtrToIntOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PtrToIntOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PtrToIntOp::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange PtrToIntOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PtrToIntOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PtrToIntOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PtrToIntOp::getRes() {
  return *getODSResults(0).begin();
}

void PtrToIntOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void PtrToIntOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void PtrToIntOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PtrToIntOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PtrToIntOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PtrToIntOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PtrToIntOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PtrToIntOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void PtrToIntOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::PtrToIntOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ResumeOp definitions
//===----------------------------------------------------------------------===//

ResumeOpAdaptor::ResumeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.resume", odsAttrs.getContext());
}

ResumeOpAdaptor::ResumeOpAdaptor(ResumeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ResumeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ResumeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ResumeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResumeOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ResumeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ResumeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ResumeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ResumeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ResumeOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ResumeOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ResumeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ResumeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ResumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value) {
  odsState.addOperands(value);
}

void ResumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value) {
  odsState.addOperands(value);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ResumeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ResumeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ResumeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ResumeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ResumeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ResumeOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ReturnOp definitions
//===----------------------------------------------------------------------===//

ReturnOpAdaptor::ReturnOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.return", odsAttrs.getContext());
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

::mlir::ValueRange ReturnOpAdaptor::getArgs() {
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

::mlir::Operation::operand_range ReturnOp::getArgs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::getArgsMutable() {
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

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange args) {
  odsState.addOperands(args);
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
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
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
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> argsOperands;
  ::llvm::SMLoc argsOperandsLoc;
  (void)argsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> argsTypes;
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();

  argsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(argsOperands))
    return ::mlir::failure();
  if (!argsOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(argsTypes))
    return ::mlir::failure();
  }
  if (parser.resolveOperands(argsOperands, argsTypes, argsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReturnOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  if (!getArgs().empty()) {
  _odsPrinter << ' ';
  _odsPrinter << getArgs();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getArgs().getTypes();
  }
}

void ReturnOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ReturnOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SDivOp definitions
//===----------------------------------------------------------------------===//

SDivOpAdaptor::SDivOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.sdiv", odsAttrs.getContext());
}

SDivOpAdaptor::SDivOpAdaptor(SDivOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SDivOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SDivOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SDivOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SDivOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SDivOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr SDivOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SDivOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SDivOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SDivOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SDivOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SDivOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SDivOp::getRes() {
  return *getODSResults(0).begin();
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SDivOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SDivOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SDivOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SDivOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult SDivOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult SDivOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SDivOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SDivOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SDivOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SExtOp definitions
//===----------------------------------------------------------------------===//

SExtOpAdaptor::SExtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.sext", odsAttrs.getContext());
}

SExtOpAdaptor::SExtOpAdaptor(SExtOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SExtOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SExtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SExtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SExtOpAdaptor::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SExtOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SExtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SExtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SExtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SExtOp::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SExtOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SExtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SExtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SExtOp::getRes() {
  return *getODSResults(0).begin();
}

void SExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void SExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void SExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SExtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SExtOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SExtOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SExtOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SExtOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SExtOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SExtOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SIToFPOp definitions
//===----------------------------------------------------------------------===//

SIToFPOpAdaptor::SIToFPOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.sitofp", odsAttrs.getContext());
}

SIToFPOpAdaptor::SIToFPOpAdaptor(SIToFPOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SIToFPOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SIToFPOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SIToFPOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SIToFPOpAdaptor::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SIToFPOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SIToFPOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SIToFPOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SIToFPOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SIToFPOp::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SIToFPOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SIToFPOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SIToFPOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SIToFPOp::getRes() {
  return *getODSResults(0).begin();
}

void SIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void SIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void SIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SIToFPOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SIToFPOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SIToFPOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult SIToFPOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SIToFPOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SIToFPOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SIToFPOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SRemOp definitions
//===----------------------------------------------------------------------===//

SRemOpAdaptor::SRemOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.srem", odsAttrs.getContext());
}

SRemOpAdaptor::SRemOpAdaptor(SRemOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SRemOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SRemOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SRemOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SRemOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SRemOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr SRemOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SRemOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SRemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SRemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SRemOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SRemOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SRemOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SRemOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SRemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SRemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SRemOp::getRes() {
  return *getODSResults(0).begin();
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SRemOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SRemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SRemOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SRemOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SRemOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult SRemOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult SRemOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SRemOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SRemOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SRemOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SelectOp definitions
//===----------------------------------------------------------------------===//

SelectOpAdaptor::SelectOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.select", odsAttrs.getContext());
}

SelectOpAdaptor::SelectOpAdaptor(SelectOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SelectOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SelectOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SelectOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOpAdaptor::getCondition() {
  return *getODSOperands(0).begin();
}

::mlir::Value SelectOpAdaptor::getTrueValue() {
  return *getODSOperands(1).begin();
}

::mlir::Value SelectOpAdaptor::getFalseValue() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr SelectOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SelectOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SelectOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SelectOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOp::getCondition() {
  return *getODSOperands(0).begin();
}

::mlir::Value SelectOp::getTrueValue() {
  return *getODSOperands(1).begin();
}

::mlir::Value SelectOp::getFalseValue() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange SelectOp::getConditionMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SelectOp::getTrueValueMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SelectOp::getFalseValueMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SelectOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SelectOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SelectOp::getRes() {
  return *getODSResults(0).begin();
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value condition, ::mlir::Value trueValue, ::mlir::Value falseValue) {
  odsState.addOperands(condition);
  odsState.addOperands(trueValue);
  odsState.addOperands(falseValue);
  odsState.addTypes(res);
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value condition, ::mlir::Value trueValue, ::mlir::Value falseValue) {
  odsState.addOperands(condition);
  odsState.addOperands(trueValue);
  odsState.addOperands(falseValue);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SelectOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value condition, ::mlir::Value trueValue, ::mlir::Value falseValue) {
  odsState.addOperands(condition);
  odsState.addOperands(trueValue);
  odsState.addOperands(falseValue);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SelectOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SelectOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SelectOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SelectOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps15(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  if (!((::llvm::is_splat(::llvm::makeArrayRef({(*this->getODSOperands(1).begin()).getType(), (*this->getODSOperands(2).begin()).getType(), (*this->getODSResults(0).begin()).getType()})))))
    return emitOpError("failed to verify that all of {trueValue, falseValue, res} have same type");
  return ::mlir::success();
}

::mlir::LogicalResult SelectOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult SelectOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[2].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult SelectOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allOperands;
  ::mlir::Type conditionRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> conditionTypes(conditionRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    conditionRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(allOperands, ::llvm::concat<const ::mlir::Type>(::llvm::ArrayRef<::mlir::Type>(conditionTypes), ::llvm::ArrayRef<::mlir::Type>(resTypes), ::llvm::ArrayRef<::mlir::Type>(resTypes)), allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SelectOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperands();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getCondition().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SelectOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SelectOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ShlOp definitions
//===----------------------------------------------------------------------===//

ShlOpAdaptor::ShlOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.shl", odsAttrs.getContext());
}

ShlOpAdaptor::ShlOpAdaptor(ShlOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ShlOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ShlOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ShlOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShlOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShlOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ShlOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ShlOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ShlOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ShlOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShlOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShlOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ShlOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShlOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ShlOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShlOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShlOp::getRes() {
  return *getODSResults(0).begin();
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ShlOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShlOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ShlOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ShlOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ShlOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult ShlOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult ShlOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ShlOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ShlOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ShlOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ShuffleVectorOp definitions
//===----------------------------------------------------------------------===//

ShuffleVectorOpAdaptor::ShuffleVectorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.shufflevector", odsAttrs.getContext());
}

ShuffleVectorOpAdaptor::ShuffleVectorOpAdaptor(ShuffleVectorOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ShuffleVectorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ShuffleVectorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ShuffleVectorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleVectorOpAdaptor::getV1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShuffleVectorOpAdaptor::getV2() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr ShuffleVectorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr ShuffleVectorOpAdaptor::getMaskAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, ShuffleVectorOp::getMaskAttrName(*odsOpName)).cast<::mlir::ArrayAttr>();
    return attr;
}

::mlir::ArrayAttr ShuffleVectorOpAdaptor::getMask() {
  auto attr = getMaskAttr();
  return attr;
}

::mlir::LogicalResult ShuffleVectorOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_mask;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.shufflevector' op ""requires attribute 'mask'");
    if (namedAttrIt->getName() == ShuffleVectorOp::getMaskAttrName(*odsOpName)) {
      tblgen_mask = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_mask && !((tblgen_mask.isa<::mlir::ArrayAttr>())))
    return emitError(loc, "'llvm.shufflevector' op ""attribute 'mask' failed to satisfy constraint: array attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ShuffleVectorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ShuffleVectorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleVectorOp::getV1() {
  return *getODSOperands(0).begin();
}

::mlir::Value ShuffleVectorOp::getV2() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange ShuffleVectorOp::getV1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange ShuffleVectorOp::getV2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ShuffleVectorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ShuffleVectorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ShuffleVectorOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::ArrayAttr ShuffleVectorOp::getMaskAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getMaskAttrName()).cast<::mlir::ArrayAttr>();
}

::mlir::ArrayAttr ShuffleVectorOp::getMask() {
  auto attr = getMaskAttr();
  return attr;
}

void ShuffleVectorOp::setMaskAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getMaskAttrName(), attr);
}

void ShuffleVectorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value v1, ::mlir::Value v2, ::mlir::ArrayAttr mask) {
  odsState.addOperands(v1);
  odsState.addOperands(v2);
  odsState.addAttribute(getMaskAttrName(odsState.name), mask);
  odsState.addTypes(res);
}

void ShuffleVectorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value v1, ::mlir::Value v2, ::mlir::ArrayAttr mask) {
  odsState.addOperands(v1);
  odsState.addOperands(v2);
  odsState.addAttribute(getMaskAttrName(odsState.name), mask);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ShuffleVectorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ShuffleVectorOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_mask;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'mask'");
    if (namedAttrIt->getName() == getMaskAttrName()) {
      tblgen_mask = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps9(*this, tblgen_mask, "mask")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps12(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ShuffleVectorOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void ShuffleVectorOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ShuffleVectorOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::StoreOp definitions
//===----------------------------------------------------------------------===//

StoreOpAdaptor::StoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.store", odsAttrs.getContext());
}

StoreOpAdaptor::StoreOpAdaptor(StoreOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StoreOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value StoreOpAdaptor::getAddr() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr StoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ArrayAttr StoreOpAdaptor::getAccessGroupsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getAccessGroupsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StoreOpAdaptor::getAccessGroups() {
  auto attr = getAccessGroupsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StoreOpAdaptor::getAliasScopesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getAliasScopesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StoreOpAdaptor::getAliasScopes() {
  auto attr = getAliasScopesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StoreOpAdaptor::getNoaliasScopesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getNoaliasScopesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ArrayAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ArrayAttr > StoreOpAdaptor::getNoaliasScopes() {
  auto attr = getNoaliasScopesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::IntegerAttr StoreOpAdaptor::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getAlignmentAttrName(*odsOpName)).dyn_cast_or_null<::mlir::IntegerAttr>();
    return attr;
}

::llvm::Optional<uint64_t> StoreOpAdaptor::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr StoreOpAdaptor::getVolatile_Attr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getVolatile_AttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool StoreOpAdaptor::getVolatile_() {
  auto attr = getVolatile_Attr();
  return attr != nullptr;
}

::mlir::UnitAttr StoreOpAdaptor::getNontemporalAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, StoreOp::getNontemporalAttrName(*odsOpName)).dyn_cast_or_null<::mlir::UnitAttr>();
    return attr;
}

bool StoreOpAdaptor::getNontemporal() {
  auto attr = getNontemporalAttr();
  return attr != nullptr;
}

::mlir::LogicalResult StoreOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_access_groups;
  ::mlir::Attribute tblgen_alias_scopes;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_noalias_scopes;
  ::mlir::Attribute tblgen_nontemporal;
  ::mlir::Attribute tblgen_volatile_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == StoreOp::getAccessGroupsAttrName(*odsOpName)) {
      tblgen_access_groups = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StoreOp::getAliasScopesAttrName(*odsOpName)) {
      tblgen_alias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StoreOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StoreOp::getNoaliasScopesAttrName(*odsOpName)) {
      tblgen_noalias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StoreOp::getNontemporalAttrName(*odsOpName)) {
      tblgen_nontemporal = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == StoreOp::getVolatile_AttrName(*odsOpName)) {
      tblgen_volatile_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_access_groups && !(((tblgen_access_groups.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_access_groups.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.store' op ""attribute 'access_groups' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_alias_scopes && !(((tblgen_alias_scopes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_alias_scopes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.store' op ""attribute 'alias_scopes' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_noalias_scopes && !(((tblgen_noalias_scopes.isa<::mlir::ArrayAttr>())) && (::llvm::all_of(tblgen_noalias_scopes.cast<::mlir::ArrayAttr>(), [&](::mlir::Attribute attr) { return attr && ((attr.isa<::mlir::SymbolRefAttr>())); }))))
    return emitError(loc, "'llvm.store' op ""attribute 'noalias_scopes' failed to satisfy constraint: symbol ref array attribute");

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(64)))))
    return emitError(loc, "'llvm.store' op ""attribute 'alignment' failed to satisfy constraint: 64-bit signless integer attribute");

  if (tblgen_volatile_ && !((tblgen_volatile_.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.store' op ""attribute 'volatile_' failed to satisfy constraint: unit attribute");

  if (tblgen_nontemporal && !((tblgen_nontemporal.isa<::mlir::UnitAttr>())))
    return emitError(loc, "'llvm.store' op ""attribute 'nontemporal' failed to satisfy constraint: unit attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StoreOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value StoreOp::getAddr() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange StoreOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange StoreOp::getAddrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::ArrayAttr StoreOp::getAccessGroupsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAccessGroupsAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StoreOp::getAccessGroups() {
  auto attr = getAccessGroupsAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StoreOp::getAliasScopesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAliasScopesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StoreOp::getAliasScopes() {
  auto attr = getAliasScopesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::ArrayAttr StoreOp::getNoaliasScopesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNoaliasScopesAttrName()).dyn_cast_or_null<::mlir::ArrayAttr>();
}

::llvm::Optional< ::mlir::ArrayAttr > StoreOp::getNoaliasScopes() {
  auto attr = getNoaliasScopesAttr();
  return attr ? ::llvm::Optional< ::mlir::ArrayAttr >(attr) : (::llvm::None);
}

::mlir::IntegerAttr StoreOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlignmentAttrName()).dyn_cast_or_null<::mlir::IntegerAttr>();
}

::llvm::Optional<uint64_t> StoreOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr ? ::llvm::Optional<uint64_t>(attr.getValue().getZExtValue()) : (::llvm::None);
}

::mlir::UnitAttr StoreOp::getVolatile_Attr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getVolatile_AttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool StoreOp::getVolatile_() {
  auto attr = getVolatile_Attr();
  return attr != nullptr;
}

::mlir::UnitAttr StoreOp::getNontemporalAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getNontemporalAttrName()).dyn_cast_or_null<::mlir::UnitAttr>();
}

bool StoreOp::getNontemporal() {
  auto attr = getNontemporalAttr();
  return attr != nullptr;
}

void StoreOp::setAccessGroupsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAccessGroupsAttrName(), attr);
}

void StoreOp::setAliasScopesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getAliasScopesAttrName(), attr);
}

void StoreOp::setNoaliasScopesAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getNoaliasScopesAttrName(), attr);
}

void StoreOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void StoreOp::setVolatile_Attr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getVolatile_AttrName(), attr);
}

void StoreOp::setNontemporalAttr(::mlir::UnitAttr attr) {
  (*this)->setAttr(getNontemporalAttrName(), attr);
}

::mlir::Attribute StoreOp::removeAccess_groupsAttr() {
  return (*this)->removeAttr(getAccessGroupsAttrName());
}

::mlir::Attribute StoreOp::removeAlias_scopesAttr() {
  return (*this)->removeAttr(getAliasScopesAttrName());
}

::mlir::Attribute StoreOp::removeNoalias_scopesAttr() {
  return (*this)->removeAttr(getNoaliasScopesAttrName());
}

::mlir::Attribute StoreOp::removeAlignmentAttr() {
  return (*this)->removeAttr(getAlignmentAttrName());
}

::mlir::Attribute StoreOp::removeVolatile_Attr() {
  return (*this)->removeAttr(getVolatile_AttrName());
}

::mlir::Attribute StoreOp::removeNontemporalAttr() {
  return (*this)->removeAttr(getNontemporalAttrName());
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::UnitAttr nontemporal) {
  odsState.addOperands(value);
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
  odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
  odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
  odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (nontemporal) {
  odsState.addAttribute(getNontemporalAttrName(odsState.name), nontemporal);
  }
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::IntegerAttr alignment, /*optional*/::mlir::UnitAttr volatile_, /*optional*/::mlir::UnitAttr nontemporal) {
  odsState.addOperands(value);
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
  odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
  odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
  odsState.addAttribute(getVolatile_AttrName(odsState.name), volatile_);
  }
  if (nontemporal) {
  odsState.addAttribute(getNontemporalAttrName(odsState.name), nontemporal);
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/bool nontemporal) {
  odsState.addOperands(value);
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
  odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
  odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
  odsState.addAttribute(getVolatile_AttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (nontemporal) {
  odsState.addAttribute(getNontemporalAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
}

void StoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value addr, /*optional*/::mlir::ArrayAttr access_groups, /*optional*/::mlir::ArrayAttr alias_scopes, /*optional*/::mlir::ArrayAttr noalias_scopes, /*optional*/::mlir::IntegerAttr alignment, /*optional*/bool volatile_, /*optional*/bool nontemporal) {
  odsState.addOperands(value);
  odsState.addOperands(addr);
  if (access_groups) {
  odsState.addAttribute(getAccessGroupsAttrName(odsState.name), access_groups);
  }
  if (alias_scopes) {
  odsState.addAttribute(getAliasScopesAttrName(odsState.name), alias_scopes);
  }
  if (noalias_scopes) {
  odsState.addAttribute(getNoaliasScopesAttrName(odsState.name), noalias_scopes);
  }
  if (alignment) {
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  }
  if (volatile_) {
  odsState.addAttribute(getVolatile_AttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  if (nontemporal) {
  odsState.addAttribute(getNontemporalAttrName(odsState.name), odsBuilder.getUnitAttr());
  }
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StoreOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_access_groups;
  ::mlir::Attribute tblgen_alias_scopes;
  ::mlir::Attribute tblgen_alignment;
  ::mlir::Attribute tblgen_noalias_scopes;
  ::mlir::Attribute tblgen_nontemporal;
  ::mlir::Attribute tblgen_volatile_;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getAccessGroupsAttrName()) {
      tblgen_access_groups = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAliasScopesAttrName()) {
      tblgen_alias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNoaliasScopesAttrName()) {
      tblgen_noalias_scopes = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getNontemporalAttrName()) {
      tblgen_nontemporal = namedAttrIt->getValue();
    }
    else if (namedAttrIt->getName() == getVolatile_AttrName()) {
      tblgen_volatile_ = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps22(*this, tblgen_access_groups, "access_groups")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps22(*this, tblgen_alias_scopes, "alias_scopes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps22(*this, tblgen_noalias_scopes, "noalias_scopes")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps2(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_volatile_, "volatile_")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps14(*this, tblgen_nontemporal, "nontemporal")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps19(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps18(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StoreOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void StoreOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::StoreOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SubOp definitions
//===----------------------------------------------------------------------===//

SubOpAdaptor::SubOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.sub", odsAttrs.getContext());
}

SubOpAdaptor::SubOpAdaptor(SubOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SubOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SubOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SubOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr SubOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SubOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SubOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SubOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value SubOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SubOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SubOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SubOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SubOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SubOp::getRes() {
  return *getODSResults(0).begin();
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SubOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SubOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SubOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SubOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SubOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult SubOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult SubOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SubOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void SubOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SubOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SwitchOp definitions
//===----------------------------------------------------------------------===//

SwitchOpAdaptor::SwitchOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.switch", odsAttrs.getContext());
}

SwitchOpAdaptor::SwitchOpAdaptor(SwitchOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SwitchOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SwitchOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  assert(odsAttrs && "missing segment size attribute for op");
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, SwitchOp::getOperandSegmentSizesAttrName(*odsOpName)).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::ValueRange SwitchOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::ValueRange SwitchOpAdaptor::getDefaultOperands() {
  return getODSOperands(1);
}

::llvm::SmallVector<::mlir::ValueRange> SwitchOpAdaptor::getCaseOperands() {
  auto tblgenTmpOperands = getODSOperands(2);
  auto sizeAttrValues = getCaseOperandSegments().getValues<uint32_t>();
  auto sizeAttrIt = sizeAttrValues.begin();

  ::llvm::SmallVector<::mlir::ValueRange> tblgenTmpOperandGroups;
  for (int i = 0, e = ::llvm::size(sizeAttrValues); i < e; ++i, ++sizeAttrIt) {
    tblgenTmpOperandGroups.push_back(tblgenTmpOperands.take_front(*sizeAttrIt));
    tblgenTmpOperands = tblgenTmpOperands.drop_front(*sizeAttrIt);
  }
  return tblgenTmpOperandGroups;
}

::mlir::DictionaryAttr SwitchOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::ElementsAttr SwitchOpAdaptor::getCaseValuesAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, SwitchOp::getCaseValuesAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ElementsAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ElementsAttr > SwitchOpAdaptor::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

::mlir::ElementsAttr SwitchOpAdaptor::getCaseOperandSegmentsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, SwitchOp::getCaseOperandSegmentsAttrName(*odsOpName)).cast<::mlir::ElementsAttr>();
    return attr;
}

::mlir::ElementsAttr SwitchOpAdaptor::getCaseOperandSegments() {
  auto attr = getCaseOperandSegmentsAttr();
  return attr;
}

::mlir::ElementsAttr SwitchOpAdaptor::getBranchWeightsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, SwitchOp::getBranchWeightsAttrName(*odsOpName)).dyn_cast_or_null<::mlir::ElementsAttr>();
    return attr;
}

::llvm::Optional< ::mlir::ElementsAttr > SwitchOpAdaptor::getBranchWeights() {
  auto attr = getBranchWeightsAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

::mlir::LogicalResult SwitchOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_case_operand_segments;
  ::mlir::Attribute tblgen_branch_weights;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.switch' op ""requires attribute 'case_operand_segments'");
    if (namedAttrIt->getName() == SwitchOp::getCaseOperandSegmentsAttrName(*odsOpName)) {
      tblgen_case_operand_segments = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == SwitchOp::getBranchWeightsAttrName(*odsOpName)) {
      tblgen_branch_weights = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_case_values;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.switch' op ""requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == SwitchOp::getOperandSegmentSizesAttrName(*odsOpName)) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == SwitchOp::getCaseValuesAttrName(*odsOpName)) {
      tblgen_case_values = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  {
    auto sizeAttr = tblgen_operand_segment_sizes.cast<::mlir::DenseIntElementsAttr>();
    auto numElements =
        sizeAttr.getType().cast<::mlir::ShapedType>().getNumElements();
    if (numElements != 3)
      return emitError(loc, "'llvm.switch' op ""'operand_segment_sizes' attribute for specifying operand segments must have 3 "
                "elements, but got ") << numElements;
  }

  if (tblgen_case_values && !((tblgen_case_values.isa<::mlir::ElementsAttr>())))
    return emitError(loc, "'llvm.switch' op ""attribute 'case_values' failed to satisfy constraint: constant vector/tensor attribute");

  if (tblgen_case_operand_segments && !((tblgen_case_operand_segments.isa<::mlir::ElementsAttr>())))
    return emitError(loc, "'llvm.switch' op ""attribute 'case_operand_segments' failed to satisfy constraint: constant vector/tensor attribute");

  if (tblgen_branch_weights && !((tblgen_branch_weights.isa<::mlir::ElementsAttr>())))
    return emitError(loc, "'llvm.switch' op ""attribute 'branch_weights' failed to satisfy constraint: constant vector/tensor attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SwitchOp::getODSOperandIndexAndLength(unsigned index) {
  auto sizeAttr = ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName()).cast<::mlir::DenseIntElementsAttr>();

  const uint32_t *sizeAttrValueIt = &*sizeAttr.value_begin<uint32_t>();
  if (sizeAttr.isSplat())
    return {*sizeAttrValueIt * index, *sizeAttrValueIt};

  unsigned start = 0;
  for (unsigned i = 0; i < index; ++i)
    start += sizeAttrValueIt[i];
  return {start, sizeAttrValueIt[index]};
}

::mlir::Operation::operand_range SwitchOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SwitchOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::Operation::operand_range SwitchOp::getDefaultOperands() {
  return getODSOperands(1);
}

::mlir::OperandRangeRange SwitchOp::getCaseOperands() {
  return getODSOperands(2).split(getCaseOperandSegmentsAttr());
}

::mlir::MutableOperandRange SwitchOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(0u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRange SwitchOp::getDefaultOperandsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(1u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange;
}

::mlir::MutableOperandRangeRange SwitchOp::getCaseOperandsMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second, ::mlir::MutableOperandRange::OperandSegment(2u, *::mlir::impl::getNamedAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getOperandSegmentSizesAttrName())));
  return mutableRange.split(*(*this)->getAttrDictionary().getNamed(getCaseOperandSegmentsAttrName()));
}

std::pair<unsigned, unsigned> SwitchOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SwitchOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Block *SwitchOp::getDefaultDestination() {
  return (*this)->getSuccessor(0);
}

::mlir::SuccessorRange SwitchOp::getCaseDestinations() {
  return {std::next((*this)->successor_begin(), 1), (*this)->successor_end()};
}

::mlir::ElementsAttr SwitchOp::getCaseValuesAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getCaseValuesAttrName()).dyn_cast_or_null<::mlir::ElementsAttr>();
}

::llvm::Optional< ::mlir::ElementsAttr > SwitchOp::getCaseValues() {
  auto attr = getCaseValuesAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

::mlir::ElementsAttr SwitchOp::getCaseOperandSegmentsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getCaseOperandSegmentsAttrName()).cast<::mlir::ElementsAttr>();
}

::mlir::ElementsAttr SwitchOp::getCaseOperandSegments() {
  auto attr = getCaseOperandSegmentsAttr();
  return attr;
}

::mlir::ElementsAttr SwitchOp::getBranchWeightsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getBranchWeightsAttrName()).dyn_cast_or_null<::mlir::ElementsAttr>();
}

::llvm::Optional< ::mlir::ElementsAttr > SwitchOp::getBranchWeights() {
  auto attr = getBranchWeightsAttr();
  return attr ? ::llvm::Optional< ::mlir::ElementsAttr >(attr) : (::llvm::None);
}

void SwitchOp::setCaseValuesAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(getCaseValuesAttrName(), attr);
}

void SwitchOp::setCaseOperandSegmentsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(getCaseOperandSegmentsAttrName(), attr);
}

void SwitchOp::setBranchWeightsAttr(::mlir::ElementsAttr attr) {
  (*this)->setAttr(getBranchWeightsAttrName(), attr);
}

::mlir::Attribute SwitchOp::removeCase_valuesAttr() {
  return (*this)->removeAttr(getCaseValuesAttrName());
}

::mlir::Attribute SwitchOp::removeBranch_weightsAttr() {
  return (*this)->removeAttr(getBranchWeightsAttrName());
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, SuccessorRange destinations, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(operands);
    odsState.addSuccessors(destinations);
    odsState.addAttributes(attributes);
  
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::ValueRange defaultOperands, ::llvm::ArrayRef<::mlir::ValueRange> caseOperands, /*optional*/::mlir::ElementsAttr case_values, /*optional*/::mlir::ElementsAttr branch_weights, ::mlir::Block *defaultDestination, ::mlir::BlockRange caseDestinations) {
  odsState.addOperands(value);
  odsState.addOperands(defaultOperands);
  for (::mlir::ValueRange range : caseOperands)
   odsState.addOperands(range);
  {
    ::llvm::SmallVector<int32_t> rangeSegments;
    for (::mlir::ValueRange range : caseOperands)
      rangeSegments.push_back(range.size());
    odsState.addAttribute(getCaseOperandSegmentsAttrName(odsState.name), odsBuilder.getI32TensorAttr(rangeSegments));  }
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(defaultOperands.size()), static_cast<int32_t>(std::accumulate(caseOperands.begin(), caseOperands.end(), 0, [](int32_t curSum, ::mlir::ValueRange range) { return curSum + range.size(); }))}));
  if (case_values) {
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), case_values);
  }
  if (branch_weights) {
  odsState.addAttribute(getBranchWeightsAttrName(odsState.name), branch_weights);
  }
  odsState.addSuccessors(defaultDestination);
  odsState.addSuccessors(caseDestinations);
}

void SwitchOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::ValueRange defaultOperands, ::llvm::ArrayRef<::mlir::ValueRange> caseOperands, /*optional*/::mlir::ElementsAttr case_values, /*optional*/::mlir::ElementsAttr branch_weights, ::mlir::Block *defaultDestination, ::mlir::BlockRange caseDestinations) {
  odsState.addOperands(value);
  odsState.addOperands(defaultOperands);
  for (::mlir::ValueRange range : caseOperands)
   odsState.addOperands(range);
  {
    ::llvm::SmallVector<int32_t> rangeSegments;
    for (::mlir::ValueRange range : caseOperands)
      rangeSegments.push_back(range.size());
    odsState.addAttribute(getCaseOperandSegmentsAttrName(odsState.name), odsBuilder.getI32TensorAttr(rangeSegments));  }
  odsState.addAttribute(getOperandSegmentSizesAttrName(odsState.name), odsBuilder.getI32VectorAttr({1, static_cast<int32_t>(defaultOperands.size()), static_cast<int32_t>(std::accumulate(caseOperands.begin(), caseOperands.end(), 0, [](int32_t curSum, ::mlir::ValueRange range) { return curSum + range.size(); }))}));
  if (case_values) {
  odsState.addAttribute(getCaseValuesAttrName(odsState.name), case_values);
  }
  if (branch_weights) {
  odsState.addAttribute(getBranchWeightsAttrName(odsState.name), branch_weights);
  }
  odsState.addSuccessors(defaultDestination);
  odsState.addSuccessors(caseDestinations);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SwitchOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SwitchOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_case_operand_segments;
  ::mlir::Attribute tblgen_branch_weights;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'case_operand_segments'");
    if (namedAttrIt->getName() == getCaseOperandSegmentsAttrName()) {
      tblgen_case_operand_segments = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getBranchWeightsAttrName()) {
      tblgen_branch_weights = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_operand_segment_sizes;
  ::mlir::Attribute tblgen_case_values;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'operand_segment_sizes'");
    if (namedAttrIt->getName() == getOperandSegmentSizesAttrName()) {
      tblgen_operand_segment_sizes = namedAttrIt->getValue();
      break;
    }
    else if (namedAttrIt->getName() == getCaseValuesAttrName()) {
      tblgen_case_values = namedAttrIt->getValue();
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

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps7(*this, tblgen_case_values, "case_values")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps7(*this, tblgen_case_operand_segments, "case_operand_segments")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMOps7(*this, tblgen_branch_weights, "branch_weights")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps20(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);
    if (::mlir::failed(::mlir::OpTrait::impl::verifyValueSizeAttr(*this, "case_operand_segments", "caseOperands", valueGroup2.size())))
      return ::mlir::failure();

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps20(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
  }
  return ::mlir::success();
}

::mlir::LogicalResult SwitchOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult SwitchOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Block *defaultDestinationSuccessor = nullptr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> defaultOperandsOperands;
  ::llvm::SMLoc defaultOperandsOperandsLoc;
  (void)defaultOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> defaultOperandsTypes;
  ::mlir::ElementsAttr case_valuesAttr;
  ::llvm::SmallVector<::mlir::Block *, 2> caseDestinationsSuccessors;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> caseOperandsOperands;
    llvm::SmallVector<int32_t> caseOperandsOperandGroupSizes;
  ::llvm::SMLoc caseOperandsOperandsLoc;
  (void)caseOperandsOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> caseOperandsTypes;

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::IntegerType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    valueRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  if (parser.parseSuccessor(defaultDestinationSuccessor))
    return ::mlir::failure();
  if (succeeded(parser.parseOptionalLParen())) {

  defaultOperandsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(defaultOperandsOperands))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(defaultOperandsTypes))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  }
  if (parser.parseLSquare())
    return ::mlir::failure();
  {
    caseOperandsOperandsLoc = parser.getCurrentLocation();
    ::llvm::SmallVector<::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand>> caseOperandsOperandGroups;
    ::llvm::SmallVector<llvm::SmallVector<::mlir::Type>> caseOperandsTypeGroups;
    if (parseSwitchOpCases(parser, valueRawTypes[0], case_valuesAttr, caseDestinationsSuccessors, caseOperandsOperandGroups, caseOperandsTypeGroups))
      return ::mlir::failure();
    if (case_valuesAttr)
      result.addAttribute("case_values", case_valuesAttr);
    for (const auto &subRange : caseOperandsOperandGroups) {
      caseOperandsOperands.append(subRange.begin(), subRange.end());
      caseOperandsOperandGroupSizes.push_back(subRange.size());
    }
    for (const auto &subRange : caseOperandsTypeGroups)
      caseOperandsTypes.append(subRange.begin(), subRange.end());
  }
  if (parser.parseRSquare())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  result.addSuccessors(defaultDestinationSuccessor);
  result.addSuccessors(caseDestinationsSuccessors);
  result.addAttribute("operand_segment_sizes", parser.getBuilder().getI32VectorAttr({1, static_cast<int32_t>(defaultOperandsOperands.size()), static_cast<int32_t>(caseOperandsOperands.size())}));
  result.addAttribute("case_operand_segments", parser.getBuilder().getI32TensorAttr(caseOperandsOperandGroupSizes));
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(defaultOperandsOperands, defaultOperandsTypes, defaultOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(caseOperandsOperands, caseOperandsTypes, caseOperandsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void SwitchOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getValue().getType();
    if (auto validType = type.dyn_cast<::mlir::IntegerType>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultDestination();
  if (!getDefaultOperands().empty()) {
  _odsPrinter << "(";
  _odsPrinter << getDefaultOperands();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter << getDefaultOperands().getTypes();
  _odsPrinter << ")";
  }
  _odsPrinter << ' ' << "[";
  _odsPrinter.printNewline();
  printSwitchOpCases(_odsPrinter, *this, getValue().getType(), getCaseValuesAttr(), getCaseDestinations(), getCaseOperands(), getCaseOperands().getTypes());
  _odsPrinter << "]";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"operand_segment_sizes", "case_operand_segments", "case_values"});
}

void SwitchOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SwitchOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::TruncOp definitions
//===----------------------------------------------------------------------===//

TruncOpAdaptor::TruncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.trunc", odsAttrs.getContext());
}

TruncOpAdaptor::TruncOpAdaptor(TruncOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange TruncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> TruncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange TruncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TruncOpAdaptor::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr TruncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult TruncOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TruncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TruncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TruncOp::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange TruncOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TruncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TruncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value TruncOp::getRes() {
  return *getODSResults(0).begin();
}

void TruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void TruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void TruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TruncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TruncOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TruncOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult TruncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TruncOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void TruncOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::TruncOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UDivOp definitions
//===----------------------------------------------------------------------===//

UDivOpAdaptor::UDivOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.udiv", odsAttrs.getContext());
}

UDivOpAdaptor::UDivOpAdaptor(UDivOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange UDivOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UDivOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UDivOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UDivOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value UDivOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr UDivOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UDivOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UDivOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value UDivOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange UDivOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange UDivOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> UDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UDivOp::getRes() {
  return *getODSResults(0).begin();
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(UDivOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void UDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(UDivOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult UDivOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult UDivOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult UDivOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult UDivOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UDivOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void UDivOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::UDivOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UIToFPOp definitions
//===----------------------------------------------------------------------===//

UIToFPOpAdaptor::UIToFPOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.uitofp", odsAttrs.getContext());
}

UIToFPOpAdaptor::UIToFPOpAdaptor(UIToFPOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange UIToFPOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UIToFPOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UIToFPOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UIToFPOpAdaptor::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr UIToFPOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UIToFPOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UIToFPOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UIToFPOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UIToFPOp::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange UIToFPOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> UIToFPOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UIToFPOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UIToFPOp::getRes() {
  return *getODSResults(0).begin();
}

void UIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void UIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void UIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UIToFPOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UIToFPOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps14(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult UIToFPOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult UIToFPOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void UIToFPOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void UIToFPOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::UIToFPOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::URemOp definitions
//===----------------------------------------------------------------------===//

URemOpAdaptor::URemOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.urem", odsAttrs.getContext());
}

URemOpAdaptor::URemOpAdaptor(URemOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange URemOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> URemOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange URemOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value URemOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value URemOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr URemOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult URemOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> URemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range URemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value URemOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value URemOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange URemOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange URemOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> URemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range URemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value URemOp::getRes() {
  return *getODSResults(0).begin();
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(URemOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void URemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void URemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(URemOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult URemOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult URemOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult URemOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult URemOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void URemOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void URemOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::URemOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UndefOp definitions
//===----------------------------------------------------------------------===//

UndefOpAdaptor::UndefOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.mlir.undef", odsAttrs.getContext());
}

UndefOpAdaptor::UndefOpAdaptor(UndefOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange UndefOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UndefOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UndefOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr UndefOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UndefOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UndefOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UndefOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> UndefOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UndefOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UndefOp::getRes() {
  return *getODSResults(0).begin();
}

void UndefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void UndefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void UndefOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UndefOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UndefOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps10(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult UndefOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult UndefOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  return ::mlir::success();
}

void UndefOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void UndefOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::UndefOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UnreachableOp definitions
//===----------------------------------------------------------------------===//

UnreachableOpAdaptor::UnreachableOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.unreachable", odsAttrs.getContext());
}

UnreachableOpAdaptor::UnreachableOpAdaptor(UnreachableOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange UnreachableOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UnreachableOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UnreachableOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr UnreachableOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UnreachableOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UnreachableOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UnreachableOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> UnreachableOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UnreachableOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void UnreachableOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void UnreachableOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UnreachableOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UnreachableOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::mlir::LogicalResult UnreachableOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult UnreachableOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  return ::mlir::success();
}

void UnreachableOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::UnreachableOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::XOrOp definitions
//===----------------------------------------------------------------------===//

XOrOpAdaptor::XOrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.xor", odsAttrs.getContext());
}

XOrOpAdaptor::XOrOpAdaptor(XOrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange XOrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> XOrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange XOrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value XOrOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value XOrOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr XOrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult XOrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> XOrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range XOrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value XOrOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value XOrOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange XOrOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange XOrOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> XOrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range XOrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value XOrOp::getRes() {
  return *getODSResults(0).begin();
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(res);
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(XOrOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void XOrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void XOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(XOrOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult XOrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult XOrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult XOrOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

::mlir::ParseResult XOrOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  lhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(lhsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  rhsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(rhsRawOperands[0]))
    return ::mlir::failure();
  {
    if (parseLLVMOpAttrs(parser, result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, resTypes[0], result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void XOrOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter << ' ';
  printLLVMOpAttrs(_odsPrinter, *this, getOperation()->getAttrDictionary());
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void XOrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::XOrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ZExtOp definitions
//===----------------------------------------------------------------------===//

ZExtOpAdaptor::ZExtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.zext", odsAttrs.getContext());
}

ZExtOpAdaptor::ZExtOpAdaptor(ZExtOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ZExtOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ZExtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ZExtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ZExtOpAdaptor::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ZExtOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ZExtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ZExtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ZExtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ZExtOp::getArg() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ZExtOp::getArgMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ZExtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ZExtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ZExtOp::getRes() {
  return *getODSResults(0).begin();
}

void ZExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void ZExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value arg) {
  odsState.addOperands(arg);
  odsState.addTypes(res);
}

void ZExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value arg) {
  odsState.addOperands(arg);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ZExtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ZExtOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ZExtOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ZExtOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand argRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> argOperands(argRawOperands);  ::llvm::SMLoc argOperandsLoc;
  (void)argOperandsLoc;
  ::mlir::Type argRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> argTypes(argRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  argOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(argRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    argRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(argOperands, argTypes, argOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ZExtOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getArg();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getArg().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

void ZExtOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ZExtOp)


#endif  // GET_OP_CLASSES

