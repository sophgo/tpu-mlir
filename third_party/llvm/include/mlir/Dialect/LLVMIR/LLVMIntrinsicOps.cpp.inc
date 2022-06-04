/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::LLVM::AssumeOp,
::mlir::LLVM::BitReverseOp,
::mlir::LLVM::CopySignOp,
::mlir::LLVM::CoroAlignOp,
::mlir::LLVM::CoroBeginOp,
::mlir::LLVM::CoroEndOp,
::mlir::LLVM::CoroFreeOp,
::mlir::LLVM::CoroIdOp,
::mlir::LLVM::CoroResumeOp,
::mlir::LLVM::CoroSaveOp,
::mlir::LLVM::CoroSizeOp,
::mlir::LLVM::CoroSuspendOp,
::mlir::LLVM::CosOp,
::mlir::LLVM::CountLeadingZerosOp,
::mlir::LLVM::CountTrailingZerosOp,
::mlir::LLVM::CtPopOp,
::mlir::LLVM::EhTypeidForOp,
::mlir::LLVM::Exp2Op,
::mlir::LLVM::ExpOp,
::mlir::LLVM::FAbsOp,
::mlir::LLVM::FCeilOp,
::mlir::LLVM::FFloorOp,
::mlir::LLVM::FMAOp,
::mlir::LLVM::FMulAddOp,
::mlir::LLVM::GetActiveLaneMaskOp,
::mlir::LLVM::Log10Op,
::mlir::LLVM::Log2Op,
::mlir::LLVM::LogOp,
::mlir::LLVM::MaskedLoadOp,
::mlir::LLVM::MaskedStoreOp,
::mlir::LLVM::MatrixColumnMajorLoadOp,
::mlir::LLVM::MatrixColumnMajorStoreOp,
::mlir::LLVM::MatrixMultiplyOp,
::mlir::LLVM::MatrixTransposeOp,
::mlir::LLVM::MaxNumOp,
::mlir::LLVM::MaximumOp,
::mlir::LLVM::MemcpyInlineOp,
::mlir::LLVM::MemcpyOp,
::mlir::LLVM::MemmoveOp,
::mlir::LLVM::MemsetOp,
::mlir::LLVM::MinNumOp,
::mlir::LLVM::MinimumOp,
::mlir::LLVM::PowIOp,
::mlir::LLVM::PowOp,
::mlir::LLVM::Prefetch,
::mlir::LLVM::SAddWithOverflowOp,
::mlir::LLVM::SMaxOp,
::mlir::LLVM::SMinOp,
::mlir::LLVM::SMulWithOverflowOp,
::mlir::LLVM::SSubWithOverflowOp,
::mlir::LLVM::SinOp,
::mlir::LLVM::SqrtOp,
::mlir::LLVM::StackRestoreOp,
::mlir::LLVM::StackSaveOp,
::mlir::LLVM::StepVectorOp,
::mlir::LLVM::UAddWithOverflowOp,
::mlir::LLVM::UMaxOp,
::mlir::LLVM::UMinOp,
::mlir::LLVM::UMulWithOverflowOp,
::mlir::LLVM::USubWithOverflowOp,
::mlir::LLVM::VPAShrOp,
::mlir::LLVM::VPAddOp,
::mlir::LLVM::VPAndOp,
::mlir::LLVM::VPFAddOp,
::mlir::LLVM::VPFDivOp,
::mlir::LLVM::VPFMulOp,
::mlir::LLVM::VPFNegOp,
::mlir::LLVM::VPFPExtOp,
::mlir::LLVM::VPFPToSIOp,
::mlir::LLVM::VPFPToUIOp,
::mlir::LLVM::VPFPTruncOp,
::mlir::LLVM::VPFRemOp,
::mlir::LLVM::VPFSubOp,
::mlir::LLVM::VPFmaOp,
::mlir::LLVM::VPIntToPtrOp,
::mlir::LLVM::VPLShrOp,
::mlir::LLVM::VPLoadOp,
::mlir::LLVM::VPMergeMinOp,
::mlir::LLVM::VPMulOp,
::mlir::LLVM::VPOrOp,
::mlir::LLVM::VPPtrToIntOp,
::mlir::LLVM::VPReduceAddOp,
::mlir::LLVM::VPReduceAndOp,
::mlir::LLVM::VPReduceFAddOp,
::mlir::LLVM::VPReduceFMaxOp,
::mlir::LLVM::VPReduceFMinOp,
::mlir::LLVM::VPReduceFMulOp,
::mlir::LLVM::VPReduceMulOp,
::mlir::LLVM::VPReduceOrOp,
::mlir::LLVM::VPReduceSMaxOp,
::mlir::LLVM::VPReduceSMinOp,
::mlir::LLVM::VPReduceUMaxOp,
::mlir::LLVM::VPReduceUMinOp,
::mlir::LLVM::VPReduceXorOp,
::mlir::LLVM::VPSDivOp,
::mlir::LLVM::VPSExtOp,
::mlir::LLVM::VPSIToFPOp,
::mlir::LLVM::VPSRemOp,
::mlir::LLVM::VPSelectMinOp,
::mlir::LLVM::VPShlOp,
::mlir::LLVM::VPStoreOp,
::mlir::LLVM::VPStridedLoadOp,
::mlir::LLVM::VPStridedStoreOp,
::mlir::LLVM::VPSubOp,
::mlir::LLVM::VPTruncOp,
::mlir::LLVM::VPUDivOp,
::mlir::LLVM::VPUIToFPOp,
::mlir::LLVM::VPURemOp,
::mlir::LLVM::VPXorOp,
::mlir::LLVM::VPZExtOp,
::mlir::LLVM::masked_compressstore,
::mlir::LLVM::masked_expandload,
::mlir::LLVM::masked_gather,
::mlir::LLVM::masked_scatter,
::mlir::LLVM::vector_reduce_add,
::mlir::LLVM::vector_reduce_and,
::mlir::LLVM::vector_reduce_fadd,
::mlir::LLVM::vector_reduce_fmax,
::mlir::LLVM::vector_reduce_fmin,
::mlir::LLVM::vector_reduce_fmul,
::mlir::LLVM::vector_reduce_mul,
::mlir::LLVM::vector_reduce_or,
::mlir::LLVM::vector_reduce_smax,
::mlir::LLVM::vector_reduce_smin,
::mlir::LLVM::vector_reduce_umax,
::mlir::LLVM::vector_reduce_umin,
::mlir::LLVM::vector_reduce_xor,
::mlir::LLVM::vscale
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace LLVM {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMIntrinsicOps0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::mlir::LLVM::isCompatibleOuterType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM dialect-compatible type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMIntrinsicOps1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMTokenType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM token type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMIntrinsicOps2(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((type.isa<::mlir::LLVM::LLVMPointerType>())) && (((type.cast<::mlir::LLVM::LLVMPointerType>().isOpaque())) || ((type.cast<::mlir::LLVM::LLVMPointerType>().getElementType().isSignlessInteger(8)))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer to 8-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMIntrinsicOps3(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(1)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMIntrinsicOps4(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isSignlessInteger(32)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be 32-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMIntrinsicOps5(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((::mlir::LLVM::isCompatibleVectorType(type)))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM dialect-compatible vector type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMIntrinsicOps6(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::IntegerType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM dialect-compatible vector of integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMIntrinsicOps7(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isSignlessInteger(1))))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM dialect-compatible vector of 1-bit signless integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMIntrinsicOps8(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::FloatType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM dialect-compatible vector of floating-point, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMIntrinsicOps9(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::LLVM::LLVMPointerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMIntrinsicOps10(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::mlir::LLVM::isCompatibleVectorType(type))) && ((::mlir::LLVM::getVectorElementType(type).isa<::mlir::LLVM::LLVMPointerType>())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be LLVM dialect-compatible vector of LLVM pointer type, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMIntrinsicOps11(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::IntegerType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be integer, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_LLVMIntrinsicOps12(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((type.isa<::mlir::FloatType>()))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be floating-point, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 32-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMIntrinsicOps1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !(((attr.isa<::mlir::IntegerAttr>())) && ((attr.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(1))))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: 1-bit signless integer attribute";
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_LLVMIntrinsicOps2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  if (attr && !((attr.isa<::mlir::BoolAttr>()))) {
    return op->emitOpError("attribute '") << attrName
        << "' failed to satisfy constraint: bool attribute";
  }
  return ::mlir::success();
}
} // namespace LLVM
} // namespace mlir
namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::AssumeOp definitions
//===----------------------------------------------------------------------===//

AssumeOpAdaptor::AssumeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.assume", odsAttrs.getContext());
}

AssumeOpAdaptor::AssumeOpAdaptor(AssumeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange AssumeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> AssumeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange AssumeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AssumeOpAdaptor::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr AssumeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult AssumeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AssumeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AssumeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value AssumeOp::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange AssumeOp::getCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AssumeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AssumeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void AssumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value cond) {
  odsState.addOperands(cond);
}

void AssumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond) {
  odsState.addOperands(cond);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AssumeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AssumeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AssumeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::AssumeOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::BitReverseOp definitions
//===----------------------------------------------------------------------===//

BitReverseOpAdaptor::BitReverseOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.bitreverse", odsAttrs.getContext());
}

BitReverseOpAdaptor::BitReverseOpAdaptor(BitReverseOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange BitReverseOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> BitReverseOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange BitReverseOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitReverseOpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr BitReverseOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult BitReverseOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> BitReverseOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range BitReverseOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitReverseOp::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange BitReverseOp::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> BitReverseOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range BitReverseOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value BitReverseOp::getRes() {
  return *getODSResults(0).begin();
}

void BitReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void BitReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(BitReverseOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void BitReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void BitReverseOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void BitReverseOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(BitReverseOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult BitReverseOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult BitReverseOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult BitReverseOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void BitReverseOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::BitReverseOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CopySignOp definitions
//===----------------------------------------------------------------------===//

CopySignOpAdaptor::CopySignOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.copysign", odsAttrs.getContext());
}

CopySignOpAdaptor::CopySignOpAdaptor(CopySignOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CopySignOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CopySignOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CopySignOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopySignOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value CopySignOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CopySignOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CopySignOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CopySignOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CopySignOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopySignOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value CopySignOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CopySignOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CopySignOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CopySignOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CopySignOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CopySignOp::getRes() {
  return *getODSResults(0).begin();
}

void CopySignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void CopySignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(CopySignOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void CopySignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CopySignOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CopySignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(CopySignOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult CopySignOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CopySignOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult CopySignOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void CopySignOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CopySignOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroAlignOp definitions
//===----------------------------------------------------------------------===//

CoroAlignOpAdaptor::CoroAlignOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.coro.align", odsAttrs.getContext());
}

CoroAlignOpAdaptor::CoroAlignOpAdaptor(CoroAlignOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CoroAlignOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroAlignOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroAlignOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CoroAlignOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroAlignOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CoroAlignOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroAlignOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CoroAlignOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroAlignOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroAlignOp::getRes() {
  return *getODSResults(0).begin();
}

void CoroAlignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void CoroAlignOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroAlignOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroAlignOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CoroAlignOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CoroAlignOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void CoroAlignOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
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

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CoroAlignOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroBeginOp definitions
//===----------------------------------------------------------------------===//

CoroBeginOpAdaptor::CoroBeginOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.coro.begin", odsAttrs.getContext());
}

CoroBeginOpAdaptor::CoroBeginOpAdaptor(CoroBeginOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CoroBeginOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroBeginOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroBeginOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroBeginOpAdaptor::getToken() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroBeginOpAdaptor::getMem() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CoroBeginOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroBeginOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CoroBeginOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroBeginOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroBeginOp::getToken() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroBeginOp::getMem() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CoroBeginOp::getTokenMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CoroBeginOp::getMemMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CoroBeginOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroBeginOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroBeginOp::getRes() {
  return *getODSResults(0).begin();
}

void CoroBeginOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value token, ::mlir::Value mem) {
  odsState.addOperands(token);
  odsState.addOperands(mem);
  odsState.addTypes(res);
}

void CoroBeginOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value token, ::mlir::Value mem) {
  odsState.addOperands(token);
  odsState.addOperands(mem);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroBeginOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroBeginOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CoroBeginOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CoroBeginOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand tokenRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tokenOperands(tokenRawOperands);  ::llvm::SMLoc tokenOperandsLoc;
  (void)tokenOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand memRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> memOperands(memRawOperands);  ::llvm::SMLoc memOperandsLoc;
  (void)memOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  tokenOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tokenRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  memOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(memRawOperands[0]))
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
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMTokenType::get(parser.getBuilder().getContext());
  ::mlir::Type odsBuildableType1 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8), 0);
  result.addTypes(resTypes);
  if (parser.resolveOperands(tokenOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(memOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroBeginOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getToken();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMem();
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

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CoroBeginOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroEndOp definitions
//===----------------------------------------------------------------------===//

CoroEndOpAdaptor::CoroEndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.coro.end", odsAttrs.getContext());
}

CoroEndOpAdaptor::CoroEndOpAdaptor(CoroEndOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CoroEndOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroEndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroEndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroEndOpAdaptor::getHandle() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroEndOpAdaptor::getUnwind() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CoroEndOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroEndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CoroEndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroEndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroEndOp::getHandle() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroEndOp::getUnwind() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CoroEndOp::getHandleMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CoroEndOp::getUnwindMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CoroEndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroEndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroEndOp::getRes() {
  return *getODSResults(0).begin();
}

void CoroEndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value handle, ::mlir::Value unwind) {
  odsState.addOperands(handle);
  odsState.addOperands(unwind);
  odsState.addTypes(res);
}

void CoroEndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value handle, ::mlir::Value unwind) {
  odsState.addOperands(handle);
  odsState.addOperands(unwind);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroEndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroEndOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CoroEndOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CoroEndOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand handleRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> handleOperands(handleRawOperands);  ::llvm::SMLoc handleOperandsLoc;
  (void)handleOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand unwindRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> unwindOperands(unwindRawOperands);  ::llvm::SMLoc unwindOperandsLoc;
  (void)unwindOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  handleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(handleRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  unwindOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(unwindRawOperands[0]))
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
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8), 0);
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIntegerType(1);
  result.addTypes(resTypes);
  if (parser.resolveOperands(handleOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(unwindOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroEndOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getHandle();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getUnwind();
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

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CoroEndOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroFreeOp definitions
//===----------------------------------------------------------------------===//

CoroFreeOpAdaptor::CoroFreeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.coro.free", odsAttrs.getContext());
}

CoroFreeOpAdaptor::CoroFreeOpAdaptor(CoroFreeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CoroFreeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroFreeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroFreeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroFreeOpAdaptor::getId() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroFreeOpAdaptor::getHandle() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CoroFreeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroFreeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CoroFreeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroFreeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroFreeOp::getId() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroFreeOp::getHandle() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CoroFreeOp::getIdMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CoroFreeOp::getHandleMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CoroFreeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroFreeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroFreeOp::getRes() {
  return *getODSResults(0).begin();
}

void CoroFreeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value id, ::mlir::Value handle) {
  odsState.addOperands(id);
  odsState.addOperands(handle);
  odsState.addTypes(res);
}

void CoroFreeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value id, ::mlir::Value handle) {
  odsState.addOperands(id);
  odsState.addOperands(handle);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroFreeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroFreeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CoroFreeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CoroFreeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand idRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> idOperands(idRawOperands);  ::llvm::SMLoc idOperandsLoc;
  (void)idOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand handleRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> handleOperands(handleRawOperands);  ::llvm::SMLoc handleOperandsLoc;
  (void)handleOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  idOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(idRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  handleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(handleRawOperands[0]))
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
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMTokenType::get(parser.getBuilder().getContext());
  ::mlir::Type odsBuildableType1 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8), 0);
  result.addTypes(resTypes);
  if (parser.resolveOperands(idOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(handleOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroFreeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getId();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getHandle();
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

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CoroFreeOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroIdOp definitions
//===----------------------------------------------------------------------===//

CoroIdOpAdaptor::CoroIdOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.coro.id", odsAttrs.getContext());
}

CoroIdOpAdaptor::CoroIdOpAdaptor(CoroIdOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CoroIdOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroIdOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroIdOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroIdOpAdaptor::getAlign() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroIdOpAdaptor::getPromise() {
  return *getODSOperands(1).begin();
}

::mlir::Value CoroIdOpAdaptor::getCoroaddr() {
  return *getODSOperands(2).begin();
}

::mlir::Value CoroIdOpAdaptor::getFnaddrs() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr CoroIdOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroIdOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CoroIdOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroIdOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroIdOp::getAlign() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroIdOp::getPromise() {
  return *getODSOperands(1).begin();
}

::mlir::Value CoroIdOp::getCoroaddr() {
  return *getODSOperands(2).begin();
}

::mlir::Value CoroIdOp::getFnaddrs() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange CoroIdOp::getAlignMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CoroIdOp::getPromiseMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CoroIdOp::getCoroaddrMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CoroIdOp::getFnaddrsMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CoroIdOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroIdOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroIdOp::getRes() {
  return *getODSResults(0).begin();
}

void CoroIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value align, ::mlir::Value promise, ::mlir::Value coroaddr, ::mlir::Value fnaddrs) {
  odsState.addOperands(align);
  odsState.addOperands(promise);
  odsState.addOperands(coroaddr);
  odsState.addOperands(fnaddrs);
  odsState.addTypes(res);
}

void CoroIdOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value align, ::mlir::Value promise, ::mlir::Value coroaddr, ::mlir::Value fnaddrs) {
  odsState.addOperands(align);
  odsState.addOperands(promise);
  odsState.addOperands(coroaddr);
  odsState.addOperands(fnaddrs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroIdOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroIdOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CoroIdOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CoroIdOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand alignRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> alignOperands(alignRawOperands);  ::llvm::SMLoc alignOperandsLoc;
  (void)alignOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand promiseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> promiseOperands(promiseRawOperands);  ::llvm::SMLoc promiseOperandsLoc;
  (void)promiseOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand coroaddrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> coroaddrOperands(coroaddrRawOperands);  ::llvm::SMLoc coroaddrOperandsLoc;
  (void)coroaddrOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand fnaddrsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> fnaddrsOperands(fnaddrsRawOperands);  ::llvm::SMLoc fnaddrsOperandsLoc;
  (void)fnaddrsOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  alignOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(alignRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  promiseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(promiseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  coroaddrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(coroaddrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  fnaddrsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(fnaddrsRawOperands[0]))
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
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIntegerType(32);
  ::mlir::Type odsBuildableType1 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8), 0);
  result.addTypes(resTypes);
  if (parser.resolveOperands(alignOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(promiseOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(coroaddrOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(fnaddrsOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroIdOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getAlign();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPromise();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getCoroaddr();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getFnaddrs();
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

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CoroIdOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroResumeOp definitions
//===----------------------------------------------------------------------===//

CoroResumeOpAdaptor::CoroResumeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.coro.resume", odsAttrs.getContext());
}

CoroResumeOpAdaptor::CoroResumeOpAdaptor(CoroResumeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CoroResumeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroResumeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroResumeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroResumeOpAdaptor::getHandle() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CoroResumeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroResumeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CoroResumeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroResumeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroResumeOp::getHandle() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CoroResumeOp::getHandleMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CoroResumeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroResumeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void CoroResumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value handle) {
  odsState.addOperands(handle);
}

void CoroResumeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value handle) {
  odsState.addOperands(handle);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroResumeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroResumeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CoroResumeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CoroResumeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand handleRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> handleOperands(handleRawOperands);  ::llvm::SMLoc handleOperandsLoc;
  (void)handleOperandsLoc;

  handleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(handleRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8), 0);
  if (parser.resolveOperands(handleOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroResumeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getHandle();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CoroResumeOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroSaveOp definitions
//===----------------------------------------------------------------------===//

CoroSaveOpAdaptor::CoroSaveOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.coro.save", odsAttrs.getContext());
}

CoroSaveOpAdaptor::CoroSaveOpAdaptor(CoroSaveOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CoroSaveOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroSaveOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroSaveOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSaveOpAdaptor::getHandle() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CoroSaveOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroSaveOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CoroSaveOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroSaveOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSaveOp::getHandle() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CoroSaveOp::getHandleMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CoroSaveOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroSaveOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSaveOp::getRes() {
  return *getODSResults(0).begin();
}

void CoroSaveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value handle) {
  odsState.addOperands(handle);
  odsState.addTypes(res);
}

void CoroSaveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value handle) {
  odsState.addOperands(handle);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroSaveOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroSaveOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CoroSaveOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CoroSaveOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand handleRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> handleOperands(handleRawOperands);  ::llvm::SMLoc handleOperandsLoc;
  (void)handleOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  handleOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(handleRawOperands[0]))
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
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8), 0);
  result.addTypes(resTypes);
  if (parser.resolveOperands(handleOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroSaveOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getHandle();
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

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CoroSaveOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroSizeOp definitions
//===----------------------------------------------------------------------===//

CoroSizeOpAdaptor::CoroSizeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.coro.size", odsAttrs.getContext());
}

CoroSizeOpAdaptor::CoroSizeOpAdaptor(CoroSizeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CoroSizeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroSizeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroSizeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr CoroSizeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroSizeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CoroSizeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroSizeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> CoroSizeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroSizeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSizeOp::getRes() {
  return *getODSResults(0).begin();
}

void CoroSizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void CoroSizeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroSizeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroSizeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CoroSizeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CoroSizeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void CoroSizeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
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

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CoroSizeOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CoroSuspendOp definitions
//===----------------------------------------------------------------------===//

CoroSuspendOpAdaptor::CoroSuspendOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.coro.suspend", odsAttrs.getContext());
}

CoroSuspendOpAdaptor::CoroSuspendOpAdaptor(CoroSuspendOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CoroSuspendOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CoroSuspendOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CoroSuspendOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSuspendOpAdaptor::getSave() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroSuspendOpAdaptor::getFinal() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CoroSuspendOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CoroSuspendOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CoroSuspendOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CoroSuspendOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSuspendOp::getSave() {
  return *getODSOperands(0).begin();
}

::mlir::Value CoroSuspendOp::getFinal() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CoroSuspendOp::getSaveMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CoroSuspendOp::getFinalMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CoroSuspendOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CoroSuspendOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CoroSuspendOp::getRes() {
  return *getODSResults(0).begin();
}

void CoroSuspendOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value save, ::mlir::Value final) {
  odsState.addOperands(save);
  odsState.addOperands(final);
  odsState.addTypes(res);
}

void CoroSuspendOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value save, ::mlir::Value final) {
  odsState.addOperands(save);
  odsState.addOperands(final);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CoroSuspendOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CoroSuspendOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps1(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CoroSuspendOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult CoroSuspendOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand saveRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> saveOperands(saveRawOperands);  ::llvm::SMLoc saveOperandsLoc;
  (void)saveOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand finalRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> finalOperands(finalRawOperands);  ::llvm::SMLoc finalOperandsLoc;
  (void)finalOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  saveOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(saveRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  finalOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(finalRawOperands[0]))
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
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMTokenType::get(parser.getBuilder().getContext());
  ::mlir::Type odsBuildableType1 = parser.getBuilder().getIntegerType(1);
  result.addTypes(resTypes);
  if (parser.resolveOperands(saveOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(finalOperands, odsBuildableType1, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CoroSuspendOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSave();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getFinal();
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

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CoroSuspendOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CosOp definitions
//===----------------------------------------------------------------------===//

CosOpAdaptor::CosOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.cos", odsAttrs.getContext());
}

CosOpAdaptor::CosOpAdaptor(CosOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CosOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CosOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CosOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CosOpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CosOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CosOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CosOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CosOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CosOp::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CosOp::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CosOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CosOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CosOp::getRes() {
  return *getODSResults(0).begin();
}

void CosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void CosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(CosOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void CosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CosOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(CosOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult CosOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CosOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult CosOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void CosOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CosOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CountLeadingZerosOp definitions
//===----------------------------------------------------------------------===//

CountLeadingZerosOpAdaptor::CountLeadingZerosOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.ctlz", odsAttrs.getContext());
}

CountLeadingZerosOpAdaptor::CountLeadingZerosOpAdaptor(CountLeadingZerosOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CountLeadingZerosOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CountLeadingZerosOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CountLeadingZerosOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CountLeadingZerosOpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::Value CountLeadingZerosOpAdaptor::getZeroUndefined() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CountLeadingZerosOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CountLeadingZerosOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CountLeadingZerosOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CountLeadingZerosOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CountLeadingZerosOp::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::Value CountLeadingZerosOp::getZeroUndefined() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CountLeadingZerosOp::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CountLeadingZerosOp::getZeroUndefinedMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CountLeadingZerosOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CountLeadingZerosOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CountLeadingZerosOp::getRes() {
  return *getODSResults(0).begin();
}

void CountLeadingZerosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in, ::mlir::Value zero_undefined) {
  odsState.addOperands(in);
  odsState.addOperands(zero_undefined);
  odsState.addTypes(res);
}

void CountLeadingZerosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in, ::mlir::Value zero_undefined) {
  odsState.addOperands(in);
  odsState.addOperands(zero_undefined);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CountLeadingZerosOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CountLeadingZerosOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CountLeadingZerosOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void CountLeadingZerosOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CountLeadingZerosOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CountTrailingZerosOp definitions
//===----------------------------------------------------------------------===//

CountTrailingZerosOpAdaptor::CountTrailingZerosOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.cttz", odsAttrs.getContext());
}

CountTrailingZerosOpAdaptor::CountTrailingZerosOpAdaptor(CountTrailingZerosOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CountTrailingZerosOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CountTrailingZerosOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CountTrailingZerosOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CountTrailingZerosOpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::Value CountTrailingZerosOpAdaptor::getZeroUndefined() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr CountTrailingZerosOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CountTrailingZerosOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CountTrailingZerosOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CountTrailingZerosOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CountTrailingZerosOp::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::Value CountTrailingZerosOp::getZeroUndefined() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange CountTrailingZerosOp::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange CountTrailingZerosOp::getZeroUndefinedMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CountTrailingZerosOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CountTrailingZerosOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CountTrailingZerosOp::getRes() {
  return *getODSResults(0).begin();
}

void CountTrailingZerosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in, ::mlir::Value zero_undefined) {
  odsState.addOperands(in);
  odsState.addOperands(zero_undefined);
  odsState.addTypes(res);
}

void CountTrailingZerosOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in, ::mlir::Value zero_undefined) {
  odsState.addOperands(in);
  odsState.addOperands(zero_undefined);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CountTrailingZerosOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult CountTrailingZerosOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps3(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CountTrailingZerosOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void CountTrailingZerosOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CountTrailingZerosOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::CtPopOp definitions
//===----------------------------------------------------------------------===//

CtPopOpAdaptor::CtPopOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.ctpop", odsAttrs.getContext());
}

CtPopOpAdaptor::CtPopOpAdaptor(CtPopOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange CtPopOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> CtPopOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange CtPopOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CtPopOpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr CtPopOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult CtPopOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> CtPopOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range CtPopOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CtPopOp::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange CtPopOp::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> CtPopOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range CtPopOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value CtPopOp::getRes() {
  return *getODSResults(0).begin();
}

void CtPopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void CtPopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(CtPopOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void CtPopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void CtPopOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void CtPopOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(CtPopOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult CtPopOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult CtPopOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult CtPopOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void CtPopOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::CtPopOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::EhTypeidForOp definitions
//===----------------------------------------------------------------------===//

EhTypeidForOpAdaptor::EhTypeidForOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.eh.typeid.for", odsAttrs.getContext());
}

EhTypeidForOpAdaptor::EhTypeidForOpAdaptor(EhTypeidForOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange EhTypeidForOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> EhTypeidForOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange EhTypeidForOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EhTypeidForOpAdaptor::getTypeInfo() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr EhTypeidForOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult EhTypeidForOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> EhTypeidForOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range EhTypeidForOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EhTypeidForOp::getTypeInfo() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange EhTypeidForOp::getTypeInfoMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> EhTypeidForOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range EhTypeidForOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value EhTypeidForOp::getRes() {
  return *getODSResults(0).begin();
}

void EhTypeidForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value type_info) {
  odsState.addOperands(type_info);
  odsState.addTypes(res);
}

void EhTypeidForOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value type_info) {
  odsState.addOperands(type_info);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void EhTypeidForOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult EhTypeidForOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult EhTypeidForOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult EhTypeidForOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand type_infoRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> type_infoOperands(type_infoRawOperands);  ::llvm::SMLoc type_infoOperandsLoc;
  (void)type_infoOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  type_infoOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(type_infoRawOperands[0]))
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
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8), 0);
  result.addTypes(resTypes);
  if (parser.resolveOperands(type_infoOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void EhTypeidForOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getTypeInfo();
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

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::EhTypeidForOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::Exp2Op definitions
//===----------------------------------------------------------------------===//

Exp2OpAdaptor::Exp2OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.exp2", odsAttrs.getContext());
}

Exp2OpAdaptor::Exp2OpAdaptor(Exp2Op op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Exp2OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Exp2OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Exp2OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Exp2OpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr Exp2OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult Exp2OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Exp2Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Exp2Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Exp2Op::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange Exp2Op::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> Exp2Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Exp2Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Exp2Op::getRes() {
  return *getODSResults(0).begin();
}

void Exp2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void Exp2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(Exp2Op::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void Exp2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Exp2Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void Exp2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(Exp2Op::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult Exp2Op::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult Exp2Op::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult Exp2Op::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void Exp2Op::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::Exp2Op)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::ExpOp definitions
//===----------------------------------------------------------------------===//

ExpOpAdaptor::ExpOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.exp", odsAttrs.getContext());
}

ExpOpAdaptor::ExpOpAdaptor(ExpOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange ExpOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> ExpOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange ExpOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExpOpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr ExpOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult ExpOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ExpOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ExpOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExpOp::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange ExpOp::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ExpOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ExpOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value ExpOp::getRes() {
  return *getODSResults(0).begin();
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(ExpOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ExpOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void ExpOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(ExpOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult ExpOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ExpOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult ExpOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void ExpOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::ExpOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FAbsOp definitions
//===----------------------------------------------------------------------===//

FAbsOpAdaptor::FAbsOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.fabs", odsAttrs.getContext());
}

FAbsOpAdaptor::FAbsOpAdaptor(FAbsOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FAbsOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FAbsOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FAbsOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAbsOpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FAbsOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FAbsOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FAbsOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FAbsOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAbsOp::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FAbsOp::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FAbsOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FAbsOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FAbsOp::getRes() {
  return *getODSResults(0).begin();
}

void FAbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void FAbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FAbsOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FAbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FAbsOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FAbsOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FAbsOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FAbsOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FAbsOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FAbsOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void FAbsOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FAbsOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FCeilOp definitions
//===----------------------------------------------------------------------===//

FCeilOpAdaptor::FCeilOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.ceil", odsAttrs.getContext());
}

FCeilOpAdaptor::FCeilOpAdaptor(FCeilOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FCeilOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FCeilOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FCeilOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCeilOpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FCeilOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FCeilOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FCeilOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FCeilOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCeilOp::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FCeilOp::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FCeilOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FCeilOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FCeilOp::getRes() {
  return *getODSResults(0).begin();
}

void FCeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void FCeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FCeilOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FCeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FCeilOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FCeilOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FCeilOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FCeilOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FCeilOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FCeilOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void FCeilOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FCeilOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FFloorOp definitions
//===----------------------------------------------------------------------===//

FFloorOpAdaptor::FFloorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.floor", odsAttrs.getContext());
}

FFloorOpAdaptor::FFloorOpAdaptor(FFloorOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FFloorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FFloorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FFloorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FFloorOpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr FFloorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FFloorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FFloorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FFloorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FFloorOp::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange FFloorOp::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FFloorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FFloorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FFloorOp::getRes() {
  return *getODSResults(0).begin();
}

void FFloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void FFloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FFloorOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FFloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FFloorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FFloorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FFloorOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FFloorOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FFloorOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FFloorOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void FFloorOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FFloorOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FMAOp definitions
//===----------------------------------------------------------------------===//

FMAOpAdaptor::FMAOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.fma", odsAttrs.getContext());
}

FMAOpAdaptor::FMAOpAdaptor(FMAOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FMAOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FMAOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FMAOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMAOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMAOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::Value FMAOpAdaptor::getC() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr FMAOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FMAOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FMAOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FMAOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMAOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMAOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::Value FMAOp::getC() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange FMAOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FMAOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FMAOp::getCMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FMAOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FMAOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMAOp::getRes() {
  return *getODSResults(0).begin();
}

void FMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.addTypes(res);
}

void FMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FMAOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FMAOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FMAOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FMAOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FMAOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FMAOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FMAOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void FMAOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FMAOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::FMulAddOp definitions
//===----------------------------------------------------------------------===//

FMulAddOpAdaptor::FMulAddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.fmuladd", odsAttrs.getContext());
}

FMulAddOpAdaptor::FMulAddOpAdaptor(FMulAddOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange FMulAddOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> FMulAddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange FMulAddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulAddOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMulAddOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::Value FMulAddOpAdaptor::getC() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr FMulAddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult FMulAddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FMulAddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FMulAddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulAddOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value FMulAddOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::Value FMulAddOp::getC() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange FMulAddOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FMulAddOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange FMulAddOp::getCMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> FMulAddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FMulAddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value FMulAddOp::getRes() {
  return *getODSResults(0).begin();
}

void FMulAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  odsState.addTypes(res);
}

void FMulAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(FMulAddOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void FMulAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b, ::mlir::Value c) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addOperands(c);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void FMulAddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void FMulAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(FMulAddOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult FMulAddOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult FMulAddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult FMulAddOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void FMulAddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::FMulAddOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::GetActiveLaneMaskOp definitions
//===----------------------------------------------------------------------===//

GetActiveLaneMaskOpAdaptor::GetActiveLaneMaskOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.get.active.lane.mask", odsAttrs.getContext());
}

GetActiveLaneMaskOpAdaptor::GetActiveLaneMaskOpAdaptor(GetActiveLaneMaskOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange GetActiveLaneMaskOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> GetActiveLaneMaskOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange GetActiveLaneMaskOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetActiveLaneMaskOpAdaptor::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value GetActiveLaneMaskOpAdaptor::getN() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr GetActiveLaneMaskOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult GetActiveLaneMaskOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GetActiveLaneMaskOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range GetActiveLaneMaskOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetActiveLaneMaskOp::getBase() {
  return *getODSOperands(0).begin();
}

::mlir::Value GetActiveLaneMaskOp::getN() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange GetActiveLaneMaskOp::getBaseMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange GetActiveLaneMaskOp::getNMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GetActiveLaneMaskOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GetActiveLaneMaskOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value GetActiveLaneMaskOp::getRes() {
  return *getODSResults(0).begin();
}

void GetActiveLaneMaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value base, ::mlir::Value n) {
  odsState.addOperands(base);
  odsState.addOperands(n);
  odsState.addTypes(res);
}

void GetActiveLaneMaskOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value base, ::mlir::Value n) {
  odsState.addOperands(base);
  odsState.addOperands(n);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GetActiveLaneMaskOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GetActiveLaneMaskOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GetActiveLaneMaskOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GetActiveLaneMaskOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand baseRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> baseOperands(baseRawOperands);  ::llvm::SMLoc baseOperandsLoc;
  (void)baseOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand nRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> nOperands(nRawOperands);  ::llvm::SMLoc nOperandsLoc;
  (void)nOperandsLoc;
  ::mlir::Type baseRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> baseTypes(baseRawTypes);
  ::mlir::Type nRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> nTypes(nRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  baseOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(baseRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  nOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(nRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    baseRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    nRawTypes[0] = type;
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
  if (parser.resolveOperands(baseOperands, baseTypes, baseOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(nOperands, nTypes, nOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GetActiveLaneMaskOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getBase();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getN();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getBase().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getN().getType();
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

void GetActiveLaneMaskOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::GetActiveLaneMaskOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::Log10Op definitions
//===----------------------------------------------------------------------===//

Log10OpAdaptor::Log10OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.log10", odsAttrs.getContext());
}

Log10OpAdaptor::Log10OpAdaptor(Log10Op op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Log10OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Log10OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Log10OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log10OpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr Log10OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult Log10OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Log10Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Log10Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log10Op::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange Log10Op::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> Log10Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Log10Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log10Op::getRes() {
  return *getODSResults(0).begin();
}

void Log10Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void Log10Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(Log10Op::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void Log10Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Log10Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void Log10Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(Log10Op::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult Log10Op::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult Log10Op::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult Log10Op::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void Log10Op::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::Log10Op)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::Log2Op definitions
//===----------------------------------------------------------------------===//

Log2OpAdaptor::Log2OpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.log2", odsAttrs.getContext());
}

Log2OpAdaptor::Log2OpAdaptor(Log2Op op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange Log2OpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> Log2OpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange Log2OpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log2OpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr Log2OpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult Log2OpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Log2Op::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Log2Op::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log2Op::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange Log2Op::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> Log2Op::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Log2Op::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Log2Op::getRes() {
  return *getODSResults(0).begin();
}

void Log2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void Log2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(Log2Op::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void Log2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Log2Op::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void Log2Op::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(Log2Op::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult Log2Op::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult Log2Op::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult Log2Op::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void Log2Op::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::Log2Op)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::LogOp definitions
//===----------------------------------------------------------------------===//

LogOpAdaptor::LogOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.log", odsAttrs.getContext());
}

LogOpAdaptor::LogOpAdaptor(LogOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange LogOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> LogOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange LogOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogOpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr LogOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult LogOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> LogOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range LogOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogOp::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange LogOp::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> LogOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range LogOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value LogOp::getRes() {
  return *getODSResults(0).begin();
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(LogOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void LogOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void LogOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(LogOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult LogOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult LogOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult LogOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void LogOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::LogOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MaskedLoadOp definitions
//===----------------------------------------------------------------------===//

MaskedLoadOpAdaptor::MaskedLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.masked.load", odsAttrs.getContext());
}

MaskedLoadOpAdaptor::MaskedLoadOpAdaptor(MaskedLoadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MaskedLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskedLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange MaskedLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedLoadOpAdaptor::getData() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskedLoadOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange MaskedLoadOpAdaptor::getPassThru() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr MaskedLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MaskedLoadOpAdaptor::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MaskedLoadOp::getAlignmentAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t MaskedLoadOpAdaptor::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult MaskedLoadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.masked.load' op ""requires attribute 'alignment'");
    if (namedAttrIt->getName() == MaskedLoadOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.intr.masked.load' op ""attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MaskedLoadOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range MaskedLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedLoadOp::getData() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskedLoadOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range MaskedLoadOp::getPassThru() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange MaskedLoadOp::getDataMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedLoadOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedLoadOp::getPassThruMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MaskedLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskedLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedLoadOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr MaskedLoadOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlignmentAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MaskedLoadOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr.getValue().getZExtValue();
}

void MaskedLoadOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void MaskedLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value data, ::mlir::Value mask, ::mlir::ValueRange pass_thru, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  odsState.addTypes(res);
}

void MaskedLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value mask, ::mlir::ValueRange pass_thru, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskedLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value data, ::mlir::Value mask, ::mlir::ValueRange pass_thru, uint32_t alignment) {
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  odsState.addTypes(res);
}

void MaskedLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value mask, ::mlir::ValueRange pass_thru, uint32_t alignment) {
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskedLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskedLoadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'alignment'");
    if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MaskedLoadOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MaskedLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allOperands;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
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
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MaskedLoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperands();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MaskedLoadOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MaskedStoreOp definitions
//===----------------------------------------------------------------------===//

MaskedStoreOpAdaptor::MaskedStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.masked.store", odsAttrs.getContext());
}

MaskedStoreOpAdaptor::MaskedStoreOpAdaptor(MaskedStoreOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MaskedStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaskedStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaskedStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedStoreOpAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskedStoreOpAdaptor::getData() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskedStoreOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr MaskedStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MaskedStoreOpAdaptor::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, MaskedStoreOp::getAlignmentAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t MaskedStoreOpAdaptor::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult MaskedStoreOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.masked.store' op ""requires attribute 'alignment'");
    if (namedAttrIt->getName() == MaskedStoreOp::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.intr.masked.store' op ""attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MaskedStoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaskedStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaskedStoreOp::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaskedStoreOp::getData() {
  return *getODSOperands(1).begin();
}

::mlir::Value MaskedStoreOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange MaskedStoreOp::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedStoreOp::getDataMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaskedStoreOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MaskedStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaskedStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr MaskedStoreOp::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlignmentAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MaskedStoreOp::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr.getValue().getZExtValue();
}

void MaskedStoreOp::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    assert(isCompatibleType(resultType) && "result must be an LLVM type");
    assert(resultType.isa<LLVMVoidType>() &&
           "for zero-result operands, only 'void' is accepted as result type");
    build(odsBuilder, odsState, operands, attributes);
  
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value data, ::mlir::Value mask, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(value);
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value data, ::mlir::Value mask, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(value);
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value data, ::mlir::Value mask, uint32_t alignment) {
  odsState.addOperands(value);
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
}

void MaskedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value data, ::mlir::Value mask, uint32_t alignment) {
  odsState.addOperands(value);
  odsState.addOperands(data);
  odsState.addOperands(mask);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaskedStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MaskedStoreOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'alignment'");
    if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MaskedStoreOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MaskedStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand dataRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dataOperands(dataRawOperands);  ::llvm::SMLoc dataOperandsLoc;
  (void)dataOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type dataRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dataTypes(dataRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  dataOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dataRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
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
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    maskRawTypes[0] = type;
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    dataRawTypes[0] = type;
  }
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dataOperands, dataTypes, dataOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MaskedStoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getData();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMask();
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
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getMask().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getData().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MaskedStoreOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MatrixColumnMajorLoadOp definitions
//===----------------------------------------------------------------------===//

MatrixColumnMajorLoadOpAdaptor::MatrixColumnMajorLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.matrix.column.major.load", odsAttrs.getContext());
}

MatrixColumnMajorLoadOpAdaptor::MatrixColumnMajorLoadOpAdaptor(MatrixColumnMajorLoadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MatrixColumnMajorLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatrixColumnMajorLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MatrixColumnMajorLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixColumnMajorLoadOpAdaptor::getData() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatrixColumnMajorLoadOpAdaptor::getStride() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MatrixColumnMajorLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MatrixColumnMajorLoadOpAdaptor::getIsVolatileAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, MatrixColumnMajorLoadOp::getIsVolatileAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

bool MatrixColumnMajorLoadOpAdaptor::getIsVolatile() {
  auto attr = getIsVolatileAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixColumnMajorLoadOpAdaptor::getRowsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, MatrixColumnMajorLoadOp::getRowsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t MatrixColumnMajorLoadOpAdaptor::getRows() {
  auto attr = getRowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixColumnMajorLoadOpAdaptor::getColumnsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, MatrixColumnMajorLoadOp::getColumnsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t MatrixColumnMajorLoadOpAdaptor::getColumns() {
  auto attr = getColumnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult MatrixColumnMajorLoadOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_columns;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.matrix.column.major.load' op ""requires attribute 'columns'");
    if (namedAttrIt->getName() == MatrixColumnMajorLoadOp::getColumnsAttrName(*odsOpName)) {
      tblgen_columns = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_isVolatile;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.matrix.column.major.load' op ""requires attribute 'isVolatile'");
    if (namedAttrIt->getName() == MatrixColumnMajorLoadOp::getIsVolatileAttrName(*odsOpName)) {
      tblgen_isVolatile = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_rows;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.matrix.column.major.load' op ""requires attribute 'rows'");
    if (namedAttrIt->getName() == MatrixColumnMajorLoadOp::getRowsAttrName(*odsOpName)) {
      tblgen_rows = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_isVolatile && !(((tblgen_isVolatile.isa<::mlir::IntegerAttr>())) && ((tblgen_isVolatile.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(1)))))
    return emitError(loc, "'llvm.intr.matrix.column.major.load' op ""attribute 'isVolatile' failed to satisfy constraint: 1-bit signless integer attribute");

  if (tblgen_rows && !(((tblgen_rows.isa<::mlir::IntegerAttr>())) && ((tblgen_rows.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.intr.matrix.column.major.load' op ""attribute 'rows' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_columns && !(((tblgen_columns.isa<::mlir::IntegerAttr>())) && ((tblgen_columns.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.intr.matrix.column.major.load' op ""attribute 'columns' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MatrixColumnMajorLoadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MatrixColumnMajorLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixColumnMajorLoadOp::getData() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatrixColumnMajorLoadOp::getStride() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MatrixColumnMajorLoadOp::getDataMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MatrixColumnMajorLoadOp::getStrideMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MatrixColumnMajorLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MatrixColumnMajorLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixColumnMajorLoadOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr MatrixColumnMajorLoadOp::getIsVolatileAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getIsVolatileAttrName()).cast<::mlir::IntegerAttr>();
}

bool MatrixColumnMajorLoadOp::getIsVolatile() {
  auto attr = getIsVolatileAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixColumnMajorLoadOp::getRowsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getRowsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatrixColumnMajorLoadOp::getRows() {
  auto attr = getRowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixColumnMajorLoadOp::getColumnsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getColumnsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatrixColumnMajorLoadOp::getColumns() {
  auto attr = getColumnsAttr();
  return attr.getValue().getZExtValue();
}

void MatrixColumnMajorLoadOp::setIsVolatileAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIsVolatileAttrName(), attr);
}

void MatrixColumnMajorLoadOp::setRowsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getRowsAttrName(), attr);
}

void MatrixColumnMajorLoadOp::setColumnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getColumnsAttrName(), attr);
}

void MatrixColumnMajorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void MatrixColumnMajorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value data, ::mlir::Value stride, ::mlir::IntegerAttr isVolatile, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute(getIsVolatileAttrName(odsState.name), isVolatile);
  odsState.addAttribute(getRowsAttrName(odsState.name), rows);
  odsState.addAttribute(getColumnsAttrName(odsState.name), columns);
  odsState.addTypes(res);
}

void MatrixColumnMajorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value stride, ::mlir::IntegerAttr isVolatile, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute(getIsVolatileAttrName(odsState.name), isVolatile);
  odsState.addAttribute(getRowsAttrName(odsState.name), rows);
  odsState.addAttribute(getColumnsAttrName(odsState.name), columns);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixColumnMajorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value data, ::mlir::Value stride, bool isVolatile, uint32_t rows, uint32_t columns) {
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute(getIsVolatileAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(1), isVolatile));
  odsState.addAttribute(getRowsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute(getColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
  odsState.addTypes(res);
}

void MatrixColumnMajorLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value data, ::mlir::Value stride, bool isVolatile, uint32_t rows, uint32_t columns) {
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute(getIsVolatileAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(1), isVolatile));
  odsState.addAttribute(getRowsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute(getColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixColumnMajorLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MatrixColumnMajorLoadOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_columns;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'columns'");
    if (namedAttrIt->getName() == getColumnsAttrName()) {
      tblgen_columns = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_isVolatile;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'isVolatile'");
    if (namedAttrIt->getName() == getIsVolatileAttrName()) {
      tblgen_isVolatile = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_rows;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'rows'");
    if (namedAttrIt->getName() == getRowsAttrName()) {
      tblgen_rows = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps1(*this, tblgen_isVolatile, "isVolatile")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(*this, tblgen_rows, "rows")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(*this, tblgen_columns, "columns")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps5(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MatrixColumnMajorLoadOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MatrixColumnMajorLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand dataRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dataOperands(dataRawOperands);  ::llvm::SMLoc dataOperandsLoc;
  (void)dataOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand strideRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> strideOperands(strideRawOperands);  ::llvm::SMLoc strideOperandsLoc;
  (void)strideOperandsLoc;
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);
  ::mlir::Type dataRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dataTypes(dataRawTypes);
  ::mlir::Type strideRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> strideTypes(strideRawTypes);

  dataOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dataRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseLess())
    return ::mlir::failure();
  if (parser.parseKeyword("stride"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  strideOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(strideRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseGreater())
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
  if (parser.parseKeyword("from"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    dataRawTypes[0] = type;
  }
  if (parser.parseKeyword("stride"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    strideRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(dataOperands, dataTypes, dataOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(strideOperands, strideTypes, strideOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatrixColumnMajorLoadOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getData();
  _odsPrinter << ",";
  _odsPrinter << ' ' << "<";
  _odsPrinter << "stride";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter << getStride();
  _odsPrinter << ">";
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
  _odsPrinter << ' ' << "from";
  _odsPrinter << ' ';
  {
    auto type = getData().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "stride";
  _odsPrinter << ' ';
  {
    auto type = getStride().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MatrixColumnMajorLoadOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MatrixColumnMajorStoreOp definitions
//===----------------------------------------------------------------------===//

MatrixColumnMajorStoreOpAdaptor::MatrixColumnMajorStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.matrix.column.major.store", odsAttrs.getContext());
}

MatrixColumnMajorStoreOpAdaptor::MatrixColumnMajorStoreOpAdaptor(MatrixColumnMajorStoreOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MatrixColumnMajorStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatrixColumnMajorStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MatrixColumnMajorStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixColumnMajorStoreOpAdaptor::getMatrix() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatrixColumnMajorStoreOpAdaptor::getData() {
  return *getODSOperands(1).begin();
}

::mlir::Value MatrixColumnMajorStoreOpAdaptor::getStride() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr MatrixColumnMajorStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MatrixColumnMajorStoreOpAdaptor::getIsVolatileAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, MatrixColumnMajorStoreOp::getIsVolatileAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

bool MatrixColumnMajorStoreOpAdaptor::getIsVolatile() {
  auto attr = getIsVolatileAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixColumnMajorStoreOpAdaptor::getRowsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, MatrixColumnMajorStoreOp::getRowsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t MatrixColumnMajorStoreOpAdaptor::getRows() {
  auto attr = getRowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixColumnMajorStoreOpAdaptor::getColumnsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, MatrixColumnMajorStoreOp::getColumnsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t MatrixColumnMajorStoreOpAdaptor::getColumns() {
  auto attr = getColumnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult MatrixColumnMajorStoreOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_columns;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.matrix.column.major.store' op ""requires attribute 'columns'");
    if (namedAttrIt->getName() == MatrixColumnMajorStoreOp::getColumnsAttrName(*odsOpName)) {
      tblgen_columns = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_isVolatile;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.matrix.column.major.store' op ""requires attribute 'isVolatile'");
    if (namedAttrIt->getName() == MatrixColumnMajorStoreOp::getIsVolatileAttrName(*odsOpName)) {
      tblgen_isVolatile = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_rows;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.matrix.column.major.store' op ""requires attribute 'rows'");
    if (namedAttrIt->getName() == MatrixColumnMajorStoreOp::getRowsAttrName(*odsOpName)) {
      tblgen_rows = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_isVolatile && !(((tblgen_isVolatile.isa<::mlir::IntegerAttr>())) && ((tblgen_isVolatile.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(1)))))
    return emitError(loc, "'llvm.intr.matrix.column.major.store' op ""attribute 'isVolatile' failed to satisfy constraint: 1-bit signless integer attribute");

  if (tblgen_rows && !(((tblgen_rows.isa<::mlir::IntegerAttr>())) && ((tblgen_rows.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.intr.matrix.column.major.store' op ""attribute 'rows' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_columns && !(((tblgen_columns.isa<::mlir::IntegerAttr>())) && ((tblgen_columns.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.intr.matrix.column.major.store' op ""attribute 'columns' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MatrixColumnMajorStoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MatrixColumnMajorStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixColumnMajorStoreOp::getMatrix() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatrixColumnMajorStoreOp::getData() {
  return *getODSOperands(1).begin();
}

::mlir::Value MatrixColumnMajorStoreOp::getStride() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange MatrixColumnMajorStoreOp::getMatrixMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MatrixColumnMajorStoreOp::getDataMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MatrixColumnMajorStoreOp::getStrideMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MatrixColumnMajorStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MatrixColumnMajorStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr MatrixColumnMajorStoreOp::getIsVolatileAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getIsVolatileAttrName()).cast<::mlir::IntegerAttr>();
}

bool MatrixColumnMajorStoreOp::getIsVolatile() {
  auto attr = getIsVolatileAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixColumnMajorStoreOp::getRowsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getRowsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatrixColumnMajorStoreOp::getRows() {
  auto attr = getRowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixColumnMajorStoreOp::getColumnsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getColumnsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatrixColumnMajorStoreOp::getColumns() {
  auto attr = getColumnsAttr();
  return attr.getValue().getZExtValue();
}

void MatrixColumnMajorStoreOp::setIsVolatileAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getIsVolatileAttrName(), attr);
}

void MatrixColumnMajorStoreOp::setRowsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getRowsAttrName(), attr);
}

void MatrixColumnMajorStoreOp::setColumnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getColumnsAttrName(), attr);
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    assert(isCompatibleType(resultType) && "result must be an LLVM type");
    assert(resultType.isa<LLVMVoidType>() &&
           "for zero-result operands, only 'void' is accepted as result type");
    build(odsBuilder, odsState, operands, attributes);
  
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value matrix, ::mlir::Value data, ::mlir::Value stride, ::mlir::IntegerAttr isVolatile, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(matrix);
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute(getIsVolatileAttrName(odsState.name), isVolatile);
  odsState.addAttribute(getRowsAttrName(odsState.name), rows);
  odsState.addAttribute(getColumnsAttrName(odsState.name), columns);
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value matrix, ::mlir::Value data, ::mlir::Value stride, ::mlir::IntegerAttr isVolatile, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(matrix);
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute(getIsVolatileAttrName(odsState.name), isVolatile);
  odsState.addAttribute(getRowsAttrName(odsState.name), rows);
  odsState.addAttribute(getColumnsAttrName(odsState.name), columns);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value matrix, ::mlir::Value data, ::mlir::Value stride, bool isVolatile, uint32_t rows, uint32_t columns) {
  odsState.addOperands(matrix);
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute(getIsVolatileAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(1), isVolatile));
  odsState.addAttribute(getRowsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute(getColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value matrix, ::mlir::Value data, ::mlir::Value stride, bool isVolatile, uint32_t rows, uint32_t columns) {
  odsState.addOperands(matrix);
  odsState.addOperands(data);
  odsState.addOperands(stride);
  odsState.addAttribute(getIsVolatileAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(1), isVolatile));
  odsState.addAttribute(getRowsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute(getColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixColumnMajorStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MatrixColumnMajorStoreOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_columns;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'columns'");
    if (namedAttrIt->getName() == getColumnsAttrName()) {
      tblgen_columns = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_isVolatile;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'isVolatile'");
    if (namedAttrIt->getName() == getIsVolatileAttrName()) {
      tblgen_isVolatile = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_rows;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'rows'");
    if (namedAttrIt->getName() == getRowsAttrName()) {
      tblgen_rows = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps1(*this, tblgen_isVolatile, "isVolatile")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(*this, tblgen_rows, "rows")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(*this, tblgen_columns, "columns")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MatrixColumnMajorStoreOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MatrixColumnMajorStoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand matrixRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> matrixOperands(matrixRawOperands);  ::llvm::SMLoc matrixOperandsLoc;
  (void)matrixOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand dataRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> dataOperands(dataRawOperands);  ::llvm::SMLoc dataOperandsLoc;
  (void)dataOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand strideRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> strideOperands(strideRawOperands);  ::llvm::SMLoc strideOperandsLoc;
  (void)strideOperandsLoc;
  ::mlir::Type matrixRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> matrixTypes(matrixRawTypes);
  ::mlir::Type dataRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> dataTypes(dataRawTypes);
  ::mlir::Type strideRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> strideTypes(strideRawTypes);

  matrixOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(matrixRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  dataOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(dataRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();
  if (parser.parseLess())
    return ::mlir::failure();
  if (parser.parseKeyword("stride"))
    return ::mlir::failure();
  if (parser.parseEqual())
    return ::mlir::failure();

  strideOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(strideRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseGreater())
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    matrixRawTypes[0] = type;
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    dataRawTypes[0] = type;
  }
  if (parser.parseKeyword("stride"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    strideRawTypes[0] = type;
  }
  if (parser.resolveOperands(matrixOperands, matrixTypes, matrixOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(dataOperands, dataTypes, dataOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(strideOperands, strideTypes, strideOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatrixColumnMajorStoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getMatrix();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getData();
  _odsPrinter << ",";
  _odsPrinter << ' ' << "<";
  _odsPrinter << "stride";
  _odsPrinter << ' ' << "=";
  _odsPrinter << ' ';
  _odsPrinter << getStride();
  _odsPrinter << ">";
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMatrix().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  {
    auto type = getData().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "stride";
  _odsPrinter << ' ';
  {
    auto type = getStride().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MatrixColumnMajorStoreOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MatrixMultiplyOp definitions
//===----------------------------------------------------------------------===//

MatrixMultiplyOpAdaptor::MatrixMultiplyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.matrix.multiply", odsAttrs.getContext());
}

MatrixMultiplyOpAdaptor::MatrixMultiplyOpAdaptor(MatrixMultiplyOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MatrixMultiplyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatrixMultiplyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MatrixMultiplyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixMultiplyOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatrixMultiplyOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MatrixMultiplyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MatrixMultiplyOpAdaptor::getLhsRowsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 1, MatrixMultiplyOp::getLhsRowsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t MatrixMultiplyOpAdaptor::getLhsRows() {
  auto attr = getLhsRowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixMultiplyOpAdaptor::getLhsColumnsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 2, MatrixMultiplyOp::getLhsColumnsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t MatrixMultiplyOpAdaptor::getLhsColumns() {
  auto attr = getLhsColumnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixMultiplyOpAdaptor::getRhsColumnsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 2, odsAttrs.end() - 0, MatrixMultiplyOp::getRhsColumnsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t MatrixMultiplyOpAdaptor::getRhsColumns() {
  auto attr = getRhsColumnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult MatrixMultiplyOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_lhs_columns;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.matrix.multiply' op ""requires attribute 'lhs_columns'");
    if (namedAttrIt->getName() == MatrixMultiplyOp::getLhsColumnsAttrName(*odsOpName)) {
      tblgen_lhs_columns = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_lhs_rows;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.matrix.multiply' op ""requires attribute 'lhs_rows'");
    if (namedAttrIt->getName() == MatrixMultiplyOp::getLhsRowsAttrName(*odsOpName)) {
      tblgen_lhs_rows = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_rhs_columns;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.matrix.multiply' op ""requires attribute 'rhs_columns'");
    if (namedAttrIt->getName() == MatrixMultiplyOp::getRhsColumnsAttrName(*odsOpName)) {
      tblgen_rhs_columns = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_lhs_rows && !(((tblgen_lhs_rows.isa<::mlir::IntegerAttr>())) && ((tblgen_lhs_rows.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.intr.matrix.multiply' op ""attribute 'lhs_rows' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_lhs_columns && !(((tblgen_lhs_columns.isa<::mlir::IntegerAttr>())) && ((tblgen_lhs_columns.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.intr.matrix.multiply' op ""attribute 'lhs_columns' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_rhs_columns && !(((tblgen_rhs_columns.isa<::mlir::IntegerAttr>())) && ((tblgen_rhs_columns.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.intr.matrix.multiply' op ""attribute 'rhs_columns' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MatrixMultiplyOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MatrixMultiplyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixMultiplyOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value MatrixMultiplyOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MatrixMultiplyOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MatrixMultiplyOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MatrixMultiplyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MatrixMultiplyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixMultiplyOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr MatrixMultiplyOp::getLhsRowsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 1, getLhsRowsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatrixMultiplyOp::getLhsRows() {
  auto attr = getLhsRowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixMultiplyOp::getLhsColumnsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 2, getLhsColumnsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatrixMultiplyOp::getLhsColumns() {
  auto attr = getLhsColumnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixMultiplyOp::getRhsColumnsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 2, (*this)->getAttrs().end() - 0, getRhsColumnsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatrixMultiplyOp::getRhsColumns() {
  auto attr = getRhsColumnsAttr();
  return attr.getValue().getZExtValue();
}

void MatrixMultiplyOp::setLhsRowsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getLhsRowsAttrName(), attr);
}

void MatrixMultiplyOp::setLhsColumnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getLhsColumnsAttrName(), attr);
}

void MatrixMultiplyOp::setRhsColumnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getRhsColumnsAttrName(), attr);
}

void MatrixMultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void MatrixMultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::IntegerAttr lhs_rows, ::mlir::IntegerAttr lhs_columns, ::mlir::IntegerAttr rhs_columns) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getLhsRowsAttrName(odsState.name), lhs_rows);
  odsState.addAttribute(getLhsColumnsAttrName(odsState.name), lhs_columns);
  odsState.addAttribute(getRhsColumnsAttrName(odsState.name), rhs_columns);
  odsState.addTypes(res);
}

void MatrixMultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::IntegerAttr lhs_rows, ::mlir::IntegerAttr lhs_columns, ::mlir::IntegerAttr rhs_columns) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getLhsRowsAttrName(odsState.name), lhs_rows);
  odsState.addAttribute(getLhsColumnsAttrName(odsState.name), lhs_columns);
  odsState.addAttribute(getRhsColumnsAttrName(odsState.name), rhs_columns);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixMultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, uint32_t lhs_rows, uint32_t lhs_columns, uint32_t rhs_columns) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getLhsRowsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), lhs_rows));
  odsState.addAttribute(getLhsColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), lhs_columns));
  odsState.addAttribute(getRhsColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rhs_columns));
  odsState.addTypes(res);
}

void MatrixMultiplyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, uint32_t lhs_rows, uint32_t lhs_columns, uint32_t rhs_columns) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addAttribute(getLhsRowsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), lhs_rows));
  odsState.addAttribute(getLhsColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), lhs_columns));
  odsState.addAttribute(getRhsColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rhs_columns));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixMultiplyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MatrixMultiplyOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_lhs_columns;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'lhs_columns'");
    if (namedAttrIt->getName() == getLhsColumnsAttrName()) {
      tblgen_lhs_columns = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_lhs_rows;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'lhs_rows'");
    if (namedAttrIt->getName() == getLhsRowsAttrName()) {
      tblgen_lhs_rows = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_rhs_columns;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'rhs_columns'");
    if (namedAttrIt->getName() == getRhsColumnsAttrName()) {
      tblgen_rhs_columns = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(*this, tblgen_lhs_rows, "lhs_rows")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(*this, tblgen_lhs_columns, "lhs_columns")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(*this, tblgen_rhs_columns, "rhs_columns")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MatrixMultiplyOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MatrixMultiplyOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand lhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> lhsOperands(lhsRawOperands);  ::llvm::SMLoc lhsOperandsLoc;
  (void)lhsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand rhsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> rhsOperands(rhsRawOperands);  ::llvm::SMLoc rhsOperandsLoc;
  (void)rhsOperandsLoc;
  ::mlir::Type lhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> lhsTypes(lhsRawTypes);
  ::mlir::Type rhsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> rhsTypes(rhsRawTypes);
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
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();
  if (parser.parseLParen())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    lhsRawTypes[0] = type;
  }
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    rhsRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  if (parser.parseArrow())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(lhsOperands, lhsTypes, lhsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(rhsOperands, rhsTypes, rhsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatrixMultiplyOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getLhs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getRhs();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ' << "(";
  {
    auto type = getLhs().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getRhs().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MatrixMultiplyOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MatrixTransposeOp definitions
//===----------------------------------------------------------------------===//

MatrixTransposeOpAdaptor::MatrixTransposeOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.matrix.transpose", odsAttrs.getContext());
}

MatrixTransposeOpAdaptor::MatrixTransposeOpAdaptor(MatrixTransposeOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MatrixTransposeOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MatrixTransposeOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MatrixTransposeOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixTransposeOpAdaptor::getMatrix() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr MatrixTransposeOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr MatrixTransposeOpAdaptor::getRowsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 1, odsAttrs.end() - 0, MatrixTransposeOp::getRowsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t MatrixTransposeOpAdaptor::getRows() {
  auto attr = getRowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixTransposeOpAdaptor::getColumnsAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 1, MatrixTransposeOp::getColumnsAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t MatrixTransposeOpAdaptor::getColumns() {
  auto attr = getColumnsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult MatrixTransposeOpAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_columns;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.matrix.transpose' op ""requires attribute 'columns'");
    if (namedAttrIt->getName() == MatrixTransposeOp::getColumnsAttrName(*odsOpName)) {
      tblgen_columns = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_rows;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.matrix.transpose' op ""requires attribute 'rows'");
    if (namedAttrIt->getName() == MatrixTransposeOp::getRowsAttrName(*odsOpName)) {
      tblgen_rows = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_rows && !(((tblgen_rows.isa<::mlir::IntegerAttr>())) && ((tblgen_rows.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.intr.matrix.transpose' op ""attribute 'rows' failed to satisfy constraint: 32-bit signless integer attribute");

  if (tblgen_columns && !(((tblgen_columns.isa<::mlir::IntegerAttr>())) && ((tblgen_columns.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.intr.matrix.transpose' op ""attribute 'columns' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MatrixTransposeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MatrixTransposeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixTransposeOp::getMatrix() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange MatrixTransposeOp::getMatrixMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MatrixTransposeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MatrixTransposeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MatrixTransposeOp::getRes() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr MatrixTransposeOp::getRowsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 1, (*this)->getAttrs().end() - 0, getRowsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatrixTransposeOp::getRows() {
  auto attr = getRowsAttr();
  return attr.getValue().getZExtValue();
}

::mlir::IntegerAttr MatrixTransposeOp::getColumnsAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 1, getColumnsAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t MatrixTransposeOp::getColumns() {
  auto attr = getColumnsAttr();
  return attr.getValue().getZExtValue();
}

void MatrixTransposeOp::setRowsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getRowsAttrName(), attr);
}

void MatrixTransposeOp::setColumnsAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getColumnsAttrName(), attr);
}

void MatrixTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void MatrixTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value matrix, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(matrix);
  odsState.addAttribute(getRowsAttrName(odsState.name), rows);
  odsState.addAttribute(getColumnsAttrName(odsState.name), columns);
  odsState.addTypes(res);
}

void MatrixTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value matrix, ::mlir::IntegerAttr rows, ::mlir::IntegerAttr columns) {
  odsState.addOperands(matrix);
  odsState.addAttribute(getRowsAttrName(odsState.name), rows);
  odsState.addAttribute(getColumnsAttrName(odsState.name), columns);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value matrix, uint32_t rows, uint32_t columns) {
  odsState.addOperands(matrix);
  odsState.addAttribute(getRowsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute(getColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
  odsState.addTypes(res);
}

void MatrixTransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value matrix, uint32_t rows, uint32_t columns) {
  odsState.addOperands(matrix);
  odsState.addAttribute(getRowsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), rows));
  odsState.addAttribute(getColumnsAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), columns));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MatrixTransposeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MatrixTransposeOp::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_columns;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'columns'");
    if (namedAttrIt->getName() == getColumnsAttrName()) {
      tblgen_columns = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }
  ::mlir::Attribute tblgen_rows;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'rows'");
    if (namedAttrIt->getName() == getRowsAttrName()) {
      tblgen_rows = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(*this, tblgen_rows, "rows")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(*this, tblgen_columns, "columns")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MatrixTransposeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult MatrixTransposeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand matrixRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> matrixOperands(matrixRawOperands);  ::llvm::SMLoc matrixOperandsLoc;
  (void)matrixOperandsLoc;
  ::mlir::Type matrixRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> matrixTypes(matrixRawTypes);
  ::mlir::Type resRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> resTypes(resRawTypes);

  matrixOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(matrixRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    matrixRawTypes[0] = type;
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    resRawTypes[0] = type;
  }
  result.addTypes(resTypes);
  if (parser.resolveOperands(matrixOperands, matrixTypes, matrixOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void MatrixTransposeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getMatrix();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getMatrix().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getRes().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MatrixTransposeOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MaxNumOp definitions
//===----------------------------------------------------------------------===//

MaxNumOpAdaptor::MaxNumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.maxnum", odsAttrs.getContext());
}

MaxNumOpAdaptor::MaxNumOpAdaptor(MaxNumOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MaxNumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaxNumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaxNumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaxNumOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaxNumOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MaxNumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaxNumOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MaxNumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaxNumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaxNumOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaxNumOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MaxNumOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaxNumOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MaxNumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaxNumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaxNumOp::getRes() {
  return *getODSResults(0).begin();
}

void MaxNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void MaxNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(MaxNumOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void MaxNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaxNumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MaxNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(MaxNumOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult MaxNumOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MaxNumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult MaxNumOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void MaxNumOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MaxNumOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MaximumOp definitions
//===----------------------------------------------------------------------===//

MaximumOpAdaptor::MaximumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.maximum", odsAttrs.getContext());
}

MaximumOpAdaptor::MaximumOpAdaptor(MaximumOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MaximumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MaximumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MaximumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaximumOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaximumOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MaximumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MaximumOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MaximumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MaximumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaximumOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value MaximumOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MaximumOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MaximumOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MaximumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MaximumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MaximumOp::getRes() {
  return *getODSResults(0).begin();
}

void MaximumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void MaximumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(MaximumOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void MaximumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MaximumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MaximumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(MaximumOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult MaximumOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MaximumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult MaximumOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void MaximumOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MaximumOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MemcpyInlineOp definitions
//===----------------------------------------------------------------------===//

MemcpyInlineOpAdaptor::MemcpyInlineOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.memcpy.inline", odsAttrs.getContext());
}

MemcpyInlineOpAdaptor::MemcpyInlineOpAdaptor(MemcpyInlineOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MemcpyInlineOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MemcpyInlineOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MemcpyInlineOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemcpyInlineOpAdaptor::getDst() {
  return *getODSOperands(0).begin();
}

::mlir::Value MemcpyInlineOpAdaptor::getSrc() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemcpyInlineOpAdaptor::getLen() {
  return *getODSOperands(2).begin();
}

::mlir::Value MemcpyInlineOpAdaptor::getIsVolatile() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr MemcpyInlineOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MemcpyInlineOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MemcpyInlineOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MemcpyInlineOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemcpyInlineOp::getDst() {
  return *getODSOperands(0).begin();
}

::mlir::Value MemcpyInlineOp::getSrc() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemcpyInlineOp::getLen() {
  return *getODSOperands(2).begin();
}

::mlir::Value MemcpyInlineOp::getIsVolatile() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange MemcpyInlineOp::getDstMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemcpyInlineOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemcpyInlineOp::getLenMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemcpyInlineOp::getIsVolatileMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MemcpyInlineOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MemcpyInlineOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void MemcpyInlineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value dst, ::mlir::Value src, ::mlir::Value len, ::mlir::Value isVolatile) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addOperands(len);
  odsState.addOperands(isVolatile);
}

void MemcpyInlineOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value src, ::mlir::Value len, ::mlir::Value isVolatile) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addOperands(len);
  odsState.addOperands(isVolatile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MemcpyInlineOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MemcpyInlineOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MemcpyInlineOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void MemcpyInlineOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MemcpyInlineOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MemcpyOp definitions
//===----------------------------------------------------------------------===//

MemcpyOpAdaptor::MemcpyOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.memcpy", odsAttrs.getContext());
}

MemcpyOpAdaptor::MemcpyOpAdaptor(MemcpyOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MemcpyOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MemcpyOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MemcpyOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemcpyOpAdaptor::getDst() {
  return *getODSOperands(0).begin();
}

::mlir::Value MemcpyOpAdaptor::getSrc() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemcpyOpAdaptor::getLen() {
  return *getODSOperands(2).begin();
}

::mlir::Value MemcpyOpAdaptor::getIsVolatile() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr MemcpyOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MemcpyOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MemcpyOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MemcpyOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemcpyOp::getDst() {
  return *getODSOperands(0).begin();
}

::mlir::Value MemcpyOp::getSrc() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemcpyOp::getLen() {
  return *getODSOperands(2).begin();
}

::mlir::Value MemcpyOp::getIsVolatile() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange MemcpyOp::getDstMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemcpyOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemcpyOp::getLenMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemcpyOp::getIsVolatileMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MemcpyOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MemcpyOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void MemcpyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value dst, ::mlir::Value src, ::mlir::Value len, ::mlir::Value isVolatile) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addOperands(len);
  odsState.addOperands(isVolatile);
}

void MemcpyOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value src, ::mlir::Value len, ::mlir::Value isVolatile) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addOperands(len);
  odsState.addOperands(isVolatile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MemcpyOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MemcpyOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MemcpyOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void MemcpyOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MemcpyOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MemmoveOp definitions
//===----------------------------------------------------------------------===//

MemmoveOpAdaptor::MemmoveOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.memmove", odsAttrs.getContext());
}

MemmoveOpAdaptor::MemmoveOpAdaptor(MemmoveOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MemmoveOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MemmoveOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MemmoveOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemmoveOpAdaptor::getDst() {
  return *getODSOperands(0).begin();
}

::mlir::Value MemmoveOpAdaptor::getSrc() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemmoveOpAdaptor::getLen() {
  return *getODSOperands(2).begin();
}

::mlir::Value MemmoveOpAdaptor::getIsVolatile() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr MemmoveOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MemmoveOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MemmoveOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MemmoveOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemmoveOp::getDst() {
  return *getODSOperands(0).begin();
}

::mlir::Value MemmoveOp::getSrc() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemmoveOp::getLen() {
  return *getODSOperands(2).begin();
}

::mlir::Value MemmoveOp::getIsVolatile() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange MemmoveOp::getDstMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemmoveOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemmoveOp::getLenMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemmoveOp::getIsVolatileMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MemmoveOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MemmoveOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void MemmoveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value dst, ::mlir::Value src, ::mlir::Value len, ::mlir::Value isVolatile) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addOperands(len);
  odsState.addOperands(isVolatile);
}

void MemmoveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value src, ::mlir::Value len, ::mlir::Value isVolatile) {
  odsState.addOperands(dst);
  odsState.addOperands(src);
  odsState.addOperands(len);
  odsState.addOperands(isVolatile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MemmoveOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MemmoveOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MemmoveOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void MemmoveOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
  for (::mlir::Value value : getODSOperands(1))
    effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MemmoveOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MemsetOp definitions
//===----------------------------------------------------------------------===//

MemsetOpAdaptor::MemsetOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.memset", odsAttrs.getContext());
}

MemsetOpAdaptor::MemsetOpAdaptor(MemsetOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MemsetOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MemsetOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MemsetOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemsetOpAdaptor::getDst() {
  return *getODSOperands(0).begin();
}

::mlir::Value MemsetOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemsetOpAdaptor::getLen() {
  return *getODSOperands(2).begin();
}

::mlir::Value MemsetOpAdaptor::getIsVolatile() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr MemsetOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MemsetOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MemsetOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MemsetOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MemsetOp::getDst() {
  return *getODSOperands(0).begin();
}

::mlir::Value MemsetOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value MemsetOp::getLen() {
  return *getODSOperands(2).begin();
}

::mlir::Value MemsetOp::getIsVolatile() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange MemsetOp::getDstMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemsetOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemsetOp::getLenMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MemsetOp::getIsVolatileMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MemsetOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MemsetOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void MemsetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value len, ::mlir::Value isVolatile) {
  odsState.addOperands(dst);
  odsState.addOperands(val);
  odsState.addOperands(len);
  odsState.addOperands(isVolatile);
}

void MemsetOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value dst, ::mlir::Value val, ::mlir::Value len, ::mlir::Value isVolatile) {
  odsState.addOperands(dst);
  odsState.addOperands(val);
  odsState.addOperands(len);
  odsState.addOperands(isVolatile);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MemsetOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MemsetOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MemsetOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void MemsetOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
  for (::mlir::Value value : getODSOperands(0))
    effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, ::mlir::SideEffects::DefaultResource::get());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MemsetOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MinNumOp definitions
//===----------------------------------------------------------------------===//

MinNumOpAdaptor::MinNumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.minnum", odsAttrs.getContext());
}

MinNumOpAdaptor::MinNumOpAdaptor(MinNumOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MinNumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MinNumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MinNumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MinNumOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value MinNumOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MinNumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MinNumOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MinNumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MinNumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MinNumOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value MinNumOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MinNumOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MinNumOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MinNumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MinNumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MinNumOp::getRes() {
  return *getODSResults(0).begin();
}

void MinNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void MinNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(MinNumOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void MinNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MinNumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MinNumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(MinNumOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult MinNumOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MinNumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult MinNumOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void MinNumOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MinNumOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::MinimumOp definitions
//===----------------------------------------------------------------------===//

MinimumOpAdaptor::MinimumOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.minimum", odsAttrs.getContext());
}

MinimumOpAdaptor::MinimumOpAdaptor(MinimumOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange MinimumOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> MinimumOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange MinimumOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MinimumOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value MinimumOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr MinimumOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult MinimumOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MinimumOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MinimumOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MinimumOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value MinimumOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange MinimumOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MinimumOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MinimumOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MinimumOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value MinimumOp::getRes() {
  return *getODSResults(0).begin();
}

void MinimumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void MinimumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(MinimumOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void MinimumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MinimumOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void MinimumOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(MinimumOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult MinimumOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MinimumOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult MinimumOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void MinimumOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::MinimumOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::PowIOp definitions
//===----------------------------------------------------------------------===//

PowIOpAdaptor::PowIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.powi", odsAttrs.getContext());
}

PowIOpAdaptor::PowIOpAdaptor(PowIOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PowIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PowIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PowIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PowIOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value PowIOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr PowIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult PowIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PowIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PowIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PowIOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value PowIOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange PowIOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PowIOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PowIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PowIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PowIOp::getRes() {
  return *getODSResults(0).begin();
}

void PowIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void PowIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PowIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PowIOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PowIOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void PowIOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::PowIOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::PowOp definitions
//===----------------------------------------------------------------------===//

PowOpAdaptor::PowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.pow", odsAttrs.getContext());
}

PowOpAdaptor::PowOpAdaptor(PowOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PowOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value PowOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr PowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult PowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PowOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value PowOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange PowOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange PowOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PowOp::getRes() {
  return *getODSResults(0).begin();
}

void PowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void PowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(PowOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void PowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void PowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(PowOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult PowOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PowOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult PowOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void PowOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::PowOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::Prefetch definitions
//===----------------------------------------------------------------------===//

PrefetchAdaptor::PrefetchAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.prefetch", odsAttrs.getContext());
}

PrefetchAdaptor::PrefetchAdaptor(Prefetch op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange PrefetchAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> PrefetchAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange PrefetchAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value PrefetchAdaptor::getAddr() {
  return *getODSOperands(0).begin();
}

::mlir::Value PrefetchAdaptor::getRw() {
  return *getODSOperands(1).begin();
}

::mlir::Value PrefetchAdaptor::getHint() {
  return *getODSOperands(2).begin();
}

::mlir::Value PrefetchAdaptor::getCache() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr PrefetchAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult PrefetchAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> Prefetch::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range Prefetch::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value Prefetch::getAddr() {
  return *getODSOperands(0).begin();
}

::mlir::Value Prefetch::getRw() {
  return *getODSOperands(1).begin();
}

::mlir::Value Prefetch::getHint() {
  return *getODSOperands(2).begin();
}

::mlir::Value Prefetch::getCache() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange Prefetch::getAddrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange Prefetch::getRwMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange Prefetch::getHintMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange Prefetch::getCacheMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> Prefetch::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range Prefetch::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void Prefetch::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value addr, ::mlir::Value rw, ::mlir::Value hint, ::mlir::Value cache) {
  odsState.addOperands(addr);
  odsState.addOperands(rw);
  odsState.addOperands(hint);
  odsState.addOperands(cache);
}

void Prefetch::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value addr, ::mlir::Value rw, ::mlir::Value hint, ::mlir::Value cache) {
  odsState.addOperands(addr);
  odsState.addOperands(rw);
  odsState.addOperands(hint);
  odsState.addOperands(cache);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void Prefetch::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult Prefetch::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult Prefetch::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::Prefetch)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SAddWithOverflowOp definitions
//===----------------------------------------------------------------------===//

SAddWithOverflowOpAdaptor::SAddWithOverflowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.sadd.with.overflow", odsAttrs.getContext());
}

SAddWithOverflowOpAdaptor::SAddWithOverflowOpAdaptor(SAddWithOverflowOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SAddWithOverflowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SAddWithOverflowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SAddWithOverflowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr SAddWithOverflowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SAddWithOverflowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SAddWithOverflowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SAddWithOverflowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> SAddWithOverflowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SAddWithOverflowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SAddWithOverflowOp::getRes() {
  return *getODSResults(0).begin();
}

void SAddWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addTypes(res);
}

void SAddWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SAddWithOverflowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SAddWithOverflowOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SAddWithOverflowOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SAddWithOverflowOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SMaxOp definitions
//===----------------------------------------------------------------------===//

SMaxOpAdaptor::SMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.smax", odsAttrs.getContext());
}

SMaxOpAdaptor::SMaxOpAdaptor(SMaxOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMaxOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value SMaxOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr SMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SMaxOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMaxOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value SMaxOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SMaxOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SMaxOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMaxOp::getRes() {
  return *getODSResults(0).begin();
}

void SMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void SMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SMaxOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SMaxOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SMaxOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult SMaxOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void SMaxOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SMaxOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SMinOp definitions
//===----------------------------------------------------------------------===//

SMinOpAdaptor::SMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.smin", odsAttrs.getContext());
}

SMinOpAdaptor::SMinOpAdaptor(SMinOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMinOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value SMinOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr SMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SMinOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMinOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value SMinOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange SMinOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange SMinOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMinOp::getRes() {
  return *getODSResults(0).begin();
}

void SMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void SMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SMinOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SMinOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SMinOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult SMinOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void SMinOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SMinOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SMulWithOverflowOp definitions
//===----------------------------------------------------------------------===//

SMulWithOverflowOpAdaptor::SMulWithOverflowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.smul.with.overflow", odsAttrs.getContext());
}

SMulWithOverflowOpAdaptor::SMulWithOverflowOpAdaptor(SMulWithOverflowOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SMulWithOverflowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SMulWithOverflowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SMulWithOverflowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr SMulWithOverflowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SMulWithOverflowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SMulWithOverflowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SMulWithOverflowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> SMulWithOverflowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SMulWithOverflowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SMulWithOverflowOp::getRes() {
  return *getODSResults(0).begin();
}

void SMulWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addTypes(res);
}

void SMulWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SMulWithOverflowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SMulWithOverflowOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SMulWithOverflowOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SMulWithOverflowOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SSubWithOverflowOp definitions
//===----------------------------------------------------------------------===//

SSubWithOverflowOpAdaptor::SSubWithOverflowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.ssub.with.overflow", odsAttrs.getContext());
}

SSubWithOverflowOpAdaptor::SSubWithOverflowOpAdaptor(SSubWithOverflowOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SSubWithOverflowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SSubWithOverflowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SSubWithOverflowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr SSubWithOverflowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SSubWithOverflowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SSubWithOverflowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SSubWithOverflowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> SSubWithOverflowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SSubWithOverflowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SSubWithOverflowOp::getRes() {
  return *getODSResults(0).begin();
}

void SSubWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addTypes(res);
}

void SSubWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SSubWithOverflowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult SSubWithOverflowOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SSubWithOverflowOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SSubWithOverflowOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SinOp definitions
//===----------------------------------------------------------------------===//

SinOpAdaptor::SinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.sin", odsAttrs.getContext());
}

SinOpAdaptor::SinOpAdaptor(SinOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SinOpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SinOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SinOp::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SinOp::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SinOp::getRes() {
  return *getODSResults(0).begin();
}

void SinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void SinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SinOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SinOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SinOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult SinOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void SinOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SinOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::SqrtOp definitions
//===----------------------------------------------------------------------===//

SqrtOpAdaptor::SqrtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.sqrt", odsAttrs.getContext());
}

SqrtOpAdaptor::SqrtOpAdaptor(SqrtOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange SqrtOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> SqrtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange SqrtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SqrtOpAdaptor::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr SqrtOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult SqrtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> SqrtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range SqrtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SqrtOp::getIn() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange SqrtOp::getInMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> SqrtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range SqrtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value SqrtOp::getRes() {
  return *getODSResults(0).begin();
}

void SqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value in) {
  odsState.addOperands(in);
  odsState.addTypes(res);
}

void SqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value in) {
  odsState.addOperands(in);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(SqrtOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void SqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value in) {
  odsState.addOperands(in);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void SqrtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void SqrtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(SqrtOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult SqrtOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult SqrtOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult SqrtOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void SqrtOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::SqrtOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::StackRestoreOp definitions
//===----------------------------------------------------------------------===//

StackRestoreOpAdaptor::StackRestoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.stackrestore", odsAttrs.getContext());
}

StackRestoreOpAdaptor::StackRestoreOpAdaptor(StackRestoreOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StackRestoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StackRestoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StackRestoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StackRestoreOpAdaptor::getPtr() {
  return *getODSOperands(0).begin();
}

::mlir::DictionaryAttr StackRestoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult StackRestoreOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StackRestoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StackRestoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StackRestoreOp::getPtr() {
  return *getODSOperands(0).begin();
}

::mlir::MutableOperandRange StackRestoreOp::getPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> StackRestoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StackRestoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void StackRestoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value ptr) {
  odsState.addOperands(ptr);
}

void StackRestoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr) {
  odsState.addOperands(ptr);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StackRestoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StackRestoreOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps2(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StackRestoreOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult StackRestoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand ptrRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> ptrOperands(ptrRawOperands);  ::llvm::SMLoc ptrOperandsLoc;
  (void)ptrOperandsLoc;

  ptrOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(ptrRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseOptionalAttrDict(result.attributes))
    return ::mlir::failure();
  ::mlir::Type odsBuildableType0 = ::mlir::LLVM::LLVMPointerType::get(::mlir::IntegerType::get(parser.getBuilder().getContext(), 8), 0);
  if (parser.resolveOperands(ptrOperands, odsBuildableType0, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void StackRestoreOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getPtr();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::StackRestoreOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::StackSaveOp definitions
//===----------------------------------------------------------------------===//

StackSaveOpAdaptor::StackSaveOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.stacksave", odsAttrs.getContext());
}

StackSaveOpAdaptor::StackSaveOpAdaptor(StackSaveOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StackSaveOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StackSaveOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StackSaveOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr StackSaveOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult StackSaveOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StackSaveOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StackSaveOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> StackSaveOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StackSaveOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StackSaveOp::getRes() {
  return *getODSResults(0).begin();
}

void StackSaveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void StackSaveOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StackSaveOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StackSaveOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StackSaveOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult StackSaveOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void StackSaveOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
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

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::StackSaveOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::StepVectorOp definitions
//===----------------------------------------------------------------------===//

StepVectorOpAdaptor::StepVectorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.experimental.stepvector", odsAttrs.getContext());
}

StepVectorOpAdaptor::StepVectorOpAdaptor(StepVectorOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange StepVectorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> StepVectorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange StepVectorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr StepVectorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult StepVectorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> StepVectorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range StepVectorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> StepVectorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range StepVectorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value StepVectorOp::getRes() {
  return *getODSResults(0).begin();
}

void StepVectorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void StepVectorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void StepVectorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult StepVectorOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult StepVectorOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult StepVectorOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
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

void StepVectorOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
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

void StepVectorOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::StepVectorOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UAddWithOverflowOp definitions
//===----------------------------------------------------------------------===//

UAddWithOverflowOpAdaptor::UAddWithOverflowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.uadd.with.overflow", odsAttrs.getContext());
}

UAddWithOverflowOpAdaptor::UAddWithOverflowOpAdaptor(UAddWithOverflowOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange UAddWithOverflowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UAddWithOverflowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UAddWithOverflowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr UAddWithOverflowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UAddWithOverflowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UAddWithOverflowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UAddWithOverflowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> UAddWithOverflowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UAddWithOverflowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UAddWithOverflowOp::getRes() {
  return *getODSResults(0).begin();
}

void UAddWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addTypes(res);
}

void UAddWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UAddWithOverflowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UAddWithOverflowOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult UAddWithOverflowOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::UAddWithOverflowOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UMaxOp definitions
//===----------------------------------------------------------------------===//

UMaxOpAdaptor::UMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.umax", odsAttrs.getContext());
}

UMaxOpAdaptor::UMaxOpAdaptor(UMaxOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange UMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UMaxOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value UMaxOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr UMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UMaxOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UMaxOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value UMaxOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange UMaxOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange UMaxOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> UMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UMaxOp::getRes() {
  return *getODSResults(0).begin();
}

void UMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void UMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(UMaxOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void UMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void UMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(UMaxOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult UMaxOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult UMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult UMaxOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void UMaxOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::UMaxOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UMinOp definitions
//===----------------------------------------------------------------------===//

UMinOpAdaptor::UMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.umin", odsAttrs.getContext());
}

UMinOpAdaptor::UMinOpAdaptor(UMinOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange UMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UMinOpAdaptor::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value UMinOpAdaptor::getB() {
  return *getODSOperands(1).begin();
}

::mlir::DictionaryAttr UMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UMinOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UMinOp::getA() {
  return *getODSOperands(0).begin();
}

::mlir::Value UMinOp::getB() {
  return *getODSOperands(1).begin();
}

::mlir::MutableOperandRange UMinOp::getAMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange UMinOp::getBMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> UMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UMinOp::getRes() {
  return *getODSResults(0).begin();
}

void UMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  odsState.addTypes(res);
}

void UMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);

        ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
        if (::mlir::succeeded(UMinOp::inferReturnTypes(odsBuilder.getContext(),
                      odsState.location, odsState.operands,
                      odsState.attributes.getDictionary(odsState.getContext()),
                      /*regions=*/{}, inferredReturnTypes)))
          odsState.addTypes(inferredReturnTypes);
        else
          ::llvm::report_fatal_error("Failed to infer result type(s).");
}

void UMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value a, ::mlir::Value b) {
  odsState.addOperands(a);
  odsState.addOperands(b);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

void UMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);

  ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
  if (::mlir::succeeded(UMinOp::inferReturnTypes(odsBuilder.getContext(),
          odsState.location, operands,
          odsState.attributes.getDictionary(odsState.getContext()),
          odsState.regions, inferredReturnTypes))) {
    assert(inferredReturnTypes.size() == 1u && "mismatched number of return types");
    odsState.addTypes(inferredReturnTypes);
  } else {
    ::llvm::report_fatal_error("Failed to infer result type(s).");
  }
}

::mlir::LogicalResult UMinOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult UMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::LogicalResult UMinOp::inferReturnTypes(::mlir::MLIRContext *context, ::llvm::Optional<::mlir::Location> location, ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions, ::llvm::SmallVectorImpl<::mlir::Type>&inferredReturnTypes) {
  inferredReturnTypes.resize(1);
  ::mlir::Builder odsBuilder(context);
  ::mlir::Type odsInferredType0 = operands[0].getType();
  inferredReturnTypes[0] = odsInferredType0;
  return ::mlir::success();
}

void UMinOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::UMinOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::UMulWithOverflowOp definitions
//===----------------------------------------------------------------------===//

UMulWithOverflowOpAdaptor::UMulWithOverflowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.umul.with.overflow", odsAttrs.getContext());
}

UMulWithOverflowOpAdaptor::UMulWithOverflowOpAdaptor(UMulWithOverflowOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange UMulWithOverflowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> UMulWithOverflowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange UMulWithOverflowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr UMulWithOverflowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult UMulWithOverflowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> UMulWithOverflowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range UMulWithOverflowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> UMulWithOverflowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range UMulWithOverflowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value UMulWithOverflowOp::getRes() {
  return *getODSResults(0).begin();
}

void UMulWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addTypes(res);
}

void UMulWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void UMulWithOverflowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult UMulWithOverflowOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult UMulWithOverflowOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::UMulWithOverflowOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::USubWithOverflowOp definitions
//===----------------------------------------------------------------------===//

USubWithOverflowOpAdaptor::USubWithOverflowOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.usub.with.overflow", odsAttrs.getContext());
}

USubWithOverflowOpAdaptor::USubWithOverflowOpAdaptor(USubWithOverflowOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange USubWithOverflowOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> USubWithOverflowOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange USubWithOverflowOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr USubWithOverflowOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult USubWithOverflowOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> USubWithOverflowOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range USubWithOverflowOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> USubWithOverflowOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range USubWithOverflowOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value USubWithOverflowOp::getRes() {
  return *getODSResults(0).begin();
}

void USubWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addTypes(res);
}

void USubWithOverflowOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void USubWithOverflowOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult USubWithOverflowOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult USubWithOverflowOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::USubWithOverflowOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPAShrOp definitions
//===----------------------------------------------------------------------===//

VPAShrOpAdaptor::VPAShrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.ashr", odsAttrs.getContext());
}

VPAShrOpAdaptor::VPAShrOpAdaptor(VPAShrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPAShrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPAShrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPAShrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPAShrOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPAShrOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPAShrOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPAShrOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPAShrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPAShrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPAShrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPAShrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPAShrOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPAShrOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPAShrOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPAShrOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPAShrOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPAShrOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPAShrOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPAShrOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPAShrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPAShrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPAShrOp::getRes() {
  return *getODSResults(0).begin();
}

void VPAShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPAShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPAShrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPAShrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPAShrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPAShrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPAShrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPAddOp definitions
//===----------------------------------------------------------------------===//

VPAddOpAdaptor::VPAddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.add", odsAttrs.getContext());
}

VPAddOpAdaptor::VPAddOpAdaptor(VPAddOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPAddOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPAddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPAddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPAddOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPAddOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPAddOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPAddOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPAddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPAddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPAddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPAddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPAddOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPAddOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPAddOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPAddOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPAddOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPAddOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPAddOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPAddOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPAddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPAddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPAddOp::getRes() {
  return *getODSResults(0).begin();
}

void VPAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPAddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPAddOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPAddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPAddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPAddOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPAndOp definitions
//===----------------------------------------------------------------------===//

VPAndOpAdaptor::VPAndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.and", odsAttrs.getContext());
}

VPAndOpAdaptor::VPAndOpAdaptor(VPAndOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPAndOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPAndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPAndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPAndOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPAndOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPAndOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPAndOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPAndOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPAndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPAndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPAndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPAndOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPAndOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPAndOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPAndOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPAndOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPAndOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPAndOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPAndOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPAndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPAndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPAndOp::getRes() {
  return *getODSResults(0).begin();
}

void VPAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPAndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPAndOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPAndOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPAndOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPAndOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPFAddOp definitions
//===----------------------------------------------------------------------===//

VPFAddOpAdaptor::VPFAddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.fadd", odsAttrs.getContext());
}

VPFAddOpAdaptor::VPFAddOpAdaptor(VPFAddOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPFAddOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPFAddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPFAddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFAddOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFAddOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFAddOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPFAddOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPFAddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPFAddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPFAddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPFAddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFAddOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFAddOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFAddOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPFAddOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPFAddOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFAddOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFAddOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFAddOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPFAddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPFAddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFAddOp::getRes() {
  return *getODSResults(0).begin();
}

void VPFAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPFAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPFAddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPFAddOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPFAddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPFAddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPFAddOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPFDivOp definitions
//===----------------------------------------------------------------------===//

VPFDivOpAdaptor::VPFDivOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.fdiv", odsAttrs.getContext());
}

VPFDivOpAdaptor::VPFDivOpAdaptor(VPFDivOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPFDivOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPFDivOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPFDivOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFDivOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFDivOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFDivOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPFDivOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPFDivOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPFDivOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPFDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPFDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFDivOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFDivOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFDivOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPFDivOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPFDivOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFDivOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFDivOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFDivOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPFDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPFDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFDivOp::getRes() {
  return *getODSResults(0).begin();
}

void VPFDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPFDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPFDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPFDivOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPFDivOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPFDivOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPFDivOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPFMulOp definitions
//===----------------------------------------------------------------------===//

VPFMulOpAdaptor::VPFMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.fmul", odsAttrs.getContext());
}

VPFMulOpAdaptor::VPFMulOpAdaptor(VPFMulOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPFMulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPFMulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPFMulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFMulOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFMulOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFMulOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPFMulOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPFMulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPFMulOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPFMulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPFMulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFMulOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFMulOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFMulOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPFMulOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPFMulOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFMulOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFMulOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFMulOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPFMulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPFMulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFMulOp::getRes() {
  return *getODSResults(0).begin();
}

void VPFMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPFMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPFMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPFMulOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPFMulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPFMulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPFMulOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPFNegOp definitions
//===----------------------------------------------------------------------===//

VPFNegOpAdaptor::VPFNegOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.fneg", odsAttrs.getContext());
}

VPFNegOpAdaptor::VPFNegOpAdaptor(VPFNegOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPFNegOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPFNegOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPFNegOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFNegOpAdaptor::getOp() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFNegOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFNegOpAdaptor::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr VPFNegOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPFNegOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPFNegOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPFNegOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFNegOp::getOp() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFNegOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFNegOp::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange VPFNegOp::getOpMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFNegOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFNegOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPFNegOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPFNegOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFNegOp::getRes() {
  return *getODSResults(0).begin();
}

void VPFNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value op, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(op);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPFNegOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value op, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(op);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPFNegOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPFNegOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPFNegOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPFNegOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPFNegOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPFPExtOp definitions
//===----------------------------------------------------------------------===//

VPFPExtOpAdaptor::VPFPExtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.fpext", odsAttrs.getContext());
}

VPFPExtOpAdaptor::VPFPExtOpAdaptor(VPFPExtOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPFPExtOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPFPExtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPFPExtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFPExtOpAdaptor::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFPExtOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFPExtOpAdaptor::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr VPFPExtOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPFPExtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPFPExtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPFPExtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFPExtOp::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFPExtOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFPExtOp::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange VPFPExtOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFPExtOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFPExtOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPFPExtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPFPExtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFPExtOp::getRes() {
  return *getODSResults(0).begin();
}

void VPFPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPFPExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPFPExtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPFPExtOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPFPExtOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPFPExtOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPFPExtOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPFPToSIOp definitions
//===----------------------------------------------------------------------===//

VPFPToSIOpAdaptor::VPFPToSIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.fptosi", odsAttrs.getContext());
}

VPFPToSIOpAdaptor::VPFPToSIOpAdaptor(VPFPToSIOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPFPToSIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPFPToSIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPFPToSIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFPToSIOpAdaptor::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFPToSIOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFPToSIOpAdaptor::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr VPFPToSIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPFPToSIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPFPToSIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPFPToSIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFPToSIOp::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFPToSIOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFPToSIOp::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange VPFPToSIOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFPToSIOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFPToSIOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPFPToSIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPFPToSIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFPToSIOp::getRes() {
  return *getODSResults(0).begin();
}

void VPFPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPFPToSIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPFPToSIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPFPToSIOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPFPToSIOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPFPToSIOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPFPToSIOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPFPToUIOp definitions
//===----------------------------------------------------------------------===//

VPFPToUIOpAdaptor::VPFPToUIOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.fptoui", odsAttrs.getContext());
}

VPFPToUIOpAdaptor::VPFPToUIOpAdaptor(VPFPToUIOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPFPToUIOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPFPToUIOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPFPToUIOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFPToUIOpAdaptor::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFPToUIOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFPToUIOpAdaptor::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr VPFPToUIOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPFPToUIOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPFPToUIOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPFPToUIOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFPToUIOp::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFPToUIOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFPToUIOp::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange VPFPToUIOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFPToUIOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFPToUIOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPFPToUIOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPFPToUIOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFPToUIOp::getRes() {
  return *getODSResults(0).begin();
}

void VPFPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPFPToUIOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPFPToUIOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPFPToUIOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPFPToUIOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPFPToUIOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPFPToUIOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPFPTruncOp definitions
//===----------------------------------------------------------------------===//

VPFPTruncOpAdaptor::VPFPTruncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.fptrunc", odsAttrs.getContext());
}

VPFPTruncOpAdaptor::VPFPTruncOpAdaptor(VPFPTruncOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPFPTruncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPFPTruncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPFPTruncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFPTruncOpAdaptor::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFPTruncOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFPTruncOpAdaptor::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr VPFPTruncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPFPTruncOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPFPTruncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPFPTruncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFPTruncOp::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFPTruncOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFPTruncOp::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange VPFPTruncOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFPTruncOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFPTruncOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPFPTruncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPFPTruncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFPTruncOp::getRes() {
  return *getODSResults(0).begin();
}

void VPFPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPFPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPFPTruncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPFPTruncOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPFPTruncOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPFPTruncOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPFPTruncOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPFRemOp definitions
//===----------------------------------------------------------------------===//

VPFRemOpAdaptor::VPFRemOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.frem", odsAttrs.getContext());
}

VPFRemOpAdaptor::VPFRemOpAdaptor(VPFRemOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPFRemOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPFRemOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPFRemOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFRemOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFRemOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFRemOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPFRemOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPFRemOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPFRemOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPFRemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPFRemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFRemOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFRemOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFRemOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPFRemOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPFRemOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFRemOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFRemOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFRemOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPFRemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPFRemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFRemOp::getRes() {
  return *getODSResults(0).begin();
}

void VPFRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPFRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPFRemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPFRemOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPFRemOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPFRemOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPFRemOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPFSubOp definitions
//===----------------------------------------------------------------------===//

VPFSubOpAdaptor::VPFSubOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.fsub", odsAttrs.getContext());
}

VPFSubOpAdaptor::VPFSubOpAdaptor(VPFSubOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPFSubOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPFSubOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPFSubOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFSubOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFSubOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFSubOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPFSubOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPFSubOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPFSubOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPFSubOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPFSubOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFSubOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFSubOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFSubOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPFSubOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPFSubOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFSubOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFSubOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFSubOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPFSubOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPFSubOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFSubOp::getRes() {
  return *getODSResults(0).begin();
}

void VPFSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPFSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPFSubOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPFSubOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPFSubOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPFSubOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPFSubOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPFmaOp definitions
//===----------------------------------------------------------------------===//

VPFmaOpAdaptor::VPFmaOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.fma", odsAttrs.getContext());
}

VPFmaOpAdaptor::VPFmaOpAdaptor(VPFmaOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPFmaOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPFmaOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPFmaOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFmaOpAdaptor::getOp1() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFmaOpAdaptor::getOp2() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFmaOpAdaptor::getOp3() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPFmaOpAdaptor::getMask() {
  return *getODSOperands(3).begin();
}

::mlir::Value VPFmaOpAdaptor::getEvl() {
  return *getODSOperands(4).begin();
}

::mlir::DictionaryAttr VPFmaOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPFmaOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPFmaOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPFmaOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFmaOp::getOp1() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPFmaOp::getOp2() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPFmaOp::getOp3() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPFmaOp::getMask() {
  return *getODSOperands(3).begin();
}

::mlir::Value VPFmaOp::getEvl() {
  return *getODSOperands(4).begin();
}

::mlir::MutableOperandRange VPFmaOp::getOp1Mutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFmaOp::getOp2Mutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFmaOp::getOp3Mutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFmaOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPFmaOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPFmaOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPFmaOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPFmaOp::getRes() {
  return *getODSResults(0).begin();
}

void VPFmaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value op1, ::mlir::Value op2, ::mlir::Value op3, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(op1);
  odsState.addOperands(op2);
  odsState.addOperands(op3);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPFmaOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value op1, ::mlir::Value op2, ::mlir::Value op3, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(op1);
  odsState.addOperands(op2);
  odsState.addOperands(op3);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPFmaOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPFmaOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPFmaOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPFmaOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPFmaOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPIntToPtrOp definitions
//===----------------------------------------------------------------------===//

VPIntToPtrOpAdaptor::VPIntToPtrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.inttoptr", odsAttrs.getContext());
}

VPIntToPtrOpAdaptor::VPIntToPtrOpAdaptor(VPIntToPtrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPIntToPtrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPIntToPtrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPIntToPtrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPIntToPtrOpAdaptor::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPIntToPtrOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPIntToPtrOpAdaptor::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr VPIntToPtrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPIntToPtrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPIntToPtrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPIntToPtrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPIntToPtrOp::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPIntToPtrOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPIntToPtrOp::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange VPIntToPtrOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPIntToPtrOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPIntToPtrOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPIntToPtrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPIntToPtrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPIntToPtrOp::getRes() {
  return *getODSResults(0).begin();
}

void VPIntToPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPIntToPtrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPIntToPtrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPIntToPtrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPIntToPtrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPIntToPtrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPIntToPtrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPLShrOp definitions
//===----------------------------------------------------------------------===//

VPLShrOpAdaptor::VPLShrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.lshr", odsAttrs.getContext());
}

VPLShrOpAdaptor::VPLShrOpAdaptor(VPLShrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPLShrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPLShrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPLShrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPLShrOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPLShrOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPLShrOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPLShrOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPLShrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPLShrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPLShrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPLShrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPLShrOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPLShrOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPLShrOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPLShrOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPLShrOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPLShrOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPLShrOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPLShrOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPLShrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPLShrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPLShrOp::getRes() {
  return *getODSResults(0).begin();
}

void VPLShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPLShrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPLShrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPLShrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPLShrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPLShrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPLShrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPLoadOp definitions
//===----------------------------------------------------------------------===//

VPLoadOpAdaptor::VPLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.load", odsAttrs.getContext());
}

VPLoadOpAdaptor::VPLoadOpAdaptor(VPLoadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPLoadOpAdaptor::getPtr() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPLoadOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPLoadOpAdaptor::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr VPLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPLoadOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPLoadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPLoadOp::getPtr() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPLoadOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPLoadOp::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange VPLoadOp::getPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPLoadOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPLoadOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPLoadOp::getRes() {
  return *getODSResults(0).begin();
}

void VPLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(ptr);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(ptr);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPLoadOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPLoadOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPLoadOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPMergeMinOp definitions
//===----------------------------------------------------------------------===//

VPMergeMinOpAdaptor::VPMergeMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.merge", odsAttrs.getContext());
}

VPMergeMinOpAdaptor::VPMergeMinOpAdaptor(VPMergeMinOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPMergeMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPMergeMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPMergeMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPMergeMinOpAdaptor::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPMergeMinOpAdaptor::getTrueVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPMergeMinOpAdaptor::getFalseVal() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPMergeMinOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPMergeMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPMergeMinOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPMergeMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPMergeMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPMergeMinOp::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPMergeMinOp::getTrueVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPMergeMinOp::getFalseVal() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPMergeMinOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPMergeMinOp::getCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPMergeMinOp::getTrueValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPMergeMinOp::getFalseValMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPMergeMinOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPMergeMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPMergeMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPMergeMinOp::getRes() {
  return *getODSResults(0).begin();
}

void VPMergeMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value cond, ::mlir::Value true_val, ::mlir::Value false_val, ::mlir::Value evl) {
  odsState.addOperands(cond);
  odsState.addOperands(true_val);
  odsState.addOperands(false_val);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPMergeMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::Value true_val, ::mlir::Value false_val, ::mlir::Value evl) {
  odsState.addOperands(cond);
  odsState.addOperands(true_val);
  odsState.addOperands(false_val);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPMergeMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPMergeMinOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPMergeMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPMergeMinOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPMergeMinOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPMulOp definitions
//===----------------------------------------------------------------------===//

VPMulOpAdaptor::VPMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.mul", odsAttrs.getContext());
}

VPMulOpAdaptor::VPMulOpAdaptor(VPMulOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPMulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPMulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPMulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPMulOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPMulOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPMulOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPMulOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPMulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPMulOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPMulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPMulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPMulOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPMulOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPMulOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPMulOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPMulOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPMulOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPMulOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPMulOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPMulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPMulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPMulOp::getRes() {
  return *getODSResults(0).begin();
}

void VPMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPMulOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPMulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPMulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPMulOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPOrOp definitions
//===----------------------------------------------------------------------===//

VPOrOpAdaptor::VPOrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.or", odsAttrs.getContext());
}

VPOrOpAdaptor::VPOrOpAdaptor(VPOrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPOrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPOrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPOrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPOrOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPOrOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPOrOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPOrOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPOrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPOrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPOrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPOrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPOrOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPOrOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPOrOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPOrOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPOrOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPOrOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPOrOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPOrOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPOrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPOrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPOrOp::getRes() {
  return *getODSResults(0).begin();
}

void VPOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPOrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPOrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPOrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPOrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPOrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPPtrToIntOp definitions
//===----------------------------------------------------------------------===//

VPPtrToIntOpAdaptor::VPPtrToIntOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.ptrtoint", odsAttrs.getContext());
}

VPPtrToIntOpAdaptor::VPPtrToIntOpAdaptor(VPPtrToIntOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPPtrToIntOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPPtrToIntOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPPtrToIntOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPPtrToIntOpAdaptor::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPPtrToIntOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPPtrToIntOpAdaptor::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr VPPtrToIntOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPPtrToIntOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPPtrToIntOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPPtrToIntOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPPtrToIntOp::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPPtrToIntOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPPtrToIntOp::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange VPPtrToIntOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPPtrToIntOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPPtrToIntOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPPtrToIntOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPPtrToIntOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPPtrToIntOp::getRes() {
  return *getODSResults(0).begin();
}

void VPPtrToIntOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPPtrToIntOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPPtrToIntOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPPtrToIntOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps10(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPPtrToIntOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPPtrToIntOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPPtrToIntOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPReduceAddOp definitions
//===----------------------------------------------------------------------===//

VPReduceAddOpAdaptor::VPReduceAddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.reduce.add", odsAttrs.getContext());
}

VPReduceAddOpAdaptor::VPReduceAddOpAdaptor(VPReduceAddOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPReduceAddOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPReduceAddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPReduceAddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceAddOpAdaptor::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceAddOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceAddOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceAddOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPReduceAddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPReduceAddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPReduceAddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPReduceAddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceAddOp::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceAddOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceAddOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceAddOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPReduceAddOp::getSatrtValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceAddOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceAddOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceAddOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPReduceAddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPReduceAddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceAddOp::getRes() {
  return *getODSResults(0).begin();
}

void VPReduceAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPReduceAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPReduceAddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPReduceAddOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPReduceAddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPReduceAddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPReduceAddOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPReduceAndOp definitions
//===----------------------------------------------------------------------===//

VPReduceAndOpAdaptor::VPReduceAndOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.reduce.and", odsAttrs.getContext());
}

VPReduceAndOpAdaptor::VPReduceAndOpAdaptor(VPReduceAndOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPReduceAndOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPReduceAndOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPReduceAndOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceAndOpAdaptor::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceAndOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceAndOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceAndOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPReduceAndOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPReduceAndOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPReduceAndOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPReduceAndOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceAndOp::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceAndOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceAndOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceAndOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPReduceAndOp::getSatrtValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceAndOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceAndOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceAndOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPReduceAndOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPReduceAndOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceAndOp::getRes() {
  return *getODSResults(0).begin();
}

void VPReduceAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPReduceAndOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPReduceAndOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPReduceAndOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPReduceAndOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPReduceAndOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPReduceAndOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPReduceFAddOp definitions
//===----------------------------------------------------------------------===//

VPReduceFAddOpAdaptor::VPReduceFAddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.reduce.fadd", odsAttrs.getContext());
}

VPReduceFAddOpAdaptor::VPReduceFAddOpAdaptor(VPReduceFAddOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPReduceFAddOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPReduceFAddOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPReduceFAddOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceFAddOpAdaptor::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceFAddOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceFAddOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceFAddOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPReduceFAddOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPReduceFAddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPReduceFAddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPReduceFAddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceFAddOp::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceFAddOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceFAddOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceFAddOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPReduceFAddOp::getSatrtValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceFAddOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceFAddOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceFAddOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPReduceFAddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPReduceFAddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceFAddOp::getRes() {
  return *getODSResults(0).begin();
}

void VPReduceFAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPReduceFAddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPReduceFAddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPReduceFAddOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPReduceFAddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPReduceFAddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPReduceFAddOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPReduceFMaxOp definitions
//===----------------------------------------------------------------------===//

VPReduceFMaxOpAdaptor::VPReduceFMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.reduce.fmax", odsAttrs.getContext());
}

VPReduceFMaxOpAdaptor::VPReduceFMaxOpAdaptor(VPReduceFMaxOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPReduceFMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPReduceFMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPReduceFMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceFMaxOpAdaptor::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceFMaxOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceFMaxOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceFMaxOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPReduceFMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPReduceFMaxOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPReduceFMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPReduceFMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceFMaxOp::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceFMaxOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceFMaxOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceFMaxOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPReduceFMaxOp::getSatrtValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceFMaxOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceFMaxOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceFMaxOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPReduceFMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPReduceFMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceFMaxOp::getRes() {
  return *getODSResults(0).begin();
}

void VPReduceFMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPReduceFMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPReduceFMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPReduceFMaxOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPReduceFMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPReduceFMaxOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPReduceFMaxOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPReduceFMinOp definitions
//===----------------------------------------------------------------------===//

VPReduceFMinOpAdaptor::VPReduceFMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.reduce.fmin", odsAttrs.getContext());
}

VPReduceFMinOpAdaptor::VPReduceFMinOpAdaptor(VPReduceFMinOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPReduceFMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPReduceFMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPReduceFMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceFMinOpAdaptor::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceFMinOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceFMinOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceFMinOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPReduceFMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPReduceFMinOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPReduceFMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPReduceFMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceFMinOp::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceFMinOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceFMinOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceFMinOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPReduceFMinOp::getSatrtValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceFMinOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceFMinOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceFMinOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPReduceFMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPReduceFMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceFMinOp::getRes() {
  return *getODSResults(0).begin();
}

void VPReduceFMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPReduceFMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPReduceFMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPReduceFMinOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPReduceFMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPReduceFMinOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPReduceFMinOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPReduceFMulOp definitions
//===----------------------------------------------------------------------===//

VPReduceFMulOpAdaptor::VPReduceFMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.reduce.fmul", odsAttrs.getContext());
}

VPReduceFMulOpAdaptor::VPReduceFMulOpAdaptor(VPReduceFMulOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPReduceFMulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPReduceFMulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPReduceFMulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceFMulOpAdaptor::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceFMulOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceFMulOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceFMulOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPReduceFMulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPReduceFMulOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPReduceFMulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPReduceFMulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceFMulOp::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceFMulOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceFMulOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceFMulOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPReduceFMulOp::getSatrtValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceFMulOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceFMulOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceFMulOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPReduceFMulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPReduceFMulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceFMulOp::getRes() {
  return *getODSResults(0).begin();
}

void VPReduceFMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPReduceFMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPReduceFMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPReduceFMulOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps12(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps8(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPReduceFMulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPReduceFMulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPReduceFMulOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPReduceMulOp definitions
//===----------------------------------------------------------------------===//

VPReduceMulOpAdaptor::VPReduceMulOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.reduce.mul", odsAttrs.getContext());
}

VPReduceMulOpAdaptor::VPReduceMulOpAdaptor(VPReduceMulOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPReduceMulOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPReduceMulOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPReduceMulOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceMulOpAdaptor::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceMulOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceMulOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceMulOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPReduceMulOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPReduceMulOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPReduceMulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPReduceMulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceMulOp::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceMulOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceMulOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceMulOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPReduceMulOp::getSatrtValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceMulOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceMulOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceMulOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPReduceMulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPReduceMulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceMulOp::getRes() {
  return *getODSResults(0).begin();
}

void VPReduceMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPReduceMulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPReduceMulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPReduceMulOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPReduceMulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPReduceMulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPReduceMulOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPReduceOrOp definitions
//===----------------------------------------------------------------------===//

VPReduceOrOpAdaptor::VPReduceOrOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.reduce.or", odsAttrs.getContext());
}

VPReduceOrOpAdaptor::VPReduceOrOpAdaptor(VPReduceOrOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPReduceOrOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPReduceOrOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPReduceOrOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceOrOpAdaptor::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceOrOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceOrOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceOrOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPReduceOrOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPReduceOrOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPReduceOrOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPReduceOrOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceOrOp::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceOrOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceOrOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceOrOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPReduceOrOp::getSatrtValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceOrOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceOrOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceOrOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPReduceOrOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPReduceOrOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceOrOp::getRes() {
  return *getODSResults(0).begin();
}

void VPReduceOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPReduceOrOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPReduceOrOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPReduceOrOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPReduceOrOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPReduceOrOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPReduceOrOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPReduceSMaxOp definitions
//===----------------------------------------------------------------------===//

VPReduceSMaxOpAdaptor::VPReduceSMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.reduce.smax", odsAttrs.getContext());
}

VPReduceSMaxOpAdaptor::VPReduceSMaxOpAdaptor(VPReduceSMaxOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPReduceSMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPReduceSMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPReduceSMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceSMaxOpAdaptor::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceSMaxOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceSMaxOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceSMaxOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPReduceSMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPReduceSMaxOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPReduceSMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPReduceSMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceSMaxOp::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceSMaxOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceSMaxOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceSMaxOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPReduceSMaxOp::getSatrtValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceSMaxOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceSMaxOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceSMaxOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPReduceSMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPReduceSMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceSMaxOp::getRes() {
  return *getODSResults(0).begin();
}

void VPReduceSMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPReduceSMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPReduceSMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPReduceSMaxOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPReduceSMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPReduceSMaxOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPReduceSMaxOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPReduceSMinOp definitions
//===----------------------------------------------------------------------===//

VPReduceSMinOpAdaptor::VPReduceSMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.reduce.smin", odsAttrs.getContext());
}

VPReduceSMinOpAdaptor::VPReduceSMinOpAdaptor(VPReduceSMinOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPReduceSMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPReduceSMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPReduceSMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceSMinOpAdaptor::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceSMinOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceSMinOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceSMinOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPReduceSMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPReduceSMinOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPReduceSMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPReduceSMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceSMinOp::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceSMinOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceSMinOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceSMinOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPReduceSMinOp::getSatrtValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceSMinOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceSMinOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceSMinOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPReduceSMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPReduceSMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceSMinOp::getRes() {
  return *getODSResults(0).begin();
}

void VPReduceSMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPReduceSMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPReduceSMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPReduceSMinOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPReduceSMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPReduceSMinOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPReduceSMinOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPReduceUMaxOp definitions
//===----------------------------------------------------------------------===//

VPReduceUMaxOpAdaptor::VPReduceUMaxOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.reduce.umax", odsAttrs.getContext());
}

VPReduceUMaxOpAdaptor::VPReduceUMaxOpAdaptor(VPReduceUMaxOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPReduceUMaxOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPReduceUMaxOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPReduceUMaxOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceUMaxOpAdaptor::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceUMaxOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceUMaxOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceUMaxOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPReduceUMaxOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPReduceUMaxOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPReduceUMaxOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPReduceUMaxOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceUMaxOp::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceUMaxOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceUMaxOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceUMaxOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPReduceUMaxOp::getSatrtValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceUMaxOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceUMaxOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceUMaxOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPReduceUMaxOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPReduceUMaxOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceUMaxOp::getRes() {
  return *getODSResults(0).begin();
}

void VPReduceUMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPReduceUMaxOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPReduceUMaxOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPReduceUMaxOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPReduceUMaxOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPReduceUMaxOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPReduceUMaxOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPReduceUMinOp definitions
//===----------------------------------------------------------------------===//

VPReduceUMinOpAdaptor::VPReduceUMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.reduce.umin", odsAttrs.getContext());
}

VPReduceUMinOpAdaptor::VPReduceUMinOpAdaptor(VPReduceUMinOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPReduceUMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPReduceUMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPReduceUMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceUMinOpAdaptor::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceUMinOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceUMinOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceUMinOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPReduceUMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPReduceUMinOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPReduceUMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPReduceUMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceUMinOp::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceUMinOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceUMinOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceUMinOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPReduceUMinOp::getSatrtValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceUMinOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceUMinOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceUMinOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPReduceUMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPReduceUMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceUMinOp::getRes() {
  return *getODSResults(0).begin();
}

void VPReduceUMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPReduceUMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPReduceUMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPReduceUMinOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPReduceUMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPReduceUMinOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPReduceUMinOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPReduceXorOp definitions
//===----------------------------------------------------------------------===//

VPReduceXorOpAdaptor::VPReduceXorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.reduce.xor", odsAttrs.getContext());
}

VPReduceXorOpAdaptor::VPReduceXorOpAdaptor(VPReduceXorOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPReduceXorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPReduceXorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPReduceXorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceXorOpAdaptor::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceXorOpAdaptor::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceXorOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceXorOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPReduceXorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPReduceXorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPReduceXorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPReduceXorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceXorOp::getSatrtValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPReduceXorOp::getVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPReduceXorOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPReduceXorOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPReduceXorOp::getSatrtValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceXorOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceXorOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPReduceXorOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPReduceXorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPReduceXorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPReduceXorOp::getRes() {
  return *getODSResults(0).begin();
}

void VPReduceXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPReduceXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value satrt_value, ::mlir::Value val, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(satrt_value);
  odsState.addOperands(val);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPReduceXorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPReduceXorOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPReduceXorOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPReduceXorOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPReduceXorOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPSDivOp definitions
//===----------------------------------------------------------------------===//

VPSDivOpAdaptor::VPSDivOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.sdiv", odsAttrs.getContext());
}

VPSDivOpAdaptor::VPSDivOpAdaptor(VPSDivOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPSDivOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPSDivOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPSDivOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSDivOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPSDivOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPSDivOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPSDivOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPSDivOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPSDivOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPSDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPSDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSDivOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPSDivOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPSDivOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPSDivOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPSDivOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSDivOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSDivOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSDivOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPSDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPSDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSDivOp::getRes() {
  return *getODSResults(0).begin();
}

void VPSDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPSDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPSDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPSDivOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPSDivOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPSDivOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPSDivOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPSExtOp definitions
//===----------------------------------------------------------------------===//

VPSExtOpAdaptor::VPSExtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.sext", odsAttrs.getContext());
}

VPSExtOpAdaptor::VPSExtOpAdaptor(VPSExtOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPSExtOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPSExtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPSExtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSExtOpAdaptor::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPSExtOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPSExtOpAdaptor::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr VPSExtOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPSExtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPSExtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPSExtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSExtOp::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPSExtOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPSExtOp::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange VPSExtOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSExtOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSExtOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPSExtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPSExtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSExtOp::getRes() {
  return *getODSResults(0).begin();
}

void VPSExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPSExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPSExtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPSExtOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPSExtOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPSExtOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPSExtOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPSIToFPOp definitions
//===----------------------------------------------------------------------===//

VPSIToFPOpAdaptor::VPSIToFPOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.sitofp", odsAttrs.getContext());
}

VPSIToFPOpAdaptor::VPSIToFPOpAdaptor(VPSIToFPOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPSIToFPOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPSIToFPOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPSIToFPOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSIToFPOpAdaptor::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPSIToFPOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPSIToFPOpAdaptor::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr VPSIToFPOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPSIToFPOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPSIToFPOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPSIToFPOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSIToFPOp::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPSIToFPOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPSIToFPOp::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange VPSIToFPOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSIToFPOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSIToFPOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPSIToFPOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPSIToFPOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSIToFPOp::getRes() {
  return *getODSResults(0).begin();
}

void VPSIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPSIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPSIToFPOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPSIToFPOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPSIToFPOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPSIToFPOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPSIToFPOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPSRemOp definitions
//===----------------------------------------------------------------------===//

VPSRemOpAdaptor::VPSRemOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.srem", odsAttrs.getContext());
}

VPSRemOpAdaptor::VPSRemOpAdaptor(VPSRemOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPSRemOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPSRemOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPSRemOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSRemOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPSRemOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPSRemOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPSRemOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPSRemOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPSRemOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPSRemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPSRemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSRemOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPSRemOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPSRemOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPSRemOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPSRemOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSRemOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSRemOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSRemOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPSRemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPSRemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSRemOp::getRes() {
  return *getODSResults(0).begin();
}

void VPSRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPSRemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPSRemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPSRemOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPSRemOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPSRemOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPSRemOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPSelectMinOp definitions
//===----------------------------------------------------------------------===//

VPSelectMinOpAdaptor::VPSelectMinOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.select", odsAttrs.getContext());
}

VPSelectMinOpAdaptor::VPSelectMinOpAdaptor(VPSelectMinOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPSelectMinOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPSelectMinOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPSelectMinOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSelectMinOpAdaptor::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPSelectMinOpAdaptor::getTrueVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPSelectMinOpAdaptor::getFalseVal() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPSelectMinOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPSelectMinOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPSelectMinOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPSelectMinOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPSelectMinOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSelectMinOp::getCond() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPSelectMinOp::getTrueVal() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPSelectMinOp::getFalseVal() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPSelectMinOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPSelectMinOp::getCondMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSelectMinOp::getTrueValMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSelectMinOp::getFalseValMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSelectMinOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPSelectMinOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPSelectMinOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSelectMinOp::getRes() {
  return *getODSResults(0).begin();
}

void VPSelectMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value cond, ::mlir::Value true_val, ::mlir::Value false_val, ::mlir::Value evl) {
  odsState.addOperands(cond);
  odsState.addOperands(true_val);
  odsState.addOperands(false_val);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPSelectMinOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value cond, ::mlir::Value true_val, ::mlir::Value false_val, ::mlir::Value evl) {
  odsState.addOperands(cond);
  odsState.addOperands(true_val);
  odsState.addOperands(false_val);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPSelectMinOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPSelectMinOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPSelectMinOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPSelectMinOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPSelectMinOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPShlOp definitions
//===----------------------------------------------------------------------===//

VPShlOpAdaptor::VPShlOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.shl", odsAttrs.getContext());
}

VPShlOpAdaptor::VPShlOpAdaptor(VPShlOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPShlOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPShlOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPShlOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPShlOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPShlOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPShlOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPShlOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPShlOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPShlOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPShlOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPShlOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPShlOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPShlOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPShlOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPShlOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPShlOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPShlOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPShlOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPShlOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPShlOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPShlOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPShlOp::getRes() {
  return *getODSResults(0).begin();
}

void VPShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPShlOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPShlOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPShlOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPShlOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPShlOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPShlOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPStoreOp definitions
//===----------------------------------------------------------------------===//

VPStoreOpAdaptor::VPStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.store", odsAttrs.getContext());
}

VPStoreOpAdaptor::VPStoreOpAdaptor(VPStoreOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPStoreOpAdaptor::getVal() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPStoreOpAdaptor::getPtr() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPStoreOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPStoreOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPStoreOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPStoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPStoreOp::getVal() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPStoreOp::getPtr() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPStoreOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPStoreOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPStoreOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPStoreOp::getPtrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPStoreOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPStoreOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void VPStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value val, ::mlir::Value ptr, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(val);
  odsState.addOperands(ptr);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
}

void VPStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value val, ::mlir::Value ptr, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(val);
  odsState.addOperands(ptr);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPStoreOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPStoreOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPStoreOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPStridedLoadOp definitions
//===----------------------------------------------------------------------===//

VPStridedLoadOpAdaptor::VPStridedLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.experimental.vp.strided.load", odsAttrs.getContext());
}

VPStridedLoadOpAdaptor::VPStridedLoadOpAdaptor(VPStridedLoadOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPStridedLoadOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPStridedLoadOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPStridedLoadOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPStridedLoadOpAdaptor::getPtr() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPStridedLoadOpAdaptor::getStride() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPStridedLoadOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPStridedLoadOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPStridedLoadOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPStridedLoadOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPStridedLoadOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPStridedLoadOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPStridedLoadOp::getPtr() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPStridedLoadOp::getStride() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPStridedLoadOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPStridedLoadOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPStridedLoadOp::getPtrMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPStridedLoadOp::getStrideMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPStridedLoadOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPStridedLoadOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPStridedLoadOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPStridedLoadOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPStridedLoadOp::getRes() {
  return *getODSResults(0).begin();
}

void VPStridedLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptr, ::mlir::Value stride, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(ptr);
  odsState.addOperands(stride);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPStridedLoadOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptr, ::mlir::Value stride, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(ptr);
  odsState.addOperands(stride);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPStridedLoadOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPStridedLoadOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPStridedLoadOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPStridedLoadOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPStridedStoreOp definitions
//===----------------------------------------------------------------------===//

VPStridedStoreOpAdaptor::VPStridedStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.experimental.vp.strided.store", odsAttrs.getContext());
}

VPStridedStoreOpAdaptor::VPStridedStoreOpAdaptor(VPStridedStoreOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPStridedStoreOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPStridedStoreOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPStridedStoreOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPStridedStoreOpAdaptor::getVal() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPStridedStoreOpAdaptor::getPtr() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPStridedStoreOpAdaptor::getStride() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPStridedStoreOpAdaptor::getMask() {
  return *getODSOperands(3).begin();
}

::mlir::Value VPStridedStoreOpAdaptor::getEvl() {
  return *getODSOperands(4).begin();
}

::mlir::DictionaryAttr VPStridedStoreOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPStridedStoreOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPStridedStoreOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPStridedStoreOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPStridedStoreOp::getVal() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPStridedStoreOp::getPtr() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPStridedStoreOp::getStride() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPStridedStoreOp::getMask() {
  return *getODSOperands(3).begin();
}

::mlir::Value VPStridedStoreOp::getEvl() {
  return *getODSOperands(4).begin();
}

::mlir::MutableOperandRange VPStridedStoreOp::getValMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPStridedStoreOp::getPtrMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPStridedStoreOp::getStrideMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPStridedStoreOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPStridedStoreOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(4);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPStridedStoreOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPStridedStoreOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void VPStridedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value val, ::mlir::Value ptr, ::mlir::Value stride, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(val);
  odsState.addOperands(ptr);
  odsState.addOperands(stride);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
}

void VPStridedStoreOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value val, ::mlir::Value ptr, ::mlir::Value stride, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(val);
  odsState.addOperands(ptr);
  odsState.addOperands(stride);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPStridedStoreOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 5u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPStridedStoreOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps9(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps11(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup4 = getODSOperands(4);

    for (auto v : valueGroup4) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPStridedStoreOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPStridedStoreOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPSubOp definitions
//===----------------------------------------------------------------------===//

VPSubOpAdaptor::VPSubOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.sub", odsAttrs.getContext());
}

VPSubOpAdaptor::VPSubOpAdaptor(VPSubOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPSubOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPSubOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPSubOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSubOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPSubOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPSubOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPSubOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPSubOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPSubOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPSubOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPSubOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSubOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPSubOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPSubOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPSubOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPSubOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSubOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSubOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPSubOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPSubOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPSubOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPSubOp::getRes() {
  return *getODSResults(0).begin();
}

void VPSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPSubOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPSubOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPSubOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPSubOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPSubOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPSubOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPTruncOp definitions
//===----------------------------------------------------------------------===//

VPTruncOpAdaptor::VPTruncOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.trunc", odsAttrs.getContext());
}

VPTruncOpAdaptor::VPTruncOpAdaptor(VPTruncOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPTruncOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPTruncOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPTruncOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPTruncOpAdaptor::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPTruncOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPTruncOpAdaptor::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr VPTruncOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPTruncOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPTruncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPTruncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPTruncOp::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPTruncOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPTruncOp::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange VPTruncOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPTruncOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPTruncOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPTruncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPTruncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPTruncOp::getRes() {
  return *getODSResults(0).begin();
}

void VPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPTruncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPTruncOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPTruncOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPTruncOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPTruncOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPTruncOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPUDivOp definitions
//===----------------------------------------------------------------------===//

VPUDivOpAdaptor::VPUDivOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.udiv", odsAttrs.getContext());
}

VPUDivOpAdaptor::VPUDivOpAdaptor(VPUDivOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPUDivOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPUDivOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPUDivOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPUDivOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPUDivOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPUDivOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPUDivOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPUDivOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPUDivOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPUDivOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPUDivOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPUDivOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPUDivOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPUDivOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPUDivOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPUDivOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPUDivOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPUDivOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPUDivOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPUDivOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPUDivOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPUDivOp::getRes() {
  return *getODSResults(0).begin();
}

void VPUDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPUDivOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPUDivOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPUDivOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPUDivOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPUDivOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPUDivOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPUIToFPOp definitions
//===----------------------------------------------------------------------===//

VPUIToFPOpAdaptor::VPUIToFPOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.uitofp", odsAttrs.getContext());
}

VPUIToFPOpAdaptor::VPUIToFPOpAdaptor(VPUIToFPOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPUIToFPOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPUIToFPOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPUIToFPOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPUIToFPOpAdaptor::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPUIToFPOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPUIToFPOpAdaptor::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr VPUIToFPOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPUIToFPOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPUIToFPOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPUIToFPOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPUIToFPOp::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPUIToFPOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPUIToFPOp::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange VPUIToFPOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPUIToFPOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPUIToFPOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPUIToFPOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPUIToFPOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPUIToFPOp::getRes() {
  return *getODSResults(0).begin();
}

void VPUIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPUIToFPOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPUIToFPOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPUIToFPOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPUIToFPOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPUIToFPOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPUIToFPOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPURemOp definitions
//===----------------------------------------------------------------------===//

VPURemOpAdaptor::VPURemOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.urem", odsAttrs.getContext());
}

VPURemOpAdaptor::VPURemOpAdaptor(VPURemOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPURemOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPURemOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPURemOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPURemOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPURemOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPURemOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPURemOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPURemOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPURemOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPURemOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPURemOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPURemOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPURemOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPURemOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPURemOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPURemOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPURemOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPURemOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPURemOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPURemOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPURemOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPURemOp::getRes() {
  return *getODSResults(0).begin();
}

void VPURemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPURemOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPURemOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPURemOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPURemOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPURemOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPURemOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPXorOp definitions
//===----------------------------------------------------------------------===//

VPXorOpAdaptor::VPXorOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.xor", odsAttrs.getContext());
}

VPXorOpAdaptor::VPXorOpAdaptor(VPXorOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPXorOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPXorOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPXorOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPXorOpAdaptor::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPXorOpAdaptor::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPXorOpAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPXorOpAdaptor::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::DictionaryAttr VPXorOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPXorOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPXorOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPXorOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPXorOp::getLhs() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPXorOp::getRhs() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPXorOp::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::Value VPXorOp::getEvl() {
  return *getODSOperands(3).begin();
}

::mlir::MutableOperandRange VPXorOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPXorOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPXorOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPXorOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(3);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPXorOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPXorOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPXorOp::getRes() {
  return *getODSResults(0).begin();
}

void VPXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPXorOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPXorOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 4u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPXorOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup3 = getODSOperands(3);

    for (auto v : valueGroup3) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPXorOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPXorOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPXorOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::VPZExtOp definitions
//===----------------------------------------------------------------------===//

VPZExtOpAdaptor::VPZExtOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vp.zext", odsAttrs.getContext());
}

VPZExtOpAdaptor::VPZExtOpAdaptor(VPZExtOp op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange VPZExtOpAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> VPZExtOpAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange VPZExtOpAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPZExtOpAdaptor::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPZExtOpAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPZExtOpAdaptor::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr VPZExtOpAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult VPZExtOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> VPZExtOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range VPZExtOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPZExtOp::getSrc() {
  return *getODSOperands(0).begin();
}

::mlir::Value VPZExtOp::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Value VPZExtOp::getEvl() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange VPZExtOp::getSrcMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPZExtOp::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange VPZExtOp::getEvlMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> VPZExtOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range VPZExtOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value VPZExtOp::getRes() {
  return *getODSResults(0).begin();
}

void VPZExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  odsState.addTypes(res);
}

void VPZExtOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value src, ::mlir::Value mask, ::mlir::Value evl) {
  odsState.addOperands(src);
  odsState.addOperands(mask);
  odsState.addOperands(evl);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void VPZExtOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult VPZExtOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps6(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps7(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps4(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult VPZExtOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void VPZExtOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::VPZExtOp)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::masked_compressstore definitions
//===----------------------------------------------------------------------===//

masked_compressstoreAdaptor::masked_compressstoreAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.masked.compressstore", odsAttrs.getContext());
}

masked_compressstoreAdaptor::masked_compressstoreAdaptor(masked_compressstore op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange masked_compressstoreAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> masked_compressstoreAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange masked_compressstoreAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr masked_compressstoreAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult masked_compressstoreAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> masked_compressstore::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range masked_compressstore::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> masked_compressstore::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range masked_compressstore::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void masked_compressstore::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
}

void masked_compressstore::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void masked_compressstore::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult masked_compressstore::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult masked_compressstore::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::masked_compressstore)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::masked_expandload definitions
//===----------------------------------------------------------------------===//

masked_expandloadAdaptor::masked_expandloadAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.masked.expandload", odsAttrs.getContext());
}

masked_expandloadAdaptor::masked_expandloadAdaptor(masked_expandload op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange masked_expandloadAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> masked_expandloadAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange masked_expandloadAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr masked_expandloadAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult masked_expandloadAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> masked_expandload::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range masked_expandload::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> masked_expandload::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range masked_expandload::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value masked_expandload::getRes() {
  return *getODSResults(0).begin();
}

void masked_expandload::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  odsState.addTypes(res);
}

void masked_expandload::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::Value odsArg_2) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addOperands(odsArg_2);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void masked_expandload::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult masked_expandload::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult masked_expandload::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::masked_expandload)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::masked_gather definitions
//===----------------------------------------------------------------------===//

masked_gatherAdaptor::masked_gatherAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.masked.gather", odsAttrs.getContext());
}

masked_gatherAdaptor::masked_gatherAdaptor(masked_gather op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange masked_gatherAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> masked_gatherAdaptor::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperands.size() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::ValueRange masked_gatherAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value masked_gatherAdaptor::getPtrs() {
  return *getODSOperands(0).begin();
}

::mlir::Value masked_gatherAdaptor::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::ValueRange masked_gatherAdaptor::getPassThru() {
  return getODSOperands(2);
}

::mlir::DictionaryAttr masked_gatherAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr masked_gatherAdaptor::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, masked_gather::getAlignmentAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t masked_gatherAdaptor::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult masked_gatherAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.masked.gather' op ""requires attribute 'alignment'");
    if (namedAttrIt->getName() == masked_gather::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.intr.masked.gather' op ""attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> masked_gather::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {false, false, true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 2) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range masked_gather::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value masked_gather::getPtrs() {
  return *getODSOperands(0).begin();
}

::mlir::Value masked_gather::getMask() {
  return *getODSOperands(1).begin();
}

::mlir::Operation::operand_range masked_gather::getPassThru() {
  return getODSOperands(2);
}

::mlir::MutableOperandRange masked_gather::getPtrsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange masked_gather::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange masked_gather::getPassThruMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> masked_gather::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range masked_gather::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value masked_gather::getRes() {
  return *getODSResults(0).begin();
}

::mlir::IntegerAttr masked_gather::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlignmentAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t masked_gather::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr.getValue().getZExtValue();
}

void masked_gather::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void masked_gather::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    if (resultType) odsState.addTypes(resultType);
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void masked_gather::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptrs, ::mlir::Value mask, ::mlir::ValueRange pass_thru, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  odsState.addTypes(res);
}

void masked_gather::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptrs, ::mlir::Value mask, ::mlir::ValueRange pass_thru, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void masked_gather::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value ptrs, ::mlir::Value mask, ::mlir::ValueRange pass_thru, uint32_t alignment) {
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  odsState.addTypes(res);
}

void masked_gather::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value ptrs, ::mlir::Value mask, ::mlir::ValueRange pass_thru, uint32_t alignment) {
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addOperands(pass_thru);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void masked_gather::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() >= 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult masked_gather::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'alignment'");
    if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps5(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult masked_gather::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult masked_gather::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> allOperands;
  ::llvm::ArrayRef<::mlir::Type> allOperandTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;
  ::llvm::SMLoc allOperandLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(allOperands))
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
  if (parser.resolveOperands(allOperands, allOperandTypes, allOperandLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void masked_gather::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getOperands();
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{});
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getOperation()->getOperandTypes(), getOperation()->getResultTypes());
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::masked_gather)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::masked_scatter definitions
//===----------------------------------------------------------------------===//

masked_scatterAdaptor::masked_scatterAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.masked.scatter", odsAttrs.getContext());
}

masked_scatterAdaptor::masked_scatterAdaptor(masked_scatter op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange masked_scatterAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> masked_scatterAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange masked_scatterAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::Value masked_scatterAdaptor::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value masked_scatterAdaptor::getPtrs() {
  return *getODSOperands(1).begin();
}

::mlir::Value masked_scatterAdaptor::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::DictionaryAttr masked_scatterAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::IntegerAttr masked_scatterAdaptor::getAlignmentAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, masked_scatter::getAlignmentAttrName(*odsOpName)).cast<::mlir::IntegerAttr>();
    return attr;
}

uint32_t masked_scatterAdaptor::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr.getValue().getZExtValue();
}

::mlir::LogicalResult masked_scatterAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitError(loc, "'llvm.intr.masked.scatter' op ""requires attribute 'alignment'");
    if (namedAttrIt->getName() == masked_scatter::getAlignmentAttrName(*odsOpName)) {
      tblgen_alignment = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (tblgen_alignment && !(((tblgen_alignment.isa<::mlir::IntegerAttr>())) && ((tblgen_alignment.cast<::mlir::IntegerAttr>().getType().isSignlessInteger(32)))))
    return emitError(loc, "'llvm.intr.masked.scatter' op ""attribute 'alignment' failed to satisfy constraint: 32-bit signless integer attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> masked_scatter::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range masked_scatter::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value masked_scatter::getValue() {
  return *getODSOperands(0).begin();
}

::mlir::Value masked_scatter::getPtrs() {
  return *getODSOperands(1).begin();
}

::mlir::Value masked_scatter::getMask() {
  return *getODSOperands(2).begin();
}

::mlir::MutableOperandRange masked_scatter::getValueMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange masked_scatter::getPtrsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange masked_scatter::getMaskMutable() {
  auto range = getODSOperandIndexAndLength(2);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> masked_scatter::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range masked_scatter::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::IntegerAttr masked_scatter::getAlignmentAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getAlignmentAttrName()).cast<::mlir::IntegerAttr>();
}

uint32_t masked_scatter::getAlignment() {
  auto attr = getAlignmentAttr();
  return attr.getValue().getZExtValue();
}

void masked_scatter::setAlignmentAttr(::mlir::IntegerAttr attr) {
  (*this)->setAttr(getAlignmentAttrName(), attr);
}

void masked_scatter::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Type resultType, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    assert(isCompatibleType(resultType) && "result must be an LLVM type");
    assert(resultType.isa<LLVMVoidType>() &&
           "for zero-result operands, only 'void' is accepted as result type");
    build(odsBuilder, odsState, operands, attributes);
  
}

void masked_scatter::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange operands, ArrayRef<NamedAttribute> attributes) {
    odsState.addOperands(operands);
    for (auto namedAttr : attributes)
      odsState.addAttribute(namedAttr.getName(), namedAttr.getValue());
  
}

void masked_scatter::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value ptrs, ::mlir::Value mask, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(value);
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
}

void masked_scatter::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value ptrs, ::mlir::Value mask, ::mlir::IntegerAttr alignment) {
  odsState.addOperands(value);
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), alignment);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void masked_scatter::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value ptrs, ::mlir::Value mask, uint32_t alignment) {
  odsState.addOperands(value);
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
}

void masked_scatter::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value ptrs, ::mlir::Value mask, uint32_t alignment) {
  odsState.addOperands(value);
  odsState.addOperands(ptrs);
  odsState.addOperands(mask);
  odsState.addAttribute(getAlignmentAttrName(odsState.name), odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), alignment));
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void masked_scatter::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 3u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult masked_scatter::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_alignment;
  while (true) {
    if (namedAttrIt == namedAttrRange.end())
      return emitOpError("requires attribute 'alignment'");
    if (namedAttrIt->getName() == getAlignmentAttrName()) {
      tblgen_alignment = namedAttrIt->getValue();
      break;
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps0(*this, tblgen_alignment, "alignment")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup2 = getODSOperands(2);

    for (auto v : valueGroup2) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult masked_scatter::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult masked_scatter::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand valueRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> valueOperands(valueRawOperands);  ::llvm::SMLoc valueOperandsLoc;
  (void)valueOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand ptrsRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> ptrsOperands(ptrsRawOperands);  ::llvm::SMLoc ptrsOperandsLoc;
  (void)ptrsOperandsLoc;
  ::mlir::OpAsmParser::UnresolvedOperand maskRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> maskOperands(maskRawOperands);  ::llvm::SMLoc maskOperandsLoc;
  (void)maskOperandsLoc;
  ::mlir::Type valueRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> valueTypes(valueRawTypes);
  ::mlir::Type maskRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> maskTypes(maskRawTypes);
  ::mlir::Type ptrsRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> ptrsTypes(ptrsRawTypes);

  valueOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(valueRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  ptrsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(ptrsRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseComma())
    return ::mlir::failure();

  maskOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(maskRawOperands[0]))
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
  if (parser.parseComma())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    maskRawTypes[0] = type;
  }
  if (parser.parseKeyword("into"))
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    ptrsRawTypes[0] = type;
  }
  if (parser.resolveOperands(valueOperands, valueTypes, valueOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(ptrsOperands, ptrsTypes, ptrsOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(maskOperands, maskTypes, maskOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void masked_scatter::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getValue();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getPtrs();
  _odsPrinter << ",";
  _odsPrinter << ' ';
  _odsPrinter << getMask();
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
  _odsPrinter << ",";
  _odsPrinter << ' ';
  {
    auto type = getMask().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ' ' << "into";
  _odsPrinter << ' ';
  {
    auto type = getPtrs().getType();
    if (auto validType = type.dyn_cast<::mlir::Type>())
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::masked_scatter)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_add definitions
//===----------------------------------------------------------------------===//

vector_reduce_addAdaptor::vector_reduce_addAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vector.reduce.add", odsAttrs.getContext());
}

vector_reduce_addAdaptor::vector_reduce_addAdaptor(vector_reduce_add op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vector_reduce_addAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_addAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_addAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_addAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_addAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vector_reduce_add::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_add::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_add::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_add::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_add::getRes() {
  return *getODSResults(0).begin();
}

void vector_reduce_add::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_add::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_add::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_add::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vector_reduce_add::verifyInvariants() {
  return verifyInvariantsImpl();
}

void vector_reduce_add::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vector_reduce_add)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_and definitions
//===----------------------------------------------------------------------===//

vector_reduce_andAdaptor::vector_reduce_andAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vector.reduce.and", odsAttrs.getContext());
}

vector_reduce_andAdaptor::vector_reduce_andAdaptor(vector_reduce_and op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vector_reduce_andAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_andAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_andAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_andAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_andAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vector_reduce_and::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_and::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_and::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_and::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_and::getRes() {
  return *getODSResults(0).begin();
}

void vector_reduce_and::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_and::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_and::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_and::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vector_reduce_and::verifyInvariants() {
  return verifyInvariantsImpl();
}

void vector_reduce_and::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vector_reduce_and)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_fadd definitions
//===----------------------------------------------------------------------===//

vector_reduce_faddAdaptor::vector_reduce_faddAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vector.reduce.fadd", odsAttrs.getContext());
}

vector_reduce_faddAdaptor::vector_reduce_faddAdaptor(vector_reduce_fadd op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vector_reduce_faddAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_faddAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_faddAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_faddAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr vector_reduce_faddAdaptor::getReassocAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, vector_reduce_fadd::getReassocAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
    return attr;
}

bool vector_reduce_faddAdaptor::getReassoc() {
  auto attr = getReassocAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

::mlir::LogicalResult vector_reduce_faddAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_reassoc;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == vector_reduce_fadd::getReassocAttrName(*odsOpName)) {
      tblgen_reassoc = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_reassoc && !((tblgen_reassoc.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'llvm.intr.vector.reduce.fadd' op ""attribute 'reassoc' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vector_reduce_fadd::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_fadd::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_fadd::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_fadd::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_fadd::getRes() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr vector_reduce_fadd::getReassocAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getReassocAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool vector_reduce_fadd::getReassoc() {
  auto attr = getReassocAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

void vector_reduce_fadd::setReassocAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getReassocAttrName(), attr);
}

void vector_reduce_fadd::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::BoolAttr reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  if (reassoc) {
  odsState.addAttribute(getReassocAttrName(odsState.name), reassoc);
  }
  odsState.addTypes(res);
}

void vector_reduce_fadd::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::BoolAttr reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  if (reassoc) {
  odsState.addAttribute(getReassocAttrName(odsState.name), reassoc);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_fadd::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, bool reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addAttribute(getReassocAttrName(odsState.name), odsBuilder.getBoolAttr(reassoc));
  odsState.addTypes(res);
}

void vector_reduce_fadd::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, bool reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addAttribute(getReassocAttrName(odsState.name), odsBuilder.getBoolAttr(reassoc));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_fadd::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_fadd::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_reassoc;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getReassocAttrName()) {
      tblgen_reassoc = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps2(*this, tblgen_reassoc, "reassoc")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vector_reduce_fadd::verifyInvariants() {
  return verifyInvariantsImpl();
}

void vector_reduce_fadd::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vector_reduce_fadd)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_fmax definitions
//===----------------------------------------------------------------------===//

vector_reduce_fmaxAdaptor::vector_reduce_fmaxAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vector.reduce.fmax", odsAttrs.getContext());
}

vector_reduce_fmaxAdaptor::vector_reduce_fmaxAdaptor(vector_reduce_fmax op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vector_reduce_fmaxAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_fmaxAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_fmaxAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_fmaxAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_fmaxAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vector_reduce_fmax::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_fmax::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_fmax::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_fmax::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_fmax::getRes() {
  return *getODSResults(0).begin();
}

void vector_reduce_fmax::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_fmax::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_fmax::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_fmax::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vector_reduce_fmax::verifyInvariants() {
  return verifyInvariantsImpl();
}

void vector_reduce_fmax::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vector_reduce_fmax)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_fmin definitions
//===----------------------------------------------------------------------===//

vector_reduce_fminAdaptor::vector_reduce_fminAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vector.reduce.fmin", odsAttrs.getContext());
}

vector_reduce_fminAdaptor::vector_reduce_fminAdaptor(vector_reduce_fmin op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vector_reduce_fminAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_fminAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_fminAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_fminAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_fminAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vector_reduce_fmin::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_fmin::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_fmin::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_fmin::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_fmin::getRes() {
  return *getODSResults(0).begin();
}

void vector_reduce_fmin::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_fmin::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_fmin::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_fmin::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vector_reduce_fmin::verifyInvariants() {
  return verifyInvariantsImpl();
}

void vector_reduce_fmin::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vector_reduce_fmin)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_fmul definitions
//===----------------------------------------------------------------------===//

vector_reduce_fmulAdaptor::vector_reduce_fmulAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vector.reduce.fmul", odsAttrs.getContext());
}

vector_reduce_fmulAdaptor::vector_reduce_fmulAdaptor(vector_reduce_fmul op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vector_reduce_fmulAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_fmulAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_fmulAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_fmulAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::BoolAttr vector_reduce_fmulAdaptor::getReassocAttr() {
  assert(odsAttrs && "no attributes when constructing adapter");
  auto attr = ::mlir::impl::getAttrFromSortedRange(odsAttrs.begin() + 0, odsAttrs.end() - 0, vector_reduce_fmul::getReassocAttrName(*odsOpName)).dyn_cast_or_null<::mlir::BoolAttr>();
    if (!attr)
      attr = ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false);
    return attr;
}

bool vector_reduce_fmulAdaptor::getReassoc() {
  auto attr = getReassocAttr();
    if (!attr)
      return ::mlir::Builder(odsAttrs.getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

::mlir::LogicalResult vector_reduce_fmulAdaptor::verify(::mlir::Location loc) {
  auto namedAttrRange = odsAttrs;
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_reassoc;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == vector_reduce_fmul::getReassocAttrName(*odsOpName)) {
      tblgen_reassoc = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (tblgen_reassoc && !((tblgen_reassoc.isa<::mlir::BoolAttr>())))
    return emitError(loc, "'llvm.intr.vector.reduce.fmul' op ""attribute 'reassoc' failed to satisfy constraint: bool attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vector_reduce_fmul::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_fmul::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_fmul::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_fmul::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_fmul::getRes() {
  return *getODSResults(0).begin();
}

::mlir::BoolAttr vector_reduce_fmul::getReassocAttr() {
  return ::mlir::impl::getAttrFromSortedRange((*this)->getAttrs().begin() + 0, (*this)->getAttrs().end() - 0, getReassocAttrName()).dyn_cast_or_null<::mlir::BoolAttr>();
}

bool vector_reduce_fmul::getReassoc() {
  auto attr = getReassocAttr();
    if (!attr)
      return ::mlir::Builder((*this)->getContext()).getBoolAttr(false).getValue();
  return attr.getValue();
}

void vector_reduce_fmul::setReassocAttr(::mlir::BoolAttr attr) {
  (*this)->setAttr(getReassocAttrName(), attr);
}

void vector_reduce_fmul::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::BoolAttr reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  if (reassoc) {
  odsState.addAttribute(getReassocAttrName(odsState.name), reassoc);
  }
  odsState.addTypes(res);
}

void vector_reduce_fmul::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, ::mlir::BoolAttr reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  if (reassoc) {
  odsState.addAttribute(getReassocAttrName(odsState.name), reassoc);
  }
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_fmul::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, bool reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addAttribute(getReassocAttrName(odsState.name), odsBuilder.getBoolAttr(reassoc));
  odsState.addTypes(res);
}

void vector_reduce_fmul::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0, ::mlir::Value odsArg_1, bool reassoc) {
  odsState.addOperands(odsArg_0);
  odsState.addOperands(odsArg_1);
  odsState.addAttribute(getReassocAttrName(odsState.name), odsBuilder.getBoolAttr(reassoc));
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_fmul::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_fmul::verifyInvariantsImpl() {
  auto namedAttrRange = (*this)->getAttrs();
  auto namedAttrIt = namedAttrRange.begin();
  ::mlir::Attribute tblgen_reassoc;
  while (true) {
    if (namedAttrIt == namedAttrRange.end()) {
      break;
    }
    else if (namedAttrIt->getName() == getReassocAttrName()) {
      tblgen_reassoc = namedAttrIt->getValue();
    }
    ++namedAttrIt;
  }

  if (::mlir::failed(__mlir_ods_local_attr_constraint_LLVMIntrinsicOps2(*this, tblgen_reassoc, "reassoc")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vector_reduce_fmul::verifyInvariants() {
  return verifyInvariantsImpl();
}

void vector_reduce_fmul::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vector_reduce_fmul)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_mul definitions
//===----------------------------------------------------------------------===//

vector_reduce_mulAdaptor::vector_reduce_mulAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vector.reduce.mul", odsAttrs.getContext());
}

vector_reduce_mulAdaptor::vector_reduce_mulAdaptor(vector_reduce_mul op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vector_reduce_mulAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_mulAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_mulAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_mulAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_mulAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vector_reduce_mul::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_mul::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_mul::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_mul::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_mul::getRes() {
  return *getODSResults(0).begin();
}

void vector_reduce_mul::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_mul::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_mul::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_mul::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vector_reduce_mul::verifyInvariants() {
  return verifyInvariantsImpl();
}

void vector_reduce_mul::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vector_reduce_mul)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_or definitions
//===----------------------------------------------------------------------===//

vector_reduce_orAdaptor::vector_reduce_orAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vector.reduce.or", odsAttrs.getContext());
}

vector_reduce_orAdaptor::vector_reduce_orAdaptor(vector_reduce_or op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vector_reduce_orAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_orAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_orAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_orAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_orAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vector_reduce_or::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_or::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_or::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_or::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_or::getRes() {
  return *getODSResults(0).begin();
}

void vector_reduce_or::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_or::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_or::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_or::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vector_reduce_or::verifyInvariants() {
  return verifyInvariantsImpl();
}

void vector_reduce_or::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vector_reduce_or)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_smax definitions
//===----------------------------------------------------------------------===//

vector_reduce_smaxAdaptor::vector_reduce_smaxAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vector.reduce.smax", odsAttrs.getContext());
}

vector_reduce_smaxAdaptor::vector_reduce_smaxAdaptor(vector_reduce_smax op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vector_reduce_smaxAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_smaxAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_smaxAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_smaxAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_smaxAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vector_reduce_smax::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_smax::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_smax::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_smax::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_smax::getRes() {
  return *getODSResults(0).begin();
}

void vector_reduce_smax::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_smax::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_smax::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_smax::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vector_reduce_smax::verifyInvariants() {
  return verifyInvariantsImpl();
}

void vector_reduce_smax::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vector_reduce_smax)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_smin definitions
//===----------------------------------------------------------------------===//

vector_reduce_sminAdaptor::vector_reduce_sminAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vector.reduce.smin", odsAttrs.getContext());
}

vector_reduce_sminAdaptor::vector_reduce_sminAdaptor(vector_reduce_smin op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vector_reduce_sminAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_sminAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_sminAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_sminAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_sminAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vector_reduce_smin::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_smin::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_smin::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_smin::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_smin::getRes() {
  return *getODSResults(0).begin();
}

void vector_reduce_smin::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_smin::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_smin::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_smin::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vector_reduce_smin::verifyInvariants() {
  return verifyInvariantsImpl();
}

void vector_reduce_smin::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vector_reduce_smin)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_umax definitions
//===----------------------------------------------------------------------===//

vector_reduce_umaxAdaptor::vector_reduce_umaxAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vector.reduce.umax", odsAttrs.getContext());
}

vector_reduce_umaxAdaptor::vector_reduce_umaxAdaptor(vector_reduce_umax op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vector_reduce_umaxAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_umaxAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_umaxAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_umaxAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_umaxAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vector_reduce_umax::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_umax::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_umax::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_umax::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_umax::getRes() {
  return *getODSResults(0).begin();
}

void vector_reduce_umax::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_umax::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_umax::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_umax::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vector_reduce_umax::verifyInvariants() {
  return verifyInvariantsImpl();
}

void vector_reduce_umax::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vector_reduce_umax)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_umin definitions
//===----------------------------------------------------------------------===//

vector_reduce_uminAdaptor::vector_reduce_uminAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vector.reduce.umin", odsAttrs.getContext());
}

vector_reduce_uminAdaptor::vector_reduce_uminAdaptor(vector_reduce_umin op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vector_reduce_uminAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_uminAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_uminAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_uminAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_uminAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vector_reduce_umin::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_umin::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_umin::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_umin::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_umin::getRes() {
  return *getODSResults(0).begin();
}

void vector_reduce_umin::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_umin::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_umin::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_umin::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vector_reduce_umin::verifyInvariants() {
  return verifyInvariantsImpl();
}

void vector_reduce_umin::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vector_reduce_umin)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vector_reduce_xor definitions
//===----------------------------------------------------------------------===//

vector_reduce_xorAdaptor::vector_reduce_xorAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vector.reduce.xor", odsAttrs.getContext());
}

vector_reduce_xorAdaptor::vector_reduce_xorAdaptor(vector_reduce_xor op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vector_reduce_xorAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vector_reduce_xorAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vector_reduce_xorAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vector_reduce_xorAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vector_reduce_xorAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vector_reduce_xor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vector_reduce_xor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vector_reduce_xor::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vector_reduce_xor::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vector_reduce_xor::getRes() {
  return *getODSResults(0).begin();
}

void vector_reduce_xor::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  odsState.addTypes(res);
}

void vector_reduce_xor::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value odsArg_0) {
  odsState.addOperands(odsArg_0);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vector_reduce_xor::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vector_reduce_xor::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vector_reduce_xor::verifyInvariants() {
  return verifyInvariantsImpl();
}

void vector_reduce_xor::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vector_reduce_xor)

namespace mlir {
namespace LLVM {

//===----------------------------------------------------------------------===//
// ::mlir::LLVM::vscale definitions
//===----------------------------------------------------------------------===//

vscaleAdaptor::vscaleAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs, ::mlir::RegionRange regions) : odsOperands(values), odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("llvm.intr.vscale", odsAttrs.getContext());
}

vscaleAdaptor::vscaleAdaptor(vscale op) : odsOperands(op->getOperands()), odsAttrs(op->getAttrDictionary()), odsRegions(op->getRegions()), odsOpName(op->getName()) {}

::mlir::ValueRange vscaleAdaptor::getOperands() {
  return odsOperands;
}

std::pair<unsigned, unsigned> vscaleAdaptor::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::ValueRange vscaleAdaptor::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(odsOperands.begin(), valueRange.first),
           std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
}

::mlir::DictionaryAttr vscaleAdaptor::getAttributes() {
  return odsAttrs;
}

::mlir::LogicalResult vscaleAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> vscale::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range vscale::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> vscale::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range vscale::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Value vscale::getRes() {
  return *getODSResults(0).begin();
}

void vscale::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type res) {
  odsState.addTypes(res);
}

void vscale::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void vscale::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult vscale::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_LLVMIntrinsicOps0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult vscale::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace LLVM
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVM::vscale)


#endif  // GET_OP_CLASSES

