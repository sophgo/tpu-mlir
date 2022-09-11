/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace amdgpu {
class LDSBarrierOp;
} // namespace amdgpu
} // namespace mlir
namespace mlir {
namespace amdgpu {
class MFMAOp;
} // namespace amdgpu
} // namespace mlir
namespace mlir {
namespace amdgpu {
class RawBufferAtomicFaddOp;
} // namespace amdgpu
} // namespace mlir
namespace mlir {
namespace amdgpu {
class RawBufferLoadOp;
} // namespace amdgpu
} // namespace mlir
namespace mlir {
namespace amdgpu {
class RawBufferStoreOp;
} // namespace amdgpu
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::LDSBarrierOp declarations
//===----------------------------------------------------------------------===//

class LDSBarrierOpAdaptor {
public:
  LDSBarrierOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  LDSBarrierOpAdaptor(LDSBarrierOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class LDSBarrierOp : public ::mlir::Op<LDSBarrierOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = LDSBarrierOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("amdgpu.lds_barrier");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::LDSBarrierOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::MFMAOp declarations
//===----------------------------------------------------------------------===//

class MFMAOpAdaptor {
public:
  MFMAOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  MFMAOpAdaptor(MFMAOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getSourceA();
  ::mlir::Value getSourceB();
  ::mlir::Value getDestC();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::IntegerAttr getMAttr();
  uint32_t getM();
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
  ::mlir::IntegerAttr getKAttr();
  uint32_t getK();
  ::mlir::IntegerAttr getBlocksAttr();
  uint32_t getBlocks();
  ::mlir::IntegerAttr getCbszAttr();
  uint32_t getCbsz();
  ::mlir::IntegerAttr getAbidAttr();
  uint32_t getAbid();
  ::mlir::amdgpu::MFMAPermBAttr getBlgpAttr();
  ::mlir::amdgpu::MFMAPermB getBlgp();
  ::mlir::UnitAttr getReducePrecisionAttr();
  bool getReducePrecision();
  ::mlir::UnitAttr getNegateAAttr();
  bool getNegateA();
  ::mlir::UnitAttr getNegateBAttr();
  bool getNegateB();
  ::mlir::UnitAttr getNegateCAttr();
  bool getNegateC();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class MFMAOp : public ::mlir::Op<MFMAOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<3>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MFMAOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("abid"), ::llvm::StringRef("blgp"), ::llvm::StringRef("blocks"), ::llvm::StringRef("cbsz"), ::llvm::StringRef("k"), ::llvm::StringRef("m"), ::llvm::StringRef("n"), ::llvm::StringRef("negateA"), ::llvm::StringRef("negateB"), ::llvm::StringRef("negateC"), ::llvm::StringRef("reducePrecision")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getAbidAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getAbidAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getBlgpAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getBlgpAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getBlocksAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getBlocksAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getCbszAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getCbszAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  ::mlir::StringAttr getKAttrName() {
    return getAttributeNameForIndex(4);
  }

  static ::mlir::StringAttr getKAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 4);
  }

  ::mlir::StringAttr getMAttrName() {
    return getAttributeNameForIndex(5);
  }

  static ::mlir::StringAttr getMAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 5);
  }

  ::mlir::StringAttr getNAttrName() {
    return getAttributeNameForIndex(6);
  }

  static ::mlir::StringAttr getNAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 6);
  }

  ::mlir::StringAttr getNegateAAttrName() {
    return getAttributeNameForIndex(7);
  }

  static ::mlir::StringAttr getNegateAAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 7);
  }

  ::mlir::StringAttr getNegateBAttrName() {
    return getAttributeNameForIndex(8);
  }

  static ::mlir::StringAttr getNegateBAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 8);
  }

  ::mlir::StringAttr getNegateCAttrName() {
    return getAttributeNameForIndex(9);
  }

  static ::mlir::StringAttr getNegateCAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 9);
  }

  ::mlir::StringAttr getReducePrecisionAttrName() {
    return getAttributeNameForIndex(10);
  }

  static ::mlir::StringAttr getReducePrecisionAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 10);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("amdgpu.mfma");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getSourceA();
  ::mlir::Value getSourceB();
  ::mlir::Value getDestC();
  ::mlir::MutableOperandRange getSourceAMutable();
  ::mlir::MutableOperandRange getSourceBMutable();
  ::mlir::MutableOperandRange getDestCMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getDestD();
  ::mlir::IntegerAttr getMAttr();
  uint32_t getM();
  ::mlir::IntegerAttr getNAttr();
  uint32_t getN();
  ::mlir::IntegerAttr getKAttr();
  uint32_t getK();
  ::mlir::IntegerAttr getBlocksAttr();
  uint32_t getBlocks();
  ::mlir::IntegerAttr getCbszAttr();
  uint32_t getCbsz();
  ::mlir::IntegerAttr getAbidAttr();
  uint32_t getAbid();
  ::mlir::amdgpu::MFMAPermBAttr getBlgpAttr();
  ::mlir::amdgpu::MFMAPermB getBlgp();
  ::mlir::UnitAttr getReducePrecisionAttr();
  bool getReducePrecision();
  ::mlir::UnitAttr getNegateAAttr();
  bool getNegateA();
  ::mlir::UnitAttr getNegateBAttr();
  bool getNegateB();
  ::mlir::UnitAttr getNegateCAttr();
  bool getNegateC();
  void setMAttr(::mlir::IntegerAttr attr);
  void setNAttr(::mlir::IntegerAttr attr);
  void setKAttr(::mlir::IntegerAttr attr);
  void setBlocksAttr(::mlir::IntegerAttr attr);
  void setCbszAttr(::mlir::IntegerAttr attr);
  void setAbidAttr(::mlir::IntegerAttr attr);
  void setBlgpAttr(::mlir::amdgpu::MFMAPermBAttr attr);
  void setReducePrecisionAttr(::mlir::UnitAttr attr);
  void setNegateAAttr(::mlir::UnitAttr attr);
  void setNegateBAttr(::mlir::UnitAttr attr);
  void setNegateCAttr(::mlir::UnitAttr attr);
  ::mlir::Attribute removeReducePrecisionAttr();
  ::mlir::Attribute removeNegateAAttr();
  ::mlir::Attribute removeNegateBAttr();
  ::mlir::Attribute removeNegateCAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type destD, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::IntegerAttr blocks, ::mlir::Value sourceA, ::mlir::Value sourceB, ::mlir::Value destC, ::mlir::IntegerAttr cbsz, ::mlir::IntegerAttr abid, ::mlir::amdgpu::MFMAPermBAttr blgp, /*optional*/::mlir::UnitAttr reducePrecision, /*optional*/::mlir::UnitAttr negateA, /*optional*/::mlir::UnitAttr negateB, /*optional*/::mlir::UnitAttr negateC);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::IntegerAttr m, ::mlir::IntegerAttr n, ::mlir::IntegerAttr k, ::mlir::IntegerAttr blocks, ::mlir::Value sourceA, ::mlir::Value sourceB, ::mlir::Value destC, ::mlir::IntegerAttr cbsz, ::mlir::IntegerAttr abid, ::mlir::amdgpu::MFMAPermBAttr blgp, /*optional*/::mlir::UnitAttr reducePrecision, /*optional*/::mlir::UnitAttr negateA, /*optional*/::mlir::UnitAttr negateB, /*optional*/::mlir::UnitAttr negateC);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type destD, uint32_t m, uint32_t n, uint32_t k, uint32_t blocks, ::mlir::Value sourceA, ::mlir::Value sourceB, ::mlir::Value destC, uint32_t cbsz, uint32_t abid, ::mlir::amdgpu::MFMAPermB blgp, /*optional*/bool reducePrecision, /*optional*/bool negateA, /*optional*/bool negateB, /*optional*/bool negateC);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, uint32_t m, uint32_t n, uint32_t k, uint32_t blocks, ::mlir::Value sourceA, ::mlir::Value sourceB, ::mlir::Value destC, uint32_t cbsz, uint32_t abid, ::mlir::amdgpu::MFMAPermB blgp, /*optional*/bool reducePrecision, /*optional*/bool negateA, /*optional*/bool negateB, /*optional*/bool negateC);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 11 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::MFMAOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferAtomicFaddOp declarations
//===----------------------------------------------------------------------===//

class RawBufferAtomicFaddOpAdaptor {
public:
  RawBufferAtomicFaddOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  RawBufferAtomicFaddOpAdaptor(RawBufferAtomicFaddOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getValue();
  ::mlir::Value getMemref();
  ::mlir::ValueRange getIndices();
  ::mlir::Value getSgprOffset();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::llvm::Optional<uint32_t> getIndexOffset();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class RawBufferAtomicFaddOp : public ::mlir::Op<RawBufferAtomicFaddOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RawBufferAtomicFaddOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("boundsCheck"), ::llvm::StringRef("indexOffset"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBoundsCheckAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBoundsCheckAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexOffsetAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexOffsetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("amdgpu.raw_buffer_atomic_fadd");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::FloatType> getValue();
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::TypedValue<::mlir::IntegerType> getSgprOffset();
  ::mlir::MutableOperandRange getValueMutable();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  ::mlir::MutableOperandRange getSgprOffsetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::llvm::Optional<uint32_t> getIndexOffset();
  void setBoundsCheckAttr(::mlir::BoolAttr attr);
  void setIndexOffsetAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeIndexOffsetAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferAtomicFaddOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferLoadOp declarations
//===----------------------------------------------------------------------===//

class RawBufferLoadOpAdaptor {
public:
  RawBufferLoadOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  RawBufferLoadOpAdaptor(RawBufferLoadOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getMemref();
  ::mlir::ValueRange getIndices();
  ::mlir::Value getSgprOffset();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::llvm::Optional<uint32_t> getIndexOffset();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class RawBufferLoadOp : public ::mlir::Op<RawBufferLoadOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::Type>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<1>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RawBufferLoadOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("boundsCheck"), ::llvm::StringRef("indexOffset"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBoundsCheckAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBoundsCheckAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexOffsetAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexOffsetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("amdgpu.raw_buffer_load");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::TypedValue<::mlir::IntegerType> getSgprOffset();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  ::mlir::MutableOperandRange getSgprOffsetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Value getValue();
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::llvm::Optional<uint32_t> getIndexOffset();
  void setBoundsCheckAttr(::mlir::BoolAttr attr);
  void setIndexOffsetAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeIndexOffsetAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferLoadOp)

namespace mlir {
namespace amdgpu {

//===----------------------------------------------------------------------===//
// ::mlir::amdgpu::RawBufferStoreOp declarations
//===----------------------------------------------------------------------===//

class RawBufferStoreOpAdaptor {
public:
  RawBufferStoreOpAdaptor(::mlir::ValueRange values, ::mlir::DictionaryAttr attrs = nullptr, ::mlir::RegionRange regions = {});

  RawBufferStoreOpAdaptor(RawBufferStoreOp op);

  ::mlir::ValueRange getOperands();
  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::ValueRange getODSOperands(unsigned index);
  ::mlir::Value getValue();
  ::mlir::Value getMemref();
  ::mlir::ValueRange getIndices();
  ::mlir::Value getSgprOffset();
  ::mlir::DictionaryAttr getAttributes();
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::llvm::Optional<uint32_t> getIndexOffset();
  ::mlir::LogicalResult verify(::mlir::Location loc);
private:
  ::mlir::ValueRange odsOperands;
  ::mlir::DictionaryAttr odsAttrs;
  ::mlir::RegionRange odsRegions;
  ::llvm::Optional<::mlir::OperationName> odsOpName;
};
class RawBufferStoreOp : public ::mlir::Op<RawBufferStoreOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::AtLeastNOperands<2>::Impl, ::mlir::OpTrait::AttrSizedOperandSegments, ::mlir::OpTrait::OpInvariants, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = RawBufferStoreOpAdaptor;
public:
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("boundsCheck"), ::llvm::StringRef("indexOffset"), ::llvm::StringRef("operand_segment_sizes")};
    return ::llvm::makeArrayRef(attrNames);
  }

  ::mlir::StringAttr getBoundsCheckAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getBoundsCheckAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getIndexOffsetAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getIndexOffsetAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getOperandSegmentSizesAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getOperandSegmentSizesAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("amdgpu.raw_buffer_store");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Value getValue();
  ::mlir::TypedValue<::mlir::MemRefType> getMemref();
  ::mlir::Operation::operand_range getIndices();
  ::mlir::TypedValue<::mlir::IntegerType> getSgprOffset();
  ::mlir::MutableOperandRange getValueMutable();
  ::mlir::MutableOperandRange getMemrefMutable();
  ::mlir::MutableOperandRange getIndicesMutable();
  ::mlir::MutableOperandRange getSgprOffsetMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::BoolAttr getBoundsCheckAttr();
  bool getBoundsCheck();
  ::mlir::IntegerAttr getIndexOffsetAttr();
  ::llvm::Optional<uint32_t> getIndexOffset();
  void setBoundsCheckAttr(::mlir::BoolAttr attr);
  void setIndexOffsetAttr(::mlir::IntegerAttr attr);
  ::mlir::Attribute removeIndexOffsetAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, ::mlir::BoolAttr boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value value, ::mlir::Value memref, ::mlir::ValueRange indices, bool boundsCheck, /*optional*/::mlir::IntegerAttr indexOffset, /*optional*/::mlir::Value sgprOffset);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static void populateDefaultAttrs(const ::mlir::RegisteredOperationName &opName, ::mlir::NamedAttrList &attributes);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 3 && "invalid attribute index");
    return name.getRegisteredInfo()->getAttributeNames()[index];
  }

public:
};
} // namespace amdgpu
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::amdgpu::RawBufferStoreOp)


#endif  // GET_OP_CLASSES

