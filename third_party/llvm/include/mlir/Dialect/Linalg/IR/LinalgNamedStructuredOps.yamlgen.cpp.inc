
//===----------------------------------------------------------------------===//
// Implementation of CopyOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> CopyOp::getIteratorTypesArray() {
  int64_t rank = getRank(getDpsInitOperand(0));
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

ArrayAttr CopyOp::getIndexingMaps() {
  MLIRContext *context = getContext();
  AffineMap scalarMap = AffineMap::get(getNumParallelLoops(), 0, context);
  AffineMap tensorMap = AffineMap::getMultiDimIdentityMap(
    getNumParallelLoops(), context);
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand &opOperand : getOperation()->getOpOperands())
    indexingMaps.push_back(getRank(&opOperand) == 0 ? scalarMap : tensorMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

unsigned CopyOp::getNumRegionArgs() { return 2; }

std::string CopyOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void CopyOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(2 > 0 && block.getNumArguments() == 2 &&
         "CopyOp regionBuilder expects 2 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
TypeFn castVal = TypeFn::cast_signed;
auto castIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                              return attr.getName() == "cast"; });
if (castIter != attrs.end()) {
  if (auto attr = castIter->getValue().dyn_cast<TypeFnAttr>())
    castVal = attr.getValue();
}

  Value value1 = helper.buildTypeFn(castVal, block.getArgument(1).getType(), block.getArgument(0));
  yields.push_back(value1);
  helper.yieldOutputs(yields);
}

ParseResult CopyOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    CopyOp::getNumRegionArgs(), CopyOp::getRegionBuilder());
}
void CopyOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult CopyOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void CopyOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of ElemwiseUnaryOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> ElemwiseUnaryOp::getIteratorTypesArray() {
  int64_t rank = getRank(getDpsInitOperand(0));
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

ArrayAttr ElemwiseUnaryOp::getIndexingMaps() {
  MLIRContext *context = getContext();
  AffineMap scalarMap = AffineMap::get(getNumParallelLoops(), 0, context);
  AffineMap tensorMap = AffineMap::getMultiDimIdentityMap(
    getNumParallelLoops(), context);
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand &opOperand : getOperation()->getOpOperands())
    indexingMaps.push_back(getRank(&opOperand) == 0 ? scalarMap : tensorMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

unsigned ElemwiseUnaryOp::getNumRegionArgs() { return 2; }

std::string ElemwiseUnaryOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void ElemwiseUnaryOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(2 > 0 && block.getNumArguments() == 2 &&
         "ElemwiseUnaryOp regionBuilder expects 2 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
UnaryFn funVal = UnaryFn::exp;
auto funIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                              return attr.getName() == "fun"; });
if (funIter != attrs.end()) {
  if (auto attr = funIter->getValue().dyn_cast<UnaryFnAttr>())
    funVal = attr.getValue();
}

  
TypeFn castVal = TypeFn::cast_signed;
auto castIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                              return attr.getName() == "cast"; });
if (castIter != attrs.end()) {
  if (auto attr = castIter->getValue().dyn_cast<TypeFnAttr>())
    castVal = attr.getValue();
}

  Value value1 = helper.buildTypeFn(castVal, block.getArgument(1).getType(), block.getArgument(0));
  Value value2 = helper.buildUnaryFn(funVal, value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult ElemwiseUnaryOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    ElemwiseUnaryOp::getNumRegionArgs(), ElemwiseUnaryOp::getRegionBuilder());
}
void ElemwiseUnaryOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult ElemwiseUnaryOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void ElemwiseUnaryOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of ElemwiseBinaryOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> ElemwiseBinaryOp::getIteratorTypesArray() {
  int64_t rank = getRank(getDpsInitOperand(0));
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

ArrayAttr ElemwiseBinaryOp::getIndexingMaps() {
  MLIRContext *context = getContext();
  AffineMap scalarMap = AffineMap::get(getNumParallelLoops(), 0, context);
  AffineMap tensorMap = AffineMap::getMultiDimIdentityMap(
    getNumParallelLoops(), context);
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand &opOperand : getOperation()->getOpOperands())
    indexingMaps.push_back(getRank(&opOperand) == 0 ? scalarMap : tensorMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

unsigned ElemwiseBinaryOp::getNumRegionArgs() { return 3; }

std::string ElemwiseBinaryOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void ElemwiseBinaryOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "ElemwiseBinaryOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
BinaryFn funVal = BinaryFn::add;
auto funIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                              return attr.getName() == "fun"; });
if (funIter != attrs.end()) {
  if (auto attr = funIter->getValue().dyn_cast<BinaryFnAttr>())
    funVal = attr.getValue();
}

  
TypeFn castVal = TypeFn::cast_signed;
auto castIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                              return attr.getName() == "cast"; });
if (castIter != attrs.end()) {
  if (auto attr = castIter->getValue().dyn_cast<TypeFnAttr>())
    castVal = attr.getValue();
}

  Value value1 = helper.buildTypeFn(castVal, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(castVal, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(funVal, value1, value2);
  yields.push_back(value3);
  helper.yieldOutputs(yields);
}

ParseResult ElemwiseBinaryOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    ElemwiseBinaryOp::getNumRegionArgs(), ElemwiseBinaryOp::getRegionBuilder());
}
void ElemwiseBinaryOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult ElemwiseBinaryOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void ElemwiseBinaryOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of MatmulOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> MatmulOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(MatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

ArrayAttr MatmulOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned MatmulOp::getNumRegionArgs() { return 3; }

std::string MatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void MatmulOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "MatmulOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
TypeFn castVal = TypeFn::cast_signed;
auto castIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                              return attr.getName() == "cast"; });
if (castIter != attrs.end()) {
  if (auto attr = castIter->getValue().dyn_cast<TypeFnAttr>())
    castVal = attr.getValue();
}

  Value value1 = helper.buildTypeFn(castVal, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(castVal, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult MatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    MatmulOp::getNumRegionArgs(), MatmulOp::getRegionBuilder());
}
void MatmulOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult MatmulOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void MatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of MatmulUnsignedOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> MatmulUnsignedOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(MatmulUnsignedOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

ArrayAttr MatmulUnsignedOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned MatmulUnsignedOp::getNumRegionArgs() { return 3; }

std::string MatmulUnsignedOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void MatmulUnsignedOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "MatmulUnsignedOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_unsigned, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_unsigned, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult MatmulUnsignedOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    MatmulUnsignedOp::getNumRegionArgs(), MatmulUnsignedOp::getRegionBuilder());
}
void MatmulUnsignedOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult MatmulUnsignedOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void MatmulUnsignedOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of QuantizedMatmulOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> QuantizedMatmulOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(QuantizedMatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

ArrayAttr QuantizedMatmulOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned QuantizedMatmulOp::getNumRegionArgs() { return 5; }

std::string QuantizedMatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void QuantizedMatmulOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "QuantizedMatmulOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.buildBinaryFn(BinaryFn::sub, value1, value2);
  Value value4 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.buildBinaryFn(BinaryFn::sub, value4, value5);
  Value value7 = helper.buildBinaryFn(BinaryFn::mul, value3, value6);
  Value value8 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

ParseResult QuantizedMatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    QuantizedMatmulOp::getNumRegionArgs(), QuantizedMatmulOp::getRegionBuilder());
}
void QuantizedMatmulOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult QuantizedMatmulOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void QuantizedMatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of Mmt4DOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Mmt4DOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(Mmt4DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  exprs.push_back(getAffineSymbolExpr(4, context));
  exprs.push_back(getAffineSymbolExpr(5, context));
  return exprs;
}

ArrayAttr Mmt4DOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0, d2, d3, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d1, d2, d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Mmt4DOp::getNumRegionArgs() { return 3; }

std::string Mmt4DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void Mmt4DOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Mmt4DOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Mmt4DOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    Mmt4DOp::getNumRegionArgs(), Mmt4DOp::getRegionBuilder());
}
void Mmt4DOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Mmt4DOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void Mmt4DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of MatmulTransposeBOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> MatmulTransposeBOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(MatmulTransposeBOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

ArrayAttr MatmulTransposeBOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned MatmulTransposeBOp::getNumRegionArgs() { return 3; }

std::string MatmulTransposeBOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void MatmulTransposeBOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "MatmulTransposeBOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
TypeFn castVal = TypeFn::cast_signed;
auto castIter = llvm::find_if(attrs, [&](const NamedAttribute &attr) {
                              return attr.getName() == "cast"; });
if (castIter != attrs.end()) {
  if (auto attr = castIter->getValue().dyn_cast<TypeFnAttr>())
    castVal = attr.getValue();
}

  Value value1 = helper.buildTypeFn(castVal, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(castVal, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult MatmulTransposeBOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    MatmulTransposeBOp::getNumRegionArgs(), MatmulTransposeBOp::getRegionBuilder());
}
void MatmulTransposeBOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult MatmulTransposeBOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void MatmulTransposeBOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of BatchMatmulOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> BatchMatmulOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(BatchMatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  return exprs;
}

ArrayAttr BatchMatmulOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d3, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned BatchMatmulOp::getNumRegionArgs() { return 3; }

std::string BatchMatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void BatchMatmulOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "BatchMatmulOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult BatchMatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    BatchMatmulOp::getNumRegionArgs(), BatchMatmulOp::getRegionBuilder());
}
void BatchMatmulOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult BatchMatmulOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void BatchMatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of QuantizedBatchMatmulOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> QuantizedBatchMatmulOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(QuantizedBatchMatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  return exprs;
}

ArrayAttr QuantizedBatchMatmulOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d3, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned QuantizedBatchMatmulOp::getNumRegionArgs() { return 5; }

std::string QuantizedBatchMatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void QuantizedBatchMatmulOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "QuantizedBatchMatmulOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.buildBinaryFn(BinaryFn::sub, value1, value2);
  Value value4 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.buildBinaryFn(BinaryFn::sub, value4, value5);
  Value value7 = helper.buildBinaryFn(BinaryFn::mul, value3, value6);
  Value value8 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

ParseResult QuantizedBatchMatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    QuantizedBatchMatmulOp::getNumRegionArgs(), QuantizedBatchMatmulOp::getRegionBuilder());
}
void QuantizedBatchMatmulOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult QuantizedBatchMatmulOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void QuantizedBatchMatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of BatchMatmulTransposeBOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> BatchMatmulTransposeBOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(BatchMatmulTransposeBOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  return exprs;
}

ArrayAttr BatchMatmulTransposeBOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned BatchMatmulTransposeBOp::getNumRegionArgs() { return 3; }

std::string BatchMatmulTransposeBOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void BatchMatmulTransposeBOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "BatchMatmulTransposeBOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult BatchMatmulTransposeBOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    BatchMatmulTransposeBOp::getNumRegionArgs(), BatchMatmulTransposeBOp::getRegionBuilder());
}
void BatchMatmulTransposeBOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult BatchMatmulTransposeBOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void BatchMatmulTransposeBOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of BatchReduceMatmulOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> BatchReduceMatmulOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::reduction, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(BatchReduceMatmulOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  return exprs;
}

ArrayAttr BatchReduceMatmulOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d3, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned BatchReduceMatmulOp::getNumRegionArgs() { return 3; }

std::string BatchReduceMatmulOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void BatchReduceMatmulOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "BatchReduceMatmulOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value2 = helper.buildBinaryFn(BinaryFn::mul, block.getArgument(0), value1);
  Value value3 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult BatchReduceMatmulOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    BatchReduceMatmulOp::getNumRegionArgs(), BatchReduceMatmulOp::getRegionBuilder());
}
void BatchReduceMatmulOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult BatchReduceMatmulOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void BatchReduceMatmulOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of MatvecOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> MatvecOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(MatvecOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr MatvecOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned MatvecOp::getNumRegionArgs() { return 3; }

std::string MatvecOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void MatvecOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "MatvecOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult MatvecOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    MatvecOp::getNumRegionArgs(), MatvecOp::getRegionBuilder());
}
void MatvecOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult MatvecOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void MatvecOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of VecmatOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> VecmatOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(VecmatOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr VecmatOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1, d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned VecmatOp::getNumRegionArgs() { return 3; }

std::string VecmatOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void VecmatOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "VecmatOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult VecmatOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    VecmatOp::getNumRegionArgs(), VecmatOp::getRegionBuilder());
}
void VecmatOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult VecmatOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void VecmatOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of BatchMatvecOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> BatchMatvecOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(BatchMatvecOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  return exprs;
}

ArrayAttr BatchMatvecOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 3, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned BatchMatvecOp::getNumRegionArgs() { return 3; }

std::string BatchMatvecOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void BatchMatvecOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "BatchMatvecOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult BatchMatvecOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    BatchMatvecOp::getNumRegionArgs(), BatchMatvecOp::getRegionBuilder());
}
void BatchMatvecOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult BatchMatvecOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void BatchMatvecOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of DotOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> DotOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(DotOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  return exprs;
}

ArrayAttr DotOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0)[s0] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 1, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0)[s0] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 1, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0)[s0] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 1, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DotOp::getNumRegionArgs() { return 3; }

std::string DotOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void DotOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DotOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult DotOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    DotOp::getNumRegionArgs(), DotOp::getRegionBuilder());
}
void DotOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult DotOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void DotOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of Conv1DOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Conv1DOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(Conv1DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr Conv1DOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0 + d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv1DOp::getNumRegionArgs() { return 3; }

std::string Conv1DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void Conv1DOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv1DOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv1DOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    Conv1DOp::getNumRegionArgs(), Conv1DOp::getRegionBuilder());
}
void Conv1DOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Conv1DOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void Conv1DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Conv2DOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  return exprs;
}

ArrayAttr Conv2DOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 + d2, d1 + d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DOp::getNumRegionArgs() { return 3; }

std::string Conv2DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void Conv2DOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv2DOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv2DOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    Conv2DOp::getNumRegionArgs(), Conv2DOp::getRegionBuilder());
}
void Conv2DOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Conv2DOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void Conv2DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of Conv3DOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Conv3DOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(Conv3DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));
  exprs.push_back(getAffineSymbolExpr(4, context));
  exprs.push_back(getAffineSymbolExpr(5, context));
  return exprs;
}

ArrayAttr Conv3DOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0 + d3, d1 + d4, d2 + d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d3, d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv3DOp::getNumRegionArgs() { return 3; }

std::string Conv3DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void Conv3DOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv3DOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv3DOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    Conv3DOp::getNumRegionArgs(), Conv3DOp::getRegionBuilder());
}
void Conv3DOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Conv3DOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void Conv3DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of Conv1DNwcWcfOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Conv1DNwcWcfOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(Conv1DNwcWcfOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));
  exprs.push_back(getAffineSymbolExpr(6, context));
  return exprs;
}

ArrayAttr Conv1DNwcWcfOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d0, d1 * s2 + d3 * s4, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d3, d4, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv1DNwcWcfOp::getNumRegionArgs() { return 3; }

std::string Conv1DNwcWcfOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv1DNwcWcfOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv1DNwcWcfOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv1DNwcWcfOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv1DNwcWcfOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv1DNwcWcfOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    Conv1DNwcWcfOp::getNumRegionArgs(), Conv1DNwcWcfOp::getRegionBuilder());
}
void Conv1DNwcWcfOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Conv1DNwcWcfOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void Conv1DNwcWcfOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of Conv1DNcwFcwOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Conv1DNcwFcwOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(Conv1DNcwFcwOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));

int64_t cst3 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst3, context));

  exprs.push_back(getAffineSymbolExpr(4, context));

int64_t cst5 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst5, context));

  exprs.push_back(getAffineSymbolExpr(6, context));
  return exprs;
}

ArrayAttr Conv1DNcwFcwOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d0, d3, d2 * s3 + d4 * s5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d1, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv1DNcwFcwOp::getNumRegionArgs() { return 3; }

std::string Conv1DNcwFcwOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv1DNcwFcwOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv1DNcwFcwOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv1DNcwFcwOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv1DNcwFcwOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv1DNcwFcwOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    Conv1DNcwFcwOp::getNumRegionArgs(), Conv1DNcwFcwOp::getRegionBuilder());
}
void Conv1DNcwFcwOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Conv1DNcwFcwOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void Conv1DNcwFcwOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DNhwcHwcfOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Conv2DNhwcHwcfOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DNhwcHwcfOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr Conv2DNhwcHwcfOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d4, d5, d6, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DNhwcHwcfOp::getNumRegionArgs() { return 3; }

std::string Conv2DNhwcHwcfOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv2DNhwcHwcfOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv2DNhwcHwcfOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv2DNhwcHwcfOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv2DNhwcHwcfOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv2DNhwcHwcfOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    Conv2DNhwcHwcfOp::getNumRegionArgs(), Conv2DNhwcHwcfOp::getRegionBuilder());
}
void Conv2DNhwcHwcfOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Conv2DNhwcHwcfOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void Conv2DNhwcHwcfOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DNhwcFhwcOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Conv2DNhwcFhwcOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DNhwcFhwcOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr Conv2DNhwcFhwcOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d3, d4, d5, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DNhwcFhwcOp::getNumRegionArgs() { return 3; }

std::string Conv2DNhwcFhwcOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv2DNhwcFhwcOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv2DNhwcFhwcOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv2DNhwcFhwcOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv2DNhwcFhwcOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv2DNhwcFhwcOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    Conv2DNhwcFhwcOp::getNumRegionArgs(), Conv2DNhwcFhwcOp::getRegionBuilder());
}
void Conv2DNhwcFhwcOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Conv2DNhwcFhwcOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void Conv2DNhwcFhwcOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DNhwcHwcfQOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Conv2DNhwcHwcfQOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DNhwcHwcfQOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr Conv2DNhwcHwcfQOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d4, d5, d6, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DNhwcHwcfQOp::getNumRegionArgs() { return 5; }

std::string Conv2DNhwcHwcfQOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv2DNhwcHwcfQOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv2DNhwcHwcfQOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv2DNhwcHwcfQOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "Conv2DNhwcHwcfQOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.buildBinaryFn(BinaryFn::sub, value1, value2);
  Value value4 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.buildBinaryFn(BinaryFn::sub, value4, value5);
  Value value7 = helper.buildBinaryFn(BinaryFn::mul, value3, value6);
  Value value8 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

ParseResult Conv2DNhwcHwcfQOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    Conv2DNhwcHwcfQOp::getNumRegionArgs(), Conv2DNhwcHwcfQOp::getRegionBuilder());
}
void Conv2DNhwcHwcfQOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Conv2DNhwcHwcfQOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void Conv2DNhwcHwcfQOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DNchwFchwOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Conv2DNchwFchwOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DNchwFchwOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));

int64_t cst3 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst3, context));

  exprs.push_back(getAffineSymbolExpr(4, context));

int64_t cst5 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst5, context));

  exprs.push_back(getAffineSymbolExpr(6, context));

int64_t cst7 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst7, context));

  exprs.push_back(getAffineSymbolExpr(8, context));

int64_t cst9 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst9, context));

  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr Conv2DNchwFchwOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d4, d2 * s3 + d5 * s5, d3 * s7 + d6 * s9)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d1, d4, d5, d6)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DNchwFchwOp::getNumRegionArgs() { return 3; }

std::string Conv2DNchwFchwOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv2DNchwFchwOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv2DNchwFchwOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv2DNchwFchwOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv2DNchwFchwOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv2DNchwFchwOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    Conv2DNchwFchwOp::getNumRegionArgs(), Conv2DNchwFchwOp::getRegionBuilder());
}
void Conv2DNchwFchwOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Conv2DNchwFchwOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void Conv2DNchwFchwOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of Conv2DNgchwFgchwOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Conv2DNgchwFgchwOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(Conv2DNgchwFgchwOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));
  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));
  return exprs;
}

ArrayAttr Conv2DNgchwFgchwOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11] -> (d0, d1, d5, d3 * s4 + d6 * s6, d4 * s8 + d7 * s10)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11] -> (d1, d2, d5, d6, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv2DNgchwFgchwOp::getNumRegionArgs() { return 3; }

std::string Conv2DNgchwFgchwOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv2DNgchwFgchwOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv2DNgchwFgchwOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv2DNgchwFgchwOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv2DNgchwFgchwOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv2DNgchwFgchwOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    Conv2DNgchwFgchwOp::getNumRegionArgs(), Conv2DNgchwFgchwOp::getRegionBuilder());
}
void Conv2DNgchwFgchwOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Conv2DNgchwFgchwOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void Conv2DNgchwFgchwOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of Conv3DNdhwcDhwcfOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Conv3DNdhwcDhwcfOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(Conv3DNdhwcDhwcfOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.getStrides().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.getDilations().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  exprs.push_back(getAffineSymbolExpr(14, context));
  return exprs;
}

ArrayAttr Conv3DNdhwcDhwcfOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3 * s10 + d7 * s12, d8)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d5, d6, d7, d8, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv3DNdhwcDhwcfOp::getNumRegionArgs() { return 3; }

std::string Conv3DNdhwcDhwcfOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv3DNdhwcDhwcfOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv3DNdhwcDhwcfOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv3DNdhwcDhwcfOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "Conv3DNdhwcDhwcfOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult Conv3DNdhwcDhwcfOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    Conv3DNdhwcDhwcfOp::getNumRegionArgs(), Conv3DNdhwcDhwcfOp::getRegionBuilder());
}
void Conv3DNdhwcDhwcfOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Conv3DNdhwcDhwcfOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void Conv3DNdhwcDhwcfOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of Conv3DNdhwcDhwcfQOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> Conv3DNdhwcDhwcfQOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(Conv3DNdhwcDhwcfQOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.getStrides().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.getDilations().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  exprs.push_back(getAffineSymbolExpr(14, context));
  return exprs;
}

ArrayAttr Conv3DNdhwcDhwcfQOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3 * s10 + d7 * s12, d8)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d5, d6, d7, d8, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned Conv3DNdhwcDhwcfQOp::getNumRegionArgs() { return 5; }

std::string Conv3DNdhwcDhwcfQOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool Conv3DNdhwcDhwcfQOp::hasDynamicIndexingMaps() { return true; }
LogicalResult Conv3DNdhwcDhwcfQOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void Conv3DNdhwcDhwcfQOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "Conv3DNdhwcDhwcfQOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.buildBinaryFn(BinaryFn::sub, value1, value2);
  Value value4 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.buildBinaryFn(BinaryFn::sub, value4, value5);
  Value value7 = helper.buildBinaryFn(BinaryFn::mul, value3, value6);
  Value value8 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

ParseResult Conv3DNdhwcDhwcfQOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    Conv3DNdhwcDhwcfQOp::getNumRegionArgs(), Conv3DNdhwcDhwcfQOp::getRegionBuilder());
}
void Conv3DNdhwcDhwcfQOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult Conv3DNdhwcDhwcfQOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void Conv3DNdhwcDhwcfQOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv1DNwcWcOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> DepthwiseConv1DNwcWcOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv1DNwcWcOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));
  return exprs;
}

ArrayAttr DepthwiseConv1DNwcWcOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1 * s2 + d3 * s4, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d3, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv1DNwcWcOp::getNumRegionArgs() { return 3; }

std::string DepthwiseConv1DNwcWcOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv1DNwcWcOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv1DNwcWcOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv1DNwcWcOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DepthwiseConv1DNwcWcOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv1DNwcWcOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    DepthwiseConv1DNwcWcOp::getNumRegionArgs(), DepthwiseConv1DNwcWcOp::getRegionBuilder());
}
void DepthwiseConv1DNwcWcOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult DepthwiseConv1DNwcWcOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void DepthwiseConv1DNwcWcOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv1DNwcWcmOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> DepthwiseConv1DNwcWcmOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv1DNwcWcmOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));
  exprs.push_back(getAffineSymbolExpr(6, context));
  return exprs;
}

ArrayAttr DepthwiseConv1DNwcWcmOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d0, d1 * s2 + d4 * s4, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d4, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4)[s0, s1, s2, s3, s4, s5, s6] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 5, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv1DNwcWcmOp::getNumRegionArgs() { return 3; }

std::string DepthwiseConv1DNwcWcmOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv1DNwcWcmOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv1DNwcWcmOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv1DNwcWcmOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DepthwiseConv1DNwcWcmOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv1DNwcWcmOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    DepthwiseConv1DNwcWcmOp::getNumRegionArgs(), DepthwiseConv1DNwcWcmOp::getRegionBuilder());
}
void DepthwiseConv1DNwcWcmOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult DepthwiseConv1DNwcWcmOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void DepthwiseConv1DNwcWcmOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv2DNhwcHwcOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> DepthwiseConv2DNhwcHwcOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv2DNhwcHwcOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr DepthwiseConv2DNhwcHwcOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv2DNhwcHwcOp::getNumRegionArgs() { return 3; }

std::string DepthwiseConv2DNhwcHwcOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv2DNhwcHwcOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv2DNhwcHwcOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv2DNhwcHwcOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DepthwiseConv2DNhwcHwcOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv2DNhwcHwcOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    DepthwiseConv2DNhwcHwcOp::getNumRegionArgs(), DepthwiseConv2DNhwcHwcOp::getRegionBuilder());
}
void DepthwiseConv2DNhwcHwcOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult DepthwiseConv2DNhwcHwcOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void DepthwiseConv2DNhwcHwcOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv2DNchwChwOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> DepthwiseConv2DNchwChwOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv2DNchwChwOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));

int64_t cst3 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst3, context));

  exprs.push_back(getAffineSymbolExpr(4, context));

int64_t cst5 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst5, context));

  exprs.push_back(getAffineSymbolExpr(6, context));

int64_t cst7 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst7, context));

  exprs.push_back(getAffineSymbolExpr(8, context));

int64_t cst9 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst9, context));

  return exprs;
}

ArrayAttr DepthwiseConv2DNchwChwOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d3, d1 * s3 + d4 * s5, d2 * s7 + d5 * s9)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d3, d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d3, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv2DNchwChwOp::getNumRegionArgs() { return 3; }

std::string DepthwiseConv2DNchwChwOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv2DNchwChwOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv2DNchwChwOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv2DNchwChwOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DepthwiseConv2DNchwChwOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv2DNchwChwOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    DepthwiseConv2DNchwChwOp::getNumRegionArgs(), DepthwiseConv2DNchwChwOp::getRegionBuilder());
}
void DepthwiseConv2DNchwChwOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult DepthwiseConv2DNchwChwOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void DepthwiseConv2DNchwChwOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv2DNhwcHwcQOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> DepthwiseConv2DNhwcHwcQOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv2DNhwcHwcQOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr DepthwiseConv2DNhwcHwcQOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv2DNhwcHwcQOp::getNumRegionArgs() { return 5; }

std::string DepthwiseConv2DNhwcHwcQOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv2DNhwcHwcQOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv2DNhwcHwcQOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv2DNhwcHwcQOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "DepthwiseConv2DNhwcHwcQOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.buildBinaryFn(BinaryFn::sub, value1, value2);
  Value value4 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.buildBinaryFn(BinaryFn::sub, value4, value5);
  Value value7 = helper.buildBinaryFn(BinaryFn::mul, value3, value6);
  Value value8 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv2DNhwcHwcQOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    DepthwiseConv2DNhwcHwcQOp::getNumRegionArgs(), DepthwiseConv2DNhwcHwcQOp::getRegionBuilder());
}
void DepthwiseConv2DNhwcHwcQOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult DepthwiseConv2DNhwcHwcQOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void DepthwiseConv2DNhwcHwcQOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv2DNhwcHwcmOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> DepthwiseConv2DNhwcHwcmOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv2DNhwcHwcmOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr DepthwiseConv2DNhwcHwcmOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d5, d6, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv2DNhwcHwcmOp::getNumRegionArgs() { return 3; }

std::string DepthwiseConv2DNhwcHwcmOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv2DNhwcHwcmOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv2DNhwcHwcmOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv2DNhwcHwcmOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DepthwiseConv2DNhwcHwcmOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv2DNhwcHwcmOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    DepthwiseConv2DNhwcHwcmOp::getNumRegionArgs(), DepthwiseConv2DNhwcHwcmOp::getRegionBuilder());
}
void DepthwiseConv2DNhwcHwcmOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult DepthwiseConv2DNhwcHwcmOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void DepthwiseConv2DNhwcHwcmOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv2DNhwcHwcmQOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> DepthwiseConv2DNhwcHwcmQOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv2DNhwcHwcmQOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  exprs.push_back(getAffineSymbolExpr(10, context));
  return exprs;
}

ArrayAttr DepthwiseConv2DNhwcHwcmQOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d5, d6, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 7, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv2DNhwcHwcmQOp::getNumRegionArgs() { return 5; }

std::string DepthwiseConv2DNhwcHwcmQOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv2DNhwcHwcmQOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv2DNhwcHwcmQOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv2DNhwcHwcmQOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(5 > 0 && block.getNumArguments() == 5 &&
         "DepthwiseConv2DNhwcHwcmQOp regionBuilder expects 5 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(2));
  Value value3 = helper.buildBinaryFn(BinaryFn::sub, value1, value2);
  Value value4 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(1));
  Value value5 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(4).getType(), block.getArgument(3));
  Value value6 = helper.buildBinaryFn(BinaryFn::sub, value4, value5);
  Value value7 = helper.buildBinaryFn(BinaryFn::mul, value3, value6);
  Value value8 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(4), value7);
  yields.push_back(value8);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv2DNhwcHwcmQOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    DepthwiseConv2DNhwcHwcmQOp::getNumRegionArgs(), DepthwiseConv2DNhwcHwcmQOp::getRegionBuilder());
}
void DepthwiseConv2DNhwcHwcmQOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult DepthwiseConv2DNhwcHwcmQOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void DepthwiseConv2DNhwcHwcmQOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv3DNdhwcDhwcOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> DepthwiseConv3DNdhwcDhwcOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::parallel };
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv3DNdhwcDhwcOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.getStrides().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.getDilations().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  return exprs;
}

ArrayAttr DepthwiseConv3DNdhwcDhwcOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3 * s10 + d6 * s12, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d4, d5, d6, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1, d2, d3, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv3DNdhwcDhwcOp::getNumRegionArgs() { return 3; }

std::string DepthwiseConv3DNdhwcDhwcOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv3DNdhwcDhwcOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv3DNdhwcDhwcOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv3DNdhwcDhwcOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DepthwiseConv3DNdhwcDhwcOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv3DNdhwcDhwcOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    DepthwiseConv3DNdhwcDhwcOp::getNumRegionArgs(), DepthwiseConv3DNdhwcDhwcOp::getRegionBuilder());
}
void DepthwiseConv3DNdhwcDhwcOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult DepthwiseConv3DNdhwcDhwcOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void DepthwiseConv3DNdhwcDhwcOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of DepthwiseConv3DNdhwcDhwcmOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> DepthwiseConv3DNdhwcDhwcmOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::parallel };
}

static SmallVector<AffineExpr> getSymbolBindings(DepthwiseConv3DNdhwcDhwcmOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.getStrides().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.getDilations().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  exprs.push_back(getAffineSymbolExpr(14, context));
  return exprs;
}

ArrayAttr DepthwiseConv3DNdhwcDhwcmOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3 * s10 + d7 * s12, d8)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d5, d6, d7, d8, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14] -> (d0, d1, d2, d3, d8, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 9, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned DepthwiseConv3DNdhwcDhwcmOp::getNumRegionArgs() { return 3; }

std::string DepthwiseConv3DNdhwcDhwcmOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool DepthwiseConv3DNdhwcDhwcmOp::hasDynamicIndexingMaps() { return true; }
LogicalResult DepthwiseConv3DNdhwcDhwcmOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void DepthwiseConv3DNdhwcDhwcmOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "DepthwiseConv3DNdhwcDhwcmOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(1));
  Value value3 = helper.buildBinaryFn(BinaryFn::mul, value1, value2);
  Value value4 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value3);
  yields.push_back(value4);
  helper.yieldOutputs(yields);
}

ParseResult DepthwiseConv3DNdhwcDhwcmOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    DepthwiseConv3DNdhwcDhwcmOp::getNumRegionArgs(), DepthwiseConv3DNdhwcDhwcmOp::getRegionBuilder());
}
void DepthwiseConv3DNdhwcDhwcmOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult DepthwiseConv3DNdhwcDhwcmOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void DepthwiseConv3DNdhwcDhwcmOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNhwcSumOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNhwcSumOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNhwcSumOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr PoolingNhwcSumOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNhwcSumOp::getNumRegionArgs() { return 3; }

std::string PoolingNhwcSumOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNhwcSumOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNhwcSumOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNhwcSumOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNhwcSumOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNhwcSumOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNhwcSumOp::getNumRegionArgs(), PoolingNhwcSumOp::getRegionBuilder());
}
void PoolingNhwcSumOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNhwcSumOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNhwcSumOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNchwSumOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNchwSumOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNchwSumOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));

int64_t cst3 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst3, context));

  exprs.push_back(getAffineSymbolExpr(4, context));

int64_t cst5 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst5, context));

  exprs.push_back(getAffineSymbolExpr(6, context));

int64_t cst7 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst7, context));

  exprs.push_back(getAffineSymbolExpr(8, context));

int64_t cst9 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst9, context));

  return exprs;
}

ArrayAttr PoolingNchwSumOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2 * s3 + d4 * s5, d3 * s7 + d5 * s9)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNchwSumOp::getNumRegionArgs() { return 3; }

std::string PoolingNchwSumOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNchwSumOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNchwSumOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNchwSumOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNchwSumOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNchwSumOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNchwSumOp::getNumRegionArgs(), PoolingNchwSumOp::getRegionBuilder());
}
void PoolingNchwSumOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNchwSumOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNchwSumOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNhwcMaxOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNhwcMaxOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNhwcMaxOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr PoolingNhwcMaxOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNhwcMaxOp::getNumRegionArgs() { return 3; }

std::string PoolingNhwcMaxOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNhwcMaxOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNhwcMaxOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNhwcMaxOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNhwcMaxOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::max_signed, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNhwcMaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNhwcMaxOp::getNumRegionArgs(), PoolingNhwcMaxOp::getRegionBuilder());
}
void PoolingNhwcMaxOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNhwcMaxOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNhwcMaxOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNhwcMaxUnsignedOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNhwcMaxUnsignedOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNhwcMaxUnsignedOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr PoolingNhwcMaxUnsignedOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNhwcMaxUnsignedOp::getNumRegionArgs() { return 3; }

std::string PoolingNhwcMaxUnsignedOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNhwcMaxUnsignedOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNhwcMaxUnsignedOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNhwcMaxUnsignedOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNhwcMaxUnsignedOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_unsigned, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::max_unsigned, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNhwcMaxUnsignedOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNhwcMaxUnsignedOp::getNumRegionArgs(), PoolingNhwcMaxUnsignedOp::getRegionBuilder());
}
void PoolingNhwcMaxUnsignedOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNhwcMaxUnsignedOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNhwcMaxUnsignedOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNchwMaxOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNchwMaxOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNchwMaxOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));

int64_t cst3 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst3, context));

  exprs.push_back(getAffineSymbolExpr(4, context));

int64_t cst5 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst5, context));

  exprs.push_back(getAffineSymbolExpr(6, context));

int64_t cst7 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst7, context));

  exprs.push_back(getAffineSymbolExpr(8, context));

int64_t cst9 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst9, context));

  return exprs;
}

ArrayAttr PoolingNchwMaxOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2 * s3 + d4 * s5, d3 * s7 + d5 * s9)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNchwMaxOp::getNumRegionArgs() { return 3; }

std::string PoolingNchwMaxOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNchwMaxOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNchwMaxOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNchwMaxOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNchwMaxOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::max_signed, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNchwMaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNchwMaxOp::getNumRegionArgs(), PoolingNchwMaxOp::getRegionBuilder());
}
void PoolingNchwMaxOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNchwMaxOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNchwMaxOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNhwcMinOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNhwcMinOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNhwcMinOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr PoolingNhwcMinOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNhwcMinOp::getNumRegionArgs() { return 3; }

std::string PoolingNhwcMinOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNhwcMinOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNhwcMinOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNhwcMinOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNhwcMinOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::min_signed, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNhwcMinOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNhwcMinOp::getNumRegionArgs(), PoolingNhwcMinOp::getRegionBuilder());
}
void PoolingNhwcMinOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNhwcMinOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNhwcMinOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNhwcMinUnsignedOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNhwcMinUnsignedOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNhwcMinUnsignedOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));
  return exprs;
}

ArrayAttr PoolingNhwcMinUnsignedOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1 * s2 + d4 * s4, d2 * s6 + d5 * s8, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d4, d5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9] -> (d0, d1, d2, d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 6, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNhwcMinUnsignedOp::getNumRegionArgs() { return 3; }

std::string PoolingNhwcMinUnsignedOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNhwcMinUnsignedOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNhwcMinUnsignedOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 2 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNhwcMinUnsignedOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNhwcMinUnsignedOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_unsigned, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::min_unsigned, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNhwcMinUnsignedOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNhwcMinUnsignedOp::getNumRegionArgs(), PoolingNhwcMinUnsignedOp::getRegionBuilder());
}
void PoolingNhwcMinUnsignedOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNhwcMinUnsignedOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNhwcMinUnsignedOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNwcSumOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNwcSumOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNwcSumOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));
  return exprs;
}

ArrayAttr PoolingNwcSumOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1 * s2 + d3 * s4, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNwcSumOp::getNumRegionArgs() { return 3; }

std::string PoolingNwcSumOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNwcSumOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNwcSumOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNwcSumOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNwcSumOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNwcSumOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNwcSumOp::getNumRegionArgs(), PoolingNwcSumOp::getRegionBuilder());
}
void PoolingNwcSumOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNwcSumOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNwcSumOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNcwSumOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNcwSumOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNcwSumOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));

int64_t cst3 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst3, context));

  exprs.push_back(getAffineSymbolExpr(4, context));

int64_t cst5 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst5, context));

  return exprs;
}

ArrayAttr PoolingNcwSumOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2 * s3 + d3 * s5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNcwSumOp::getNumRegionArgs() { return 3; }

std::string PoolingNcwSumOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNcwSumOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNcwSumOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNcwSumOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNcwSumOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNcwSumOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNcwSumOp::getNumRegionArgs(), PoolingNcwSumOp::getRegionBuilder());
}
void PoolingNcwSumOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNcwSumOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNcwSumOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNwcMaxOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNwcMaxOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNwcMaxOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));
  return exprs;
}

ArrayAttr PoolingNwcMaxOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1 * s2 + d3 * s4, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNwcMaxOp::getNumRegionArgs() { return 3; }

std::string PoolingNwcMaxOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNwcMaxOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNwcMaxOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNwcMaxOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNwcMaxOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::max_signed, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNwcMaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNwcMaxOp::getNumRegionArgs(), PoolingNwcMaxOp::getRegionBuilder());
}
void PoolingNwcMaxOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNwcMaxOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNwcMaxOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNwcMaxUnsignedOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNwcMaxUnsignedOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNwcMaxUnsignedOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));
  return exprs;
}

ArrayAttr PoolingNwcMaxUnsignedOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1 * s2 + d3 * s4, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNwcMaxUnsignedOp::getNumRegionArgs() { return 3; }

std::string PoolingNwcMaxUnsignedOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNwcMaxUnsignedOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNwcMaxUnsignedOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNwcMaxUnsignedOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNwcMaxUnsignedOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_unsigned, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::max_unsigned, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNwcMaxUnsignedOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNwcMaxUnsignedOp::getNumRegionArgs(), PoolingNwcMaxUnsignedOp::getRegionBuilder());
}
void PoolingNwcMaxUnsignedOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNwcMaxUnsignedOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNwcMaxUnsignedOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNcwMaxOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNcwMaxOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNcwMaxOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  exprs.push_back(getAffineSymbolExpr(2, context));

int64_t cst3 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst3, context));

  exprs.push_back(getAffineSymbolExpr(4, context));

int64_t cst5 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst5, context));

  return exprs;
}

ArrayAttr PoolingNcwMaxOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2 * s3 + d3 * s5)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNcwMaxOp::getNumRegionArgs() { return 3; }

std::string PoolingNcwMaxOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNcwMaxOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNcwMaxOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNcwMaxOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNcwMaxOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::max_signed, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNcwMaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNcwMaxOp::getNumRegionArgs(), PoolingNcwMaxOp::getRegionBuilder());
}
void PoolingNcwMaxOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNcwMaxOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNcwMaxOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNwcMinOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNwcMinOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNwcMinOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));
  return exprs;
}

ArrayAttr PoolingNwcMinOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1 * s2 + d3 * s4, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNwcMinOp::getNumRegionArgs() { return 3; }

std::string PoolingNwcMinOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNwcMinOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNwcMinOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNwcMinOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNwcMinOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::min_signed, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNwcMinOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNwcMinOp::getNumRegionArgs(), PoolingNwcMinOp::getRegionBuilder());
}
void PoolingNwcMinOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNwcMinOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNwcMinOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNwcMinUnsignedOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNwcMinUnsignedOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNwcMinUnsignedOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));
  return exprs;
}

ArrayAttr PoolingNwcMinUnsignedOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1 * s2 + d3 * s4, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d3)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5] -> (d0, d1, d2)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 4, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNwcMinUnsignedOp::getNumRegionArgs() { return 3; }

std::string PoolingNwcMinUnsignedOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNwcMinUnsignedOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNwcMinUnsignedOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 1 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNwcMinUnsignedOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNwcMinUnsignedOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_unsigned, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::min_unsigned, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNwcMinUnsignedOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNwcMinUnsignedOp::getNumRegionArgs(), PoolingNwcMinUnsignedOp::getRegionBuilder());
}
void PoolingNwcMinUnsignedOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNwcMinUnsignedOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNwcMinUnsignedOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNdhwcSumOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNdhwcSumOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNdhwcSumOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.getStrides().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.getDilations().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  return exprs;
}

ArrayAttr PoolingNdhwcSumOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3 * s10 + d7 * s12, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d5, d6, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNdhwcSumOp::getNumRegionArgs() { return 3; }

std::string PoolingNdhwcSumOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNdhwcSumOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNdhwcSumOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNdhwcSumOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNdhwcSumOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::add, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNdhwcSumOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNdhwcSumOp::getNumRegionArgs(), PoolingNdhwcSumOp::getRegionBuilder());
}
void PoolingNdhwcSumOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNdhwcSumOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNdhwcSumOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNdhwcMaxOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNdhwcMaxOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNdhwcMaxOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.getStrides().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.getDilations().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  return exprs;
}

ArrayAttr PoolingNdhwcMaxOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3 * s10 + d7 * s12, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d5, d6, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNdhwcMaxOp::getNumRegionArgs() { return 3; }

std::string PoolingNdhwcMaxOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNdhwcMaxOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNdhwcMaxOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNdhwcMaxOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNdhwcMaxOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::max_signed, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNdhwcMaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNdhwcMaxOp::getNumRegionArgs(), PoolingNdhwcMaxOp::getRegionBuilder());
}
void PoolingNdhwcMaxOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNdhwcMaxOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNdhwcMaxOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of PoolingNdhwcMinOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> PoolingNdhwcMinOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::parallel, utils::IteratorType::reduction, utils::IteratorType::reduction, utils::IteratorType::reduction };
}

static SmallVector<AffineExpr> getSymbolBindings(PoolingNdhwcMinOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));

int64_t cst2 = self.getStrides().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst2, context));

  exprs.push_back(getAffineSymbolExpr(3, context));

int64_t cst4 = self.getDilations().getValues<int64_t>()[0];
exprs.push_back(getAffineConstantExpr(cst4, context));

  exprs.push_back(getAffineSymbolExpr(5, context));

int64_t cst6 = self.getStrides().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst6, context));

  exprs.push_back(getAffineSymbolExpr(7, context));

int64_t cst8 = self.getDilations().getValues<int64_t>()[1];
exprs.push_back(getAffineConstantExpr(cst8, context));

  exprs.push_back(getAffineSymbolExpr(9, context));

int64_t cst10 = self.getStrides().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst10, context));

  exprs.push_back(getAffineSymbolExpr(11, context));

int64_t cst12 = self.getDilations().getValues<int64_t>()[2];
exprs.push_back(getAffineConstantExpr(cst12, context));

  exprs.push_back(getAffineSymbolExpr(13, context));
  return exprs;
}

ArrayAttr PoolingNdhwcMinOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1 * s2 + d5 * s4, d2 * s6 + d6 * s8, d3 * s10 + d7 * s12, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d5, d6, d7)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1, d2, d3, d4, d5, d6, d7)[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13] -> (d0, d1, d2, d3, d4)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 8, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned PoolingNdhwcMinOp::getNumRegionArgs() { return 3; }

std::string PoolingNdhwcMinOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

bool PoolingNdhwcMinOp::hasDynamicIndexingMaps() { return true; }
LogicalResult PoolingNdhwcMinOp::verifyIndexingMapRequiredAttributes() {
  Operation *op = getOperation();
  
if (auto attr = op->getAttrOfType<DenseElementsAttr>("strides")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'strides'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'strides'");
}


if (auto attr = op->getAttrOfType<DenseElementsAttr>("dilations")) {
  if (!attr.getType().getElementType().isInteger(64))
    return op->emitError("incorrect element type for index attribute 'dilations'");
  if (attr.getType().getShape() != ArrayRef<int64_t>{ 3 })
    return op->emitError("incorrect shape for index attribute 'dilations'");
}

  return success();
}

void PoolingNdhwcMinOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(3 > 0 && block.getNumArguments() == 3 &&
         "PoolingNdhwcMinOp regionBuilder expects 3 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(2).getType(), block.getArgument(0));
  Value value2 = helper.buildBinaryFn(BinaryFn::min_signed, block.getArgument(2), value1);
  yields.push_back(value2);
  helper.yieldOutputs(yields);
}

ParseResult PoolingNdhwcMinOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    PoolingNdhwcMinOp::getNumRegionArgs(), PoolingNdhwcMinOp::getRegionBuilder());
}
void PoolingNdhwcMinOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult PoolingNdhwcMinOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void PoolingNdhwcMinOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of FillOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> FillOp::getIteratorTypesArray() {
  int64_t rank = getRank(getDpsInitOperand(0));
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

ArrayAttr FillOp::getIndexingMaps() {
  MLIRContext *context = getContext();
  AffineMap scalarMap = AffineMap::get(getNumParallelLoops(), 0, context);
  AffineMap tensorMap = AffineMap::getMultiDimIdentityMap(
    getNumParallelLoops(), context);
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand &opOperand : getOperation()->getOpOperands())
    indexingMaps.push_back(getRank(&opOperand) == 0 ? scalarMap : tensorMap);
  return Builder(getContext()).getAffineMapArrayAttr(indexingMaps);
}

unsigned FillOp::getNumRegionArgs() { return 2; }

std::string FillOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void FillOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(2 > 0 && block.getNumArguments() == 2 &&
         "FillOp regionBuilder expects 2 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(1).getType(), block.getArgument(0));
  yields.push_back(value1);
  helper.yieldOutputs(yields);
}

ParseResult FillOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    FillOp::getNumRegionArgs(), FillOp::getRegionBuilder());
}
void FillOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult FillOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void FillOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

//===----------------------------------------------------------------------===//
// Implementation of FillRng2DOp
//===----------------------------------------------------------------------===//

SmallVector<utils::IteratorType> FillRng2DOp::getIteratorTypesArray() {
  return SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel };
}

static SmallVector<AffineExpr> getSymbolBindings(FillRng2DOp self) {
  MLIRContext *context = self.getContext();
  SmallVector<AffineExpr> exprs;
  exprs.push_back(getAffineSymbolExpr(0, context));
  exprs.push_back(getAffineSymbolExpr(1, context));
  return exprs;
}

ArrayAttr FillRng2DOp::getIndexingMaps() {
  static const char memoizeAttr[] = "linalg.memoized_indexing_maps";
  ArrayAttr cached = getOperation()->getAttrOfType<ArrayAttr>(memoizeAttr);
  if (cached)
    return cached;

  MLIRContext *context = getContext();
  auto symbolBindings = getSymbolBindings(*this);
  SmallVector<AffineMap> maps;
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> ()>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  maps.push_back(mlir::parseAttribute("affine_map<(d0, d1)[s0, s1] -> (d0, d1)>", context).cast<AffineMapAttr>().getValue());
  maps.back() = simplifyAffineMap(maps.back().replaceDimsAndSymbols({}, symbolBindings, 2, 0));
  cached = Builder(context).getAffineMapArrayAttr(maps);
  getOperation()->setAttr(memoizeAttr, cached);
  return cached;
}

unsigned FillRng2DOp::getNumRegionArgs() { return 4; }

std::string FillRng2DOp::getLibraryCallName() {
  return generateLibraryCallName(getOperation());
}

void FillRng2DOp::regionBuilder(ImplicitLocOpBuilder &b,
                        Block &block, ArrayRef<NamedAttribute> attrs) {
  assert(4 > 0 && block.getNumArguments() == 4 &&
         "FillRng2DOp regionBuilder expects 4 (>=0) args");
  RegionBuilderHelper helper(block.getArgument(0).getContext(), block);
  SmallVector<Value> yields;
  
  Value value1 = helper.constant("2147483647 : i64");
  Value value2 = helper.buildTypeFn(TypeFn::cast_signed, helper.getFloat64Type(), value1);
  Value value3 = helper.index(1);
  Value value4 = helper.buildTypeFn(TypeFn::cast_signed, helper.getIntegerType(32), value3);
  Value value5 = helper.index(0);
  Value value6 = helper.buildTypeFn(TypeFn::cast_signed, helper.getIntegerType(32), value5);
  Value value7 = helper.buildBinaryFn(BinaryFn::add, value6, block.getArgument(2));
  Value value8 = helper.constant("1103515245 : i64");
  Value value9 = helper.buildTypeFn(TypeFn::cast_signed, helper.getIntegerType(32), value8);
  Value value10 = helper.buildBinaryFn(BinaryFn::mul, value7, value9);
  Value value11 = helper.constant("12345 : i64");
  Value value12 = helper.buildTypeFn(TypeFn::cast_signed, helper.getIntegerType(32), value11);
  Value value13 = helper.buildBinaryFn(BinaryFn::add, value10, value12);
  Value value14 = helper.buildBinaryFn(BinaryFn::add, value4, value13);
  Value value15 = helper.constant("1103515245 : i64");
  Value value16 = helper.buildTypeFn(TypeFn::cast_signed, helper.getIntegerType(32), value15);
  Value value17 = helper.buildBinaryFn(BinaryFn::mul, value14, value16);
  Value value18 = helper.constant("12345 : i64");
  Value value19 = helper.buildTypeFn(TypeFn::cast_signed, helper.getIntegerType(32), value18);
  Value value20 = helper.buildBinaryFn(BinaryFn::add, value17, value19);
  Value value21 = helper.buildTypeFn(TypeFn::cast_signed, helper.getFloat64Type(), value20);
  Value value22 = helper.buildBinaryFn(BinaryFn::add, value2, value21);
  Value value23 = helper.buildBinaryFn(BinaryFn::sub, block.getArgument(1), block.getArgument(0));
  Value value24 = helper.constant("2.3283063999999999E-10 : f64");
  Value value25 = helper.buildTypeFn(TypeFn::cast_signed, helper.getFloat64Type(), value24);
  Value value26 = helper.buildBinaryFn(BinaryFn::mul, value23, value25);
  Value value27 = helper.buildBinaryFn(BinaryFn::mul, value22, value26);
  Value value28 = helper.buildBinaryFn(BinaryFn::add, value27, block.getArgument(0));
  Value value29 = helper.buildTypeFn(TypeFn::cast_signed, block.getArgument(3).getType(), value28);
  yields.push_back(value29);
  helper.yieldOutputs(yields);
}

ParseResult FillRng2DOp::parse(OpAsmParser &parser, OperationState &result) {
  return ::parseNamedStructuredOp(parser, result,
    FillRng2DOp::getNumRegionArgs(), FillRng2DOp::getRegionBuilder());
}
void FillRng2DOp::print(OpAsmPrinter &p) {
  ::printNamedStructuredOp(p, getOperation(), getInputs(), getOutputs());
}

LogicalResult FillRng2DOp::fold(FoldAdaptor,
                        SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this);
}
void FillRng2DOp::getEffects(SmallVectorImpl<
    SideEffects::EffectInstance<MemoryEffects::Effect> >&effects) {
      if (hasTensorSemantics()) return;
      getGenericEffectsImpl(effects,
        getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}
