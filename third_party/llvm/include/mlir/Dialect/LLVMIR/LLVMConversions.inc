if (auto op = dyn_cast<::mlir::LLVM::AShrOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateAShr(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AddOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateAdd(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AddrSpaceCastOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateAddrSpaceCast(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AllocaOp>(opInst)) {

    auto addrSpace = moduleTranslation.convertType(op.getResult().getType())->getPointerAddressSpace();
    llvm::Type *elementType = moduleTranslation.convertType(
        op.getElemType() ? *op.getElemType()
                   : op.getType().cast<LLVMPointerType>().getElementType());
    auto *inst = builder.CreateAlloca(elementType, addrSpace, moduleTranslation.lookupValue(op.getArraySize()));
    
    if (op.getAlignment().hasValue()) {
      auto align = op.getAlignment().getValue();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    moduleTranslation.mapValue(op.getRes()) = inst;
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AndOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateAnd(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AtomicCmpXchgOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateAtomicCmpXchg(moduleTranslation.lookupValue(op.getPtr()), moduleTranslation.lookupValue(op.getCmp()), moduleTranslation.lookupValue(op.getVal()), llvm::MaybeAlign(),
                   getLLVMAtomicOrdering(op.getSuccessOrdering()),
                   getLLVMAtomicOrdering(op.getFailureOrdering()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::AtomicRMWOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateAtomicRMW(getLLVMAtomicBinOp(op.getBinOp()), moduleTranslation.lookupValue(op.getPtr()), moduleTranslation.lookupValue(op.getVal()),
                                   llvm::MaybeAlign(),
                                   getLLVMAtomicOrdering(op.getOrdering()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::BitcastOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateBitCast(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ConstantOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = getLLVMConstant(moduleTranslation.convertType(op.getResult().getType()), op.getValue(), opInst.getLoc(),
                                            moduleTranslation);
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExtractElementOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateExtractElement(moduleTranslation.lookupValue(op.getVector()), moduleTranslation.lookupValue(op.getPosition()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ExtractValueOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateExtractValue(moduleTranslation.lookupValue(op.getContainer()), extractPosition(op.getPosition()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FAddOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFAdd(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FCmpOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateFCmp(getLLVMCmpPredicate(op.getPredicate()), moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FDivOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFDiv(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FMulOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFMul(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FNegOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFNeg(moduleTranslation.lookupValue(op.getOperand()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPExtOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFPExt(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPToSIOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFPToSI(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPToUIOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFPToUI(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FPTruncOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFPTrunc(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FRemOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFRem(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FSubOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateFSub(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FenceOp>(opInst)) {

    llvm::LLVMContext &llvmContext = builder.getContext();
    builder.CreateFence(getLLVMAtomicOrdering(op.getOrdering()),
      llvmContext.getOrInsertSyncScopeID(op.getSyncscope()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::FreezeOp>(opInst)) {
builder.CreateFreeze(moduleTranslation.lookupValue(op.getVal()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::GEPOp>(opInst)) {

    SmallVector<llvm::Value *> indices;
    indices.reserve(op.getStructIndices().size());
    unsigned operandIdx = 0;
    for (int32_t structIndex : op.getStructIndices().getValues<int32_t>()) {
      if (structIndex == GEPOp::kDynamicIndex)
        indices.push_back(moduleTranslation.lookupValues(op.getIndices())[operandIdx++]);
      else
        indices.push_back(builder.getInt32(structIndex));
    }
    Type baseElementType = op.getSourceElementType();
    llvm::Type *elementType = moduleTranslation.convertType(baseElementType);
    moduleTranslation.mapValue(op.getRes()) = builder.CreateGEP(elementType, moduleTranslation.lookupValue(op.getBase()), indices);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ICmpOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateICmp(getLLVMCmpPredicate(op.getPredicate()), moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::InsertElementOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateInsertElement(moduleTranslation.lookupValue(op.getVector()), moduleTranslation.lookupValue(op.getValue()), moduleTranslation.lookupValue(op.getPosition()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::InsertValueOp>(opInst)) {

    moduleTranslation.mapValue(op.getRes()) = builder.CreateInsertValue(moduleTranslation.lookupValue(op.getContainer()), moduleTranslation.lookupValue(op.getValue()),
                                     extractPosition(op.getPosition()));
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::IntToPtrOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateIntToPtr(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LShrOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateLShr(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::LoadOp>(opInst)) {

    auto *inst = builder.CreateLoad(moduleTranslation.convertType(op.getResult().getType()), moduleTranslation.lookupValue(op.getAddr()), op.getVolatile_());
  
    if (op.getAlignment().hasValue()) {
      auto align = op.getAlignment().getValue();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    if (op.getNontemporal()) {
      llvm::Module *module = builder.GetInsertBlock()->getModule();
      llvm::MDNode *metadata = llvm::MDNode::get(
          inst->getContext(), llvm::ConstantAsMetadata::get(
              builder.getInt32(1)));
      inst->setMetadata(module->getMDKindID("nontemporal"), metadata);
    }
  
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  
    moduleTranslation.setAliasScopeMetadata(op, inst);
  
    moduleTranslation.mapValue(op.getRes()) = inst;
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::MulOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateMul(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::NullOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = llvm::ConstantPointerNull::get(    cast<llvm::PointerType>(moduleTranslation.convertType(op.getResult().getType())));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::OrOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateOr(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::PtrToIntOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreatePtrToInt(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ResumeOp>(opInst)) {
 builder.CreateResume(moduleTranslation.lookupValue(op.getValue())); 
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ReturnOp>(opInst)) {

    if (opInst.getNumOperands() != 0)
      builder.CreateRet(moduleTranslation.lookupValues(op.getArgs())[0]);
    else
      builder.CreateRetVoid();
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SDivOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateSDiv(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SExtOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateSExt(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SIToFPOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateSIToFP(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SRemOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateSRem(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SelectOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateSelect(moduleTranslation.lookupValue(op.getCondition()), moduleTranslation.lookupValue(op.getTrueValue()), moduleTranslation.lookupValue(op.getFalseValue()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ShlOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateShl(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ShuffleVectorOp>(opInst)) {

      SmallVector<unsigned, 4> position = extractPosition(op.getMask());
      SmallVector<int, 4> mask(position.begin(), position.end());
      moduleTranslation.mapValue(op.getRes()) = builder.CreateShuffleVector(moduleTranslation.lookupValue(op.getV1()), moduleTranslation.lookupValue(op.getV2()), mask);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::StoreOp>(opInst)) {

    auto *inst = builder.CreateStore(moduleTranslation.lookupValue(op.getValue()), moduleTranslation.lookupValue(op.getAddr()), op.getVolatile_());
  
    if (op.getAlignment().hasValue()) {
      auto align = op.getAlignment().getValue();
      if (align != 0)
        inst->setAlignment(llvm::Align(align));
    }
  
    if (op.getNontemporal()) {
      llvm::Module *module = builder.GetInsertBlock()->getModule();
      llvm::MDNode *metadata = llvm::MDNode::get(
          inst->getContext(), llvm::ConstantAsMetadata::get(
              builder.getInt32(1)));
      inst->setMetadata(module->getMDKindID("nontemporal"), metadata);
    }
  
    moduleTranslation.setAccessGroupsMetadata(op, inst);
  
    moduleTranslation.setAliasScopeMetadata(op, inst);
  
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::SubOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateSub(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::TruncOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateTrunc(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UDivOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateUDiv(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UIToFPOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateUIToFP(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::URemOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateURem(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UndefOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = llvm::UndefValue::get(moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::UnreachableOp>(opInst)) {
 builder.CreateUnreachable(); 
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::XOrOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateXor(moduleTranslation.lookupValue(op.getLhs()), moduleTranslation.lookupValue(op.getRhs()));
  return success();
}
if (auto op = dyn_cast<::mlir::LLVM::ZExtOp>(opInst)) {
moduleTranslation.mapValue(op.getRes()) = builder.CreateZExt(moduleTranslation.lookupValue(op.getArg()), moduleTranslation.convertType(op.getResult().getType()));
  return success();
}
